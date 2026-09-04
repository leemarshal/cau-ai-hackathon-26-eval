#!/usr/bin/env python3
"""Convert and score one submission with the pinned native Python runtime.

The watcher/worker pipeline invokes this program with the physical CUDA device
number.  The scorer subprocess sees only that device through
``CUDA_VISIBLE_DEVICES`` and therefore always scores on ``cuda:0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence


AUDIT_SCHEMA_VERSION = "finalist-grading-audit-v3"
RUNTIME_SCHEMA_VERSION = "native-grader-runtime-v1"
SCORE_VERSION = "unlearning-v2"
PINNED_TEST_DATASET_REVISION = (
    "f7938fad4be1b9559433adf6f3edfab6088750ba003371de7c7505b5da05353b"
)
REQUIRED_TORCH_VERSION = "2.8.0"
REQUIRED_TORCHVISION_VERSION = "0.23.0"
REQUIRED_TORCH_CUDA = "12.8"
MAX_CONVERTED_BYTES = 512 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RUNTIME_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = PROJECT_ROOT / "grading_docker" / "convert_checkpoint.py"
SCORER = PROJECT_ROOT / "grading_docker" / "score_unlearning.py"
MODEL_CODE = PROJECT_ROOT / "grading_docker" / "imagenet_vit.py"
FINALIZER = PROJECT_ROOT / "ops" / "finalize-test-reference.py"
RUNTIME_CODE = (Path(__file__).resolve(), CONVERTER, SCORER, MODEL_CODE, FINALIZER)


class GradingError(RuntimeError):
    """Raised when a submission cannot be graded safely and completely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def runtime_info() -> dict[str, Any]:
    """Return and validate the exact native runtime used by grading."""

    try:
        import PIL
        import numpy
        import safetensors
        import timm
        import torch
        import torchvision  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - installation dependent
        raise GradingError(f"cannot import the native grader runtime: {exc}") from exc

    torch_version = str(torch.__version__)
    torchvision_version = str(torchvision.__version__)
    timm_version = str(timm.__version__)
    numpy_version = str(numpy.__version__)
    pillow_version = str(PIL.__version__)
    safetensors_version = str(getattr(safetensors, "__version__", ""))
    if not safetensors_version:
        raise GradingError("cannot determine the installed safetensors version")
    torch_cuda = str(torch.version.cuda) if torch.version.cuda is not None else None
    if _base_version(torch_version) != REQUIRED_TORCH_VERSION:
        raise GradingError(
            f"native grader requires torch {REQUIRED_TORCH_VERSION}, "
            f"found {torch_version}"
        )
    if _base_version(torchvision_version) != REQUIRED_TORCHVISION_VERSION:
        raise GradingError(
            f"native grader requires torchvision {REQUIRED_TORCHVISION_VERSION}, "
            f"found {torchvision_version}"
        )
    if torch_cuda != REQUIRED_TORCH_CUDA:
        raise GradingError(
            f"native grader requires torch CUDA {REQUIRED_TORCH_CUDA}, "
            f"found {torch_cuda!r}"
        )

    code_hashes: dict[str, str] = {}
    for path in RUNTIME_CODE:
        if path.is_symlink() or not path.is_file():
            raise GradingError(f"native grader code is missing or unsafe: {path}")
        code_hashes[path.relative_to(PROJECT_ROOT).as_posix()] = _sha256(path)

    python_version = platform.python_version()
    identity = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "python": python_version,
        "python_implementation": platform.python_implementation(),
        "torch": torch_version,
        "torchvision": torchvision_version,
        "torch_cuda": torch_cuda,
        "timm": timm_version,
        "numpy": numpy_version,
        "pillow": pillow_version,
        "safetensors": safetensors_version,
        "code_sha256": code_hashes,
    }
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_id": f"sha256:{_canonical_sha256(identity)}",
        "python": python_version,
        "python_implementation": platform.python_implementation(),
        "torch": torch_version,
        "torchvision": torchvision_version,
        "torch_cuda": torch_cuda,
        "timm": timm_version,
        "numpy": numpy_version,
        "pillow": pillow_version,
        "safetensors": safetensors_version,
        "code_sha256": code_hashes,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def _positive_timeout(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise GradingError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise GradingError(f"{name} must be a positive integer")
    return value


def _tail(path: Path, limit: int = 8 * 1024) -> str:
    with path.open("rb") as source:
        size = source.seek(0, os.SEEK_END)
        source.seek(max(0, size - limit))
        return source.read().decode("utf-8", errors="replace")


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_checked(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    environment: dict[str, str],
    cwd: Path,
    log_path: Path,
    description: str,
) -> None:
    with log_path.open("xb") as output:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        def forward_signal(signum: int, _frame: object) -> None:
            _terminate_group(process)
            raise GradingError(f"{description} interrupted by signal {signum}")

        previous_term = signal.signal(signal.SIGTERM, forward_signal)
        previous_int = signal.signal(signal.SIGINT, forward_signal)
        try:
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_group(process)
                raise GradingError(
                    f"{description} timed out after {timeout_seconds} seconds: "
                    f"{_tail(log_path)}"
                ) from exc
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
    if return_code != 0:
        raise GradingError(
            f"{description} exited with {return_code}: {_tail(log_path)}"
        )


def _child_environment(*, cuda_visible_devices: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _require_regular(path: Path, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GradingError(f"{description} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GradingError(f"{description} must be a regular non-symlink file")
    return metadata


def _require_real_directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GradingError(f"{description} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GradingError(f"{description} must be a real directory")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp(path: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_pair(
    staged_report: Path, report_path: Path, audit: dict[str, object]
) -> Path:
    audit_path = (
        report_path.with_suffix(".audit.json")
        if report_path.suffix == ".json"
        else Path(f"{report_path}.audit.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_real_directory(report_path.parent, "report directory")
    if report_path.exists() or report_path.is_symlink():
        raise GradingError(f"refusing to overwrite report: {report_path}")
    if audit_path.exists() or audit_path.is_symlink():
        raise GradingError(f"refusing to overwrite audit: {audit_path}")

    report_payload = staged_report.read_bytes()
    audit_payload = (
        json.dumps(audit, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    report_temporary = _write_temp(report_path, report_payload)
    audit_temporary = _write_temp(audit_path, audit_payload)
    report_published = False
    audit_published = False
    publication_complete = False
    try:
        os.link(report_temporary, report_path, follow_symlinks=False)
        report_published = True
        try:
            os.link(audit_temporary, audit_path, follow_symlinks=False)
            audit_published = True
        except BaseException:
            report_path.unlink(missing_ok=True)
            report_published = False
            raise
        _fsync_file(report_path)
        _fsync_file(audit_path)
        _fsync_directory(report_path.parent)
        publication_complete = True
    except FileExistsError as exc:
        raise GradingError("refusing to overwrite an existing grading result") from exc
    finally:
        if not publication_complete:
            if report_published:
                report_path.unlink(missing_ok=True)
            if audit_published:
                audit_path.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)
    return audit_path


def _validate_report(path: Path, submission_id: str) -> None:
    _require_regular(path, "scorer report")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GradingError(f"scorer report is invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise GradingError("scorer report must be a JSON object")
    expected = {
        "phase": "test",
        "tag": submission_id,
        "score_version": SCORE_VERSION,
        "dataset_revision": PINNED_TEST_DATASET_REVISION,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise GradingError(f"scorer report field {key!r} does not match its pin")


def grade(args: argparse.Namespace) -> tuple[Path, Path]:
    runtime = runtime_info()
    if not RUNTIME_ID_RE.fullmatch(args.expected_runtime_id):
        raise GradingError("expected runtime ID must be sha256:<64 lowercase hex>")
    if runtime["runtime_id"] != args.expected_runtime_id:
        raise GradingError(
            f"native grader runtime mismatch: expected {args.expected_runtime_id}, "
            f"found {runtime['runtime_id']}"
        )
    if not runtime["cuda_available"]:
        raise GradingError("CUDA is unavailable in the native grader runtime")
    if args.gpu < 0 or args.gpu >= runtime["cuda_device_count"]:
        raise GradingError(f"physical CUDA GPU {args.gpu} is unavailable")

    checkpoint = args.checkpoint.absolute()
    grading_root = args.grading_root.absolute()
    report_path = args.report.absolute()
    if not args.checkpoint.is_absolute() or not args.grading_root.is_absolute():
        raise GradingError("checkpoint and grading root paths must be absolute")
    if not args.report.is_absolute():
        raise GradingError("report path must be absolute")
    try:
        parsed_id = uuid.UUID(args.submission_id)
    except ValueError as exc:
        raise GradingError("submission ID must be a canonical lowercase UUID") from exc
    if str(parsed_id) != args.submission_id:
        raise GradingError("submission ID must be a canonical lowercase UUID")
    if checkpoint.name != f"{args.submission_id}.pt":
        raise GradingError("checkpoint basename must be <submission-id>.pt")
    if not SHA256_RE.fullmatch(args.expected_sha256):
        raise GradingError("expected checkpoint SHA-256 is invalid")

    checkpoint_metadata = _require_regular(checkpoint, "checkpoint")
    _require_real_directory(grading_root, "private grading root")
    if stat.S_IMODE(grading_root.stat().st_mode) & 0o077:
        raise GradingError("private grading root must have owner-only permissions")
    if _sha256(checkpoint) != args.expected_sha256:
        raise GradingError("checkpoint SHA-256 does not match the receipt")
    if checkpoint_metadata.st_size <= 0:
        raise GradingError("checkpoint is empty")

    verifier_timeout = _positive_timeout("FINALIST_VERIFIER_TIMEOUT_SECONDS", 300)
    conversion_timeout = _positive_timeout("FINALIST_CONVERSION_TIMEOUT_SECONDS", 360)
    scoring_timeout = _positive_timeout("FINALIST_SCORING_TIMEOUT_SECONDS", 1800)
    cpu_environment = _child_environment(cuda_visible_devices="-1")

    with tempfile.TemporaryDirectory(prefix="hackathon-finalist-") as raw_run_root:
        run_root = Path(raw_run_root)
        staged_checkpoint = run_root / "submission.pt"
        safe_checkpoint = run_root / "submission.safetensors"
        staged_report = run_root / "report.json"
        shutil.copyfile(checkpoint, staged_checkpoint, follow_symlinks=False)
        staged_checkpoint.chmod(0o400)
        if _sha256(staged_checkpoint) != args.expected_sha256:
            raise GradingError("locally staged checkpoint SHA-256 mismatch")

        verifier_command = [
            sys.executable,
            str(FINALIZER),
            "--grading-root",
            str(grading_root),
            "--verify-only",
        ]
        _run_checked(
            verifier_command,
            timeout_seconds=verifier_timeout,
            environment=cpu_environment,
            cwd=PROJECT_ROOT,
            log_path=run_root / "verify-before.log",
            description="private grading bundle verification",
        )

        _run_checked(
            [
                sys.executable,
                str(CONVERTER),
                "--input",
                str(staged_checkpoint),
                "--output",
                str(safe_checkpoint),
                "--max-bytes",
                str(MAX_CONVERTED_BYTES),
            ],
            timeout_seconds=conversion_timeout,
            environment=cpu_environment,
            cwd=PROJECT_ROOT,
            log_path=run_root / "convert.log",
            description="checkpoint conversion",
        )
        safe_metadata = _require_regular(safe_checkpoint, "converted checkpoint")
        if not 0 < safe_metadata.st_size <= MAX_CONVERTED_BYTES:
            raise GradingError("converted checkpoint has an invalid size")
        safe_sha256 = _sha256(safe_checkpoint)
        safe_checkpoint.chmod(0o400)

        scorer_environment = _child_environment(cuda_visible_devices=str(args.gpu))
        _run_checked(
            [
                sys.executable,
                str(SCORER),
                "score",
                "--phase",
                "test",
                "--split",
                str(grading_root / "splits" / "test_split.pt"),
                "--refs",
                str(grading_root / "score_cache" / "refs.pt"),
                "--image-root",
                str(grading_root / "imagenet_test"),
                "--ckpt",
                str(safe_checkpoint),
                "--mo-cache",
                str(grading_root / "score_cache" / "M_o__test.npz"),
                "--tag",
                args.submission_id,
                "--report",
                str(staged_report),
                "--device",
                "cuda:0",
            ],
            timeout_seconds=scoring_timeout,
            environment=scorer_environment,
            cwd=PROJECT_ROOT,
            log_path=run_root / "score.log",
            description="private test scoring",
        )
        _validate_report(staged_report, args.submission_id)
        if _sha256(safe_checkpoint) != safe_sha256:
            raise GradingError("converted checkpoint changed during scoring")

        _run_checked(
            verifier_command,
            timeout_seconds=verifier_timeout,
            environment=cpu_environment,
            cwd=PROJECT_ROOT,
            log_path=run_root / "verify-after.log",
            description="post-score private grading bundle verification",
        )

        report_sha256 = _sha256(staged_report)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "submission_id": args.submission_id,
            "original_checkpoint_sha256": args.expected_sha256,
            "converted_safetensors_sha256": safe_sha256,
            "final_report_sha256": report_sha256,
            "score_version": SCORE_VERSION,
            "test_dataset_revision": PINNED_TEST_DATASET_REVISION,
            "grader_runtime_id": args.expected_runtime_id,
        }
        audit_path = _publish_pair(staged_report, report_path, audit)

    return report_path, audit_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-info", action="store_true")
    parser.add_argument("--expected-runtime-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--submission-id")
    parser.add_argument("--grading-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gpu", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        grading_values = (
            args.expected_runtime_id,
            args.checkpoint,
            args.expected_sha256,
            args.submission_id,
            args.grading_root,
            args.report,
            args.gpu,
        )
        if args.runtime_info:
            if any(value is not None for value in grading_values):
                parser.error("--runtime-info cannot be combined with grading arguments")
            print(json.dumps(runtime_info(), sort_keys=True), flush=True)
            return 0
        if any(value is None for value in grading_values):
            parser.error(
                "grading requires --expected-runtime-id, --checkpoint, "
                "--expected-sha256, --submission-id, --grading-root, --report, "
                "and --gpu"
            )
        report_path, audit_path = grade(args)
        print(f"wrote {report_path}")
        print(f"wrote {audit_path}")
        return 0
    except (GradingError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"grade-finalist: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
