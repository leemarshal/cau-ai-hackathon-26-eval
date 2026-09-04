from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import re
import socket
import stat
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .config import Settings
from .database import Database, LostClaimError
from .metrics import validate_report
from .publish import publish_state
from . import SCORE_VERSION, TEST_DATASET_REVISION


LOGGER = logging.getLogger("ta-grader.worker")
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CLAIM_TOKEN = re.compile(r"[0-9a-f]{64}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("database contains an unsafe relative path")
    root = root.resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("database path traverses a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError("database path escapes the backup root") from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_marker(settings: Settings, row: dict) -> tuple[Path, Path]:
    marker_path = _safe_path(settings.backup_root, row["marker_relative_path"])
    if marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError("ready marker is missing or unsafe")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "state": "ready",
        "submission_id": row["id"],
        "team_name": row["team_name"],
        "submission_number": row["submission_number"],
        "model_name": row["model_name"],
        "size_bytes": row["source_size_bytes"],
        "sha256": row["source_sha256"],
        "artifact_relative_path": row["artifact_relative_path"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"ready marker field {key!r} does not match the database")
    checkpoint = _safe_path(settings.backup_root, row["artifact_relative_path"])
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError("backed-up checkpoint is missing or unsafe")
    file_stat = checkpoint.stat()
    if file_stat.st_size != row["source_size_bytes"]:
        raise RuntimeError("backed-up checkpoint size does not match the receipt")
    if checkpoint.name != f"{row['id']}.pt":
        raise RuntimeError("backed-up checkpoint name does not match submission UUID")
    if _sha256(checkpoint) != row["source_sha256"]:
        raise RuntimeError("backed-up checkpoint SHA-256 does not match the receipt")
    return marker_path, checkpoint


def _validate_audit(
    settings: Settings, audit_path: Path, report_path: Path, row: dict
) -> None:
    if audit_path.is_symlink() or not audit_path.is_file():
        raise RuntimeError("grading audit is missing or unsafe")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "submission_id",
        "original_checkpoint_sha256",
        "converted_safetensors_sha256",
        "final_report_sha256",
        "score_version",
        "test_dataset_revision",
        "grader_runtime_id",
    }
    if not isinstance(audit, dict) or set(audit) != expected_keys:
        raise RuntimeError("grading audit has an unexpected schema")
    expected = {
        "schema_version": "finalist-grading-audit-v3",
        "submission_id": row["id"],
        "original_checkpoint_sha256": row["source_sha256"],
        "final_report_sha256": _sha256(report_path),
        "score_version": SCORE_VERSION,
        "test_dataset_revision": TEST_DATASET_REVISION,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(f"grading audit field {key!r} is invalid")
    if not isinstance(audit["converted_safetensors_sha256"], str) or not LOWER_SHA256.fullmatch(
        audit["converted_safetensors_sha256"]
    ):
        raise RuntimeError("grading audit converted checkpoint SHA-256 is invalid")
    runtime_id = audit["grader_runtime_id"]
    if (
        not isinstance(runtime_id, str)
        or not runtime_id.startswith("sha256:")
        or not LOWER_SHA256.fullmatch(runtime_id.removeprefix("sha256:"))
    ):
        raise RuntimeError("grading audit runtime ID is invalid")
    if not settings.grader_runtime_id:
        raise RuntimeError("grader runtime is not pinned")
    if runtime_id != settings.grader_runtime_id:
        raise RuntimeError("grading audit runtime ID does not match the pinned runtime")


def _write_attempt_log(settings: Settings, row: dict, gpu: int, content: str) -> Path:
    log_root = (
        settings.backup_root
        / "logs"
        / row["team_name"]
        / f"submission-{row['submission_number']:04d}-{row['id']}"
    )
    log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = _claim_token(row)
    log_path = (
        log_root
        / f"attempt-{row['attempt_count']:03d}-{token[:12]}-gpu-{gpu}.log"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(log_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)
        if not content.endswith("\n"):
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return log_path


def _claim_token(row: dict) -> str:
    token = row.get("claim_token")
    if not isinstance(token, str) or not CLAIM_TOKEN.fullmatch(token):
        raise RuntimeError("database row has no valid claim token")
    return token


def _attempt_paths(checkpoint: Path, row: dict) -> tuple[Path, Path]:
    token = _claim_token(row)
    attempt_number = row.get("attempt_count")
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
    ):
        raise RuntimeError("database row has no valid attempt number")
    attempts_root = checkpoint.parent / "grading-attempts"
    attempts_root.mkdir(mode=0o700, exist_ok=True)
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise RuntimeError("grading attempts path is not a real directory")
    attempts_root.chmod(0o700)
    attempt_root = attempts_root / f"attempt-{attempt_number:03d}-{token}"
    attempt_root.mkdir(mode=0o700, exist_ok=True)
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise RuntimeError("grading attempt path is not a real directory")
    attempt_root.chmod(0o700)
    return attempt_root / "score.json", attempt_root / "score.audit.json"


@contextmanager
def _gpu_lifetime_lock(settings: Settings, gpu: int):
    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if settings.state_root.is_symlink() or not settings.state_root.is_dir():
        raise RuntimeError("worker state path is not a real directory")
    settings.state_root.chmod(0o700)
    lock_path = settings.state_root / f"worker-cuda-{gpu}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    lock = os.fdopen(descriptor, "a+b")
    try:
        metadata = os.fstat(lock.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("GPU worker lock is not a regular file")
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another worker already owns CUDA GPU {gpu}") from exc
        yield
    finally:
        lock.close()


def process_one(settings: Settings, database: Database, row: dict, gpu: int) -> None:
    report_path: Path | None = None
    audit_path: Path | None = None
    claim_token = _claim_token(row)
    try:
        _, checkpoint = _validate_marker(settings, row)
        report_path, audit_path = _attempt_paths(checkpoint, row)

        # Each claim owns a distinct immutable output directory. A partial or
        # invalid pair poisons only this attempt; retrying creates a fresh token
        # and directory without deleting forensic artifacts from the old one.
        if report_path.exists() or audit_path.exists():
            if not (report_path.is_file() and audit_path.is_file()):
                raise RuntimeError("only one member of the report/audit pair exists")
            metrics = validate_report(report_path, row["id"])
            _validate_audit(settings, audit_path, report_path, row)
        else:
            command = [
                str(settings.grader_python),
                str(settings.grade_script),
                "--expected-runtime-id",
                settings.grader_runtime_id,
                "--checkpoint",
                str(checkpoint),
                "--expected-sha256",
                row["source_sha256"],
                "--submission-id",
                row["id"],
                "--grading-root",
                str(settings.grading_root),
                "--report",
                str(report_path),
                "--gpu",
                str(gpu),
            ]
            started = utc_now()
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_text = (
                f"started_at={started}\nfinished_at={utc_now()}\n"
                f"worker_gpu={gpu}\nreturncode={result.returncode}\n"
                f"command={' '.join(command[:1])} [arguments redacted]\n\n"
                f"{result.stdout}"
            )
            _write_attempt_log(settings, row, gpu, log_text)
            if result.returncode != 0:
                raise RuntimeError(
                    f"grade-finalist exited {result.returncode}: {result.stdout[-2000:]}"
                )
            metrics = validate_report(report_path, row["id"])
            _validate_audit(settings, audit_path, report_path, row)

        report_relative = report_path.relative_to(settings.backup_root).as_posix()
        audit_relative = audit_path.relative_to(settings.backup_root).as_posix()
        database.mark_done(
            row["id"],
            claim_token,
            utc_now(),
            metrics,
            report_relative,
            audit_relative,
        )
        try:
            publish_state(settings, database)
        except Exception:
            LOGGER.exception("score was committed but Admin summary publication failed")
        LOGGER.info(
            "done team=%s submission=%04d model=%s gpu=%d "
            "CKA_f_o=%.6f CKA_r_o=%.6f AUS=%.6f RUS_o=%.6f final=%.6f",
            row["team_name"],
            row["submission_number"],
            row["model_name"],
            gpu,
            metrics["cka_f_o"],
            metrics["cka_r_o"],
            metrics["aus"],
            metrics["rus_o"],
            metrics["final_score"],
        )
    except LostClaimError:
        LOGGER.warning(
            "discarding output from lost claim team=%s submission=%04d gpu=%d",
            row["team_name"],
            row["submission_number"],
            gpu,
        )
    except Exception as exc:
        LOGGER.exception(
            "grading failed team=%s submission=%04d gpu=%d",
            row["team_name"],
            row["submission_number"],
            gpu,
        )
        try:
            database.mark_error(row["id"], claim_token, utc_now(), str(exc))
        except LostClaimError:
            LOGGER.warning(
                "discarding error from lost claim team=%s submission=%04d gpu=%d",
                row["team_name"],
                row["submission_number"],
                gpu,
            )
            return
        try:
            publish_state(settings, database)
        except Exception:
            LOGGER.exception("error was committed but Admin summary publication failed")


def worker_loop(settings: Settings, gpu: int, *, once: bool = False) -> bool:
    if gpu not in settings.gpu_ids:
        raise ValueError(f"GPU {gpu} is not in automatic worker set {settings.gpu_ids}")
    with _gpu_lifetime_lock(settings, gpu):
        database = Database(settings.database_path)
        database.initialize()
        worker_id = f"{socket.gethostname()}:{os.getpid()}:cuda-{gpu}"
        processed = False
        while True:
            row = database.claim_next(gpu, worker_id, utc_now())
            if row is None:
                if once:
                    return processed
                time.sleep(settings.worker_poll_seconds)
                continue
            processed = True
            process_one(settings, database, row, gpu)
            if once:
                return True
