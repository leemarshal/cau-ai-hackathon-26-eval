from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, Settings
from .database import Database
from .publish import publish_state
from .storage import discover_teams
from .storage import atomic_write_json_no_clobber, fsync_dir
from .watcher import SubmissionWatcher, watcher_lock, watcher_loop
from .worker import worker_loop


LOGGER = logging.getLogger("ta-grader")
LOWER_SHA256_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("TA_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _real_directory(path: Path, description: str) -> None:
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{description} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise RuntimeError(f"{description} must be a real directory: {path}")


def _resolve_grading_image(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", "--", image],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    image_id = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"grader image is unavailable: {image}: {result.stderr.strip()}"
        )
    if not LOWER_SHA256_IMAGE.fullmatch(image_id):
        raise RuntimeError(f"Docker returned an invalid grader image ID: {image_id!r}")
    return image_id


def _pin_grading_image(settings: Settings) -> Settings:
    """Keep one immutable grader image ID for the lifetime of this state DB."""

    image_id = _resolve_grading_image(settings.grading_image)
    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if settings.state_root.is_symlink() or not settings.state_root.is_dir():
        raise RuntimeError("TA state root must be a real local directory")
    settings.state_root.chmod(0o700)
    pin_path = settings.state_root / "grader-image.json"
    expected = {"schema_version": 1, "grader_image_id": image_id}
    try:
        atomic_write_json_no_clobber(pin_path, expected, mode=0o600)
    except FileExistsError:
        if pin_path.is_symlink() or not pin_path.is_file():
            raise RuntimeError("grader image pin is not a regular file")
        try:
            pinned = json.loads(pin_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("grader image pin is unreadable") from exc
        if pinned != expected:
            raise RuntimeError(
                "grader image ID changed for this grading database; "
                "restore the pinned image or use a deliberate new TA_STATE_ROOT"
            )
    return replace(settings, grading_image=image_id)


def check_environment(settings: Settings) -> dict:
    _real_directory(settings.mnt_root, "shared mount root")
    _real_directory(settings.admin_root, "Admin storage")
    admin_candidates = []
    for candidate in settings.mnt_root.glob("Admin-Storage_*"):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            admin_candidates.append(candidate.resolve(strict=True))
    if admin_candidates != [settings.admin_root.resolve(strict=True)]:
        raise RuntimeError(
            "exactly one Admin-Storage_* directory must exist and match TA_ADMIN_ROOT"
        )
    teams = discover_teams(settings.mnt_root, settings.expected_team_count)

    SubmissionWatcher(settings, Database(settings.database_path)).ensure_layout()
    probe_id = str(uuid.uuid4())
    probe_directory = settings.backup_root / f".storage-probe-{probe_id}"
    probe_directory.mkdir(mode=0o700)
    try:
        part = probe_directory / ".payload.part"
        final = probe_directory / "payload"
        with part.open("xb") as output:
            output.write(b"TA storage atomicity probe\n")
            output.flush()
            os.fchmod(output.fileno(), 0o600)
            os.fsync(output.fileno())
        os.rename(part, final)
        fsync_dir(probe_directory)
        marker = probe_directory / "ready.json"
        atomic_write_json_no_clobber(marker, {"probe_id": probe_id}, mode=0o600)
        if json.loads(marker.read_text(encoding="utf-8")) != {"probe_id": probe_id}:
            raise RuntimeError("Admin storage atomic marker probe returned wrong content")
    finally:
        shutil.rmtree(probe_directory, ignore_errors=True)
        fsync_dir(settings.backup_root)
    free_bytes = shutil.disk_usage(settings.backup_root).free
    if free_bytes < settings.max_checkpoint_bytes * 2:
        raise RuntimeError("Admin storage has less than two checkpoint slots free")

    if not settings.grade_script.is_file() or settings.grade_script.is_symlink():
        raise RuntimeError(f"grade script is missing or unsafe: {settings.grade_script}")
    if not os.access(settings.grade_script, os.X_OK):
        raise RuntimeError(f"grade script is not executable: {settings.grade_script}")
    _real_directory(settings.grading_root, "private grading root")
    for relative in (
        "splits/test_split.pt",
        "score_cache/refs.pt",
        "score_cache/M_o__test.npz",
        "imagenet_test",
    ):
        if not (settings.grading_root / relative).exists():
            raise RuntimeError(f"private grading asset is missing: {relative}")

    grading_image_id = _resolve_grading_image(settings.grading_image)
    gpu_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if gpu_check.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {gpu_check.stderr.strip()}")
    available = {
        int(line.strip())
        for line in gpu_check.stdout.splitlines()
        if line.strip().isdigit()
    }
    missing = {0, *settings.gpu_ids}.difference(available)
    if missing:
        raise RuntimeError(f"automatic worker GPUs are missing: {sorted(missing)}")
    return {
        "mnt_root": str(settings.mnt_root),
        "admin_root": str(settings.admin_root),
        "backup_root": str(settings.backup_root),
        "database_path": str(settings.database_path),
        "grading_root": str(settings.grading_root),
        "grading_image": settings.grading_image,
        "grading_image_id": grading_image_id,
        "teams": list(teams),
        "automatic_gpus": list(settings.gpu_ids),
        "reserved_gpus": [0],
    }


def _terminate_process_groups(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _prepare_supervised_child(supervisor_pid: int) -> None:
    """Start a process group and ask Linux to terminate it if its parent dies."""

    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        os._exit(125)
    # The parent can die between fork() and prctl().
    if os.getppid() != supervisor_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _install_child_group_signal_forwarding() -> None:
    if os.environ.get("TA_SUPERVISED_CHILD") != "1":
        return

    def forward(signum, _frame) -> None:
        signal.signal(signum, signal.SIG_DFL)
        try:
            os.killpg(os.getpgrp(), signum)
        except ProcessLookupError:
            os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)


def supervise(settings: Settings, *, skip_check: bool = False) -> int:
    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.state_root.chmod(0o700)
    lock_path = settings.state_root / "supervisor.lock"
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another TA grading supervisor is already running") from exc

    settings = _pin_grading_image(settings)
    database = Database(settings.database_path)
    database.initialize()
    if not skip_check:
        check_environment(settings)
    recovered = database.requeue_running()
    if recovered:
        LOGGER.warning("requeued %d interrupted grading jobs", recovered)
    publish_state(settings, database)

    commands = [[sys.executable, "-m", "ta_grading.cli", "watch"]]
    commands.extend(
        [sys.executable, "-m", "ta_grading.cli", "worker", "--gpu", str(gpu)]
        for gpu in settings.gpu_ids
    )
    children: list[subprocess.Popen] = []
    stopping = False
    child_failure = 0

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        child_environment = os.environ.copy()
        child_environment["TA_GRADING_IMAGE"] = settings.grading_image
        child_environment["TA_SUPERVISED_CHILD"] = "1"
        supervisor_pid = os.getpid()
        for command in commands:
            children.append(
                subprocess.Popen(
                    command,
                    cwd=settings.project_root,
                    env=child_environment,
                    preexec_fn=lambda: _prepare_supervised_child(supervisor_pid),
                    pass_fds=(lock.fileno(),),
                )
            )
        LOGGER.info(
            "started watcher and CUDA workers %s; GPU 0 remains reserved",
            settings.gpu_ids,
        )
        while not stopping:
            for child in children:
                return_code = child.poll()
                if return_code is not None:
                    LOGGER.error("pipeline child %d exited with %d", child.pid, return_code)
                    child_failure = return_code if return_code != 0 else 1
                    stopping = True
                    break
            if not stopping:
                time.sleep(1)
    finally:
        _terminate_process_groups(children)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        lock.close()
    return child_failure


def _status(settings: Settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    summary = database.summary()
    summary["submissions"] = [
        Database.row_for_json(row) for row in summary["submissions"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate mounts, private data, image, and GPUs")
    watch = commands.add_parser("watch", help="watch Team directories and enqueue backups")
    watch.add_argument("--once", action="store_true")
    worker = commands.add_parser("worker", help="run one automatic GPU worker")
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--once", action="store_true")
    run = commands.add_parser("run", help="run watcher plus GPU 1/2/3 workers")
    run.add_argument("--skip-check", action="store_true")
    commands.add_parser("status", help="print the local queue and scores as JSON")
    retry = commands.add_parser("retry", help="requeue an errored submission")
    retry.add_argument("submission_id")
    commands.add_parser("reconcile", help="rebuild queue entries from ready markers")
    return root


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        _install_child_group_signal_forwarding()
        database = Database(settings.database_path)
        database.initialize()
        if args.command == "check":
            print(json.dumps(check_environment(settings), indent=2, sort_keys=True))
        elif args.command == "watch":
            watcher_loop(settings, once=args.once)
        elif args.command == "worker":
            settings = _pin_grading_image(settings)
            worker_loop(settings, args.gpu, once=args.once)
        elif args.command == "run":
            return supervise(settings, skip_check=args.skip_check)
        elif args.command == "status":
            _status(settings)
        elif args.command == "retry":
            if not database.retry(args.submission_id):
                raise RuntimeError("submission is not in error state or does not exist")
            publish_state(settings, database)
        elif args.command == "reconcile":
            with watcher_lock(settings):
                watcher = SubmissionWatcher(settings, database)
                watcher.reconcile()
                publish_state(settings, database)
        return 0
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
