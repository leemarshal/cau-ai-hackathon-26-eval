from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


RUNTIME_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default).strip()
    if not raw:
        raise ConfigError(f"{name} must not be empty")
    return Path(raw).expanduser().resolve(strict=False)


def _executable_path(name: str, default: str) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    raw = os.environ.get(name, default).strip()
    if not raw:
        raise ConfigError(f"{name} must not be empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ConfigError(f"{name} must be an absolute path")
    # Resolving /opt/venv/bin/python3 to its base interpreter bypasses the
    # venv's pyvenv.cfg and site-packages.  absolute() normalizes the path
    # while deliberately preserving the executable symlink.
    return path.absolute()


def _integer(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return value


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be 0/1, true/false, yes/no, or on/off")


def _https_url(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value or any(character.isspace() for character in value):
        raise ConfigError(f"{name} must be a valid HTTPS URL")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ConfigError(f"{name} must be a valid HTTPS URL") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ConfigError(f"{name} must be an HTTPS URL without credentials or fragment")
    return value


def _gpus() -> tuple[int, ...]:
    raw = os.environ.get("TA_GPU_IDS", "1,2,3")
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigError("TA_GPU_IDS must be a comma-separated list of integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ConfigError("TA_GPU_IDS must contain distinct non-negative GPU IDs")
    if 0 in values:
        raise ConfigError("GPU 0 is reserved for manual TA work and cannot be automatic")
    if values != (1, 2, 3):
        raise ConfigError("TA_GPU_IDS must be exactly 1,2,3 for this server")
    return values


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mnt_root: Path
    admin_root: Path
    backup_root: Path
    state_root: Path
    database_path: Path
    grading_root: Path
    grade_script: Path
    grader_python: Path
    grader_runtime_id: str
    expected_team_count: int
    max_team_number: int
    poll_seconds: float
    stable_confirmations: int
    post_copy_seconds: float
    min_checkpoint_bytes: int
    max_checkpoint_bytes: int
    recursive_scan: bool
    worker_poll_seconds: float
    gpu_ids: tuple[int, ...]
    max_submissions_per_team: int = 30
    max_pending_captures: int = 6
    score_post_url: str = "https://api.minds.ai.kr/submit"
    score_post_timeout_seconds: float = 10.0
    score_post_retry_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        mnt_root = _path("TA_MNT_ROOT", "/mnt")
        admin_root = _path("TA_ADMIN_ROOT", "/mnt/Admin-Storage_7ed0d")
        backup_root = _path(
            "TA_BACKUP_ROOT", str(admin_root / "submission-backups")
        )
        state_root = _path(
            "TA_STATE_ROOT", str(Path.home() / ".local/state/hackathon-ta-grader")
        )
        database_path = _path(
            "TA_DATABASE_PATH", str(state_root / "grading.sqlite3")
        )
        grading_root = _path(
            "TA_GRADING_ROOT", str(Path.home() / "private-grading/assets")
        )
        grade_script = _path(
            "TA_GRADE_SCRIPT", str(project_root / "ops/grade-finalist.py")
        )
        grader_python = _executable_path("TA_GRADER_PYTHON", sys.executable)
        min_bytes = _integer("TA_MIN_CHECKPOINT_BYTES", 300_000_000, minimum=1)
        max_bytes = _integer(
            "TA_MAX_CHECKPOINT_BYTES", 512 * 1024 * 1024, minimum=1
        )
        if min_bytes >= max_bytes:
            raise ConfigError(
                "TA_MIN_CHECKPOINT_BYTES must be smaller than TA_MAX_CHECKPOINT_BYTES"
            )
        recursive_scan = _boolean("TA_RECURSIVE_SCAN", False)
        if recursive_scan:
            raise ConfigError(
                "TA_RECURSIVE_SCAN=1 is disabled: checkpoints must be direct Team children"
            )
        try:
            database_path.relative_to(mnt_root)
        except ValueError:
            pass
        else:
            raise ConfigError("SQLite database must be on local disk, not under TA_MNT_ROOT")
        for local_path, name in (
            (state_root, "TA_STATE_ROOT"),
            (grading_root, "TA_GRADING_ROOT"),
        ):
            try:
                local_path.relative_to(mnt_root)
            except ValueError:
                continue
            raise ConfigError(f"{name} must be on local disk, not under TA_MNT_ROOT")
        try:
            backup_root.relative_to(admin_root)
        except ValueError as exc:
            raise ConfigError("TA_BACKUP_ROOT must be inside TA_ADMIN_ROOT") from exc
        try:
            admin_root.relative_to(mnt_root)
        except ValueError as exc:
            raise ConfigError("TA_ADMIN_ROOT must be inside TA_MNT_ROOT") from exc
        grader_runtime_id = os.environ.get("TA_GRADER_RUNTIME_ID", "").strip()
        if grader_runtime_id and not RUNTIME_ID.fullmatch(grader_runtime_id):
            raise ConfigError("TA_GRADER_RUNTIME_ID must be sha256:<64 lowercase hex>")
        expected_team_count = _integer(
            "TA_EXPECTED_TEAM_COUNT", 22, minimum=1
        )
        max_team_number = _integer("TA_MAX_TEAM_NUMBER", 26, minimum=1)
        if max_team_number < expected_team_count:
            raise ConfigError(
                "TA_MAX_TEAM_NUMBER must be at least TA_EXPECTED_TEAM_COUNT"
            )
        return cls(
            project_root=project_root,
            mnt_root=mnt_root,
            admin_root=admin_root,
            backup_root=backup_root,
            state_root=state_root,
            database_path=database_path,
            grading_root=grading_root,
            grade_script=grade_script,
            grader_python=grader_python,
            grader_runtime_id=grader_runtime_id,
            expected_team_count=expected_team_count,
            max_team_number=max_team_number,
            poll_seconds=_float("TA_POLL_SECONDS", 20.0, minimum=1.0),
            stable_confirmations=_integer(
                "TA_STABLE_CONFIRMATIONS", 3, minimum=1
            ),
            post_copy_seconds=_float("TA_POST_COPY_SECONDS", 20.0, minimum=0.0),
            min_checkpoint_bytes=min_bytes,
            max_checkpoint_bytes=max_bytes,
            recursive_scan=recursive_scan,
            worker_poll_seconds=_float(
                "TA_WORKER_POLL_SECONDS", 2.0, minimum=0.1
            ),
            gpu_ids=_gpus(),
            max_submissions_per_team=_integer(
                "TA_MAX_SUBMISSIONS_PER_TEAM", 30, minimum=1
            ),
            max_pending_captures=_integer(
                "TA_MAX_PENDING_CAPTURES", 6, minimum=1
            ),
            score_post_url=_https_url(
                "TA_SCORE_POST_URL", "https://api.minds.ai.kr/submit"
            ),
            score_post_timeout_seconds=_float(
                "TA_SCORE_POST_TIMEOUT_SECONDS", 10.0, minimum=0.1
            ),
            score_post_retry_seconds=_float(
                "TA_SCORE_POST_RETRY_SECONDS", 60.0, minimum=1.0
            ),
        )
