from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default).strip()
    if not raw:
        raise ConfigError(f"{name} must not be empty")
    return Path(raw).expanduser().resolve(strict=False)


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
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be 0/1, true/false, yes/no, or on/off")


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
    grading_image: str
    expected_team_count: int
    poll_seconds: float
    stable_confirmations: int
    post_copy_seconds: float
    min_checkpoint_bytes: int
    max_checkpoint_bytes: int
    recursive_scan: bool
    worker_poll_seconds: float
    gpu_ids: tuple[int, ...]
    max_submissions_per_team: int = 10
    max_pending_captures: int = 6

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
            "TA_GRADE_SCRIPT", str(project_root / "ops/grade-finalist.sh")
        )
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
        grading_image = os.environ.get(
            "TA_GRADING_IMAGE", "hackathon/private-test-grader:2026.09"
        ).strip()
        if not grading_image:
            raise ConfigError("TA_GRADING_IMAGE must not be empty")
        return cls(
            project_root=project_root,
            mnt_root=mnt_root,
            admin_root=admin_root,
            backup_root=backup_root,
            state_root=state_root,
            database_path=database_path,
            grading_root=grading_root,
            grade_script=grade_script,
            grading_image=grading_image,
            expected_team_count=_integer("TA_EXPECTED_TEAM_COUNT", 22, minimum=1),
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
                "TA_MAX_SUBMISSIONS_PER_TEAM", 10, minimum=1
            ),
            max_pending_captures=_integer(
                "TA_MAX_PENDING_CAPTURES", 6, minimum=1
            ),
        )
