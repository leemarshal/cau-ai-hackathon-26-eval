from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import SCORE_VERSION, TEST_DATASET_REVISION
from .config import Settings
from .database import Database, SubmissionLimitError
from .publish import publish_state
from .storage import (
    CheckpointSizeError,
    FileSignature,
    SourceChangedError,
    StagedCopy,
    StorageError,
    atomic_write_json_no_clobber,
    discover_teams,
    fsync_dir,
    iter_checkpoints,
    stage_copy,
    stat_regular,
    verify_staged_source,
)


LOGGER = logging.getLogger("ta-grader.watcher")
TEAM_NUMBER = re.compile(r"Team([1-9][0-9]*)_[A-Za-z0-9]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Observation:
    signature: FileSignature
    unchanged_confirmations: int = 0


@contextmanager
def watcher_lock(settings: Settings):
    """Prevent standalone/reconcile commands from racing the active watcher."""

    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if settings.state_root.is_symlink() or not settings.state_root.is_dir():
        raise RuntimeError("watcher state path is not a real directory")
    settings.state_root.chmod(0o700)
    lock_path = settings.state_root / "watcher.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    lock = os.fdopen(descriptor, "a+b")
    try:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise RuntimeError("watcher lock is not a regular file")
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another submission watcher is already running") from exc
        yield
    finally:
        lock.close()


class SubmissionWatcher:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.observations: dict[tuple[str, str], Observation] = {}
        # Pin directory identities before this process creates any backup
        # children.  If the network mount disappears and exposes a local
        # mountpoint, scans fail closed instead of writing into the host disk.
        self._storage_identities = {
            settings.mnt_root: self._directory_identity(settings.mnt_root),
            settings.admin_root: self._directory_identity(settings.admin_root),
        }

    @staticmethod
    def _directory_identity(path: Path) -> tuple[int, int]:
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"storage path must be a real directory: {path}")
        return metadata.st_dev, metadata.st_ino

    def _assert_storage_identity(self) -> None:
        for path, expected in self._storage_identities.items():
            try:
                actual = self._directory_identity(path)
            except OSError as exc:
                raise RuntimeError(f"storage path is unavailable: {path}: {exc}") from exc
            if actual != expected:
                raise RuntimeError(
                    f"storage mount identity changed; refusing all writes: {path}"
                )

    @property
    def incoming_root(self) -> Path:
        return self.settings.backup_root / ".incoming"

    @property
    def artifacts_root(self) -> Path:
        return self.settings.backup_root / "artifacts"

    @property
    def ready_root(self) -> Path:
        return self.settings.backup_root / "ready"

    def ensure_layout(self) -> None:
        self._assert_storage_identity()
        for path in (
            self.settings.backup_root,
            self.incoming_root,
            self.artifacts_root,
            self.ready_root,
            self.settings.backup_root / "logs",
            self.settings.backup_root / "results",
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"backup path must be a real directory: {path}")
            path.chmod(0o700)
        self._assert_storage_identity()

    @staticmethod
    def _read_json(path: Path) -> dict:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"JSON record is missing or unsafe: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"JSON record must be an object: {path}")
        return value

    @staticmethod
    def _relative_source(value: str) -> PurePosixPath:
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError("source_relative_path is unsafe")
        return relative

    def _relative_backup(self, path: Path) -> str:
        return path.relative_to(self.settings.backup_root).as_posix()

    def _existing_backup_path(self, relative_value: str) -> Path:
        relative = self._relative_source(relative_value)
        root = self.settings.backup_root.resolve(strict=True)
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError("backup record traverses a symlink")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        return resolved

    @staticmethod
    def _hash_file(path: Path) -> tuple[int, str]:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("O_NOFOLLOW is required for artifact verification")
        descriptor = os.open(
            path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            before = os.fstat(descriptor)
            path_before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"artifact is not a regular file: {path}")

            def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_mode,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            if identity(before) != identity(path_before):
                raise RuntimeError(f"artifact path changed before hashing: {path}")
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            path_after = path.lstat()
            if not (identity(before) == identity(after) == identity(path_after)):
                raise RuntimeError(f"artifact changed while hashing: {path}")
            return size, digest.hexdigest()
        finally:
            os.close(descriptor)

    def _validate_marker_shape(self, marker: dict, marker_path: Path) -> None:
        if marker.get("schema_version") != 1 or marker.get("state") != "ready":
            raise RuntimeError(f"invalid ready marker identity: {marker_path}")
        try:
            canonical = str(uuid.UUID(marker["submission_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid ready marker UUID: {marker_path}") from exc
        if canonical != marker["submission_id"] or marker_path.name != f"{canonical}.json":
            raise RuntimeError(f"ready marker UUID/path mismatch: {marker_path}")
        team_name = marker.get("team_name")
        if not isinstance(team_name, str):
            raise RuntimeError(f"ready marker team name is invalid: {marker_path}")
        match = TEAM_NUMBER.fullmatch(team_name)
        if match is None or int(match.group(1)) != marker.get("team_number"):
            raise RuntimeError(f"ready marker team identity mismatch: {marker_path}")
        if not 1 <= marker["team_number"] <= self.settings.max_team_number:
            raise RuntimeError(f"ready marker team number is out of range: {marker_path}")
        submission_number = marker.get("submission_number")
        if (
            isinstance(submission_number, bool)
            or not isinstance(submission_number, int)
            or submission_number < 1
            or submission_number > self.settings.max_submissions_per_team
        ):
            raise RuntimeError(f"ready marker submission number is invalid: {marker_path}")
        model_name = marker.get("model_name")
        if (
            not isinstance(model_name, str)
            or not model_name
            or model_name != PurePosixPath(model_name).name
            or not model_name.endswith(".pt")
        ):
            raise RuntimeError(f"ready marker model name is invalid: {marker_path}")
        source_relative = self._relative_source(marker.get("source_relative_path", ""))
        if source_relative.name != model_name:
            raise RuntimeError(f"ready marker source/model name mismatch: {marker_path}")
        size = marker.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= self.settings.min_checkpoint_bytes
            or size > self.settings.max_checkpoint_bytes
        ):
            raise RuntimeError(f"ready marker size is invalid: {marker_path}")
        if not isinstance(marker.get("sha256"), str) or not SHA256.fullmatch(
            marker["sha256"]
        ):
            raise RuntimeError(f"ready marker SHA-256 is invalid: {marker_path}")
        expected_directory = (
            self.artifacts_root
            / team_name
            / f"submission-{submission_number:04d}-{canonical}"
        )
        expected_artifact = self._relative_backup(expected_directory / f"{canonical}.pt")
        expected_receipt = self._relative_backup(expected_directory / "receipt.json")
        if marker.get("artifact_relative_path") != expected_artifact:
            raise RuntimeError(f"ready marker artifact path mismatch: {marker_path}")
        if marker.get("receipt_relative_path") != expected_receipt:
            raise RuntimeError(f"ready marker receipt path mismatch: {marker_path}")
        if marker.get("marker_relative_path") != self._relative_backup(marker_path):
            raise RuntimeError(f"ready marker self-path mismatch: {marker_path}")
        if marker.get("score_version") != SCORE_VERSION:
            raise RuntimeError(f"ready marker score version mismatch: {marker_path}")
        if marker.get("test_dataset_revision") != TEST_DATASET_REVISION:
            raise RuntimeError(f"ready marker dataset revision mismatch: {marker_path}")

    def _marker_artifact(self, marker: dict) -> Path:
        resolved = self._existing_backup_path(marker["artifact_relative_path"])
        if resolved.name != f"{marker['submission_id']}.pt":
            raise RuntimeError("marker artifact name does not match submission UUID")
        return resolved

    def _ingest_marker(self, marker_path: Path) -> bool:
        marker = self._read_json(marker_path)
        self._validate_marker_shape(marker, marker_path)
        if self.database.submission_exists(marker["submission_id"]):
            return self.database.ingest_marker(marker)
        artifact = self._marker_artifact(marker)
        size, digest = self._hash_file(artifact)
        if size != marker["size_bytes"] or digest != marker["sha256"]:
            raise RuntimeError(f"marker artifact size/SHA mismatch: {artifact}")
        return self.database.ingest_marker(marker)

    def _recover_orphan_receipts(self) -> None:
        for receipt_path in sorted(self.artifacts_root.glob("*/*/receipt.json")):
            try:
                receipt = self._read_json(receipt_path)
                if receipt.get("schema_version") != 1:
                    raise RuntimeError("receipt schema mismatch")
                submission_id = str(uuid.UUID(receipt["submission_id"]))
                if submission_id != receipt["submission_id"]:
                    raise RuntimeError("receipt submission UUID is not canonical")
                expected_receipt = self._relative_backup(receipt_path)
                if receipt.get("receipt_relative_path") != expected_receipt:
                    raise RuntimeError("receipt self-path mismatch")
                marker_path = self.ready_root / f"{submission_id}.json"
                if receipt.get("marker_relative_path") != self._relative_backup(marker_path):
                    raise RuntimeError("receipt marker path mismatch")
                if marker_path.exists() or marker_path.is_symlink():
                    continue
                artifact = self._existing_backup_path(receipt["artifact_relative_path"])
                size, digest = self._hash_file(artifact)
                if size != receipt["size_bytes"] or digest != receipt["sha256"]:
                    raise RuntimeError("orphan receipt artifact hash mismatch")
                marker = dict(receipt)
                marker["state"] = "ready"
                marker["ready_at"] = utc_now()
                self._validate_marker_shape(marker, marker_path)
                atomic_write_json_no_clobber(marker_path, marker)
                LOGGER.warning("recovered ready marker for %s", receipt["submission_id"])
            except Exception:
                LOGGER.exception("could not reconcile orphan receipt %s", receipt_path)

    def reconcile(self) -> int:
        self.ensure_layout()
        self._recover_orphan_receipts()
        inserted = 0
        for marker_path in sorted(self.ready_root.glob("*.json")):
            try:
                if self._ingest_marker(marker_path):
                    inserted += 1
            except Exception:
                LOGGER.exception("invalid ready marker %s", marker_path)
        return inserted

    def _pending_records(self) -> list[tuple[Path, dict]]:
        records: list[tuple[Path, dict]] = []
        for stage_dir in sorted(self.incoming_root.glob(".*.incoming")):
            capture_path = stage_dir / "capture.json"
            try:
                capture = self._read_json(capture_path)
                if capture.get("schema_version") != 1 or capture.get("state") != "captured":
                    raise RuntimeError("capture record identity mismatch")
                records.append((stage_dir, capture))
            except Exception:
                LOGGER.exception("invalid pending capture %s", stage_dir)
        return records

    def _reconstruct_staged(self, stage_dir: Path, capture: dict) -> StagedCopy:
        signature = FileSignature(**capture["source_signature"])
        submission_id = capture["submission_id"]
        return StagedCopy(
            stage_dir=stage_dir,
            checkpoint_path=stage_dir / f"{submission_id}.pt",
            source_signature=signature,
            sha256=capture["sha256"],
            size_bytes=capture["size_bytes"],
            captured_at=capture["captured_at"],
        )

    def _remove_stage(self, stage_dir: Path) -> None:
        try:
            stage_dir.relative_to(self.incoming_root)
        except ValueError as exc:
            raise RuntimeError("refusing to remove stage outside incoming root") from exc
        shutil.rmtree(stage_dir)
        fsync_dir(self.incoming_root)

    def _finalize_staged(self, staged: StagedCopy, capture: dict) -> None:
        self._assert_storage_identity()
        team_name = capture["team_name"]
        duplicate = self.database.find_team_sha(team_name, staged.sha256)
        if duplicate is not None:
            self.database.record_source_version(
                team_name=team_name,
                source_relative_path=capture["source_relative_path"],
                size_bytes=staged.size_bytes,
                mtime_ns=capture["source_signature"]["mtime_ns"],
                ctime_ns=capture["source_signature"]["ctime_ns"],
                sha256=staged.sha256,
                disposition="duplicate",
                submission_id=duplicate["id"],
            )
            LOGGER.info(
                "duplicate ignored team=%s model=%s existing_submission=%04d sha256=%s",
                team_name,
                capture["model_name"],
                duplicate["submission_number"],
                staged.sha256,
            )
            self._remove_stage(staged.stage_dir)
            return

        staged_receipt_path = staged.stage_dir / "receipt.json"
        if staged_receipt_path.exists() or staged_receipt_path.is_symlink():
            receipt = self._read_json(staged_receipt_path)
            for field, expected in (
                ("submission_id", capture["submission_id"]),
                ("team_name", team_name),
                ("model_name", capture["model_name"]),
                ("source_relative_path", capture["source_relative_path"]),
                ("size_bytes", staged.size_bytes),
                ("sha256", staged.sha256),
            ):
                if receipt.get(field) != expected:
                    raise RuntimeError(f"staged receipt field {field!r} mismatch")
            submission_number = receipt["submission_number"]
            if (
                isinstance(submission_number, bool)
                or not isinstance(submission_number, int)
                or submission_number < 1
                or submission_number > self.settings.max_submissions_per_team
            ):
                raise RuntimeError("staged receipt has invalid submission number")
            artifact_directory = (
                self.artifacts_root
                / team_name
                / f"submission-{submission_number:04d}-{capture['submission_id']}"
            )
            expected_artifact = self._relative_backup(
                artifact_directory / f"{capture['submission_id']}.pt"
            )
            expected_receipt = self._relative_backup(artifact_directory / "receipt.json")
            marker_path = self.ready_root / f"{capture['submission_id']}.json"
            if receipt.get("artifact_relative_path") != expected_artifact:
                raise RuntimeError("staged receipt has invalid artifact path")
            if receipt.get("receipt_relative_path") != expected_receipt:
                raise RuntimeError("staged receipt has invalid receipt path")
            if receipt.get("marker_relative_path") != self._relative_backup(marker_path):
                raise RuntimeError("staged receipt has invalid marker path")
        else:
            try:
                submission_number = self.database.allocate_submission_number(
                    team_name, self.settings.max_submissions_per_team
                )
            except SubmissionLimitError:
                self.database.record_source_version(
                    team_name=team_name,
                    source_relative_path=capture["source_relative_path"],
                    size_bytes=staged.size_bytes,
                    mtime_ns=capture["source_signature"]["mtime_ns"],
                    ctime_ns=capture["source_signature"]["ctime_ns"],
                    sha256=staged.sha256,
                    disposition="limit_exceeded",
                    submission_id=capture["submission_id"],
                )
                LOGGER.warning(
                    "submission limit reached; ignored team=%s model=%s limit=%d",
                    team_name,
                    capture["model_name"],
                    self.settings.max_submissions_per_team,
                )
                self._remove_stage(staged.stage_dir)
                return
            artifact_directory = (
                self.artifacts_root
                / team_name
                / f"submission-{submission_number:04d}-{capture['submission_id']}"
            )
            artifact_relative = (
                artifact_directory / f"{capture['submission_id']}.pt"
            ).relative_to(self.settings.backup_root).as_posix()
            receipt_relative = (artifact_directory / "receipt.json").relative_to(
                self.settings.backup_root
            ).as_posix()
            marker_path = self.ready_root / f"{capture['submission_id']}.json"
            marker_relative = self._relative_backup(marker_path)
            receipt = {
                "schema_version": 1,
                "state": "captured",
                "submission_id": capture["submission_id"],
                "team_name": team_name,
                "team_number": capture["team_number"],
                "submission_number": submission_number,
                "model_name": capture["model_name"],
                "source_relative_path": capture["source_relative_path"],
                "source_mtime_ns": capture["source_signature"]["mtime_ns"],
                "source_ctime_ns": capture["source_signature"]["ctime_ns"],
                "source_signature": capture["source_signature"],
                "size_bytes": staged.size_bytes,
                "sha256": staged.sha256,
                "captured_at": staged.captured_at,
                "ready_at": utc_now(),
                "artifact_relative_path": artifact_relative,
                "receipt_relative_path": receipt_relative,
                "marker_relative_path": marker_relative,
                "score_version": SCORE_VERSION,
                "test_dataset_revision": TEST_DATASET_REVISION,
            }
            atomic_write_json_no_clobber(staged_receipt_path, receipt)

        artifact_directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._assert_storage_identity()
        fsync_dir(staged.stage_dir)
        os.rename(staged.stage_dir, artifact_directory)
        fsync_dir(artifact_directory.parent)

        marker = dict(receipt)
        marker["state"] = "ready"
        atomic_write_json_no_clobber(marker_path, marker)
        capture_at_destination = artifact_directory / "capture.json"
        capture_at_destination.unlink(missing_ok=True)
        fsync_dir(artifact_directory)
        # Re-open and hash the artifact at its final path before the queue can
        # observe it.  This catches storage corruption between staging and the
        # ready-marker boundary as well as during crash recovery.
        if not self._ingest_marker(marker_path):
            raise RuntimeError("new ready marker could not be inserted into SQLite")
        publish_state(self.settings, self.database)
        LOGGER.info(
            "queued team=%s submission=%04d model=%s bytes=%d sha256=%s",
            team_name,
            submission_number,
            capture["model_name"],
            staged.size_bytes,
            staged.sha256,
        )

    def _verify_pending(self, teams: dict[str, Path]) -> set[tuple[str, str]]:
        pending_keys: set[tuple[str, str]] = set()
        for stage_dir, capture in self._pending_records():
            key = (capture.get("team_name", ""), capture.get("source_relative_path", ""))
            pending_keys.add(key)
            if time.time() - float(capture["captured_epoch"]) < self.settings.post_copy_seconds:
                continue
            try:
                team_root = teams[capture["team_name"]]
                relative = self._relative_source(capture["source_relative_path"])
                source = team_root.joinpath(*relative.parts)
                staged = self._reconstruct_staged(stage_dir, capture)
                if not verify_staged_source(source, staged):
                    LOGGER.warning(
                        "source changed after capture; discarding and observing again: %s",
                        source,
                    )
                    self._remove_stage(stage_dir)
                    self.observations.pop(key, None)
                    pending_keys.discard(key)
                    continue
                try:
                    staged_size, staged_digest = self._hash_file(
                        staged.checkpoint_path
                    )
                except (OSError, RuntimeError):
                    staged_size, staged_digest = -1, ""
                if (
                    staged_size != staged.size_bytes
                    or staged_digest != staged.sha256
                ):
                    LOGGER.warning(
                        "staged artifact changed after capture; discarding: %s",
                        staged.checkpoint_path,
                    )
                    self._remove_stage(stage_dir)
                    self.observations.pop(key, None)
                    pending_keys.discard(key)
                    continue
                self._finalize_staged(staged, capture)
                pending_keys.discard(key)
            except Exception:
                LOGGER.exception("pending checkpoint verification failed: %s", stage_dir)
                self.observations.pop(key, None)
        return pending_keys

    def _capture(self, team_name: str, team_root: Path, source: Path) -> None:
        free_bytes = shutil.disk_usage(self.settings.backup_root).free
        reserve_bytes = self.settings.max_checkpoint_bytes * 2
        if free_bytes < reserve_bytes:
            raise StorageError(
                "Admin storage is below the two-checkpoint free-space reserve"
            )
        submission_id = str(uuid.uuid4())
        relative = source.relative_to(team_root).as_posix()
        staged = stage_copy(
            source,
            self.incoming_root,
            submission_id,
            self.settings.min_checkpoint_bytes,
            self.settings.max_checkpoint_bytes,
        )
        match = TEAM_NUMBER.match(team_name)
        if match is None:
            self._remove_stage(staged.stage_dir)
            raise RuntimeError(f"invalid team name after discovery: {team_name}")
        capture = {
            "schema_version": 1,
            "state": "captured",
            "submission_id": submission_id,
            "team_name": team_name,
            "team_number": int(match.group(1)),
            "model_name": source.name,
            "source_relative_path": relative,
            "source_signature": asdict(staged.source_signature),
            "size_bytes": staged.size_bytes,
            "sha256": staged.sha256,
            "captured_at": staged.captured_at,
            "captured_epoch": time.time(),
        }
        atomic_write_json_no_clobber(
            staged.stage_dir / "capture.json", capture, mode=0o600
        )
        LOGGER.info(
            "captured; waiting %.0fs for second source hash: %s/%s",
            self.settings.post_copy_seconds,
            team_name,
            relative,
        )

    def run_once(self) -> None:
        self.ensure_layout()
        self.reconcile()
        teams = discover_teams(
            self.settings.mnt_root,
            self.settings.expected_team_count,
            self.settings.max_team_number,
        )
        pending = self._verify_pending(teams)
        capture_budget = max(
            self.settings.max_pending_captures - len(pending), 0
        )
        captured_teams: set[str] = set()
        seen: set[tuple[str, str]] = set()

        for team_name, team_root in teams.items():
            for source in iter_checkpoints(
                team_root, recursive=self.settings.recursive_scan
            ):
                relative = source.relative_to(team_root).as_posix()
                key = (team_name, relative)
                seen.add(key)
                if key in pending:
                    continue
                try:
                    signature = stat_regular(source)
                except StorageError:
                    self.observations.pop(key, None)
                    continue
                if (
                    signature.size <= self.settings.min_checkpoint_bytes
                    or signature.size > self.settings.max_checkpoint_bytes
                ):
                    self.observations[key] = Observation(signature)
                    continue
                if self.database.has_source_version(
                    team_name,
                    relative,
                    signature.size,
                    signature.mtime_ns,
                    signature.ctime_ns,
                ):
                    self.observations.pop(key, None)
                    continue

                observation = self.observations.get(key)
                if observation is None or observation.signature != signature:
                    self.observations[key] = Observation(signature)
                    continue
                observation.unchanged_confirmations += 1
                if (
                    observation.unchanged_confirmations
                    < self.settings.stable_confirmations
                ):
                    continue
                if capture_budget <= 0 or team_name in captured_teams:
                    continue
                try:
                    self._capture(team_name, team_root, source)
                    capture_budget -= 1
                    captured_teams.add(team_name)
                    self.observations.pop(key, None)
                except (CheckpointSizeError, SourceChangedError, StorageError, OSError):
                    LOGGER.exception("checkpoint capture deferred: %s", source)
                    self.observations.pop(key, None)

        for key in set(self.observations).difference(seen):
            self.observations.pop(key, None)


def watcher_loop(settings: Settings, *, once: bool = False) -> None:
    with watcher_lock(settings):
        database = Database(settings.database_path)
        database.initialize()
        watcher = SubmissionWatcher(settings, database)
        while True:
            try:
                watcher.run_once()
            except Exception:
                LOGGER.exception("submission scan failed")
                if once:
                    raise
            if once:
                return
            time.sleep(settings.poll_seconds)
