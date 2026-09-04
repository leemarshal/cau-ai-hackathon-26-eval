from __future__ import annotations

import csv
import errno
import fcntl
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path

from .config import Settings
from .database import Database


class PublishError(RuntimeError):
    """Raised when result files cannot be published safely."""


def _secure_makedirs(path: Path) -> Path:
    """Create *path* without following a symlink in any existing component."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    nofollow_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, nofollow_flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator won the race. The O_NOFOLLOW open
                    # below still verifies that it created a real directory.
                    pass
                try:
                    child = os.open(component, nofollow_flags, dir_fd=descriptor)
                except OSError as exc:
                    raise PublishError(
                        f"unsafe publish directory component: {absolute}"
                    ) from exc
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise PublishError(
                        f"refusing symlinked publish directory: {absolute}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return absolute


def _reject_symlink_target(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise PublishError(f"refusing symlinked publish target: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise PublishError(f"publish target is not a regular file: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, writer) -> None:
    path = _secure_makedirs(path.parent) / path.name
    _reject_symlink_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer(output)
            output.flush()
            os.fchmod(output.fileno(), 0o600)
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_cell(value):
    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" \t\r\n\ufeff")
    if value.startswith(("\t", "\r", "\n")) or candidate.startswith(
        ("=", "+", "-", "@")
    ):
        # CSV quoting does not disable formulas in spreadsheet programs. A
        # leading apostrophe forces the imported value to remain literal text.
        return "'" + value
    return value


def _snapshot_rows(snapshot: Path) -> list[dict]:
    connection = sqlite3.connect(snapshot)
    connection.row_factory = sqlite3.Row
    try:
        return [
            Database.row_for_json(dict(row))
            for row in connection.execute(
                "SELECT * FROM submissions "
                "ORDER BY team_number, submission_number"
            )
        ]
    finally:
        connection.close()


def _create_snapshot(database: Database, results_root: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".grading.sqlite3.", suffix=".tmp", dir=results_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source = sqlite3.connect(database.path, timeout=30)
        destination = sqlite3.connect(temporary)
        try:
            source.execute("PRAGMA busy_timeout=30000")
            source.backup(destination)
            destination.commit()
        finally:
            destination.close()
            source.close()
        temporary.chmod(0o600)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_state(settings: Settings, database: Database) -> None:
    settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = settings.state_root / "publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results_root = _secure_makedirs(settings.backup_root / "results")
        json_path = results_root / "submissions.json"
        csv_path = results_root / "submissions.csv"
        snapshot = results_root / "grading.sqlite3.snapshot"
        for target in (json_path, csv_path, snapshot):
            _reject_symlink_target(target)

        # Take one SQLite backup first, then derive every human-readable output
        # from that exact database image. Concurrent grading commits can only
        # appear in the next publication, never in just one of these files.
        temporary = _create_snapshot(database, results_root)
        try:
            rows = _snapshot_rows(temporary)
            _atomic_replace(
                json_path,
                lambda output: json.dump(
                    {"schema_version": 1, "submissions": rows},
                    output,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )

            columns = list(rows[0].keys()) if rows else list(
                Database.row_for_json({}).keys()
            )

            def write_csv(output) -> None:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(
                    {
                        column: _csv_cell(row.get(column))
                        for column in columns
                    }
                    for row in rows
                )

            _atomic_replace(csv_path, write_csv)

            _reject_symlink_target(snapshot)
            os.replace(temporary, snapshot)
            _fsync_directory(results_root)
        finally:
            temporary.unlink(missing_ok=True)
