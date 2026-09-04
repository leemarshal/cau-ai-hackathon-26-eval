"""Filesystem primitives for conservative checkpoint intake.

The participant-facing tree is always treated as mutable.  A checkpoint only
becomes a staging artifact after a descriptor-based copy has observed the same
regular file before and after the read.  Callers are expected to wait and call
``verify_staged_source`` once more before publishing a ready marker.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


_TEAM_RE = re.compile(r"Team([1-9][0-9]*)_([A-Za-z0-9]+)\Z")
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


class StorageError(RuntimeError):
    """Base class for intake storage failures."""


class UnsafePathError(StorageError):
    """A path did not resolve to the expected real file or directory type."""


class SourceChangedError(StorageError):
    """The source changed while an immutable staging copy was being made."""


class CheckpointSizeError(StorageError):
    """A checkpoint was outside the configured byte limits."""


@dataclass(frozen=True, slots=True)
class FileSignature:
    dev: int
    ino: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class StagedCopy:
    stage_dir: Path
    checkpoint_path: Path
    source_signature: FileSignature
    sha256: str
    size_bytes: int
    captured_at: str


def _signature(metadata: os.stat_result) -> FileSignature:
    return FileSignature(
        dev=metadata.st_dev,
        ino=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _require_real_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"directory does not exist: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafePathError(f"path must be a real non-symlink directory: {path}")


def _canonical_submission_id(raw: str) -> str:
    try:
        canonical = str(uuid.UUID(raw))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("submission_id must be a UUID") from exc
    if raw != canonical:
        raise ValueError("submission_id must be a canonical lowercase UUID")
    return canonical


def _validate_limits(min_bytes: int, max_bytes: int) -> None:
    if (
        isinstance(min_bytes, bool)
        or isinstance(max_bytes, bool)
        or not isinstance(min_bytes, int)
        or not isinstance(max_bytes, int)
        or min_bytes < 0
        or max_bytes <= min_bytes
    ):
        raise ValueError("byte limits must satisfy 0 <= min_bytes < max_bytes")


def _validate_size(size: int, min_bytes: int, max_bytes: int) -> None:
    if size <= min_bytes:
        raise CheckpointSizeError(
            f"checkpoint must be larger than {min_bytes} bytes (got {size})"
        )
    if size > max_bytes:
        raise CheckpointSizeError(
            f"checkpoint exceeds {max_bytes} bytes (got {size})"
        )


def discover_teams(
    mnt_root: Path,
    expected_count: int,
    max_team_number: int | None = None,
) -> dict[str, Path]:
    """Return allowed direct ``Team<number>_<suffix>`` directories.

    Team numbers ``1..expected_count`` are required.  Higher team numbers are
    optional through ``max_team_number``; when no maximum is supplied, it
    defaults to ``expected_count`` and preserves the original exact-set
    behavior.  A matching name that is a symlink or non-directory is rejected
    rather than silently ignored, and duplicate numeric team IDs are rejected
    even when their suffixes differ.
    """

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    if max_team_number is None:
        max_team_number = expected_count
    elif (
        isinstance(max_team_number, bool)
        or not isinstance(max_team_number, int)
        or max_team_number < expected_count
    ):
        raise ValueError(
            "max_team_number must be an integer at least expected_count"
        )

    root = Path(mnt_root)
    _require_real_directory(root)
    by_number: dict[int, tuple[str, Path]] = {}
    invalid_team_entries: list[str] = []

    with os.scandir(root) as entries:
        for entry in entries:
            match = _TEAM_RE.fullmatch(entry.name)
            if match is None:
                if entry.name.startswith("Team"):
                    invalid_team_entries.append(entry.name)
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise StorageError(
                    f"team entry disappeared during discovery: {entry.name}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                invalid_team_entries.append(entry.name)
                continue
            number = int(match.group(1))
            if number in by_number:
                other = by_number[number][0]
                raise StorageError(
                    f"duplicate Team{number} directories: {other!r}, {entry.name!r}"
                )
            by_number[number] = (entry.name, root / entry.name)

    if invalid_team_entries:
        names = ", ".join(repr(name) for name in sorted(invalid_team_entries))
        raise StorageError(f"invalid or unsafe Team entries under {root}: {names}")

    required = set(range(1, expected_count + 1))
    actual = set(by_number)
    missing = sorted(required - actual)
    unexpected = sorted(number for number in actual if number > max_team_number)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise StorageError(
            f"team directories must include Team1..Team{expected_count} and "
            f"not exceed Team{max_team_number}: "
            + ", ".join(details)
        )

    return {
        name: path
        for _number, (name, path) in sorted(by_number.items(), key=lambda item: item[0])
    }


def iter_checkpoints(team_dir: Path, recursive: bool = False) -> Iterator[Path]:
    """Yield regular, non-symlink files whose suffix is exactly lowercase .pt."""

    root = Path(team_dir)
    _require_real_directory(root)

    if not recursive:
        candidates: list[Path] = []
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.endswith(".pt") or entry.name == ".pt":
                    continue
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    candidates.append(root / entry.name)
        yield from sorted(candidates, key=lambda path: path.name)
        return

    candidates = []
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        safe_directories: list[str] = []
        for name in directory_names:
            child = current / name
            try:
                metadata = child.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode):
                safe_directories.append(name)
        directory_names[:] = safe_directories

        for name in file_names:
            if not name.endswith(".pt") or name == ".pt":
                continue
            candidate = current / name
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                candidates.append(candidate)

    candidates.sort(key=lambda path: path.relative_to(root).as_posix())
    yield from candidates


def stat_regular(path: Path) -> FileSignature:
    """Return a no-follow signature for a regular file."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"file does not exist: {candidate}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafePathError(f"path must be a regular non-symlink file: {candidate}")
    return _signature(metadata)


def fsync_dir(path: Path) -> None:
    """Synchronize a real directory after an entry creation or rename."""

    directory = Path(path)
    _require_real_directory(directory)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_source(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required for checkpoint intake")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise UnsafePathError(f"cannot safely open checkpoint: {path}") from exc


def _regular_descriptor_signature(descriptor: int, path: Path) -> FileSignature:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafePathError(f"checkpoint descriptor is not a regular file: {path}")
    return _signature(metadata)


def _remove_failed_stage(stage_dir: Path, *known_files: Path) -> None:
    for path in known_files:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        stage_dir.rmdir()
    except OSError:
        pass


def stage_copy(
    source: Path,
    incoming_root: Path,
    submission_id: str,
    min_bytes: int,
    max_bytes: int,
) -> StagedCopy:
    """Copy a stable source into a private, uniquely named incoming directory.

    The source is opened with ``O_NOFOLLOW`` and is never renamed or deleted.
    The returned checkpoint has already been atomically renamed from ``.part``
    after its contents and metadata were checked against the open source.
    """

    canonical_id = _canonical_submission_id(submission_id)
    _validate_limits(min_bytes, max_bytes)
    source_path = Path(source)
    root = Path(incoming_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_real_directory(root)

    stage_dir = root / f".{canonical_id}.incoming"
    stage_dir.mkdir(mode=0o700)
    part_path = stage_dir / f".{canonical_id}.pt.part"
    checkpoint_path = stage_dir / f"{canonical_id}.pt"

    source_descriptor: int | None = None
    output_descriptor: int | None = None
    try:
        fsync_dir(root)
        source_descriptor = _open_source(source_path)
        before = _regular_descriptor_signature(source_descriptor, source_path)
        _validate_size(before.size, min_bytes, max_bytes)

        output_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        output_descriptor = os.open(part_path, output_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_bytes:
                raise CheckpointSizeError(
                    f"checkpoint grew beyond {max_bytes} bytes during copy"
                )
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_descriptor, view)
                if written <= 0:
                    raise OSError("checkpoint staging write made no progress")
                view = view[written:]

        after = _regular_descriptor_signature(source_descriptor, source_path)
        path_after = stat_regular(source_path)
        if before != after or before != path_after or copied != before.size:
            raise SourceChangedError("checkpoint changed while it was being copied")
        _validate_size(copied, min_bytes, max_bytes)

        os.fchmod(output_descriptor, 0o400)
        os.fsync(output_descriptor)
        os.close(output_descriptor)
        output_descriptor = None
        os.replace(part_path, checkpoint_path)
        fsync_dir(stage_dir)

        return StagedCopy(
            stage_dir=stage_dir,
            checkpoint_path=checkpoint_path,
            source_signature=before,
            sha256=digest.hexdigest(),
            size_bytes=copied,
            captured_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
    except BaseException:
        _remove_failed_stage(stage_dir, part_path, checkpoint_path)
        raise
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def verify_staged_source(source: Path, staged: StagedCopy) -> bool:
    """Re-read the mutable source and compare it with the captured copy.

    ``False`` means a confirmed identity/content mismatch.  Transient storage
    errors are raised so the caller retains the good staging copy and retries
    on a later poll instead of destroying evidence during an NFS outage.
    """

    source_path = Path(source)
    descriptor: int | None = None
    try:
        descriptor = _open_source(source_path)
        before = _regular_descriptor_signature(descriptor, source_path)
        if before != staged.source_signature or before.size != staged.size_bytes:
            return False

        digest = hashlib.sha256()
        read_bytes = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            read_bytes += len(chunk)
            if read_bytes > staged.size_bytes:
                return False
            digest.update(chunk)

        after = _regular_descriptor_signature(descriptor, source_path)
        path_after = stat_regular(source_path)
        return (
            before == after == path_after == staged.source_signature
            and read_bytes == staged.size_bytes
            and digest.hexdigest() == staged.sha256
        )
    except UnsafePathError as exc:
        cause = exc.__cause__
        if cause is None or (
            isinstance(cause, OSError)
            and cause.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}
        ):
            return False
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_write_json_no_clobber(
    path: Path, payload: Mapping[str, object], mode: int = 0o444
) -> None:
    """Atomically publish JSON without replacing any existing directory entry."""

    destination = Path(path)
    parent = destination.parent
    _require_real_directory(parent)
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777:
        raise ValueError("mode must be an integer in 0o000..0o777")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fchmod(output.fileno(), mode)
            os.fsync(output.fileno())

        # Linking a complete inode supplies create-if-absent semantics that
        # os.replace lacks.  The target becomes visible atomically and an
        # existing file or symlink causes FileExistsError.
        os.link(temporary, destination, follow_symlinks=False)
        published = True
        fsync_dir(parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            if published:
                fsync_dir(parent)
