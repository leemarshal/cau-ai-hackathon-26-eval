#!/usr/bin/env python3
"""Install the pinned private-test archive without making it world-readable."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence


EXPECTED_ARCHIVE_BYTES = 1_075_558_400
EXPECTED_ARCHIVE_SHA256 = (
    "0caa77605652dd213ea967b944e9168e3a5c3f5ebd4847af168fc7f849da55af"
)
ARCHIVE_PREFIX = "grading_docker"
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class InstallError(RuntimeError):
    """Raised when the archive or target violates the private-data contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_path(path: Path, expected_bytes: int, expected_sha256: str) -> Path:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise InstallError(f"cannot inspect archive: {exc}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise InstallError("archive must be a regular, non-symlink file")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise InstallError("private grading archive must have owner-only permissions")
    if file_stat.st_size != expected_bytes:
        raise InstallError(
            f"archive byte count mismatch: expected {expected_bytes}, got {file_stat.st_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise InstallError("archive SHA-256 does not match the pinned v2 package")
    return path


def _validate_destination(destination: Path) -> tuple[Path, Path]:
    if not destination.is_absolute() or not destination.name:
        raise InstallError("destination must be an absolute, non-root path")
    if destination.exists() or destination.is_symlink():
        raise InstallError(f"destination already exists: {destination}")

    lexical_parent = Path(os.path.abspath(destination.parent))
    try:
        parent = destination.parent.resolve(strict=True)
        parent_stat = parent.lstat()
    except OSError as exc:
        raise InstallError(f"destination parent must already exist: {exc}") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise InstallError("destination parent must be a real directory")
    if parent != lexical_parent:
        raise InstallError("destination must not traverse a symlinked parent")
    if stat.S_IMODE(parent_stat.st_mode) & 0o077:
        raise InstallError("destination parent must have owner-only permissions")
    if parent_stat.st_uid != os.geteuid():
        raise InstallError("destination parent must be owned by the installing user")

    normalized = parent / destination.name
    if normalized.exists() or normalized.is_symlink():
        raise InstallError(f"destination already exists: {normalized}")
    return parent, normalized


def _member_relative_path(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name
    pure = PurePosixPath(name)
    if (
        not name
        or pure.is_absolute()
        or pure.as_posix() != name
        or "\\" in name
        or "\x00" in name
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise InstallError(f"unsafe archive member path: {name!r}")
    if not pure.parts or pure.parts[0] != ARCHIVE_PREFIX:
        raise InstallError(f"archive member is outside {ARCHIVE_PREFIX}/: {name!r}")
    return PurePosixPath(*pure.parts[1:])


def _validate_members(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    members = archive.getmembers()
    names = [member.name for member in members]
    if not members or names != sorted(names) or len(names) != len(set(names)):
        raise InstallError("archive members must be non-empty, unique, and sorted")

    validated: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
    for member in members:
        relative = _member_relative_path(member)
        if not (member.isdir() or member.isreg()):
            raise InstallError(f"archive contains a link or special file: {member.name!r}")
        expected_mode = 0o755 if member.isdir() else 0o644
        if (
            member.mode != expected_mode
            or member.uid != 0
            or member.gid != 0
            or member.uname != ""
            or member.gname != ""
            or member.mtime != 0
            or member.linkname
            or member.pax_headers
        ):
            raise InstallError(f"archive member metadata is not canonical: {member.name!r}")
        if not relative.parts and not member.isdir():
            raise InstallError("archive prefix must be a directory")
        validated.append((member, relative))
    if validated[0][0].name != ARCHIVE_PREFIX or not validated[0][0].isdir():
        raise InstallError("archive must begin with its single grading_docker directory")
    return validated


def _copy_member(source: BinaryIO, destination: Path, expected_size: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    copied = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                copied += len(chunk)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o600)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if copied != expected_size:
        destination.unlink(missing_ok=True)
        raise InstallError(
            f"archive member size mismatch for {destination.name}: "
            f"expected {expected_size}, got {copied}"
        )


def _extract_private(archive_path: Path, stage: Path) -> None:
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = _validate_members(archive)
            for member, relative in members:
                if not relative.parts:
                    continue
                target = stage.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=0o700)
                    target.chmod(0o700)
                    continue
                if not target.parent.is_dir():
                    raise InstallError(
                        f"archive file precedes its directory: {member.name!r}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise InstallError(f"cannot read archive member: {member.name!r}")
                with source:
                    _copy_member(source, target, member.size)
    except InstallError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise InstallError(f"cannot extract private grading archive: {exc}") from exc


def _verify_owner_only_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            raise InstallError(f"installed tree contains a symlink: {path}")
        if file_stat.st_uid != os.geteuid():
            raise InstallError(f"installed path has an unexpected owner: {path}")
        expected_mode = 0o700 if stat.S_ISDIR(file_stat.st_mode) else 0o600
        if not (stat.S_ISDIR(file_stat.st_mode) or stat.S_ISREG(file_stat.st_mode)):
            raise InstallError(f"installed tree contains a special file: {path}")
        if stat.S_IMODE(file_stat.st_mode) != expected_mode:
            raise InstallError(f"installed path has unsafe permissions: {path}")


def install_grading_package(
    archive_path: Path | str,
    destination_path: Path | str,
    *,
    expected_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> Path:
    """Verify and atomically publish an owner-only extracted grading tree."""

    archive = _validate_archive_path(
        Path(archive_path).expanduser(), expected_bytes, expected_sha256
    )
    parent, destination = _validate_destination(Path(destination_path).expanduser())
    stage = parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        stage.mkdir(mode=0o700)
        stage.chmod(0o700)
        _extract_private(archive, stage)
        _verify_owner_only_tree(stage)
        if destination.exists() or destination.is_symlink():
            raise InstallError(f"destination appeared during install: {destination}")
        os.rename(stage, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _verify_owner_only_tree(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = install_grading_package(args.archive, args.destination)
    except InstallError as exc:
        print(f"install-grading-package: {exc}", file=sys.stderr)
        return 1
    print(f"private grading data installed at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
