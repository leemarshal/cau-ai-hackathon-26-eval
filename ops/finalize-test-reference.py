#!/usr/bin/env python3
"""Validate a v2 test reference cache and mark its manifest ready.

Run this after ``score_unlearning.py prepare-reference`` has written
``grading_docker/score_cache/M_o__test.npz``.  The cache, test split, refs,
images, and manifest must agree exactly before the manifest is atomically
changed from ``pending-prepare-reference`` to ``ready``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 2
SCORE_VERSION = "unlearning-v2"
PHASE = "test"
DEPTHS = ["b4", "b8", "b12", "pre"]
CACHE_RELATIVE_PATH = "score_cache/M_o__test.npz"
SPLIT_RELATIVE_PATH = "splits/test_split.pt"
REFS_RELATIVE_PATH = "score_cache/refs.pt"
IMAGE_RELATIVE_ROOT = "imagenet_test"
TEST_RUNTIME_ROOT = "/grading-data/assets/imagenet_test"
FEATURE_WIDTH = 768

# Immutable pins for the generated v2 private-test bundle. Verification checks
# these before parsing either torch metadata file.
KNOWN_TEST_REVISION = "f7938fad4be1b9559433adf6f3edfab6088750ba003371de7c7505b5da05353b"
KNOWN_MANIFEST_SHA256 = "31c2ea78fb8adbb95921a3a314e863a8789d0b86b42082ee0495b91aee61ede3"
KNOWN_SPLIT_SHA256 = "7691c5b6d9401fb963247f7bda7b580ca0319ac170b381f5448e21d125f5eb12"
KNOWN_REFS_SHA256 = "3a0fa9a03babd69bfa6beee7a0fb59a119a903a793c91f26003cd4a1d374d6f0"
KNOWN_MODEL_SHA256 = "b67e6091d53e8bbd2d72e627c80f6d472127538c9b87352f3f6afd13491385ab"
KNOWN_CACHE_SHA256 = "d8cb99d511359a05e38d36fc12c0895a753683f232e4278987e54f8b6ee15849"
KNOWN_CACHE_BYTES = 61_484_632
KNOWN_IMAGE_TREE_SHA256 = "a72603c586443df097019e7685a930bbba9efe21b4475e50f189291c7b981450"

REF_KEYS = {
    "schema_version",
    "phase",
    "score_version",
    "dataset_revision",
    "accuracy_split",
    "representation_split",
    "reference_accuracy",
    "score_depth",
    "depths",
    "forget_labels",
    "forget_wnids",
}
CACHE_KEYS = {
    "schema_version",
    "phase",
    "dataset_revision",
    "split_name",
    "correct",
    "total",
    "labels",
    *(f"f_{depth}" for depth in DEPTHS),
}
MANIFEST_KEYS = {
    "schema_version",
    "phase",
    "score_version",
    "dataset_revision",
    "runtime_root",
    "splits",
    "assets",
}
SPLIT_META_KEYS = {
    "schema_version",
    "score_version",
    "wnids",
    "phase",
    "root",
    "dataset_revision",
    "counts",
    "sha256",
    "records_sha256",
    "content_sha256",
}

Record = tuple[str, int]


class FinalizeError(RuntimeError):
    """Raised when a grading tree or reference cache violates the v2 contract."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FinalizeError(f"{description} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FinalizeError(f"{description} must be a real directory: {path}")


def _require_regular(path: Path, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FinalizeError(f"{description} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FinalizeError(f"{description} must be a regular file: {path}")
    return metadata


def _validate_tree_shape(root: Path) -> None:
    expected_top = {
        "dataset_manifest.json",
        "imagenet_test",
        "m_o",
        "score_cache",
        "splits",
    }
    if {entry.name for entry in root.iterdir()} != expected_top:
        raise FinalizeError("grading root contains missing or unexpected entries")
    for relative, expected in (
        ("m_o", {"M_o.pt"}),
        ("score_cache", {"M_o__test.npz", "refs.pt"}),
        ("splits", {"test_split.pt"}),
    ):
        directory = root / relative
        _require_directory(directory, f"grading {relative} directory")
        if {entry.name for entry in directory.iterdir()} != expected:
            raise FinalizeError(f"grading {relative} contains unexpected entries")


def _require_file_pin(path: Path, expected: str, description: str) -> None:
    _require_regular(path, description)
    if not _is_sha256(expected) or not hmac.compare_digest(
        _sha256_file(path), expected
    ):
        raise FinalizeError(f"{description} does not match its immutable SHA-256 pin")


def _image_tree_sha256(root: Path, records: Sequence[Record]) -> str:
    digest = hashlib.sha256()
    for relative, _ in sorted(records):
        path = root / IMAGE_RELATIVE_ROOT / relative
        metadata = _require_regular(path, f"test image {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require_regular(path, "grading dataset manifest")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"cannot read grading dataset manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizeError("grading dataset manifest must be a JSON object")
    return value


def _load_torch_mapping(path: Path, description: str) -> dict[str, Any]:
    _require_regular(path, description)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch exception varies
        raise FinalizeError(f"cannot load {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizeError(f"{description} must be a mapping")
    return value


def _record_hash(records: Sequence[Record]) -> str:
    payload = "".join(
        f"{path}\t{label}\n" for path, label in sorted(records)
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _path_hash(records: Sequence[Record]) -> str:
    return hashlib.sha256(
        "".join(sorted(path for path, _ in records)).encode()
    ).hexdigest()


def _dataset_revision(split_hashes: Mapping[str, str]) -> str:
    payload = "".join(
        f"{name}\t{digest}\n" for name, digest in sorted(split_hashes.items())
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_records(raw: Any, n_classes: int) -> list[Record]:
    if not isinstance(raw, list) or not raw:
        raise FinalizeError("test split must be a non-empty list")
    records: list[Record] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise FinalizeError("test split contains a malformed record")
        relative, label = item
        if not isinstance(relative, str) or not relative:
            raise FinalizeError("test split contains an invalid path")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or not pure.parts
            or pure.parts[0] != "test"
            or pure.suffix != ".JPEG"
        ):
            raise FinalizeError(f"test split contains an unsafe path: {relative!r}")
        if relative in seen:
            raise FinalizeError(f"test split contains a duplicate path: {relative!r}")
        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < n_classes
        ):
            raise FinalizeError("test split contains an invalid label")
        seen.add(relative)
        records.append((relative, label))
    return records


def _validate_file_asset(root: Path, raw: Any, expected_path: str, name: str) -> None:
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
        raise FinalizeError(f"manifest {name} asset schema is invalid")
    if raw.get("path") != expected_path:
        raise FinalizeError(f"manifest {name} path is invalid")
    asset = root / expected_path
    metadata = _require_regular(asset, f"{name} asset")
    size = raw.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise FinalizeError(f"manifest {name} byte size is invalid")
    if size != metadata.st_size:
        raise FinalizeError(f"manifest {name} byte size mismatch")
    if not _is_sha256(raw.get("sha256")) or raw["sha256"] != _sha256_file(asset):
        raise FinalizeError(f"manifest {name} SHA-256 mismatch")


def _validate_manifest(manifest: dict[str, Any], root: Path) -> str:
    if set(manifest) != MANIFEST_KEYS:
        raise FinalizeError("grading dataset manifest schema is invalid")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("phase", PHASE),
        ("score_version", SCORE_VERSION),
    ):
        if manifest.get(key) != expected:
            raise FinalizeError(f"grading dataset manifest {key} mismatch")
    revision = manifest.get("dataset_revision")
    if not _is_sha256(revision):
        raise FinalizeError("grading dataset manifest revision is invalid")
    runtime_root = manifest.get("runtime_root")
    if runtime_root != TEST_RUNTIME_ROOT:
        raise FinalizeError("grading dataset manifest runtime_root is invalid")
    if not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) != {
        "test"
    }:
        raise FinalizeError("grading dataset manifest must describe test only")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or set(assets) != {
        "m_o",
        "refs",
        "representation_cache",
    }:
        raise FinalizeError("grading dataset manifest assets are invalid")
    _validate_file_asset(root, assets["m_o"], "m_o/M_o.pt", "m_o")
    refs_asset = assets["refs"]
    if not isinstance(refs_asset, dict) or refs_asset != {
        "path": REFS_RELATIVE_PATH,
        "schema_version": SCHEMA_VERSION,
    }:
        raise FinalizeError("manifest refs asset schema is invalid")
    cache_asset = assets["representation_cache"]
    pending = {
        "path": CACHE_RELATIVE_PATH,
        "status": "pending-prepare-reference",
    }
    if cache_asset != pending:
        if not isinstance(cache_asset, dict) or set(cache_asset) != {
            "path",
            "status",
            "bytes",
            "sha256",
        }:
            raise FinalizeError("manifest representation cache state is invalid")
        if (
            cache_asset.get("path") != CACHE_RELATIVE_PATH
            or cache_asset.get("status") != "ready"
            or isinstance(cache_asset.get("bytes"), bool)
            or not isinstance(cache_asset.get("bytes"), int)
            or cache_asset["bytes"] <= 0
            or not _is_sha256(cache_asset.get("sha256"))
        ):
            raise FinalizeError("manifest ready representation cache is invalid")
    return revision


def _validate_split(
    root: Path, manifest: Mapping[str, Any], revision: str
) -> tuple[list[str], list[Record]]:
    split = _load_torch_mapping(root / SPLIT_RELATIVE_PATH, "test split manifest")
    if set(split) != {"meta", "splits"}:
        raise FinalizeError("test split manifest must contain exactly meta/splits")
    meta, raw_splits = split["meta"], split["splits"]
    if not isinstance(meta, dict) or set(meta) != SPLIT_META_KEYS:
        raise FinalizeError("test split metadata schema is invalid")
    if not isinstance(raw_splits, dict) or set(raw_splits) != {"test"}:
        raise FinalizeError("test split manifest must expose test only")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("score_version", SCORE_VERSION),
        ("phase", PHASE),
        ("dataset_revision", revision),
        ("root", manifest["runtime_root"]),
    ):
        if meta.get(key) != expected:
            raise FinalizeError(f"test split metadata {key} mismatch")
    wnids = meta.get("wnids")
    if (
        not isinstance(wnids, list)
        or not wnids
        or len(wnids) != len(set(wnids))
        or any(not isinstance(wnid, str) or not wnid for wnid in wnids)
    ):
        raise FinalizeError("test split wnids are invalid")
    records = _safe_records(raw_splits["test"], len(wnids))
    records_digest = _record_hash(records)
    path_digest = _path_hash(records)
    expected_revision = _dataset_revision({"test": records_digest})
    if expected_revision != revision:
        raise FinalizeError("test records do not produce dataset_revision")
    if (
        not isinstance(meta.get("counts"), dict)
        or set(meta["counts"]) != {"test"}
        or isinstance(meta["counts"]["test"], bool)
        or not isinstance(meta["counts"]["test"], int)
    ):
        raise FinalizeError("test split metadata counts schema is invalid")
    for key in ("sha256", "records_sha256", "content_sha256"):
        value = meta.get(key)
        if (
            not isinstance(value, dict)
            or set(value) != {"test"}
            or not _is_sha256(value["test"])
        ):
            raise FinalizeError(f"test split metadata {key} schema is invalid")
    if meta.get("counts") != {"test": len(records)}:
        raise FinalizeError("test split metadata count mismatch")
    if meta.get("sha256") != {"test": path_digest}:
        raise FinalizeError("test split metadata path hash mismatch")
    if meta.get("records_sha256") != {"test": records_digest}:
        raise FinalizeError("test split metadata record hash mismatch")

    split_info = manifest["splits"]["test"]
    if not isinstance(split_info, dict) or set(split_info) != {
        "count",
        "bytes",
        "path_sha256",
        "records_sha256",
        "content_sha256",
    }:
        raise FinalizeError("grading manifest test split schema is invalid")
    for key in ("count", "bytes"):
        value = split_info.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FinalizeError(f"grading manifest test {key} is invalid")
    for key in ("path_sha256", "records_sha256", "content_sha256"):
        if not _is_sha256(split_info.get(key)):
            raise FinalizeError(f"grading manifest test {key} is invalid")
    expected_info = {
        "count": len(records),
        "path_sha256": path_digest,
        "records_sha256": records_digest,
    }
    for key, expected in expected_info.items():
        if split_info.get(key) != expected:
            raise FinalizeError(f"grading manifest test {key} mismatch")

    image_root = root / IMAGE_RELATIVE_ROOT
    _require_directory(image_root, "test image root")
    actual_paths: set[str] = set()
    actual_bytes = 0
    for path in image_root.rglob("*"):
        relative = path.relative_to(image_root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FinalizeError(f"symlink is forbidden in test images: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise FinalizeError(f"special file is forbidden in test images: {relative}")
        actual_paths.add(relative)
        actual_bytes += metadata.st_size
    expected_paths = {path for path, _ in records}
    if actual_paths != expected_paths:
        raise FinalizeError("test image file set does not match the split manifest")
    if split_info.get("bytes") != actual_bytes:
        raise FinalizeError("grading manifest test byte size mismatch")
    content_digest = _image_tree_sha256(root, records)
    if split_info.get("content_sha256") != content_digest:
        raise FinalizeError("grading manifest test content hash mismatch")
    if meta.get("content_sha256") != {"test": content_digest}:
        raise FinalizeError("test split metadata content hash mismatch")
    return wnids, records


def _validate_refs(root: Path, revision: str, wnids: Sequence[str]) -> dict[str, Any]:
    refs = _load_torch_mapping(root / REFS_RELATIVE_PATH, "test metric refs")
    if set(refs) != REF_KEYS:
        raise FinalizeError("test metric refs schema is invalid")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("phase", PHASE),
        ("score_version", SCORE_VERSION),
        ("dataset_revision", revision),
        ("accuracy_split", PHASE),
        ("representation_split", PHASE),
        ("score_depth", "pre"),
        ("depths", DEPTHS),
    ):
        if refs.get(key) != expected:
            raise FinalizeError(f"test metric refs {key} mismatch")
    accuracy = refs.get("reference_accuracy")
    if not isinstance(accuracy, dict) or set(accuracy) != {"acc_f", "acc_r"}:
        raise FinalizeError("test reference_accuracy schema is invalid")
    for name, value in accuracy.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise FinalizeError(f"test reference_accuracy {name} is invalid")
    labels = refs.get("forget_labels")
    if (
        not isinstance(labels, list)
        or not labels
        or len(labels) != len(set(labels))
        or any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < len(wnids)
            for label in labels
        )
    ):
        raise FinalizeError("test forget_labels are invalid")
    if refs.get("forget_wnids") != [wnids[label] for label in labels]:
        raise FinalizeError("test forget label/wnid mapping is invalid")
    return refs


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    value = archive[key]
    if value.shape != ():
        raise FinalizeError(f"reference cache {key} must be a scalar")
    return value.item()


def _validate_npz_members(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected = {f"{key}.npy" for key in CACHE_KEYS}
            if len(names) != len(set(names)) or set(names) != expected:
                raise FinalizeError("reference cache ZIP members are unexpected")
            for member in members:
                pure = PurePosixPath(member.filename)
                if (
                    member.is_dir()
                    or pure.is_absolute()
                    or len(pure.parts) != 1
                    or ".." in pure.parts
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.flag_bits & 0x1
                ):
                    raise FinalizeError(
                        f"reference cache ZIP member is unsafe: {member.filename!r}"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise FinalizeError(f"reference cache is not a valid NPZ archive: {exc}") from exc


def _validate_cache(
    path: Path,
    *,
    revision: str,
    records: Sequence[Record],
    n_classes: int,
) -> tuple[int, str]:
    before = _require_regular(path, "test M_o reference cache")
    if before.st_size <= 0:
        raise FinalizeError("test M_o reference cache is empty")
    _validate_npz_members(path)
    expected_labels = np.asarray([label for _, label in records], dtype=np.int64)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != CACHE_KEYS:
                raise FinalizeError("reference cache arrays are unexpected")
            if _scalar(archive, "schema_version") != SCHEMA_VERSION:
                raise FinalizeError("reference cache schema_version mismatch")
            if _scalar(archive, "phase") != PHASE:
                raise FinalizeError("reference cache phase mismatch")
            if _scalar(archive, "dataset_revision") != revision:
                raise FinalizeError("reference cache dataset_revision mismatch")
            if _scalar(archive, "split_name") != PHASE:
                raise FinalizeError("reference cache split_name mismatch")

            labels = archive["labels"]
            if labels.dtype != np.dtype(np.int64) or labels.ndim != 1:
                raise FinalizeError("reference cache labels must be a 1-D int64 array")
            if not np.array_equal(labels, expected_labels):
                raise FinalizeError("reference cache labels/order do not match test")

            correct = archive["correct"]
            total = archive["total"]
            for name, values in (("correct", correct), ("total", total)):
                if values.dtype != np.dtype(np.float64) or values.shape != (n_classes,):
                    raise FinalizeError(
                        f"reference cache {name} must be float64[{n_classes}]"
                    )
                if not np.isfinite(values).all():
                    raise FinalizeError(f"reference cache {name} contains non-finite values")
                if (values < 0).any() or not np.equal(values, np.floor(values)).all():
                    raise FinalizeError(
                        f"reference cache {name} must contain non-negative counts"
                    )
            if (correct > total).any():
                raise FinalizeError("reference cache correct exceeds total")
            expected_total = np.bincount(expected_labels, minlength=n_classes).astype(
                np.float64
            )
            if not np.array_equal(total, expected_total):
                raise FinalizeError("reference cache totals do not match test labels")

            feature_shape: tuple[int, int] | None = None
            for depth in DEPTHS:
                values = archive[f"f_{depth}"]
                if (
                    values.dtype != np.dtype(np.float32)
                    or values.ndim != 2
                    or values.shape != (len(records), FEATURE_WIDTH)
                ):
                    raise FinalizeError(
                        f"reference cache f_{depth} has an invalid dtype/shape"
                    )
                if feature_shape is None:
                    feature_shape = values.shape
                elif values.shape != feature_shape:
                    raise FinalizeError("reference cache feature shapes differ by depth")
                if not np.isfinite(values).all():
                    raise FinalizeError(
                        f"reference cache f_{depth} contains non-finite values"
                    )
    except FinalizeError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise FinalizeError(f"cannot load test M_o reference cache: {exc}") from exc

    digest = _sha256_file(path)
    after = _require_regular(path, "test M_o reference cache")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise FinalizeError("test M_o reference cache changed during validation")
    return after.st_size, digest


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback
            return
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(grading_root: Path | str) -> dict[str, Any]:
    """Validate and finalize one v2 ``grading_docker`` directory."""
    root = Path(grading_root).expanduser().absolute()
    _require_directory(root, "grading root")
    for relative, description in (
        ("splits", "grading splits directory"),
        ("score_cache", "grading score cache directory"),
        ("m_o", "grading model directory"),
    ):
        _require_directory(root / relative, description)

    manifest_path = root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    revision = _validate_manifest(manifest, root)
    wnids, records = _validate_split(root, manifest, revision)
    _validate_refs(root, revision, wnids)
    cache_bytes, cache_sha256 = _validate_cache(
        root / CACHE_RELATIVE_PATH,
        revision=revision,
        records=records,
        n_classes=len(wnids),
    )
    ready = {
        "path": CACHE_RELATIVE_PATH,
        "status": "ready",
        "bytes": cache_bytes,
        "sha256": cache_sha256,
    }
    current = manifest["assets"]["representation_cache"]
    if current.get("status") == "ready" and current != ready:
        raise FinalizeError("ready manifest checksum/size do not match the cache")
    if current != ready:
        manifest["assets"]["representation_cache"] = ready
        _write_json_atomic(manifest_path, manifest)
    return manifest


def verify_ready(
    grading_root: Path | str,
    *,
    expected_revision: str = KNOWN_TEST_REVISION,
    expected_manifest_sha256: str = KNOWN_MANIFEST_SHA256,
    expected_split_sha256: str = KNOWN_SPLIT_SHA256,
    expected_refs_sha256: str = KNOWN_REFS_SHA256,
    expected_model_sha256: str = KNOWN_MODEL_SHA256,
    expected_cache_sha256: str = KNOWN_CACHE_SHA256,
    expected_cache_bytes: int = KNOWN_CACHE_BYTES,
    expected_image_tree_sha256: str = KNOWN_IMAGE_TREE_SHA256,
) -> dict[str, Any]:
    """Verify a finalized private-test tree against immutable external pins."""
    root = Path(grading_root).expanduser().absolute()
    _require_directory(root, "grading root")
    _validate_tree_shape(root)

    # Verify executable torch metadata and every score-affecting asset before
    # parsing any of them in the trusted/private-data process.
    _require_file_pin(
        root / "dataset_manifest.json",
        expected_manifest_sha256,
        "grading dataset manifest",
    )
    _require_file_pin(
        root / SPLIT_RELATIVE_PATH, expected_split_sha256, "test split manifest"
    )
    _require_file_pin(root / REFS_RELATIVE_PATH, expected_refs_sha256, "test refs")
    _require_file_pin(root / "m_o/M_o.pt", expected_model_sha256, "M_o checkpoint")
    cache_path = root / CACHE_RELATIVE_PATH
    cache_metadata = _require_regular(cache_path, "test M_o reference cache")
    if cache_metadata.st_size != expected_cache_bytes:
        raise FinalizeError("test M_o reference cache byte-size pin mismatch")
    _require_file_pin(cache_path, expected_cache_sha256, "test M_o reference cache")

    manifest = _read_json(root / "dataset_manifest.json")
    revision = _validate_manifest(manifest, root)
    if not _is_sha256(expected_revision) or not hmac.compare_digest(
        revision, expected_revision
    ):
        raise FinalizeError("private test dataset revision pin mismatch")
    cache_descriptor = manifest["assets"]["representation_cache"]
    if cache_descriptor.get("status") != "ready":
        raise FinalizeError("private test representation cache is not ready")

    wnids, records = _validate_split(root, manifest, revision)
    _validate_refs(root, revision, wnids)
    cache_bytes, cache_sha256 = _validate_cache(
        cache_path,
        revision=revision,
        records=records,
        n_classes=len(wnids),
    )
    if cache_descriptor != {
        "path": CACHE_RELATIVE_PATH,
        "status": "ready",
        "bytes": cache_bytes,
        "sha256": cache_sha256,
    }:
        raise FinalizeError("ready manifest checksum/size do not match the cache")
    actual_image_tree = _image_tree_sha256(root, records)
    if not _is_sha256(expected_image_tree_sha256) or not hmac.compare_digest(
        actual_image_tree, expected_image_tree_sha256
    ):
        raise FinalizeError("private test image tree SHA-256 pin mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grading-root",
        required=True,
        type=Path,
        help="v2 grading_docker directory to finalize",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require the finalized tree to match the immutable v2 pins",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = (
            verify_ready(args.grading_root)
            if args.verify_only
            else finalize(args.grading_root)
        )
    except FinalizeError as exc:
        print(f"finalize-test-reference: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["assets"]["representation_cache"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
