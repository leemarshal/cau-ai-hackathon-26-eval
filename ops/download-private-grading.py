#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


EXPECTED_BYTES = 1_075_558_400
EXPECTED_SHA256 = "0caa77605652dd213ea967b944e9168e3a5c3f5ebd4847af168fc7f849da55af"
DEFAULT_REPO = "cau-ai-hackathon/imagenet-grading"
DEFAULT_REVISION = "5d8f84f903f177ebab5b43188a792d2436d50230"
DEFAULT_FILENAME = "grading_docker.tar"


def required_token(path: Path) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("HF token must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError("HF token must have owner-only permissions (chmod 600)")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("HF token file is empty")
    return token


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    state_root = Path(
        os.environ.get(
            "TA_STATE_ROOT", Path.home() / ".local/state/hackathon-ta-grader"
        )
    ).expanduser().resolve(strict=False)
    grading_root = Path(
        os.environ.get("TA_GRADING_ROOT", Path.home() / "private-grading/assets")
    ).expanduser().resolve(strict=False)
    token_path = Path(
        os.environ.get("TA_HF_TOKEN_FILE", Path.home() / ".config/huggingface/token")
    ).expanduser()
    config = {
        "repo": os.environ.get("TA_HF_REPO", DEFAULT_REPO),
        "revision": os.environ.get("TA_HF_REVISION", DEFAULT_REVISION),
        "filename": os.environ.get("TA_HF_FILENAME", DEFAULT_FILENAME),
        "bytes": int(os.environ.get("TA_HF_ARCHIVE_BYTES", EXPECTED_BYTES)),
        "sha256": os.environ.get("TA_HF_ARCHIVE_SHA256", EXPECTED_SHA256),
    }
    if config["bytes"] != EXPECTED_BYTES or config["sha256"] != EXPECTED_SHA256:
        raise RuntimeError("archive size/SHA overrides do not match the pinned grader code")

    parent = grading_root.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    ready_path = parent / ".grading-ready.json"
    if ready_path.exists() or ready_path.is_symlink():
        if (
            ready_path.is_file()
            and not ready_path.is_symlink()
            and grading_root.is_dir()
            and all(
                (grading_root / relative).exists()
                for relative in (
                    "splits/test_split.pt",
                    "score_cache/refs.pt",
                    "score_cache/M_o__test.npz",
                    "imagenet_test",
                )
            )
            and json.loads(ready_path.read_text(encoding="utf-8")) == config
        ):
            print(f"private grading data already installed: {grading_root}")
            return 0
        raise RuntimeError(f"existing grading ready marker does not match: {ready_path}")
    if grading_root.exists() or grading_root.is_symlink():
        raise RuntimeError(
            f"grading destination already exists without the matching pin: {grading_root}"
        )

    download_root = state_root / "downloads"
    cache_root = state_root / "huggingface-cache"
    download_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    download_root.chmod(0o700)
    cache_root.chmod(0o700)
    downloaded = Path(
        hf_hub_download(
            repo_id=config["repo"],
            repo_type="dataset",
            revision=config["revision"],
            filename=config["filename"],
            token=required_token(token_path),
            local_dir=download_root,
            cache_dir=cache_root,
        )
    )
    downloaded.chmod(0o600)
    if downloaded.stat().st_size != config["bytes"] or sha256(downloaded) != config["sha256"]:
        raise RuntimeError("downloaded private grading archive failed size/SHA verification")

    subprocess.run(
        [
            sys.executable,
            str(project_root / "ops/install-grading-package.py"),
            "--archive",
            str(downloaded),
            "--destination",
            str(grading_root),
        ],
        check=True,
    )
    temporary = parent / ".grading-ready.json.tmp"
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fchmod(output.fileno(), 0o600)
        os.fsync(output.fileno())
    os.replace(temporary, ready_path)
    directory_descriptor = os.open(
        parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    print(f"private grading data installed: {grading_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
