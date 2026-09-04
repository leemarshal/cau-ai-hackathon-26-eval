from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import stat
import sys
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ta_grading import storage  # noqa: E402


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()


class TeamDiscoveryTests(TemporaryDirectoryTestCase):
    def test_discovers_exact_numbered_directories_in_numeric_order(self) -> None:
        mnt = self.root / "mnt"
        mnt.mkdir()
        (mnt / "Admin-Storage_7ed0d").mkdir()
        (mnt / "Model-Storage_5d351").mkdir()
        for name in ("Team3_ccccc", "Team1_aaaaa", "Team2_bbbbb"):
            (mnt / name).mkdir()

        teams = storage.discover_teams(mnt, 3)

        self.assertEqual(
            list(teams), ["Team1_aaaaa", "Team2_bbbbb", "Team3_ccccc"]
        )
        self.assertEqual(teams["Team2_bbbbb"], mnt / "Team2_bbbbb")

    def test_rejects_missing_unexpected_and_duplicate_numbers(self) -> None:
        cases = (
            ("missing", ("Team1_a", "Team3_c")),
            ("unexpected", ("Team1_a", "Team2_b", "Team4_d")),
            ("duplicate", ("Team1_a", "Team1_b", "Team2_c", "Team3_d")),
        )
        for label, names in cases:
            with self.subTest(label=label):
                mnt = self.root / label
                mnt.mkdir()
                for name in names:
                    (mnt / name).mkdir()
                with self.assertRaises(storage.StorageError):
                    storage.discover_teams(mnt, 3)

    def test_rejects_malformed_or_symlink_team_entries(self) -> None:
        malformed = self.root / "malformed"
        malformed.mkdir()
        (malformed / "Team1_a").mkdir()
        (malformed / "Team2-b").mkdir()
        with self.assertRaisesRegex(storage.StorageError, "invalid or unsafe"):
            storage.discover_teams(malformed, 1)

        linked = self.root / "linked"
        linked.mkdir()
        target = self.root / "real-team"
        target.mkdir()
        (linked / "Team1_a").symlink_to(target, target_is_directory=True)
        with self.assertRaises(storage.StorageError):
            storage.discover_teams(linked, 1)

    def test_requires_a_real_root_and_positive_expected_count(self) -> None:
        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)

        with self.assertRaises(storage.UnsafePathError):
            storage.discover_teams(alias, 1)
        for invalid in (0, -1, True):
            with self.subTest(expected_count=invalid):
                with self.assertRaises(ValueError):
                    storage.discover_teams(real, invalid)


class CheckpointEnumerationTests(TemporaryDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.team = self.root / "Team1_abcde"
        self.team.mkdir()
        (self.team / "b.pt").write_bytes(b"b")
        (self.team / "a.pt").write_bytes(b"a")
        (self.team / "upper.PT").write_bytes(b"upper")
        (self.team / ".pt").write_bytes(b"empty basename")
        (self.team / "notes.txt").write_text("notes", encoding="utf-8")
        (self.team / "linked.pt").symlink_to(self.team / "a.pt")
        nested = self.team / "nested"
        nested.mkdir()
        (nested / "c.pt").write_bytes(b"c")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "escape.pt").write_bytes(b"escape")
        (self.team / "linked-directory").symlink_to(outside, target_is_directory=True)

    def test_non_recursive_only_yields_direct_lowercase_regular_files(self) -> None:
        paths = list(storage.iter_checkpoints(self.team))
        self.assertEqual(paths, [self.team / "a.pt", self.team / "b.pt"])

    def test_recursive_does_not_follow_symlink_directories(self) -> None:
        paths = list(storage.iter_checkpoints(self.team, recursive=True))
        self.assertEqual(
            paths,
            [self.team / "a.pt", self.team / "b.pt", self.team / "nested/c.pt"],
        )

    def test_stat_regular_rejects_symlinks_and_directories(self) -> None:
        signature = storage.stat_regular(self.team / "a.pt")
        actual = (self.team / "a.pt").lstat()
        self.assertEqual(signature.size, 1)
        self.assertEqual(signature.ino, actual.st_ino)
        self.assertTrue(stat.S_ISREG(signature.mode))

        for unsafe in (self.team, self.team / "linked.pt"):
            with self.subTest(path=unsafe):
                with self.assertRaises(storage.UnsafePathError):
                    storage.stat_regular(unsafe)


class StageCopyTests(TemporaryDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.source = self.root / "model.pt"
        self.payload = (b"checkpoint-data-" * 64) + b"end"
        self.source.write_bytes(self.payload)
        self.incoming = self.root / "incoming"
        self.submission_id = str(uuid.uuid4())

    def stage(self) -> storage.StagedCopy:
        return storage.stage_copy(
            self.source,
            self.incoming,
            self.submission_id,
            min_bytes=8,
            max_bytes=4096,
        )

    def test_copies_hashes_fsyncs_and_preserves_source(self) -> None:
        source_before = self.source.read_bytes()

        staged = self.stage()

        self.assertEqual(
            staged.stage_dir,
            self.incoming / f".{self.submission_id}.incoming",
        )
        self.assertEqual(
            staged.checkpoint_path,
            staged.stage_dir / f"{self.submission_id}.pt",
        )
        self.assertEqual(staged.checkpoint_path.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(staged.size_bytes, len(self.payload))
        self.assertEqual(staged.sha256, hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(staged.source_signature, storage.stat_regular(self.source))
        self.assertEqual(stat.S_IMODE(staged.checkpoint_path.stat().st_mode), 0o400)
        self.assertFalse(any(staged.stage_dir.glob("*.part")))
        self.assertTrue(staged.captured_at.endswith("Z"))
        datetime.fromisoformat(staged.captured_at.removesuffix("Z") + "+00:00")
        self.assertTrue(storage.verify_staged_source(self.source, staged))

    def test_existing_stage_is_not_clobbered(self) -> None:
        staged = self.stage()

        with self.assertRaises(FileExistsError):
            self.stage()

        self.assertEqual(staged.checkpoint_path.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_enforces_strict_minimum_and_inclusive_maximum(self) -> None:
        exactly_ten = self.root / "ten.pt"
        exactly_ten.write_bytes(b"x" * 10)
        for minimum, maximum in ((10, 20), (0, 9)):
            with self.subTest(minimum=minimum, maximum=maximum):
                submission_id = str(uuid.uuid4())
                with self.assertRaises(storage.CheckpointSizeError):
                    storage.stage_copy(
                        exactly_ten,
                        self.incoming,
                        submission_id,
                        min_bytes=minimum,
                        max_bytes=maximum,
                    )
                self.assertFalse(
                    (self.incoming / f".{submission_id}.incoming").exists()
                )
                self.assertEqual(exactly_ten.read_bytes(), b"x" * 10)

        staged = storage.stage_copy(
            exactly_ten,
            self.incoming,
            str(uuid.uuid4()),
            min_bytes=9,
            max_bytes=10,
        )
        self.assertEqual(staged.size_bytes, 10)

    def test_rejects_symlink_source_and_unsafe_submission_id(self) -> None:
        link = self.root / "link.pt"
        link.symlink_to(self.source)
        for unsafe_source in (link, self.root):
            with self.subTest(source=unsafe_source):
                with self.assertRaises(storage.UnsafePathError):
                    storage.stage_copy(
                        unsafe_source,
                        self.incoming,
                        str(uuid.uuid4()),
                        min_bytes=0,
                        max_bytes=4096,
                    )

        with self.assertRaises(ValueError):
            storage.stage_copy(
                self.source,
                self.incoming,
                "../model",
                min_bytes=0,
                max_bytes=4096,
            )

    def test_detects_source_change_during_copy_and_cleans_stage(self) -> None:
        real_read = os.read
        first_read = True

        def racing_read(descriptor: int, count: int) -> bytes:
            nonlocal first_read
            chunk = real_read(descriptor, count)
            if first_read and chunk:
                first_read = False
                replacement = self.root / "replacement-during-copy.pt"
                replacement.write_bytes(b"changed-data----" * 64 + b"end")
                os.replace(replacement, self.source)
            return chunk

        with mock.patch.object(storage, "_COPY_CHUNK_BYTES", 8), mock.patch.object(
            storage.os, "read", side_effect=racing_read
        ):
            with self.assertRaises(storage.SourceChangedError):
                self.stage()

        self.assertTrue(self.source.is_file())
        self.assertFalse(
            (self.incoming / f".{self.submission_id}.incoming").exists()
        )

    def test_second_pass_detects_modified_or_replaced_source(self) -> None:
        staged = self.stage()
        replacement = self.root / "replacement.pt"
        replacement.write_bytes(self.payload)
        os.replace(replacement, self.source)

        self.assertFalse(storage.verify_staged_source(self.source, staged))
        self.assertEqual(staged.checkpoint_path.read_bytes(), self.payload)

    def test_second_pass_returns_false_for_missing_or_symlink_source(self) -> None:
        staged = self.stage()
        self.source.unlink()
        self.assertFalse(storage.verify_staged_source(self.source, staged))
        self.source.symlink_to(staged.checkpoint_path)
        self.assertFalse(storage.verify_staged_source(self.source, staged))

    def test_second_pass_raises_on_transient_storage_error(self) -> None:
        staged = self.stage()
        failure = OSError(errno.EIO, "simulated NFS I/O error")
        with mock.patch.object(storage, "_open_source", side_effect=failure):
            with self.assertRaises(OSError) as raised:
                storage.verify_staged_source(self.source, staged)
        self.assertEqual(raised.exception.errno, errno.EIO)
        self.assertTrue(staged.checkpoint_path.is_file())


class AtomicJsonTests(TemporaryDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.marker_dir = self.root / "ready"
        self.marker_dir.mkdir()
        self.marker = self.marker_dir / "submission.ready.json"

    def test_writes_complete_json_with_requested_mode(self) -> None:
        payload = {"sha256": "abc", "size_bytes": 123, "ready": True}

        storage.atomic_write_json_no_clobber(self.marker, payload, mode=0o440)

        self.assertEqual(json.loads(self.marker.read_text(encoding="utf-8")), payload)
        self.assertTrue(self.marker.read_text(encoding="utf-8").endswith("\n"))
        self.assertEqual(stat.S_IMODE(self.marker.stat().st_mode), 0o440)
        self.assertEqual(list(self.marker_dir.glob(".*.tmp")), [])

    def test_never_overwrites_existing_file_or_symlink(self) -> None:
        self.marker.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            storage.atomic_write_json_no_clobber(self.marker, {"new": True})
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "original")

        self.marker.unlink()
        target = self.root / "target"
        target.write_text("target", encoding="utf-8")
        self.marker.symlink_to(target)
        with self.assertRaises(FileExistsError):
            storage.atomic_write_json_no_clobber(self.marker, {"new": True})
        self.assertEqual(target.read_text(encoding="utf-8"), "target")

    def test_invalid_json_is_not_published_and_temp_is_cleaned(self) -> None:
        with self.assertRaises(ValueError):
            storage.atomic_write_json_no_clobber(self.marker, {"bad": math.nan})

        self.assertFalse(self.marker.exists())
        self.assertEqual(list(self.marker_dir.iterdir()), [])

    def test_fsync_dir_rejects_non_directory(self) -> None:
        regular = self.root / "file"
        regular.write_text("x", encoding="utf-8")
        with self.assertRaises(storage.UnsafePathError):
            storage.fsync_dir(regular)


if __name__ == "__main__":
    unittest.main()
