from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.config import Settings  # noqa: E402
from ta_grading.database import Database  # noqa: E402
from ta_grading.watcher import SubmissionWatcher, watcher_lock  # noqa: E402


class SubmissionWatcherIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mnt_root = self.root / "mnt"
        self.mnt_root.mkdir()
        self.admin_root = self.mnt_root / "Admin-Storage_admin1"
        self.admin_root.mkdir()
        self.teams = {
            "Team1_aaaaa": self.mnt_root / "Team1_aaaaa",
            "Team2_bbbbb": self.mnt_root / "Team2_bbbbb",
        }
        for path in self.teams.values():
            path.mkdir()

        self.settings = Settings(
            project_root=ROOT,
            mnt_root=self.mnt_root,
            admin_root=self.admin_root,
            backup_root=self.admin_root / "submission-backups",
            state_root=self.root / "state",
            database_path=self.root / "state/grading.sqlite3",
            grading_root=self.root / "private-grading/assets",
            grade_script=ROOT / "ops/grade-finalist.sh",
            grading_image="test/grader:local",
            expected_team_count=len(self.teams),
            poll_seconds=0.0,
            stable_confirmations=1,
            post_copy_seconds=0.0,
            min_checkpoint_bytes=8,
            max_checkpoint_bytes=4096,
            recursive_scan=False,
            worker_poll_seconds=0.0,
            gpu_ids=(1, 2, 3),
        )
        self.database = Database(self.settings.database_path)
        self.database.initialize()
        self.watcher = SubmissionWatcher(self.settings, self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _poll(self, count: int = 1) -> None:
        for _ in range(count):
            self.watcher.run_once()

    def _write_model(
        self, team_name: str, model_name: str, payload: bytes
    ) -> Path:
        path = self.teams[team_name] / model_name
        path.write_bytes(payload)
        return path

    def _ready_paths(self) -> list[Path]:
        return sorted((self.settings.backup_root / "ready").glob("*.json"))

    def _artifact_paths(self) -> list[Path]:
        return sorted((self.settings.backup_root / "artifacts").glob("*/*/*.pt"))

    def _capture_dirs(self) -> list[Path]:
        return sorted((self.settings.backup_root / ".incoming").glob(".*.incoming"))

    def test_mount_identity_change_fails_before_creating_backup_layout(self) -> None:
        self.assertFalse(self.settings.backup_root.exists())

    def test_watcher_lock_rejects_a_second_mutating_watcher(self) -> None:
        with watcher_lock(self.settings):
            with self.assertRaisesRegex(RuntimeError, "another submission watcher"):
                with watcher_lock(self.settings):
                    self.fail("a second watcher unexpectedly acquired the lock")
        with mock.patch.object(
            self.watcher, "_directory_identity", return_value=(999, 999)
        ):
            with self.assertRaisesRegex(RuntimeError, "mount identity changed"):
                self.watcher.ensure_layout()
        self.assertFalse(self.settings.backup_root.exists())

    def test_observe_capture_verify_and_queue_preserves_source(self) -> None:
        payload = b"complete-checkpoint" * 8
        source = self._write_model("Team1_aaaaa", "winning.pt", payload)
        small = self._write_model("Team2_bbbbb", "too-small.pt", b"tiny")

        self._poll()
        self.assertEqual(self.database.rows(), [])
        self.assertEqual(self._capture_dirs(), [])

        self._poll()
        capture_dirs = self._capture_dirs()
        self.assertEqual(len(capture_dirs), 1)
        capture = json.loads(
            (capture_dirs[0] / "capture.json").read_text(encoding="utf-8")
        )
        self.assertEqual(capture["state"], "captured")
        self.assertEqual(capture["team_name"], "Team1_aaaaa")
        self.assertEqual(capture["model_name"], "winning.pt")
        self.assertEqual(self.database.rows(), [])
        self.assertEqual(self._ready_paths(), [])

        self._poll()

        rows = self.database.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["team_name"], "Team1_aaaaa")
        self.assertEqual(row["team_number"], 1)
        self.assertEqual(row["submission_number"], 1)
        self.assertEqual(row["model_name"], "winning.pt")
        self.assertEqual(row["source_relative_path"], "winning.pt")
        self.assertEqual(row["source_size_bytes"], len(payload))
        self.assertEqual(row["source_sha256"], hashlib.sha256(payload).hexdigest())

        markers = self._ready_paths()
        artifacts = self._artifact_paths()
        self.assertEqual(len(markers), 1)
        self.assertEqual(len(artifacts), 1)
        marker = json.loads(markers[0].read_text(encoding="utf-8"))
        self.assertEqual(marker["state"], "ready")
        self.assertEqual(marker["submission_id"], row["id"])
        self.assertEqual(marker["artifact_relative_path"], row["artifact_relative_path"])
        self.assertEqual(artifacts[0].read_bytes(), payload)
        self.assertTrue((artifacts[0].parent / "receipt.json").is_file())
        self.assertEqual(self._capture_dirs(), [])

        self.assertEqual(source.read_bytes(), payload)
        self.assertEqual(small.read_bytes(), b"tiny")
        self.assertEqual([item for item in rows if item["team_name"] == "Team2_bbbbb"], [])

    def test_changed_file_restarts_stability_observation(self) -> None:
        source = self._write_model("Team1_aaaaa", "changing.pt", b"a" * 64)
        self._poll()

        replacement = b"b" * 96
        source.write_bytes(replacement)
        self._poll()

        self.assertEqual(self.database.rows(), [])
        self.assertEqual(self._capture_dirs(), [])
        observation = self.watcher.observations[("Team1_aaaaa", "changing.pt")]
        self.assertEqual(observation.signature.size, len(replacement))
        self.assertEqual(observation.unchanged_confirmations, 0)

        self._poll()
        self.assertEqual(len(self._capture_dirs()), 1)
        self.assertEqual(self.database.rows(), [])

        self._poll()
        rows = self.database.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_size_bytes"], len(replacement))
        self.assertEqual(
            rows[0]["source_sha256"], hashlib.sha256(replacement).hexdigest()
        )
        self.assertEqual(source.read_bytes(), replacement)

    def test_same_size_staging_tamper_is_discarded_before_marker(self) -> None:
        payload = b"staged-integrity-check" * 6
        source = self._write_model("Team1_aaaaa", "stage-tamper.pt", payload)
        self._poll(2)

        stage_dir = self._capture_dirs()[0]
        staged_checkpoint = next(stage_dir.glob("*.pt"))
        staged_checkpoint.chmod(0o600)
        staged_checkpoint.write_bytes(b"x" * len(payload))

        with self.assertLogs("ta-grader.watcher", level="WARNING"):
            self._poll()

        self.assertEqual(self.database.rows(), [])
        self.assertEqual(self._ready_paths(), [])
        self.assertEqual(self._artifact_paths(), [])
        self.assertEqual(self._capture_dirs(), [])
        self.assertEqual(source.read_bytes(), payload)

    def test_capture_backpressure_allows_at_most_one_new_model_per_team_poll(self) -> None:
        self._write_model("Team1_aaaaa", "a.pt", b"a" * 64)
        self._write_model("Team1_aaaaa", "b.pt", b"b" * 64)
        self._write_model("Team2_bbbbb", "c.pt", b"c" * 64)

        self._poll(2)

        captures = [
            json.loads((stage / "capture.json").read_text(encoding="utf-8"))
            for stage in self._capture_dirs()
        ]
        self.assertEqual(len(captures), 2)
        self.assertEqual(
            {capture["team_name"] for capture in captures},
            {"Team1_aaaaa", "Team2_bbbbb"},
        )

    def test_same_team_same_sha_is_recorded_as_duplicate_without_new_job(self) -> None:
        payload = b"same-checkpoint-content" * 5
        first = self._write_model("Team1_aaaaa", "first.pt", payload)
        self._poll(3)
        original_row = self.database.rows()[0]

        duplicate = self._write_model("Team1_aaaaa", "renamed.pt", payload)
        self._poll(3)

        rows = self.database.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], original_row["id"])
        self.assertEqual(len(self._ready_paths()), 1)
        self.assertEqual(len(self._artifact_paths()), 1)
        self.assertEqual(self._capture_dirs(), [])
        self.assertEqual(first.read_bytes(), payload)
        self.assertEqual(duplicate.read_bytes(), payload)

        with self.database.connect() as connection:
            source_version = connection.execute(
                "SELECT disposition, submission_id FROM source_versions "
                "WHERE team_name = ? AND source_relative_path = ?",
                ("Team1_aaaaa", "renamed.pt"),
            ).fetchone()
            next_number = connection.execute(
                "SELECT next_submission_number FROM team_counters WHERE team_name = ?",
                ("Team1_aaaaa",),
            ).fetchone()[0]
        self.assertEqual(source_version["disposition"], "duplicate")
        self.assertEqual(source_version["submission_id"], original_row["id"])
        self.assertEqual(next_number, 2)

    def test_team_submission_limit_is_atomic_and_extra_model_is_recorded(self) -> None:
        self.settings = replace(self.settings, max_submissions_per_team=1)
        self.watcher = SubmissionWatcher(self.settings, self.database)
        self._write_model("Team1_aaaaa", "first-limit.pt", b"a" * 64)
        self._poll(3)
        self._write_model("Team1_aaaaa", "over-limit.pt", b"b" * 65)

        with self.assertLogs("ta-grader.watcher", level="WARNING"):
            self._poll(3)

        rows = self.database.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["submission_number"], 1)
        self.assertEqual(len(self._ready_paths()), 1)
        self.assertEqual(len(self._artifact_paths()), 1)
        self.assertEqual(self._capture_dirs(), [])
        with self.database.connect() as connection:
            disposition = connection.execute(
                "SELECT disposition FROM source_versions "
                "WHERE team_name = ? AND source_relative_path = ?",
                ("Team1_aaaaa", "over-limit.pt"),
            ).fetchone()[0]
            next_number = connection.execute(
                "SELECT next_submission_number FROM team_counters WHERE team_name = ?",
                ("Team1_aaaaa",),
            ).fetchone()[0]
        self.assertEqual(disposition, "limit_exceeded")
        self.assertEqual(next_number, 2)

    def test_reconcile_valid_marker_rebuilds_an_empty_database_idempotently(self) -> None:
        payload = b"checkpoint-for-reconcile" * 5
        self._write_model("Team2_bbbbb", "reconcile.pt", payload)
        self._poll(3)
        original = self.database.rows()[0]

        rebuilt_settings = replace(
            self.settings,
            state_root=self.root / "rebuilt-state",
            database_path=self.root / "rebuilt-state/grading.sqlite3",
        )
        rebuilt_database = Database(rebuilt_settings.database_path)
        rebuilt_database.initialize()
        rebuilt_watcher = SubmissionWatcher(rebuilt_settings, rebuilt_database)

        self.assertEqual(rebuilt_watcher.reconcile(), 1)
        self.assertEqual(rebuilt_watcher.reconcile(), 0)
        rebuilt_rows = rebuilt_database.rows()
        self.assertEqual(len(rebuilt_rows), 1)
        self.assertEqual(rebuilt_rows[0]["id"], original["id"])
        self.assertEqual(rebuilt_rows[0]["status"], "queued")
        self.assertEqual(rebuilt_rows[0]["source_sha256"], hashlib.sha256(payload).hexdigest())

    def test_reconcile_recovers_missing_marker_from_verified_receipt(self) -> None:
        payload = b"orphaned-receipt-model" * 5
        self._write_model("Team1_aaaaa", "orphan.pt", payload)
        self._poll(3)
        original = self.database.rows()[0]
        marker = self._ready_paths()[0]
        marker.unlink()

        rebuilt_settings = replace(
            self.settings,
            state_root=self.root / "orphan-state",
            database_path=self.root / "orphan-state/grading.sqlite3",
        )
        rebuilt_database = Database(rebuilt_settings.database_path)
        rebuilt_database.initialize()
        rebuilt_watcher = SubmissionWatcher(rebuilt_settings, rebuilt_database)

        with self.assertLogs("ta-grader.watcher", level="WARNING"):
            inserted = rebuilt_watcher.reconcile()

        self.assertEqual(inserted, 1)
        self.assertTrue(marker.is_file())
        recovered = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(recovered["state"], "ready")
        self.assertEqual(recovered["submission_id"], original["id"])
        self.assertEqual(rebuilt_database.rows()[0]["id"], original["id"])

    def test_reconcile_rejects_marker_when_artifact_size_is_wrong(self) -> None:
        payload = b"artifact-to-corrupt" * 6
        self._write_model("Team1_aaaaa", "corrupt.pt", payload)
        self._poll(3)
        artifact = self._artifact_paths()[0]
        artifact.chmod(0o600)
        artifact.write_bytes(b"short")

        rebuilt_settings = replace(
            self.settings,
            state_root=self.root / "corrupt-state",
            database_path=self.root / "corrupt-state/grading.sqlite3",
        )
        rebuilt_database = Database(rebuilt_settings.database_path)
        rebuilt_database.initialize()
        rebuilt_watcher = SubmissionWatcher(rebuilt_settings, rebuilt_database)

        with self.assertLogs("ta-grader.watcher", level="ERROR"):
            inserted = rebuilt_watcher.reconcile()

        self.assertEqual(inserted, 0)
        self.assertEqual(rebuilt_database.rows(), [])

    def test_reconcile_rejects_same_size_artifact_with_wrong_sha(self) -> None:
        payload = b"same-size-integrity-check" * 6
        self._write_model("Team1_aaaaa", "same-size-corrupt.pt", payload)
        self._poll(3)
        artifact = self._artifact_paths()[0]
        artifact.chmod(0o600)
        artifact.write_bytes(b"x" * len(payload))
        self.assertEqual(artifact.stat().st_size, len(payload))

        rebuilt_settings = replace(
            self.settings,
            state_root=self.root / "same-size-state",
            database_path=self.root / "same-size-state/grading.sqlite3",
        )
        rebuilt_database = Database(rebuilt_settings.database_path)
        rebuilt_database.initialize()
        rebuilt_watcher = SubmissionWatcher(rebuilt_settings, rebuilt_database)

        with self.assertLogs("ta-grader.watcher", level="ERROR"):
            inserted = rebuilt_watcher.reconcile()

        self.assertEqual(inserted, 0)
        self.assertEqual(rebuilt_database.rows(), [])


if __name__ == "__main__":
    unittest.main()
