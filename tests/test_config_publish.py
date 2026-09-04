from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.config import ConfigError, Settings  # noqa: E402
from ta_grading.database import Database  # noqa: E402
from ta_grading.publish import publish_state  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_defaults_match_the_22_team_four_gpu_server(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.mnt_root, Path("/mnt"))
        self.assertEqual(settings.admin_root, Path("/mnt/Admin-Storage_7ed0d"))
        self.assertEqual(settings.expected_team_count, 22)
        self.assertEqual(settings.max_team_number, 26)
        self.assertEqual(settings.poll_seconds, 20.0)
        self.assertEqual(settings.stable_confirmations, 3)
        self.assertEqual(settings.post_copy_seconds, 20.0)
        self.assertEqual(settings.min_checkpoint_bytes, 300_000_000)
        self.assertEqual(settings.max_submissions_per_team, 30)
        self.assertEqual(settings.max_pending_captures, 6)
        self.assertEqual(settings.gpu_ids, (1, 2, 3))
        self.assertEqual(
            settings.score_post_url, "https://api.minds.ai.kr/submit"
        )

    def test_rejects_gpu_zero_or_a_different_automatic_set(self) -> None:
        for value in ("0,1,2", "1,2", "2,3,4"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"TA_GPU_IDS": value}, clear=True
            ):
                with self.assertRaises(ConfigError):
                    Settings.from_env()

    def test_submission_limit_environment_overrides_default(self) -> None:
        with mock.patch.dict(
            os.environ, {"TA_MAX_SUBMISSIONS_PER_TEAM": "17"}, clear=True
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.max_submissions_per_team, 17)

    def test_preserves_virtualenv_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_python = root / "base-python"
            base_python.write_bytes(b"")
            venv_python = root / "venv-python"
            venv_python.symlink_to(base_python)
            with mock.patch.dict(
                os.environ,
                {"TA_GRADER_PYTHON": str(venv_python)},
                clear=True,
            ):
                settings = Settings.from_env()

            self.assertEqual(settings.grader_python, venv_python.absolute())
            self.assertTrue(settings.grader_python.is_symlink())

    def test_rejects_relative_grader_python_path(self) -> None:
        with mock.patch.dict(
            os.environ, {"TA_GRADER_PYTHON": "python3"}, clear=True
        ):
            with self.assertRaisesRegex(ConfigError, "absolute path"):
                Settings.from_env()

    def test_rejects_sqlite_on_shared_mount(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TA_DATABASE_PATH": "/mnt/Admin-Storage_7ed0d/grading.sqlite3"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigError, "local disk"):
                Settings.from_env()

    def test_rejects_recursive_team_scan(self) -> None:
        with mock.patch.dict(os.environ, {"TA_RECURSIVE_SCAN": "1"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "direct Team children"):
                Settings.from_env()

    def test_max_team_number_cannot_exclude_required_teams(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TA_EXPECTED_TEAM_COUNT": "22", "TA_MAX_TEAM_NUMBER": "21"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigError, "at least"):
                Settings.from_env()

    def test_rejects_non_https_score_endpoint(self) -> None:
        with mock.patch.dict(
            os.environ, {"TA_SCORE_POST_URL": "http://api.minds.ai.kr/submit"}, clear=True
        ):
            with self.assertRaisesRegex(ConfigError, "HTTPS URL"):
                Settings.from_env()

    def test_rejects_invalid_post_limits_and_endpoint_port(self) -> None:
        cases = (
            {"TA_SCORE_POST_TIMEOUT_SECONDS": "nan"},
            {"TA_SCORE_POST_RETRY_SECONDS": "inf"},
            {"TA_SCORE_POST_URL": "https://api.minds.ai.kr:99999/submit"},
        )
        for environment in cases:
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ, environment, clear=True
            ):
                with self.assertRaises(ConfigError):
                    Settings.from_env()


class PublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(
            project_root=ROOT,
            mnt_root=root / "mnt",
            admin_root=root / "mnt/Admin-Storage_a",
            backup_root=root / "mnt/Admin-Storage_a/submission-backups",
            state_root=root / "state",
            database_path=root / "state/grading.sqlite3",
            grading_root=root / "private/assets",
            grade_script=ROOT / "ops/grade-finalist.py",
            grader_python=Path(sys.executable),
            grader_runtime_id="sha256:" + "a" * 64,
            expected_team_count=1,
            max_team_number=1,
            poll_seconds=0,
            stable_confirmations=1,
            post_copy_seconds=0,
            min_checkpoint_bytes=1,
            max_checkpoint_bytes=4096,
            recursive_scan=False,
            worker_poll_seconds=0,
            gpu_ids=(1, 2, 3),
        )
        self.database = Database(self.settings.database_path)
        self.database.initialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_publishes_json_csv_and_consistent_sqlite_snapshot(self) -> None:
        submission_id = str(uuid.uuid4())
        marker = {
            "submission_id": submission_id,
            "team_name": "Team1_a",
            "team_number": 1,
            "submission_number": 1,
            "model_name": "model.pt",
            "source_relative_path": "model.pt",
            "size_bytes": 100,
            "source_mtime_ns": 10,
            "source_ctime_ns": 20,
            "sha256": "a" * 64,
            "artifact_relative_path": f"artifacts/Team1_a/submission-0001-{submission_id}/{submission_id}.pt",
            "receipt_relative_path": f"artifacts/Team1_a/submission-0001-{submission_id}/receipt.json",
            "marker_relative_path": f"ready/{submission_id}.json",
            "captured_at": "2026-09-04T00:00:00+00:00",
            "ready_at": "2026-09-04T00:01:00+00:00",
        }
        self.assertTrue(self.database.ingest_marker(marker))
        publish_state(self.settings, self.database)

        results = self.settings.backup_root / "results"
        payload = json.loads((results / "submissions.json").read_text())
        self.assertEqual(payload["submissions"][0]["id"], submission_id)
        self.assertIn("f1", payload["submissions"][0])
        with (results / "submissions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["id"], submission_id)
        self.assertIn("f1", rows[0])

        snapshot = sqlite3.connect(results / "grading.sqlite3.snapshot")
        try:
            row = snapshot.execute("SELECT id, status FROM submissions").fetchone()
        finally:
            snapshot.close()
        self.assertEqual(row, (submission_id, "queued"))


if __name__ == "__main__":
    unittest.main()
