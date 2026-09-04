from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.config import Settings  # noqa: E402
from ta_grading.database import Database  # noqa: E402
import ta_grading.publish as publish_module  # noqa: E402


class PublishHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(
            project_root=ROOT,
            mnt_root=self.root / "mnt",
            admin_root=self.root / "mnt/Admin-Storage_a",
            backup_root=self.root / "mnt/Admin-Storage_a/submission-backups",
            state_root=self.root / "state",
            database_path=self.root / "state/grading.sqlite3",
            grading_root=self.root / "private/assets",
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

    def _ingest(
        self,
        *,
        model_name: str = "model.pt",
        source_relative_path: str = "model.pt",
    ) -> str:
        submission_id = str(uuid.uuid4())
        marker = {
            "submission_id": submission_id,
            "team_name": "Team1_a",
            "team_number": 1,
            "submission_number": 1,
            "model_name": model_name,
            "source_relative_path": source_relative_path,
            "size_bytes": 100,
            "source_mtime_ns": 10,
            "source_ctime_ns": 20,
            "sha256": "a" * 64,
            "artifact_relative_path": (
                f"artifacts/Team1_a/submission-0001-{submission_id}/"
                f"{submission_id}.pt"
            ),
            "receipt_relative_path": (
                f"artifacts/Team1_a/submission-0001-{submission_id}/receipt.json"
            ),
            "marker_relative_path": f"ready/{submission_id}.json",
            "captured_at": "2026-09-04T00:00:00+00:00",
            "ready_at": "2026-09-04T00:01:00+00:00",
        }
        self.assertTrue(self.database.ingest_marker(marker))
        return submission_id

    def test_all_outputs_derive_from_the_same_sqlite_snapshot(self) -> None:
        submission_id = self._ingest()
        original_atomic_replace = publish_module._atomic_replace
        mutation_done = False

        def mutate_live_database_then_write(path, writer):
            nonlocal mutation_done
            if not mutation_done:
                mutation_done = True
                with self.database.connect() as connection:
                    connection.execute(
                        "UPDATE submissions SET status = 'error' WHERE id = ?",
                        (submission_id,),
                    )
            return original_atomic_replace(path, writer)

        with mock.patch.object(
            publish_module, "_atomic_replace", mutate_live_database_then_write
        ):
            publish_module.publish_state(self.settings, self.database)

        results = self.settings.backup_root / "results"
        payload = json.loads((results / "submissions.json").read_text())
        with (results / "submissions.csv").open(newline="") as stream:
            csv_row = next(csv.DictReader(stream))
        snapshot = sqlite3.connect(results / "grading.sqlite3.snapshot")
        try:
            snapshot_status = snapshot.execute(
                "SELECT status FROM submissions WHERE id = ?", (submission_id,)
            ).fetchone()[0]
        finally:
            snapshot.close()
        with self.database.connect() as connection:
            live_status = connection.execute(
                "SELECT status FROM submissions WHERE id = ?", (submission_id,)
            ).fetchone()[0]

        self.assertEqual(payload["submissions"][0]["status"], "queued")
        self.assertEqual(csv_row["status"], "queued")
        self.assertEqual(snapshot_status, "queued")
        self.assertEqual(live_status, "error")

    def test_csv_neutralizes_formulas_but_json_preserves_original_strings(self) -> None:
        model_name = "=HYPERLINK(\"https://example.invalid\").pt"
        source_relative_path = " \t@SUM(1+1).pt"
        self._ingest(
            model_name=model_name,
            source_relative_path=source_relative_path,
        )

        publish_module.publish_state(self.settings, self.database)

        results = self.settings.backup_root / "results"
        payload = json.loads((results / "submissions.json").read_text())
        with (results / "submissions.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))

        self.assertEqual(payload["submissions"][0]["model_name"], model_name)
        self.assertEqual(
            payload["submissions"][0]["source_relative_path"],
            source_relative_path,
        )
        self.assertEqual(row["model_name"], "'" + model_name)
        self.assertEqual(row["source_relative_path"], "'" + source_relative_path)

    def test_rejects_symlinked_results_directory(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.settings.backup_root.mkdir(parents=True)
        (self.settings.backup_root / "results").symlink_to(
            outside, target_is_directory=True
        )

        with self.assertRaisesRegex(
            publish_module.PublishError, "symlinked publish directory"
        ):
            publish_module.publish_state(self.settings, self.database)
        self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlink_in_results_ancestor(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.settings.backup_root.parent.mkdir(parents=True)
        self.settings.backup_root.symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(
            publish_module.PublishError, "symlinked publish directory"
        ):
            publish_module.publish_state(self.settings, self.database)
        self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlinked_output_file(self) -> None:
        outside = self.root / "outside.json"
        outside.write_text("do not overwrite")
        results = self.settings.backup_root / "results"
        results.mkdir(parents=True)
        (results / "submissions.json").symlink_to(outside)

        with self.assertRaisesRegex(
            publish_module.PublishError, "symlinked publish target"
        ):
            publish_module.publish_state(self.settings, self.database)
        self.assertEqual(outside.read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
