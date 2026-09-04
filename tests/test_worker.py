from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.config import Settings  # noqa: E402
from ta_grading.database import Database  # noqa: E402
from ta_grading.metrics import (  # noqa: E402
    EXPECTED_FORGET_WNIDS,
    REFERENCE_ACC_F,
    REFERENCE_ACC_R,
    TEST_DATASET_REVISION,
)
from ta_grading import worker  # noqa: E402


def harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)


def report_for(submission_id: str) -> dict:
    acc_f, acc_r = 10.0, 90.0
    reference_acc_f, reference_acc_r = REFERENCE_ACC_F, REFERENCE_ACC_R
    drop_r = max(reference_acc_r - acc_r, 0.0)
    gap_f = abs(acc_f - reference_acc_f)
    aus = (1.0 - drop_r / 100.0) / (1.0 + gap_f / 100.0)
    cka = {
        "b4": {"CKA_f_o": 0.10, "CKA_r_o": 0.90},
        "b8": {"CKA_f_o": 0.20, "CKA_r_o": 0.85},
        "b12": {"CKA_f_o": 0.30, "CKA_r_o": 0.82},
        "pre": {"CKA_f_o": 0.25, "CKA_r_o": 0.80},
    }
    rus_o = harmonic(0.75, 0.80)
    final_score = harmonic(aus, rus_o)
    return {
        "schema_version": 2,
        "score_version": "unlearning-v2",
        "dataset_revision": TEST_DATASET_REVISION,
        "phase": "test",
        "tag": submission_id,
        "accuracy_split": "test",
        "representation_split": "test",
        "score_depth": "pre",
        "split_accuracy": {"test": {"Acc_f": acc_f, "Acc_r": acc_r}},
        "accuracy_metric": {
            "Acc_f": acc_f,
            "Acc_r": acc_r,
            "reference_Acc_f": reference_acc_f,
            "reference_Acc_r": reference_acc_r,
            "drop_r": drop_r,
            "gap_f": gap_f,
            "AUS": aus,
        },
        "representation_metric": {
            "CKA_f_o": 0.25,
            "CKA_r_o": 0.80,
            "RUS_o": rus_o,
        },
        "cka_per_depth": cka,
        "AUS": aus,
        "RUS_o": rus_o,
        "final_score": final_score,
        "forget_wnids": list(EXPECTED_FORGET_WNIDS),
    }


class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        mnt = root / "mnt"
        mnt.mkdir()
        admin = mnt / "Admin-Storage_a"
        admin.mkdir()
        backup = admin / "submission-backups"
        backup.mkdir()
        self.settings = Settings(
            project_root=ROOT,
            mnt_root=mnt,
            admin_root=admin,
            backup_root=backup,
            state_root=root / "state",
            database_path=root / "state/grading.sqlite3",
            grading_root=root / "private/assets",
            grade_script=ROOT / "ops/grade-finalist.sh",
            grading_image="fixture/grader:test",
            expected_team_count=1,
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
        self.submission_id = str(uuid.uuid4())
        self.payload = b"trusted-backup-checkpoint"
        relative_dir = (
            f"artifacts/Team1_a/submission-0001-{self.submission_id}"
        )
        artifact_dir = backup / relative_dir
        artifact_dir.mkdir(parents=True)
        self.checkpoint = artifact_dir / f"{self.submission_id}.pt"
        self.checkpoint.write_bytes(self.payload)
        digest = hashlib.sha256(self.payload).hexdigest()
        self.marker = {
            "schema_version": 1,
            "state": "ready",
            "submission_id": self.submission_id,
            "team_name": "Team1_a",
            "team_number": 1,
            "submission_number": 1,
            "model_name": "winning.pt",
            "source_relative_path": "winning.pt",
            "size_bytes": len(self.payload),
            "source_mtime_ns": 10,
            "source_ctime_ns": 20,
            "sha256": digest,
            "artifact_relative_path": f"{relative_dir}/{self.submission_id}.pt",
            "receipt_relative_path": f"{relative_dir}/receipt.json",
            "marker_relative_path": f"ready/{self.submission_id}.json",
            "captured_at": "2026-09-04T00:00:00+00:00",
            "ready_at": "2026-09-04T00:01:00+00:00",
        }
        marker_path = backup / self.marker["marker_relative_path"]
        marker_path.parent.mkdir()
        marker_path.write_text(json.dumps(self.marker), encoding="utf-8")
        self.assertTrue(self.database.ingest_marker(self.marker))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _claimed(self, gpu: int = 2) -> dict:
        row = self.database.claim_next(gpu, f"test-worker-{gpu}", worker.utc_now())
        self.assertIsNotNone(row)
        return row

    def _write_outputs(self, command: list[str]) -> None:
        report_path = Path(command[command.index("--report") + 1])
        submission_id = command[command.index("--submission-id") + 1]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_for(submission_id), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_path = report_path.with_name("score.audit.json")
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "finalist-grading-audit-v2",
                    "submission_id": submission_id,
                    "original_checkpoint_sha256": self.marker["sha256"],
                    "converted_safetensors_sha256": "b" * 64,
                    "final_report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    "score_version": "unlearning-v2",
                    "test_dataset_revision": TEST_DATASET_REVISION,
                    "grader_image_id": "sha256:" + "c" * 64,
                }
            ),
            encoding="utf-8",
        )

    def test_grades_on_requested_gpu_and_persists_all_metrics(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="graded\n")

        row = self._claimed(2)
        with mock.patch.object(worker.subprocess, "run", side_effect=fake_run), mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, row, 2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("--gpu") + 1], "2")
        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "done")
        self.assertEqual(stored["model_name"], "winning.pt")
        self.assertAlmostEqual(stored["cka_f_o"], 0.25)
        self.assertAlmostEqual(stored["cka_r_o"], 0.80)
        expected_aus = (1.0 - (REFERENCE_ACC_R - 90.0) / 100.0) / 1.1
        self.assertAlmostEqual(stored["aus"], expected_aus)
        self.assertAlmostEqual(stored["rus_o"], harmonic(0.75, 0.80))
        self.assertEqual(stored["f1"], stored["final_score"])
        self.assertEqual(stored["worker_gpu"], 2)

    def test_failure_is_recorded_and_can_be_retried(self) -> None:
        row = self._claimed(1)
        failed = subprocess.CompletedProcess(
            [str(self.settings.grade_script)], 7, stdout="conversion incomplete"
        )
        with mock.patch.object(worker.subprocess, "run", return_value=failed), mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, row, 1)

        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "error")
        self.assertIn("conversion incomplete", stored["error"])
        self.assertTrue(self.database.retry(self.submission_id))

    def test_existing_valid_report_pair_is_recovered_without_rescoring(self) -> None:
        row = self._claimed(3)
        report_path, _ = worker._attempt_paths(self.checkpoint, row)
        command = [
            "fake",
            "--report",
            str(report_path),
            "--submission-id",
            self.submission_id,
        ]
        self._write_outputs(command)

        with mock.patch.object(worker.subprocess, "run") as run, mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, row, 3)

        run.assert_not_called()
        self.assertEqual(self.database.rows()[0]["status"], "done")

    def test_partial_report_pair_is_retained_but_does_not_poison_retry(self) -> None:
        first = self._claimed(1)
        first_report, first_audit = worker._attempt_paths(self.checkpoint, first)
        first_report.write_text(
            json.dumps(report_for(self.submission_id)), encoding="utf-8"
        )

        with mock.patch.object(worker.subprocess, "run") as first_run, mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, first, 1)

        first_run.assert_not_called()
        errored = self.database.rows()[0]
        self.assertEqual(errored["status"], "error")
        self.assertIn("only one member", errored["error"])
        self.assertTrue(first_report.is_file())
        self.assertFalse(first_audit.exists())

        self.assertTrue(
            self.database.retry(self.submission_id, first["claim_token"])
        )
        second = self._claimed(1)
        second_report, second_audit = worker._attempt_paths(self.checkpoint, second)
        self.assertNotEqual(first_report.parent, second_report.parent)

        def fake_run(command, **_kwargs):
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="graded\n")

        with mock.patch.object(worker.subprocess, "run", side_effect=fake_run), mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, second, 1)

        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "done")
        self.assertTrue(first_report.is_file())
        self.assertFalse(first_audit.exists())
        self.assertTrue(second_report.is_file())
        self.assertTrue(second_audit.is_file())
        self.assertEqual(
            stored["report_relative_path"],
            second_report.relative_to(self.settings.backup_root).as_posix(),
        )

    def test_invalid_report_pair_is_retained_but_does_not_poison_retry(self) -> None:
        first = self._claimed(2)
        first_report, first_audit = worker._attempt_paths(self.checkpoint, first)
        self._write_outputs(
            [
                "fake",
                "--report",
                str(first_report),
                "--submission-id",
                self.submission_id,
            ]
        )
        first_report.write_text("{not-json", encoding="utf-8")

        with mock.patch.object(worker.subprocess, "run") as first_run, mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, first, 2)
        first_run.assert_not_called()
        self.assertEqual(self.database.rows()[0]["status"], "error")

        self.assertTrue(
            self.database.retry(self.submission_id, first["claim_token"])
        )
        second = self._claimed(2)
        second_report, _ = worker._attempt_paths(self.checkpoint, second)

        def fake_run(command, **_kwargs):
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="graded\n")

        with mock.patch.object(worker.subprocess, "run", side_effect=fake_run), mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, second, 2)

        self.assertEqual(self.database.rows()[0]["status"], "done")
        self.assertEqual(first_report.read_text(encoding="utf-8"), "{not-json")
        self.assertTrue(first_audit.is_file())
        self.assertNotEqual(first_report.parent, second_report.parent)

    def test_lost_worker_output_cannot_change_the_new_claim(self) -> None:
        stale = self._claimed(1)
        self.assertEqual(self.database.requeue_running(), 1)
        current = self._claimed(2)

        def fake_run(command, **_kwargs):
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="stale result\n")

        with mock.patch.object(worker.subprocess, "run", side_effect=fake_run), mock.patch.object(
            worker, "publish_state"
        ) as publish:
            worker.process_one(self.settings, self.database, stale, 1)

        publish.assert_not_called()
        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "running")
        self.assertEqual(stored["claim_token"], current["claim_token"])
        self.assertEqual(stored["worker_id"], current["worker_id"])
        self.assertIsNone(stored["final_score"])
        self.assertIsNone(stored["error"])

    def test_checkpoint_hash_is_rechecked_before_scoring(self) -> None:
        row = self._claimed(2)
        self.checkpoint.write_bytes(b"x" * len(self.payload))
        with mock.patch.object(worker.subprocess, "run") as run, mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(self.settings, self.database, row, 2)

        run.assert_not_called()
        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "error")
        self.assertIn("SHA-256", stored["error"])

    def test_pinned_grader_image_must_match_audit(self) -> None:
        row = self._claimed(3)
        pinned = replace(self.settings, grading_image="sha256:" + "d" * 64)

        def fake_run(command, **_kwargs):
            self._write_outputs(command)
            return subprocess.CompletedProcess(command, 0, stdout="graded\n")

        with mock.patch.object(worker.subprocess, "run", side_effect=fake_run), mock.patch.object(
            worker, "publish_state"
        ):
            worker.process_one(pinned, self.database, row, 3)

        stored = self.database.rows()[0]
        self.assertEqual(stored["status"], "error")
        self.assertIn("pinned image", stored["error"])

    def test_only_one_worker_can_hold_a_gpu_lifetime_lock(self) -> None:
        with worker._gpu_lifetime_lock(self.settings, 1):
            with self.assertRaisesRegex(RuntimeError, "already owns CUDA GPU 1"):
                worker.worker_loop(self.settings, 1, once=True)
        self.assertEqual(self.database.rows()[0]["status"], "queued")

    def test_gpu_zero_is_never_an_automatic_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in automatic worker"):
            worker.worker_loop(self.settings, 0, once=True)


if __name__ == "__main__":
    unittest.main()
