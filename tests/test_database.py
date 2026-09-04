from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ta_grading.database import (  # noqa: E402
    DATABASE_SCHEMA_VERSION,
    SCHEMA,
    Database,
    LostClaimError,
    SubmissionLimitError,
)


def marker(
    token: str,
    *,
    team_name: str = "Team1_aaaaa",
    team_number: int = 1,
    submission_number: int = 1,
) -> dict:
    submission_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"database-test:{token}"))
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    artifact_dir = (
        f"artifacts/{team_name}/submission-{submission_number:04d}-{submission_id}"
    )
    return {
        "submission_id": submission_id,
        "team_name": team_name,
        "team_number": team_number,
        "submission_number": submission_number,
        "model_name": f"{token}.pt",
        "source_relative_path": f"{token}.pt",
        "size_bytes": 380_000_000 + submission_number,
        "source_mtime_ns": 1_000_000 + submission_number,
        "source_ctime_ns": 2_000_000 + submission_number,
        "sha256": digest,
        "artifact_relative_path": f"{artifact_dir}/{submission_id}.pt",
        "receipt_relative_path": f"{artifact_dir}/receipt.json",
        "marker_relative_path": f"ready/{submission_id}.json",
        "captured_at": f"2026-09-04T00:00:{submission_number:02d}+00:00",
        "ready_at": f"2026-09-04T00:01:{submission_number:02d}+00:00",
    }


def metrics() -> dict:
    cka_per_depth = {
        depth: {"CKA_f_o": 0.2, "CKA_r_o": 0.8}
        for depth in ("b4", "b8", "b12", "pre")
    }
    return {
        "score_depth": "pre",
        "acc_f": 10.0,
        "acc_r": 90.0,
        "cka_f_o": 0.2,
        "cka_r_o": 0.8,
        "cka_per_depth_json": json.dumps(cka_per_depth, sort_keys=True),
        "aus": 0.75,
        "rus_o": 0.8,
        "final_score": 24.0 / 31.0,
        "f1_alias": 24.0 / 31.0,
        "report_json": json.dumps({"phase": "test", "tag": "fixture"}),
    }


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "state" / "grading.sqlite3"
        self.database = Database(self.database_path)
        self.database.initialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initialize_migrates_legacy_claims_and_source_dispositions(self) -> None:
        legacy_path = Path(self.temporary.name) / "legacy.sqlite3"
        legacy_schema = SCHEMA.replace("    claim_token TEXT,\n", "").replace(
            ", 'limit_exceeded'", ""
        )
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(legacy_schema)

        legacy_database = Database(legacy_path)
        value = marker("legacy")
        self.assertTrue(legacy_database.ingest_marker(value))
        with legacy_database.connect() as connection:
            connection.execute(
                "UPDATE submissions SET status = 'running', worker_id = 'legacy', "
                "worker_gpu = 1, started_at = '2026-09-04T00:00:00+00:00', "
                "attempt_count = 1 WHERE id = ?",
                (value["submission_id"],),
            )

        legacy_database.initialize()

        migrated = legacy_database.rows()[0]
        self.assertEqual(migrated["status"], "queued")
        self.assertIsNone(migrated["claim_token"])
        self.assertIn("claim-token migration", migrated["error"])
        claimed = legacy_database.claim_next(
            1, "new-worker", "2026-09-04T00:10:00+00:00"
        )
        self.assertIsNotNone(claimed)
        self.assertRegex(claimed["claim_token"], r"\A[0-9a-f]{64}\Z")
        legacy_database.record_source_version(
            team_name=value["team_name"],
            source_relative_path="legacy-over-limit.pt",
            size_bytes=value["size_bytes"] + 1,
            mtime_ns=value["source_mtime_ns"] + 1,
            ctime_ns=value["source_ctime_ns"] + 1,
            sha256=hashlib.sha256(b"legacy-over-limit").hexdigest(),
            disposition="limit_exceeded",
            submission_id=value["submission_id"],
        )
        with legacy_database.connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, DATABASE_SCHEMA_VERSION)

    def test_submission_numbers_increase_independently_per_team(self) -> None:
        self.assertEqual(self.database.allocate_submission_number("Team1_a"), 1)
        self.assertEqual(self.database.allocate_submission_number("Team1_a"), 2)
        self.assertEqual(self.database.allocate_submission_number("Team2_b"), 1)
        self.assertEqual(self.database.allocate_submission_number("Team1_a"), 3)
        self.assertEqual(self.database.allocate_submission_number("Team2_b"), 2)

        recovered = marker(
            "recovered-seven",
            team_name="Team3_c",
            team_number=3,
            submission_number=7,
        )
        self.assertTrue(self.database.ingest_marker(recovered))
        self.assertEqual(self.database.allocate_submission_number("Team3_c"), 8)

    def test_submission_limit_is_atomic_and_does_not_advance_counter(self) -> None:
        self.assertEqual(
            self.database.allocate_submission_number("Team1_limit", max_count=2), 1
        )
        self.assertEqual(
            self.database.allocate_submission_number("Team1_limit", max_count=2), 2
        )
        with self.assertRaises(SubmissionLimitError):
            self.database.allocate_submission_number("Team1_limit", max_count=2)
        with self.assertRaises(SubmissionLimitError):
            self.database.allocate_submission_number("Team1_limit", max_count=2)
        self.assertEqual(
            self.database.allocate_submission_number("Team1_limit"), 3
        )

        with self.assertRaises(SubmissionLimitError):
            self.database.allocate_submission_number("Team2_limit", max_count=0)
        self.assertEqual(self.database.allocate_submission_number("Team2_limit"), 1)

    def test_marker_ingest_is_idempotent_and_tracks_source_versions(self) -> None:
        first = marker("first")

        self.assertTrue(self.database.ingest_marker(first))
        self.assertFalse(self.database.ingest_marker(first))
        self.assertEqual(len(self.database.rows()), 1)
        self.assertTrue(
            self.database.has_source_version(
                first["team_name"],
                first["source_relative_path"],
                first["size_bytes"],
                first["source_mtime_ns"],
                first["source_ctime_ns"],
            )
        )
        found = self.database.find_team_sha(first["team_name"], first["sha256"])
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], first["submission_id"])

        self.database.record_source_version(
            team_name=first["team_name"],
            source_relative_path="renamed-copy.pt",
            size_bytes=first["size_bytes"],
            mtime_ns=first["source_mtime_ns"] + 10,
            ctime_ns=first["source_ctime_ns"] + 10,
            sha256=first["sha256"],
            disposition="duplicate",
            submission_id=first["submission_id"],
        )
        self.database.record_source_version(
            team_name=first["team_name"],
            source_relative_path="over-limit.pt",
            size_bytes=first["size_bytes"] + 1,
            mtime_ns=first["source_mtime_ns"] + 20,
            ctime_ns=first["source_ctime_ns"] + 20,
            sha256=hashlib.sha256(b"over-limit").hexdigest(),
            disposition="limit_exceeded",
            submission_id=first["submission_id"],
        )
        self.assertTrue(
            self.database.has_source_version(
                first["team_name"],
                "renamed-copy.pt",
                first["size_bytes"],
                first["source_mtime_ns"] + 10,
                first["source_ctime_ns"] + 10,
            )
        )

        with self.database.connect() as connection:
            versions = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM source_versions ORDER BY source_relative_path"
                )
            ]
        self.assertEqual(len(versions), 3)
        self.assertEqual(
            {row["disposition"] for row in versions},
            {"queued", "duplicate", "limit_exceeded"},
        )
        self.assertTrue(
            all(row["submission_id"] == first["submission_id"] for row in versions)
        )

        with self.assertRaisesRegex(ValueError, "disposition"):
            self.database.record_source_version(
                team_name=first["team_name"],
                source_relative_path="bad.pt",
                size_bytes=first["size_bytes"],
                mtime_ns=3,
                ctime_ns=4,
                sha256=first["sha256"],
                disposition="unknown",
                submission_id=first["submission_id"],
            )

    def test_marker_unique_conflicts_do_not_poison_dedup_state(self) -> None:
        first = marker("original")
        self.assertTrue(self.database.ingest_marker(first))

        same_uuid_changed_metadata = dict(first)
        same_uuid_changed_metadata["model_name"] = "tampered.pt"
        with self.assertRaisesRegex(ValueError, "existing submission UUID"):
            self.database.ingest_marker(same_uuid_changed_metadata)

        same_sha_new_identity = marker("other", submission_number=50)
        same_sha_new_identity["sha256"] = first["sha256"]
        with self.assertRaisesRegex(ValueError, "another submission identity"):
            self.database.ingest_marker(same_sha_new_identity)

        rows = self.database.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], first["submission_id"])
        self.assertEqual(rows[0]["model_name"], first["model_name"])
        with self.database.connect() as connection:
            versions = [
                dict(row)
                for row in connection.execute("SELECT * FROM source_versions")
            ]
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["submission_id"], first["submission_id"])
        self.assertEqual(versions[0]["source_sha256"], first["sha256"])

        # Both rejected marker transactions must roll back their counter writes.
        self.assertEqual(self.database.allocate_submission_number(first["team_name"]), 2)

    def test_three_workers_claim_distinct_jobs_atomically(self) -> None:
        expected_ids = set()
        for number in range(1, 4):
            value = marker(f"claim-{number}", submission_number=number)
            expected_ids.add(value["submission_id"])
            self.assertTrue(self.database.ingest_marker(value))

        barrier = threading.Barrier(3)

        def claim(gpu: int) -> dict | None:
            worker_database = Database(self.database_path)
            barrier.wait(timeout=5)
            return worker_database.claim_next(
                gpu,
                f"worker-cuda-{gpu}",
                f"2026-09-04T01:00:0{gpu}+00:00",
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            claimed = list(pool.map(claim, (1, 2, 3)))

        self.assertNotIn(None, claimed)
        claimed_rows = [row for row in claimed if row is not None]
        self.assertEqual({row["id"] for row in claimed_rows}, expected_ids)
        self.assertEqual(len({row["id"] for row in claimed_rows}), 3)
        self.assertEqual({row["worker_gpu"] for row in claimed_rows}, {1, 2, 3})
        self.assertTrue(all(row["status"] == "running" for row in claimed_rows))
        self.assertTrue(all(row["attempt_count"] == 1 for row in claimed_rows))
        tokens = {row["claim_token"] for row in claimed_rows}
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", token) for token in tokens))
        self.assertIsNone(
            self.database.claim_next(1, "fourth-worker", "2026-09-04T01:01:00+00:00")
        )

    def test_mark_done_persists_metrics_and_f1_alias(self) -> None:
        value = marker("done")
        self.assertTrue(self.database.ingest_marker(value))
        claimed = self.database.claim_next(
            2, "worker-cuda-2", "2026-09-04T02:00:00+00:00"
        )
        self.assertIsNotNone(claimed)
        normalized = metrics()

        self.database.mark_done(
            value["submission_id"],
            claimed["claim_token"],
            "2026-09-04T02:30:00+00:00",
            normalized,
            "results/report.json",
            "results/report.audit.json",
        )

        row = self.database.rows()[0]
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["finished_at"], "2026-09-04T02:30:00+00:00")
        self.assertIsNone(row["error"])
        for field in (
            "score_depth",
            "acc_f",
            "acc_r",
            "cka_f_o",
            "cka_r_o",
            "cka_per_depth_json",
            "aus",
            "rus_o",
            "final_score",
            "report_json",
        ):
            self.assertEqual(row[field], normalized[field], field)
        self.assertEqual(row["f1"], normalized["f1_alias"])
        self.assertEqual(row["f1"], row["final_score"])
        self.assertEqual(row["report_relative_path"], "results/report.json")
        self.assertEqual(row["audit_relative_path"], "results/report.audit.json")

        public = Database.row_for_json(row)
        self.assertEqual(public["f1"], public["final_score"])
        self.assertEqual(public["cka_f_o"], normalized["cka_f_o"])
        self.assertEqual(public["cka_r_o"], normalized["cka_r_o"])

    def test_error_can_be_retried_and_attempt_count_increments(self) -> None:
        value = marker("retry")
        self.assertTrue(self.database.ingest_marker(value))
        first_claim = self.database.claim_next(
            1, "worker-cuda-1", "2026-09-04T03:00:00+00:00"
        )
        self.assertIsNotNone(first_claim)
        long_error = "discard-this-prefix:" + "x" * 4100
        self.database.mark_error(
            value["submission_id"],
            first_claim["claim_token"],
            "2026-09-04T03:10:00+00:00",
            long_error,
        )
        errored = self.database.rows()[0]
        self.assertEqual(errored["status"], "error")
        self.assertEqual(len(errored["error"]), 4000)
        self.assertEqual(errored["attempt_count"], 1)

        self.assertFalse(self.database.retry(value["submission_id"], "0" * 64))
        self.assertTrue(
            self.database.retry(value["submission_id"], first_claim["claim_token"])
        )
        self.assertFalse(self.database.retry(value["submission_id"]))
        queued = self.database.rows()[0]
        self.assertEqual(queued["status"], "queued")
        for field in (
            "worker_id",
            "worker_gpu",
            "claim_token",
            "started_at",
            "finished_at",
            "error",
        ):
            self.assertIsNone(queued[field], field)
        self.assertEqual(queued["attempt_count"], 1)

        second_claim = self.database.claim_next(
            3, "worker-cuda-3", "2026-09-04T03:20:00+00:00"
        )
        self.assertIsNotNone(second_claim)
        self.assertEqual(second_claim["id"], value["submission_id"])
        self.assertEqual(second_claim["attempt_count"], 2)
        self.assertEqual(second_claim["worker_gpu"], 3)
        self.assertNotEqual(
            second_claim["claim_token"], first_claim["claim_token"]
        )

    def test_stale_claim_cannot_overwrite_a_reclaimed_submission(self) -> None:
        value = marker("stale-claim")
        self.assertTrue(self.database.ingest_marker(value))
        first = self.database.claim_next(
            1, "old-worker", "2026-09-04T03:30:00+00:00"
        )
        self.assertIsNotNone(first)
        self.assertFalse(
            self.database.requeue_claim(value["submission_id"], "0" * 64)
        )
        self.assertTrue(
            self.database.requeue_claim(
                value["submission_id"], first["claim_token"]
            )
        )
        second = self.database.claim_next(
            2, "new-worker", "2026-09-04T03:31:00+00:00"
        )
        self.assertIsNotNone(second)
        self.assertNotEqual(first["claim_token"], second["claim_token"])

        with self.assertRaises(LostClaimError):
            self.database.mark_done(
                value["submission_id"],
                first["claim_token"],
                "2026-09-04T03:32:00+00:00",
                metrics(),
                "stale/score.json",
                "stale/score.audit.json",
            )
        with self.assertRaises(LostClaimError):
            self.database.mark_error(
                value["submission_id"],
                first["claim_token"],
                "2026-09-04T03:32:01+00:00",
                "stale worker failure",
            )

        running = self.database.rows()[0]
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["claim_token"], second["claim_token"])
        self.assertEqual(running["worker_id"], "new-worker")
        self.assertIsNone(running["report_relative_path"])
        self.assertIsNone(running["error"])

        self.database.mark_done(
            value["submission_id"],
            second["claim_token"],
            "2026-09-04T03:33:00+00:00",
            metrics(),
            "winner/score.json",
            "winner/score.audit.json",
        )
        self.assertEqual(self.database.rows()[0]["status"], "done")

    def test_running_jobs_are_recovered_after_restart(self) -> None:
        for number in range(1, 4):
            self.assertTrue(
                self.database.ingest_marker(
                    marker(f"recover-{number}", submission_number=number)
                )
            )
        first = self.database.claim_next(
            1, "worker-cuda-1", "2026-09-04T04:00:01+00:00"
        )
        second = self.database.claim_next(
            2, "worker-cuda-2", "2026-09-04T04:00:02+00:00"
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)

        self.assertEqual(self.database.requeue_running(), 2)
        self.assertEqual(self.database.requeue_running(), 0)
        rows = self.database.rows()
        self.assertEqual({row["status"] for row in rows}, {"queued"})
        self.assertEqual(self.database.summary()["counts"], {"queued": 3})
        recovered_ids = {first["id"], second["id"]}
        for row in rows:
            if row["id"] in recovered_ids:
                self.assertEqual(row["attempt_count"], 1)
                self.assertEqual(row["error"], "requeued after grader restart")
                self.assertIsNone(row["worker_id"])
                self.assertIsNone(row["worker_gpu"])
                self.assertIsNone(row["claim_token"])
                self.assertIsNone(row["started_at"])
            else:
                self.assertEqual(row["attempt_count"], 0)
                self.assertIsNone(row["error"])


if __name__ == "__main__":
    unittest.main()
