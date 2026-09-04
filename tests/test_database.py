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

    def _create_score_post(
        self, token: str, *, team_number: int = 1, deliver: bool = False
    ) -> tuple[dict, dict | None]:
        value = marker(
            token,
            team_name=f"Team{team_number}_{token}",
            team_number=team_number,
        )
        self.assertTrue(self.database.ingest_marker(value))
        grading_claim = self.database.claim_next(
            1, f"grader-{token}", "2026-09-04T08:00:00+00:00"
        )
        self.assertIsNotNone(grading_claim)
        assert grading_claim is not None
        self.database.mark_done(
            value["submission_id"],
            grading_claim["claim_token"],
            "2026-09-04T08:10:00+00:00",
            metrics(),
            f"results/{token}/score.json",
            f"results/{token}/score.audit.json",
        )
        if not deliver:
            return value, None
        post_claim = self.database.claim_next_score_post(
            f"poster-{token}", "2026-09-04T08:10:00+00:00"
        )
        self.assertIsNotNone(post_claim)
        assert post_claim is not None
        self.assertTrue(
            self.database.mark_score_post_delivered(
                value["submission_id"],
                post_claim["claim_token"],
                "2026-09-04T08:11:00+00:00",
            )
        )
        return value, post_claim

    def test_initialize_migrates_legacy_claims_and_source_dispositions(self) -> None:
        legacy_path = Path(self.temporary.name) / "legacy.sqlite3"
        legacy_schema = (
            SCHEMA.split("CREATE TABLE IF NOT EXISTS score_posts", 1)[0]
            .replace("    claim_token TEXT,\n", "")
            .replace(", 'limit_exceeded'", "")
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
            score_posts = connection.execute(
                "SELECT * FROM score_posts"
            ).fetchall()
        self.assertEqual(version, DATABASE_SCHEMA_VERSION)
        self.assertEqual(score_posts, [])

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

        posts = self.database.score_post_rows()
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["submission_id"], value["submission_id"])
        self.assertEqual(
            {"team_id": posts[0]["team_id"], "score": posts[0]["score"]},
            {"team_id": value["team_number"], "score": normalized["final_score"]},
        )
        self.assertEqual(posts[0]["status"], "pending")
        self.assertEqual(posts[0]["attempt_count"], 0)
        self.assertEqual(
            posts[0]["next_attempt_at"], "2026-09-04T02:30:00+00:00"
        )
        self.assertEqual(self.database.summary()["score_post_counts"], {"pending": 1})

    def test_mark_done_and_score_post_are_one_transaction(self) -> None:
        value = marker("atomic-score-post")
        self.assertTrue(self.database.ingest_marker(value))
        claimed = self.database.claim_next(
            2, "worker-cuda-2", "2026-09-04T02:40:00+00:00"
        )
        self.assertIsNotNone(claimed)
        with self.database.connect() as connection:
            connection.execute(
                "CREATE TRIGGER reject_score_post BEFORE INSERT ON score_posts "
                "BEGIN SELECT RAISE(ABORT, 'blocked score post'); END"
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "blocked score post"):
            self.database.mark_done(
                value["submission_id"],
                claimed["claim_token"],
                "2026-09-04T02:45:00+00:00",
                metrics(),
                "results/report.json",
                "results/report.audit.json",
            )

        rolled_back = self.database.rows()[0]
        self.assertEqual(rolled_back["status"], "running")
        self.assertIsNone(rolled_back["finished_at"])
        self.assertEqual(self.database.score_post_rows(), [])

        with self.database.connect() as connection:
            connection.execute("DROP TRIGGER reject_score_post")
        self.database.mark_done(
            value["submission_id"],
            claimed["claim_token"],
            "2026-09-04T02:45:00+00:00",
            metrics(),
            "results/report.json",
            "results/report.audit.json",
        )
        self.assertEqual(self.database.rows()[0]["status"], "done")
        self.assertEqual(len(self.database.score_post_rows()), 1)

    def test_score_post_retry_schedule_and_stale_claim_guards(self) -> None:
        value = marker("score-post-retry", team_number=8)
        self.assertTrue(self.database.ingest_marker(value))
        grading_claim = self.database.claim_next(
            1, "grader", "2026-09-04T05:00:00+00:00"
        )
        self.assertIsNotNone(grading_claim)
        self.database.mark_done(
            value["submission_id"],
            grading_claim["claim_token"],
            "2026-09-04T05:10:00+00:00",
            metrics(),
            "results/report.json",
            "results/report.audit.json",
        )

        self.assertIsNone(
            self.database.claim_next_score_post(
                "poster-1", "2026-09-04T05:09:59+00:00"
            )
        )
        first = self.database.claim_next_score_post(
            "poster-1", "2026-09-04T05:10:00+00:00"
        )
        self.assertIsNotNone(first)
        self.assertEqual(first["team_id"], 8)
        self.assertEqual(first["score"], metrics()["final_score"])
        self.assertEqual(first["status"], "posting")
        self.assertEqual(first["attempt_count"], 1)
        self.assertRegex(first["claim_token"], r"\A[0-9a-f]{64}\Z")
        observable = self.database.summary()["score_posts"][0]
        self.assertEqual(observable["status"], "posting")
        self.assertNotIn("claim_token", observable)
        self.assertNotIn("worker_id", observable)
        self.assertFalse(
            self.database.mark_score_post_failed(
                value["submission_id"],
                "0" * 64,
                "2026-09-04T05:10:01+00:00",
                "stale failure",
                "2026-09-04T05:20:00+00:00",
            )
        )
        long_error = "discard-this-prefix:" + "x" * 4100
        self.assertTrue(
            self.database.mark_score_post_failed(
                value["submission_id"],
                first["claim_token"],
                "2026-09-04T05:10:02+00:00",
                long_error,
                "2026-09-04T05:20:00+00:00",
            )
        )
        failed = self.database.score_post_rows()[0]
        self.assertEqual(failed["status"], "pending")
        self.assertEqual(failed["attempt_count"], 1)
        self.assertIsNone(failed["claim_token"])
        self.assertEqual(len(failed["last_error"]), 4000)
        self.assertEqual(failed["last_failed_at"], "2026-09-04T05:10:02+00:00")
        self.assertIsNone(
            self.database.claim_next_score_post(
                "poster-2", "2026-09-04T05:19:59+00:00"
            )
        )

        second = self.database.claim_next_score_post(
            "poster-2", "2026-09-04T05:20:00+00:00"
        )
        self.assertIsNotNone(second)
        self.assertEqual(second["attempt_count"], 2)
        self.assertNotEqual(second["claim_token"], first["claim_token"])
        self.assertFalse(
            self.database.mark_score_post_delivered(
                value["submission_id"],
                first["claim_token"],
                "2026-09-04T05:20:01+00:00",
            )
        )
        self.assertTrue(
            self.database.mark_score_post_delivered(
                value["submission_id"],
                second["claim_token"],
                "2026-09-04T05:20:02+00:00",
            )
        )
        delivered = self.database.score_post_rows()[0]
        self.assertEqual(delivered["status"], "delivered")
        self.assertEqual(delivered["attempt_count"], 2)
        self.assertEqual(delivered["delivered_at"], "2026-09-04T05:20:02+00:00")
        self.assertIsNone(delivered["last_error"])
        self.assertIsNone(
            self.database.claim_next_score_post(
                "poster-3", "2026-09-04T06:00:00+00:00"
            )
        )

    def test_score_post_claims_are_atomic_and_restart_requeues_them(self) -> None:
        expected_ids = set()
        for number in range(1, 4):
            value = marker(
                f"score-post-{number}",
                team_name=f"Team{number}_post",
                team_number=number,
            )
            expected_ids.add(value["submission_id"])
            self.assertTrue(self.database.ingest_marker(value))
            grading_claim = self.database.claim_next(
                number, f"grader-{number}", "2026-09-04T06:00:00+00:00"
            )
            self.assertIsNotNone(grading_claim)
            self.database.mark_done(
                value["submission_id"],
                grading_claim["claim_token"],
                f"2026-09-04T06:0{number}:00+00:00",
                metrics(),
                f"results/{number}/report.json",
                f"results/{number}/report.audit.json",
            )

        barrier = threading.Barrier(3)

        def claim(number: int) -> dict | None:
            worker_database = Database(self.database_path)
            barrier.wait(timeout=5)
            return worker_database.claim_next_score_post(
                f"poster-{number}", "2026-09-04T07:00:00+00:00"
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            claimed = list(pool.map(claim, (1, 2, 3)))

        self.assertNotIn(None, claimed)
        claimed_rows = [row for row in claimed if row is not None]
        self.assertEqual(
            {row["submission_id"] for row in claimed_rows}, expected_ids
        )
        self.assertEqual(len({row["claim_token"] for row in claimed_rows}), 3)
        self.assertEqual(self.database.requeue_posting_score_posts(), 3)
        self.assertEqual(self.database.requeue_posting_score_posts(), 0)
        recovered = self.database.score_post_rows()
        self.assertEqual({row["status"] for row in recovered}, {"pending"})
        self.assertTrue(all(row["attempt_count"] == 1 for row in recovered))
        self.assertTrue(all(row["claim_token"] is None for row in recovered))
        self.assertTrue(
            all(
                row["last_error"] == "requeued after score poster restart"
                for row in recovered
            )
        )
        for stale in claimed_rows:
            self.assertFalse(
                self.database.mark_score_post_delivered(
                    stale["submission_id"],
                    stale["claim_token"],
                    "2026-09-04T07:00:01+00:00",
                )
            )

    def test_delivered_score_post_can_be_requeued_for_repost(self) -> None:
        value, delivered_claim = self._create_score_post(
            "single-repost", team_number=8, deliver=True
        )
        self.assertIsNotNone(delivered_claim)
        before = self.database.score_post_rows()[0]
        self.assertEqual(before["status"], "delivered")
        self.assertEqual(before["attempt_count"], 1)

        # The transition clears every ownership/delivery/error field even if a
        # legacy or manually recovered delivered row retained stale metadata.
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE score_posts SET worker_id = 'stale-poster', "
                "claim_token = ?, claimed_at = ?, last_failed_at = ?, "
                "last_error = 'stale error' WHERE submission_id = ?",
                (
                    "f" * 64,
                    "2026-09-04T08:12:00+00:00",
                    "2026-09-04T08:12:01+00:00",
                    value["submission_id"],
                ),
            )

        due_at = "2026-09-04T09:00:00+00:00"
        self.assertTrue(
            self.database.requeue_delivered_score_post(
                value["submission_id"], due_at
            )
        )
        requeued = self.database.score_post_rows()[0]
        self.assertEqual(requeued["status"], "pending")
        self.assertEqual(requeued["attempt_count"], before["attempt_count"])
        self.assertEqual(requeued["next_attempt_at"], due_at)
        for field in (
            "worker_id",
            "claim_token",
            "claimed_at",
            "delivered_at",
            "last_failed_at",
            "last_error",
        ):
            self.assertIsNone(requeued[field], field)
        self.assertIsNone(
            self.database.claim_next_score_post(
                "early-poster", "2026-09-04T08:59:59+00:00"
            )
        )
        repost_claim = self.database.claim_next_score_post(
            "reposter", due_at
        )
        self.assertIsNotNone(repost_claim)
        self.assertEqual(repost_claim["submission_id"], value["submission_id"])
        self.assertEqual(repost_claim["attempt_count"], 2)

    def test_repost_requeue_rejects_non_delivered_and_missing_ids(self) -> None:
        pending, _ = self._create_score_post(
            "pending-repost", team_number=2
        )
        original = self.database.score_post_rows()[0]
        self.assertFalse(
            self.database.requeue_delivered_score_post(
                pending["submission_id"], "2026-09-04T09:00:00+00:00"
            )
        )
        self.assertEqual(self.database.score_post_rows()[0], original)

        posting = self.database.claim_next_score_post(
            "active-poster", "2026-09-04T08:10:00+00:00"
        )
        self.assertIsNotNone(posting)
        self.assertFalse(
            self.database.requeue_delivered_score_post(
                posting["submission_id"], "2026-09-04T09:00:00+00:00"
            )
        )
        still_posting = self.database.score_post_rows()[0]
        self.assertEqual(still_posting["status"], "posting")
        self.assertEqual(still_posting["claim_token"], posting["claim_token"])

        self.assertFalse(
            self.database.requeue_delivered_score_post(
                str(uuid.uuid4()), "2026-09-04T09:00:00+00:00"
            )
        )

    def test_all_delivered_score_posts_are_atomically_requeued(self) -> None:
        first, first_claim = self._create_score_post(
            "all-repost-1", team_number=3, deliver=True
        )
        second, second_claim = self._create_score_post(
            "all-repost-2", team_number=4, deliver=True
        )
        pending, _ = self._create_score_post(
            "all-repost-pending", team_number=5
        )
        self.assertIsNotNone(first_claim)
        self.assertIsNotNone(second_claim)
        before = {
            row["submission_id"]: row for row in self.database.score_post_rows()
        }

        due_at = "2026-09-04T10:00:00+00:00"
        self.assertEqual(
            self.database.requeue_all_delivered_score_posts(due_at), 2
        )
        self.assertEqual(
            self.database.requeue_all_delivered_score_posts(due_at), 0
        )
        after = {
            row["submission_id"]: row for row in self.database.score_post_rows()
        }
        for submission_id in (first["submission_id"], second["submission_id"]):
            self.assertEqual(after[submission_id]["status"], "pending")
            self.assertEqual(after[submission_id]["next_attempt_at"], due_at)
            self.assertEqual(
                after[submission_id]["attempt_count"],
                before[submission_id]["attempt_count"],
            )
            for field in (
                "worker_id",
                "claim_token",
                "claimed_at",
                "delivered_at",
                "last_failed_at",
                "last_error",
            ):
                self.assertIsNone(after[submission_id][field], field)
        self.assertEqual(
            after[pending["submission_id"]], before[pending["submission_id"]]
        )

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
