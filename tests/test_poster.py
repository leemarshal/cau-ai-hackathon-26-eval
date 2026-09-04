from __future__ import annotations

from contextlib import contextmanager
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.config import Settings  # noqa: E402
from ta_grading import poster  # noqa: E402
from ta_grading.score_post import ScorePostError  # noqa: E402


class PosterTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
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
            poll_seconds=20,
            stable_confirmations=3,
            post_copy_seconds=20,
            min_checkpoint_bytes=1,
            max_checkpoint_bytes=4096,
            recursive_scan=False,
            worker_poll_seconds=1,
            gpu_ids=(1, 2, 3),
            score_post_retry_seconds=60,
        )
        self.row = {
            "submission_id": "12345678-1234-4234-8234-123456789abc",
            "team_id": 8,
            "score": 0.7321,
            "claim_token": "b" * 64,
        }

    def test_delivers_exact_outbox_values_and_marks_claim(self) -> None:
        database = mock.Mock()
        database.claim_next_score_post.return_value = self.row
        database.mark_score_post_delivered.return_value = True
        with mock.patch.object(poster, "post_score") as post, mock.patch.object(
            poster, "publish_state"
        ):
            self.assertTrue(poster.deliver_one(self.settings, database, "poster-1"))

        post.assert_called_once_with(
            "https://api.minds.ai.kr/submit",
            8,
            0.7321,
            timeout_seconds=10.0,
        )
        delivered = database.mark_score_post_delivered.call_args.args
        self.assertEqual(delivered[:2], (self.row["submission_id"], "b" * 64))
        datetime.fromisoformat(delivered[2])

    def test_http_failure_keeps_item_pending_for_configured_retry(self) -> None:
        database = mock.Mock()
        database.claim_next_score_post.return_value = self.row
        database.mark_score_post_failed.return_value = True
        with mock.patch.object(
            poster, "post_score", side_effect=ScorePostError("offline")
        ), mock.patch.object(poster, "publish_state"):
            self.assertTrue(poster.deliver_one(self.settings, database, "poster-1"))

        failed = database.mark_score_post_failed.call_args.args
        self.assertEqual(failed[0:2], (self.row["submission_id"], "b" * 64))
        self.assertEqual(failed[3], "offline")
        failed_at = datetime.fromisoformat(failed[2])
        next_attempt = datetime.fromisoformat(failed[4])
        self.assertEqual((next_attempt - failed_at).total_seconds(), 60)
        database.mark_score_post_delivered.assert_not_called()

    def test_no_due_item_does_nothing(self) -> None:
        database = mock.Mock()
        database.claim_next_score_post.return_value = None
        with mock.patch.object(poster, "post_score") as post:
            self.assertFalse(poster.deliver_one(self.settings, database, "poster-1"))
        post.assert_not_called()

    def test_lifetime_lock_rejects_a_second_poster(self) -> None:
        with poster._poster_lifetime_lock(self.settings):
            with self.assertRaisesRegex(
                RuntimeError, "another score poster is already running"
            ):
                with poster._poster_lifetime_lock(self.settings):
                    self.fail("a second poster unexpectedly acquired the lock")

    def test_loop_recovers_interrupted_claims_only_after_locking(self) -> None:
        events: list[str] = []
        lock_held = False
        database = mock.Mock()

        @contextmanager
        def fake_lifetime_lock(_settings):
            nonlocal lock_held
            lock_held = True
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")
                lock_held = False

        def initialize() -> None:
            self.assertTrue(lock_held)
            events.append("initialize")

        def recover() -> int:
            self.assertTrue(lock_held)
            events.append("recover")
            return 2

        database.initialize.side_effect = initialize
        database.requeue_posting_score_posts.side_effect = recover
        with mock.patch.object(
            poster, "_poster_lifetime_lock", side_effect=fake_lifetime_lock
        ), mock.patch.object(
            poster, "Database", return_value=database
        ), mock.patch.object(
            poster, "deliver_one", return_value=False
        ), mock.patch.object(
            poster, "_publish_post_state"
        ) as publish:
            self.assertFalse(poster.poster_loop(self.settings, once=True))

        self.assertEqual(
            events,
            ["lock-enter", "initialize", "recover", "lock-exit"],
        )
        database.requeue_posting_score_posts.assert_called_once_with()
        publish.assert_called_once_with(self.settings, database)


if __name__ == "__main__":
    unittest.main()
