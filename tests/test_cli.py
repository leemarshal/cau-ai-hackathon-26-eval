from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading import cli  # noqa: E402
from ta_grading.config import Settings  # noqa: E402


class _FakeChild:
    def __init__(self, pid: int, return_code: int | None):
        self.pid = pid
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


class SupervisorTests(unittest.TestCase):
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
            poll_seconds=20,
            stable_confirmations=3,
            post_copy_seconds=20,
            min_checkpoint_bytes=1,
            max_checkpoint_bytes=4096,
            recursive_scan=False,
            worker_poll_seconds=1,
            gpu_ids=(1, 2, 3),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_any_unexpected_child_exit_makes_supervisor_fail(self) -> None:
        pinned = "sha256:" + "a" * 64
        runtime = {"runtime_id": pinned}
        for first_return_code, expected in ((0, 1), (7, 7)):
            with self.subTest(return_code=first_return_code):
                created: list[tuple[tuple, dict]] = []

                def fake_popen(*args, **kwargs):
                    created.append((args, kwargs))
                    code = first_return_code if len(created) == 1 else None
                    return _FakeChild(20_000 + len(created), code)

                with mock.patch.object(
                    cli, "_resolve_grader_runtime", return_value=runtime
                ), mock.patch.object(
                    cli, "publish_state"
                ), mock.patch.object(
                    cli.subprocess, "Popen", side_effect=fake_popen
                ), mock.patch.object(
                    cli, "_terminate_process_groups"
                ):
                    result = cli.supervise(self.settings, skip_check=True)

                self.assertEqual(result, expected)
                self.assertEqual(len(created), 5)
                commands = [list(args[0]) for args, _kwargs in created]
                self.assertTrue(any(command[-1] == "poster" for command in commands))
                for _args, kwargs in created:
                    self.assertEqual(kwargs["env"]["TA_GRADER_RUNTIME_ID"], pinned)
                    self.assertEqual(
                        kwargs["env"]["TA_GRADER_PYTHON"],
                        str(self.settings.grader_python),
                    )
                    self.assertEqual(kwargs["env"]["TA_SUPERVISED_CHILD"], "1")
                    self.assertIn("preexec_fn", kwargs)
                    self.assertIn("pass_fds", kwargs)

    def test_grader_runtime_cannot_drift_for_an_existing_state_database(self) -> None:
        first = "sha256:" + "a" * 64
        second = "sha256:" + "b" * 64
        with mock.patch.object(
            cli, "_resolve_grader_runtime", return_value={"runtime_id": first}
        ):
            pinned = cli._pin_grader_runtime(self.settings)
        self.assertEqual(pinned.grader_runtime_id, first)

        with mock.patch.object(
            cli, "_resolve_grader_runtime", return_value={"runtime_id": second}
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime changed"):
                cli._pin_grader_runtime(self.settings)

    def test_repost_parser_requires_one_explicit_target(self) -> None:
        submission_id = "625702e6-828c-49d1-93fc-f4597d873abf"
        single = cli.parser().parse_args(["repost", submission_id])
        self.assertEqual(single.submission_id, submission_id)
        self.assertFalse(single.all_delivered)

        all_delivered = cli.parser().parse_args(
            ["repost", "--all-delivered"]
        )
        self.assertIsNone(all_delivered.submission_id)
        self.assertTrue(all_delivered.all_delivered)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.parser().parse_args(["repost"])
            with self.assertRaises(SystemExit):
                cli.parser().parse_args(
                    ["repost", submission_id, "--all-delivered"]
                )

    def test_repost_commands_requeue_without_touching_grading_result(self) -> None:
        submission_id = "625702e6-828c-49d1-93fc-f4597d873abf"
        database = mock.Mock()
        database.requeue_delivered_score_post.return_value = True
        database.requeue_all_delivered_score_posts.return_value = 3

        with mock.patch.object(
            cli.Settings, "from_env", return_value=self.settings
        ), mock.patch.object(
            cli, "Database", return_value=database
        ), mock.patch.object(
            cli, "publish_state"
        ) as publish, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["repost", submission_id]), 0)
            self.assertEqual(cli.main(["repost", "--all-delivered"]), 0)

        single_args = database.requeue_delivered_score_post.call_args.args
        self.assertEqual(single_args[0], submission_id)
        datetime.fromisoformat(single_args[1])
        all_args = database.requeue_all_delivered_score_posts.call_args.args
        datetime.fromisoformat(all_args[0])
        self.assertEqual(publish.call_count, 2)
        self.assertIn('"requeued_score_posts": 1', output.getvalue())
        self.assertIn('"requeued_score_posts": 3', output.getvalue())


if __name__ == "__main__":
    unittest.main()
