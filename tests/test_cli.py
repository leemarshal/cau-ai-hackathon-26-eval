from __future__ import annotations

import sys
import tempfile
import unittest
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
            grade_script=ROOT / "ops/grade-finalist.sh",
            grading_image="fixture/grader:test",
            expected_team_count=1,
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
        for first_return_code, expected in ((0, 1), (7, 7)):
            with self.subTest(return_code=first_return_code):
                created: list[tuple[tuple, dict]] = []

                def fake_popen(*args, **kwargs):
                    created.append((args, kwargs))
                    code = first_return_code if len(created) == 1 else None
                    return _FakeChild(20_000 + len(created), code)

                with mock.patch.object(
                    cli, "_resolve_grading_image", return_value=pinned
                ), mock.patch.object(
                    cli, "publish_state"
                ), mock.patch.object(
                    cli.subprocess, "Popen", side_effect=fake_popen
                ), mock.patch.object(
                    cli, "_terminate_process_groups"
                ):
                    result = cli.supervise(self.settings, skip_check=True)

                self.assertEqual(result, expected)
                self.assertEqual(len(created), 4)
                for _args, kwargs in created:
                    self.assertEqual(kwargs["env"]["TA_GRADING_IMAGE"], pinned)
                    self.assertEqual(kwargs["env"]["TA_SUPERVISED_CHILD"], "1")
                    self.assertIn("preexec_fn", kwargs)
                    self.assertIn("pass_fds", kwargs)

    def test_grader_image_cannot_drift_for_an_existing_state_database(self) -> None:
        first = "sha256:" + "a" * 64
        second = "sha256:" + "b" * 64
        with mock.patch.object(cli, "_resolve_grading_image", return_value=first):
            pinned = cli._pin_grading_image(self.settings)
        self.assertEqual(pinned.grading_image, first)

        with mock.patch.object(cli, "_resolve_grading_image", return_value=second):
            with self.assertRaisesRegex(RuntimeError, "image ID changed"):
                cli._pin_grading_image(self.settings)


if __name__ == "__main__":
    unittest.main()
