from __future__ import annotations

import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "ops/download-private-grading.py"
SPEC = importlib.util.spec_from_file_location("download_private_grading", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class HuggingFaceAuthenticationTests(unittest.TestCase):
    def test_explicit_owner_only_file_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credential"
            path.write_text("hf_example\n", encoding="utf-8")
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            with mock.patch.dict(os.environ, {"TA_HF_TOKEN_FILE": str(path)}):
                self.assertEqual(download.hf_token(), "hf_example")

    def test_standard_cached_login_is_used_without_prompt(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            download, "get_token", return_value="hf_cached"
        ), mock.patch.object(download, "login") as login:
            self.assertEqual(download.hf_token(), "hf_cached")
            login.assert_not_called()

    def test_missing_legacy_file_falls_back_to_standard_cached_login(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "old-token-path"
            with mock.patch.dict(
                os.environ, {"TA_HF_TOKEN_FILE": str(missing)}, clear=True
            ), mock.patch.object(
                download, "get_token", return_value="hf_cached"
            ), mock.patch.object(download, "login") as login:
                self.assertEqual(download.hf_token(), "hf_cached")
                login.assert_not_called()

    def test_empty_legacy_file_falls_back_to_standard_cached_login(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "old-token-path"
            empty.touch(mode=0o600)
            with mock.patch.dict(
                os.environ, {"TA_HF_TOKEN_FILE": str(empty)}, clear=True
            ), mock.patch.object(
                download, "get_token", return_value="hf_cached"
            ), mock.patch.object(download, "login") as login:
                self.assertEqual(download.hf_token(), "hf_cached")
                login.assert_not_called()

    def test_interactive_first_run_logs_in_once(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            download, "get_token", side_effect=[None, "hf_new"]
        ), mock.patch.object(download.sys.stdin, "isatty", return_value=True), mock.patch.object(
            download, "login"
        ) as login:
            self.assertEqual(download.hf_token(), "hf_new")
            login.assert_called_once_with(
                add_to_git_credential=False, skip_if_logged_in=False
            )

    def test_noninteractive_first_run_has_actionable_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            download, "get_token", return_value=None
        ), mock.patch.object(download.sys.stdin, "isatty", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "interactive terminal"):
                download.hf_token()


if __name__ == "__main__":
    unittest.main()
