from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "ops" / "grade-finalist.py"
SPEC = importlib.util.spec_from_file_location("native_finalist_grader", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeGraderTests(unittest.TestCase):
    def test_runtime_info_pins_versions_and_code(self) -> None:
        fake_torch = types.SimpleNamespace(
            __version__="2.8.0+cu128",
            version=types.SimpleNamespace(cuda="12.8"),
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 4,
            ),
        )
        fake_torchvision = types.SimpleNamespace(__version__="0.23.0+cu128")
        fake_dependencies = {
            "timm": types.SimpleNamespace(__version__="0.9.10"),
            "numpy": types.SimpleNamespace(__version__="1.26.4"),
            "PIL": types.SimpleNamespace(__version__="12.2.0"),
            "safetensors": types.SimpleNamespace(__version__="0.8.0"),
        }
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torchvision": fake_torchvision,
                **fake_dependencies,
            },
        ):
            first = native.runtime_info()
            second = native.runtime_info()

        self.assertEqual(first, second)
        self.assertRegex(first["runtime_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first["schema_version"], "native-grader-runtime-v1")
        self.assertEqual(first["torch"], "2.8.0+cu128")
        self.assertEqual(first["torchvision"], "0.23.0+cu128")
        self.assertEqual(first["torch_cuda"], "12.8")
        self.assertEqual(first["timm"], "0.9.10")
        self.assertEqual(first["numpy"], "1.26.4")
        self.assertEqual(first["pillow"], "12.2.0")
        self.assertEqual(first["safetensors"], "0.8.0")
        self.assertTrue(first["cuda_available"])
        self.assertEqual(first["cuda_device_count"], 4)
        self.assertEqual(
            set(first["code_sha256"]),
            {
                "ops/grade-finalist.py",
                "grading_docker/convert_checkpoint.py",
                "grading_docker/score_unlearning.py",
                "grading_docker/imagenet_vit.py",
                "ops/finalize-test-reference.py",
            },
        )

    def test_runtime_info_rejects_wrong_torch(self) -> None:
        fake_torch = types.SimpleNamespace(
            __version__="2.7.1+cu128",
            version=types.SimpleNamespace(cuda="12.8"),
        )
        fake_torchvision = types.SimpleNamespace(__version__="0.23.0+cu128")
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torchvision": fake_torchvision,
                "timm": types.SimpleNamespace(__version__="0.9.10"),
                "numpy": types.SimpleNamespace(__version__="1.26.4"),
                "PIL": types.SimpleNamespace(__version__="12.2.0"),
                "safetensors": types.SimpleNamespace(__version__="0.8.0"),
            },
        ):
            with self.assertRaisesRegex(native.GradingError, "requires torch 2.8.0"):
                native.runtime_info()

    def test_grade_converts_then_scores_on_remapped_gpu_and_publishes_audit(self) -> None:
        submission_id = "12345678-1234-4234-8234-123456789abc"
        runtime_id = f"sha256:{'a' * 64}"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            checkpoint = root / f"{submission_id}.pt"
            checkpoint.write_bytes(b"plain checkpoint fixture")
            expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            grading_root = root / "grading"
            grading_root.mkdir(mode=0o700)
            report = root / "results" / "score.json"
            calls: list[tuple[list[str], dict[str, str]]] = []

            def fake_run(command, **kwargs) -> None:
                command = list(command)
                calls.append((command, kwargs["environment"]))
                if str(native.CONVERTER) in command:
                    output = Path(command[command.index("--output") + 1])
                    output.write_bytes(b"converted safetensors")
                elif str(native.SCORER) in command:
                    output = Path(command[command.index("--report") + 1])
                    output.write_text(
                        json.dumps(
                            {
                                "phase": "test",
                                "tag": submission_id,
                                "score_version": native.SCORE_VERSION,
                                "dataset_revision": native.PINNED_TEST_DATASET_REVISION,
                            }
                        ),
                        encoding="utf-8",
                    )

            args = argparse.Namespace(
                expected_runtime_id=runtime_id,
                checkpoint=checkpoint,
                expected_sha256=expected_sha256,
                submission_id=submission_id,
                grading_root=grading_root,
                report=report,
                gpu=2,
            )
            runtime = {
                "runtime_id": runtime_id,
                "cuda_available": True,
                "cuda_device_count": 4,
            }
            with mock.patch.object(native, "runtime_info", return_value=runtime), mock.patch.object(
                native, "_run_checked", side_effect=fake_run
            ):
                report_path, audit_path = native.grade(args)

            self.assertEqual(report_path, report)
            self.assertEqual(audit_path, report.with_suffix(".audit.json"))
            scorer_call = next(call for call in calls if str(native.SCORER) in call[0])
            self.assertEqual(scorer_call[1]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(
                scorer_call[0][scorer_call[0].index("--device") + 1], "cuda:0"
            )
            converter_call = next(
                call for call in calls if str(native.CONVERTER) in call[0]
            )
            self.assertEqual(converter_call[1]["CUDA_VISIBLE_DEVICES"], "-1")

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["schema_version"], "finalist-grading-audit-v3")
            self.assertEqual(audit["grader_runtime_id"], runtime_id)
            self.assertEqual(audit["original_checkpoint_sha256"], expected_sha256)
            self.assertEqual(
                audit["final_report_sha256"],
                hashlib.sha256(report.read_bytes()).hexdigest(),
            )
            self.assertEqual(sum(str(native.FINALIZER) in call[0] for call in calls), 2)

    def test_publish_pair_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            staged = root / "staged.json"
            staged.write_text("{}\n", encoding="utf-8")
            report = root / "published" / "score.json"
            native._publish_pair(staged, report, {"schema_version": "audit"})
            with self.assertRaisesRegex(native.GradingError, "overwrite"):
                native._publish_pair(staged, report, {"schema_version": "other"})
            self.assertEqual(report.read_text(encoding="utf-8"), "{}\n")
            self.assertEqual(
                json.loads(report.with_suffix(".audit.json").read_text(encoding="utf-8")),
                {"schema_version": "audit"},
            )


if __name__ == "__main__":
    unittest.main()
