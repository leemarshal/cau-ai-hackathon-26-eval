import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_grading.metrics import (  # noqa: E402
    EXPECTED_FORGET_WNIDS,
    REFERENCE_ACC_F,
    REFERENCE_ACC_R,
    TEST_DATASET_REVISION,
    validate_report,
)


SUBMISSION_ID = "123e4567-e89b-42d3-a456-426614174000"


def harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)


def valid_report(*, score_depth: str = "pre") -> dict:
    acc_f = 10.0
    acc_r = 90.0
    reference_acc_f = REFERENCE_ACC_F
    reference_acc_r = REFERENCE_ACC_R
    drop_r = reference_acc_r - acc_r
    gap_f = 10.0
    aus = (1.0 - drop_r / 100.0) / 1.1
    cka_per_depth = {
        "b4": {"CKA_f_o": 0.10, "CKA_r_o": 0.90},
        "b8": {"CKA_f_o": 0.20, "CKA_r_o": 0.85},
        "b12": {"CKA_f_o": 0.30, "CKA_r_o": 0.82},
        "pre": {"CKA_f_o": 0.25, "CKA_r_o": 0.80},
    }
    selected = cka_per_depth[score_depth]
    rus_o = harmonic(1.0 - selected["CKA_f_o"], selected["CKA_r_o"])
    final_score = harmonic(aus, rus_o)
    return {
        "schema_version": 2,
        "score_version": "unlearning-v2",
        "dataset_revision": TEST_DATASET_REVISION,
        "phase": "test",
        "tag": SUBMISSION_ID,
        "accuracy_split": "test",
        "representation_split": "test",
        "score_depth": score_depth,
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
            "CKA_f_o": selected["CKA_f_o"],
            "CKA_r_o": selected["CKA_r_o"],
            "RUS_o": rus_o,
        },
        "cka_per_depth": cka_per_depth,
        "AUS": aus,
        "RUS_o": rus_o,
        "final_score": final_score,
        "forget_wnids": list(EXPECTED_FORGET_WNIDS),
    }


class ValidateReportTests(unittest.TestCase):
    def test_normalizes_dict_and_path_reports(self):
        report = valid_report()
        expected_rus = harmonic(0.75, 0.80)

        normalized = validate_report(report, SUBMISSION_ID)

        self.assertAlmostEqual(
            normalized["aus"],
            (1.0 - (REFERENCE_ACC_R - 90.0) / 100.0) / 1.1,
        )
        self.assertEqual(normalized["cka_f_o"], 0.25)
        self.assertEqual(normalized["cka_r_o"], 0.80)
        self.assertAlmostEqual(normalized["rus_o"], expected_rus)
        self.assertEqual(normalized["f1_alias"], normalized["final_score"])
        self.assertEqual(normalized["score_depth"], "pre")
        self.assertEqual(normalized["acc_f"], 10.0)
        self.assertEqual(normalized["acc_r"], 90.0)
        self.assertEqual(json.loads(normalized["report_json"]), report)
        self.assertEqual(
            json.loads(normalized["cka_per_depth_json"]), report["cka_per_depth"]
        )

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            from_path = validate_report(path, SUBMISSION_ID)
        self.assertEqual(from_path, normalized)

    def test_rejects_identity_and_phase_contract_mismatches(self):
        cases = {
            "schema_version": 3,
            "score_version": "unlearning-v1",
            "dataset_revision": "0" * 64,
            "phase": "validation",
            "tag": "another-submission",
            "accuracy_split": "validation",
            "representation_split": "validation",
        }
        for field, bad_value in cases.items():
            with self.subTest(field=field):
                report = valid_report()
                report[field] = bad_value
                with self.assertRaisesRegex(ValueError, field):
                    validate_report(report, SUBMISSION_ID)

    def test_rejects_nonfinite_boolean_and_out_of_range_unit_metrics(self):
        changes = (
            ("AUS", math.nan),
            ("AUS", True),
            ("AUS", 1.0001),
            ("RUS_o", -0.01),
            ("final_score", math.inf),
        )
        for field, bad_value in changes:
            with self.subTest(field=field, value=bad_value):
                report = valid_report()
                report[field] = bad_value
                with self.assertRaises(ValueError):
                    validate_report(report, SUBMISSION_ID)

        report = valid_report()
        report["cka_per_depth"]["b4"]["CKA_f_o"] = 2.0
        with self.assertRaisesRegex(ValueError, "CKA_f_o"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_top_level_and_nested_mismatches(self):
        report = valid_report()
        report["accuracy_metric"]["AUS"] -= 0.01
        with self.assertRaisesRegex(ValueError, "nested AUS"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report()
        report["representation_metric"]["RUS_o"] -= 0.01
        with self.assertRaisesRegex(ValueError, "nested RUS_o"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report()
        report["split_accuracy"]["test"]["Acc_f"] += 1.0
        with self.assertRaisesRegex(ValueError, "forget accuracy"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_score_depth_cka_mismatch_or_incomplete_depths(self):
        report = valid_report()
        report["representation_metric"]["CKA_f_o"] = 0.21
        with self.assertRaisesRegex(ValueError, "score-depth CKA_f_o"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report()
        del report["cka_per_depth"]["b4"]
        with self.assertRaisesRegex(ValueError, "cka_per_depth"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report(score_depth="b12")
        with self.assertRaisesRegex(ValueError, "score_depth"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_unpinned_reference_or_forget_classes(self):
        report = valid_report()
        report["accuracy_metric"]["reference_Acc_r"] -= 1.0
        with self.assertRaisesRegex(ValueError, "pinned reference_Acc_r"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report()
        report["forget_wnids"] = list(reversed(EXPECTED_FORGET_WNIDS))
        with self.assertRaisesRegex(ValueError, "pinned test contract"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_incorrect_rus_and_final_harmonic_formulas(self):
        report = valid_report()
        wrong_rus = report["RUS_o"] + 0.01
        report["RUS_o"] = wrong_rus
        report["representation_metric"]["RUS_o"] = wrong_rus
        with self.assertRaisesRegex(ValueError, "RUS_o.*formula"):
            validate_report(report, SUBMISSION_ID)

        report = valid_report()
        report["final_score"] += 0.01
        with self.assertRaisesRegex(ValueError, "final_score.*formula"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_incorrect_accuracy_formula(self):
        report = valid_report()
        wrong_aus = report["AUS"] + 0.01
        report["AUS"] = wrong_aus
        report["accuracy_metric"]["AUS"] = wrong_aus
        with self.assertRaisesRegex(ValueError, "AUS.*formula"):
            validate_report(report, SUBMISSION_ID)

    def test_rejects_duplicate_or_nonfinite_json_tokens(self):
        report = valid_report()
        encoded = json.dumps(report)
        duplicate = encoded.replace(
            '"schema_version": 2,',
            '"schema_version": 2, "schema_version": 2,',
            1,
        )
        with tempfile.TemporaryDirectory() as raw:
            duplicate_path = Path(raw) / "duplicate.json"
            duplicate_path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_report(duplicate_path, SUBMISSION_ID)

            nan_path = Path(raw) / "nan.json"
            nan_path.write_text(
                encoded.replace(
                    '"AUS": 0.', '"AUS": NaN, "ignored": 0.', 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                validate_report(nan_path, SUBMISSION_ID)


if __name__ == "__main__":
    unittest.main()
