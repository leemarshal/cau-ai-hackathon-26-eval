"""Strict validation and normalization for private-test scorer reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
SCORE_VERSION = "unlearning-v2"
TEST_DATASET_REVISION = (
    "f7938fad4be1b9559433adf6f3edfab6088750ba003371de7c7505b5da05353b"
)
DEPTHS = ("b4", "b8", "b12", "pre")
SCORE_DEPTH = "pre"
REFERENCE_ACC_F = 0.0
REFERENCE_ACC_R = 95.02962962962964
EXPECTED_FORGET_WNIDS = (
    "n01558993",
    "n01950731",
    "n02129604",
    "n02256656",
    "n02361337",
    "n02799071",
    "n03649909",
    "n04162706",
    "n04252225",
    "n04371430",
)
ABSOLUTE_TOLERANCE = 1e-9

REPORT_KEYS = {
    "schema_version",
    "score_version",
    "dataset_revision",
    "phase",
    "tag",
    "accuracy_split",
    "representation_split",
    "score_depth",
    "split_accuracy",
    "accuracy_metric",
    "representation_metric",
    "cka_per_depth",
    "AUS",
    "RUS_o",
    "final_score",
    "forget_wnids",
}
ACCURACY_KEYS = {
    "Acc_f",
    "Acc_r",
    "reference_Acc_f",
    "reference_Acc_r",
    "drop_r",
    "gap_f",
    "AUS",
}
REPRESENTATION_KEYS = {"CKA_f_o", "CKA_r_o", "RUS_o"}
CKA_KEYS = {"CKA_f_o", "CKA_r_o"}
SPLIT_ACCURACY_KEYS = {"Acc_f", "Acc_r"}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"invalid scorer report JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("scorer report must be a JSON object")
    return value


def _load_report(
    path_or_dict: os.PathLike[str] | str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        try:
            raw = json.dumps(path_or_dict, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scorer report is not JSON-serializable: {exc}") from exc
        return _decode_json(raw)
    if not isinstance(path_or_dict, (str, os.PathLike)):
        raise TypeError("path_or_dict must be a path or dict")
    path = Path(path_or_dict)
    if path.is_symlink() or not path.is_file():
        raise ValueError("scorer report path must be a regular non-symlink file")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read scorer report: {exc}") from exc
    return _decode_json(raw)


def _mapping_with_keys(
    value: Any, expected: set[str], description: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{description} must contain exactly {sorted(expected)}")
    return value


def _number(value: Any, description: str, *, upper: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= upper
    ):
        raise ValueError(
            f"{description} must be a finite number in [0, {upper:g}]"
        )
    return float(value)


def _unit_number(value: Any, description: str) -> float:
    return _number(value, description, upper=1.0)


def _percentage(value: Any, description: str) -> float:
    return _number(value, description, upper=100.0)


def _require_equal(left: float, right: float, description: str) -> None:
    # Duplicated fields are emitted from the same Python float by the scorer and
    # therefore must survive JSON serialization identically.
    if left != right:
        raise ValueError(f"{description} mismatch")


def _require_formula(actual: float, expected: float, description: str) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{description} does not match the scoring formula")


def _harmonic(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2.0 * left * right / (left + right)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_report(
    path_or_dict: os.PathLike[str] | str | dict[str, Any],
    submission_id: str,
) -> dict[str, Any]:
    """Validate a v2 private-test report and return DB-ready metric values.

    ``submission_id`` is compared with the scorer-controlled ``tag``.  The
    returned JSON strings are canonical and safe to store directly in SQLite.
    """

    if not isinstance(submission_id, str) or not submission_id:
        raise ValueError("submission_id must be a non-empty string")
    report = _load_report(path_or_dict)
    _mapping_with_keys(report, REPORT_KEYS, "scorer report")

    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("scorer report schema_version mismatch")
    identity = {
        "score_version": SCORE_VERSION,
        "dataset_revision": TEST_DATASET_REVISION,
        "phase": "test",
        "tag": submission_id,
        "accuracy_split": "test",
        "representation_split": "test",
    }
    for field, expected in identity.items():
        if report[field] != expected:
            raise ValueError(f"scorer report {field} mismatch")

    score_depth = report["score_depth"]
    if score_depth != SCORE_DEPTH:
        raise ValueError(f"scorer report score_depth must be {SCORE_DEPTH!r}")

    split_accuracy = _mapping_with_keys(
        report["split_accuracy"], {"test"}, "split_accuracy"
    )
    test_accuracy = _mapping_with_keys(
        split_accuracy["test"], SPLIT_ACCURACY_KEYS, "split_accuracy.test"
    )
    split_acc_f = _percentage(test_accuracy["Acc_f"], "split_accuracy.test.Acc_f")
    split_acc_r = _percentage(test_accuracy["Acc_r"], "split_accuracy.test.Acc_r")

    accuracy = _mapping_with_keys(
        report["accuracy_metric"], ACCURACY_KEYS, "accuracy_metric"
    )
    acc_f = _percentage(accuracy["Acc_f"], "accuracy_metric.Acc_f")
    acc_r = _percentage(accuracy["Acc_r"], "accuracy_metric.Acc_r")
    reference_acc_f = _percentage(
        accuracy["reference_Acc_f"], "accuracy_metric.reference_Acc_f"
    )
    reference_acc_r = _percentage(
        accuracy["reference_Acc_r"], "accuracy_metric.reference_Acc_r"
    )
    _require_equal(reference_acc_f, REFERENCE_ACC_F, "pinned reference_Acc_f")
    _require_equal(reference_acc_r, REFERENCE_ACC_R, "pinned reference_Acc_r")
    drop_r = _percentage(accuracy["drop_r"], "accuracy_metric.drop_r")
    gap_f = _percentage(accuracy["gap_f"], "accuracy_metric.gap_f")
    nested_aus = _unit_number(accuracy["AUS"], "accuracy_metric.AUS")
    aus = _unit_number(report["AUS"], "AUS")

    _require_equal(split_acc_f, acc_f, "forget accuracy")
    _require_equal(split_acc_r, acc_r, "retain accuracy")
    _require_equal(nested_aus, aus, "top-level and nested AUS")

    expected_drop_r = max(reference_acc_r - acc_r, 0.0)
    expected_gap_f = abs(acc_f - reference_acc_f)
    expected_aus = (1.0 - expected_drop_r / 100.0) / (
        1.0 + expected_gap_f / 100.0
    )
    _require_formula(drop_r, expected_drop_r, "drop_r")
    _require_formula(gap_f, expected_gap_f, "gap_f")
    _require_formula(aus, expected_aus, "AUS")

    representation = _mapping_with_keys(
        report["representation_metric"],
        REPRESENTATION_KEYS,
        "representation_metric",
    )
    cka_f_o = _unit_number(
        representation["CKA_f_o"], "representation_metric.CKA_f_o"
    )
    cka_r_o = _unit_number(
        representation["CKA_r_o"], "representation_metric.CKA_r_o"
    )
    nested_rus_o = _unit_number(
        representation["RUS_o"], "representation_metric.RUS_o"
    )

    cka_per_depth = _mapping_with_keys(
        report["cka_per_depth"], set(DEPTHS), "cka_per_depth"
    )
    normalized_cka: dict[str, dict[str, float]] = {}
    for depth in DEPTHS:
        depth_metrics = _mapping_with_keys(
            cka_per_depth[depth], CKA_KEYS, f"cka_per_depth.{depth}"
        )
        normalized_cka[depth] = {
            "CKA_f_o": _unit_number(
                depth_metrics["CKA_f_o"], f"cka_per_depth.{depth}.CKA_f_o"
            ),
            "CKA_r_o": _unit_number(
                depth_metrics["CKA_r_o"], f"cka_per_depth.{depth}.CKA_r_o"
            ),
        }

    selected_cka = normalized_cka[score_depth]
    _require_equal(
        cka_f_o,
        selected_cka["CKA_f_o"],
        "representation_metric and score-depth CKA_f_o",
    )
    _require_equal(
        cka_r_o,
        selected_cka["CKA_r_o"],
        "representation_metric and score-depth CKA_r_o",
    )

    rus_o = _unit_number(report["RUS_o"], "RUS_o")
    final_score = _unit_number(report["final_score"], "final_score")
    _require_equal(nested_rus_o, rus_o, "top-level and nested RUS_o")
    _require_formula(rus_o, _harmonic(1.0 - cka_f_o, cka_r_o), "RUS_o")
    _require_formula(final_score, _harmonic(aus, rus_o), "final_score")

    forget_wnids = report["forget_wnids"]
    if forget_wnids != list(EXPECTED_FORGET_WNIDS):
        raise ValueError("forget_wnids do not match the pinned test contract")

    return {
        "aus": aus,
        "cka_f_o": cka_f_o,
        "cka_r_o": cka_r_o,
        "rus_o": rus_o,
        "final_score": final_score,
        "f1_alias": final_score,
        "score_depth": score_depth,
        "cka_per_depth_json": _canonical_json(normalized_cka),
        "acc_f": acc_f,
        "acc_r": acc_r,
        "report_json": _canonical_json(report),
    }
