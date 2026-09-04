"""Score ImageNet-100 unlearning checkpoints with a phase-scoped dataset.

Metric version ``unlearning-v2`` deliberately has two isolated phases:

* ``validation`` (participant side): AUS and RUS_o on one combined public
  ``validation`` split.
* ``test`` (organizer side): AUS and RUS_o on the private ``test`` split.

Only RUS_o enters the score. It compares the submitted representation with
the public original model M_o, so v2 does not distribute per-image M_r
features. The old RUS_r/probe diagnostics required holdout-A assets even
during final grading and made a true test-only bundle impossible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset

from imagenet_vit import ViTWrapper


SCHEMA_VERSION = 2
SCORE_VERSION = "unlearning-v2"
DEPTHS = {"b4": 3, "b8": 7, "b12": 11}
ALL_DEPTHS = [*DEPTHS, "pre"]
DEFAULT_SCORE_DEPTH = "pre"
FEATURE_WIDTH = 768
CACHE_KEYS = {
    "schema_version",
    "phase",
    "dataset_revision",
    "split_name",
    "correct",
    "total",
    "labels",
    *(f"f_{depth}" for depth in ALL_DEPTHS),
}


class ListDataset(Dataset):
    """Read only paths explicitly listed in the phase manifest."""

    def __init__(self, items: list[tuple[str, int]], root: Path, transform):
        self.items = items
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        relative, label = self.items[index]
        with (self.root / relative).open("rb") as image_file:
            image = Image.open(image_file).convert("RGB")
        return self.transform(image), label


def eval_transform(config: dict):
    import timm

    return timm.data.create_transform(
        input_size=config["input_size"][-1],
        is_training=False,
        crop_pct=config.get("crop_pct", 0.9),
        interpolation=config.get("interpolation", "bicubic"),
        mean=config["mean"],
        std=config["std"],
    )


def _new_model(num_classes: int) -> ViTWrapper:
    return ViTWrapper(
        num_classes=num_classes,
        pretrained=False,
        drop_path_rate=0.0,
        in_model_norm=False,
    )


def load_submission_model(
    checkpoint: Path, device: torch.device, num_classes: int
) -> ViTWrapper:
    """Load only a non-executable safetensors participant artifact."""
    if checkpoint.suffix != ".safetensors":
        raise ValueError(
            "participant checkpoints must be converted to .safetensors before scoring"
        )
    state = load_safetensors(str(checkpoint), device="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError("converted checkpoint must contain a non-empty state_dict")
    model = _new_model(num_classes)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def load_trusted_reference_model(
    checkpoint: Path, device: torch.device, num_classes: int
) -> ViTWrapper:
    """Load the organizer-built M_o checkpoint, never participant input."""
    model = ViTWrapper(
        num_classes=num_classes,
        pretrained=False,
        drop_path_rate=0.0,
        in_model_norm=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state_dict mapping")
    model.load_state_dict(state.get("model", state), strict=True)
    return model.eval().to(device)


def _safe_items(raw_items: object, split_name: str, num_classes: int) -> list[tuple[str, int]]:
    if not isinstance(raw_items, (list, tuple)) or not raw_items:
        raise ValueError(f"split {split_name!r} must be a non-empty list")
    items: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"split {split_name!r} contains an invalid record")
        relative, label = raw
        if not isinstance(relative, str):
            raise ValueError(f"split {split_name!r} contains a non-string path")
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or ".." in posix_path.parts or not posix_path.parts:
            raise ValueError(f"split {split_name!r} contains an unsafe path: {relative!r}")
        if relative in seen:
            raise ValueError(f"split {split_name!r} contains a duplicate path: {relative!r}")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_classes:
            raise ValueError(f"split {split_name!r} contains an invalid label")
        seen.add(relative)
        items.append((relative, label))
    return items


def load_phase_inputs(
    split_path: Path,
    refs_path: Path,
    phase: str,
    image_root: Path | None,
) -> tuple[Path, list[str], dict[str, list[tuple[str, int]]], dict]:
    manifest = torch.load(split_path, map_location="cpu", weights_only=True)
    refs = torch.load(refs_path, map_location="cpu", weights_only=True)
    if not isinstance(manifest, dict) or set(manifest) != {"meta", "splits"}:
        raise ValueError("split manifest must contain exactly meta and splits")
    if not isinstance(refs, dict):
        raise ValueError("metric refs must be a mapping")

    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("phase", phase),
        ("score_version", SCORE_VERSION),
    ):
        if refs.get(field) != expected:
            raise ValueError(
                f"metric refs {field} mismatch: expected {expected!r}, got {refs.get(field)!r}"
            )

    meta = manifest["meta"]
    raw_splits = manifest["splits"]
    if not isinstance(meta, dict) or not isinstance(raw_splits, dict):
        raise ValueError("split manifest meta/splits must be mappings")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("split manifest schema_version mismatch")
    if meta.get("dataset_revision") != refs.get("dataset_revision"):
        raise ValueError("split manifest and metric refs dataset_revision differ")
    if meta.get("phase") != phase:
        raise ValueError("split manifest phase mismatch")

    wnids = meta.get("wnids")
    if not isinstance(wnids, list) or not wnids or not all(isinstance(x, str) for x in wnids):
        raise ValueError("split manifest wnids must be a non-empty string list")

    accuracy_split = refs.get("accuracy_split")
    representation_split = refs.get("representation_split")
    scored_names = {accuracy_split, representation_split}
    allowed_names = scored_names | ({"released"} if phase == "validation" else set())
    if None in scored_names or set(raw_splits) != allowed_names:
        raise ValueError(
            f"{phase} manifest must contain exactly {sorted(allowed_names - {None})}"
        )
    splits = {
        name: _safe_items(raw_splits[name], name, len(wnids))
        for name in scored_names
    }

    configured_root = image_root or Path(
        os.environ.get("IMAGENET_ROOT", str(meta.get("root", "")))
    )
    if not configured_root.is_absolute():
        raise ValueError("image root must be an absolute path")
    return configured_root, wnids, splits, refs


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def extract(
    model: ViTWrapper,
    items: list[tuple[str, int]],
    root: Path,
    device: torch.device,
    *,
    capture_features: bool,
    batch_size: int,
    workers: int,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be zero or greater")
    loader = DataLoader(
        ListDataset(items, root, eval_transform(model.data_config)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    num_classes = model.head.out_features
    correct = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    labels: list[np.ndarray] = []
    features: dict[str, list[np.ndarray]] = {depth: [] for depth in ALL_DEPTHS}
    taps: dict[str, torch.Tensor] = {}
    hooks = []
    if capture_features:
        hooks = [
            model.backbone.blocks[index].register_forward_hook(
                lambda _module, _inputs, output, key=key: taps.__setitem__(
                    key, output[:, 0].float()
                )
            )
            for key, index in DEPTHS.items()
        ]
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with _autocast(device):
                encoded = model.backbone.forward_features(images)
                pre_logits = model.backbone.forward_head(encoded, pre_logits=True)
                logits = model.head(pre_logits)
            target_values = targets.numpy()
            predictions = logits.argmax(1).cpu().numpy()
            np.add.at(total, target_values, 1)
            np.add.at(correct, target_values, predictions == target_values)
            labels.append(target_values)
            if capture_features:
                features["pre"].append(pre_logits.float().cpu().numpy().astype(np.float32))
                for key in DEPTHS:
                    features[key].append(taps[key].cpu().numpy().astype(np.float32))
    finally:
        for hook in hooks:
            hook.remove()

    result = {
        "correct": correct,
        "total": total,
        "labels": np.concatenate(labels),
    }
    if capture_features:
        result["feats"] = {
            key: np.concatenate(values) for key, values in features.items()
        }
    return result


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Centered linear CKA, evaluated over the complete split slice."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("CKA inputs must have the same 2-D shape and at least two rows")
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left64 -= left64.mean(axis=0, keepdims=True)
    right64 -= right64.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(left64.T @ left64) * np.linalg.norm(right64.T @ right64)
    if denominator <= 0:
        return 0.0
    value = float(((left64.T @ right64) ** 2).sum() / denominator)
    return min(max(value, 0.0), 1.0)


def harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0 or right <= 0 else 2 * left * right / (left + right)


def split_accuracy(result: dict, forget_labels: Iterable[int]) -> tuple[float, float]:
    forget_mask = np.zeros(len(result["total"]), dtype=bool)
    forget_mask[list(forget_labels)] = True
    correct = result["correct"]
    total = result["total"]
    forget_total = total[forget_mask].sum()
    retain_total = total[~forget_mask].sum()
    if forget_total <= 0 or retain_total <= 0:
        raise ValueError("accuracy split must contain both forget and retain examples")
    return (
        float(100 * correct[forget_mask].sum() / forget_total),
        float(100 * correct[~forget_mask].sum() / retain_total),
    )


def compute_score(
    *,
    acc_f: float,
    acc_r: float,
    reference_acc_f: float,
    reference_acc_r: float,
    cka_f_o: float,
    cka_r_o: float,
) -> dict[str, float]:
    values = (acc_f, acc_r, reference_acc_f, reference_acc_r)
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 100
        for value in values
    ):
        raise ValueError("accuracy inputs must be finite values in [0, 100]")
    cka_values = (cka_f_o, cka_r_o)
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
        for value in cka_values
    ):
        raise ValueError("CKA inputs must be finite values in [0, 1]")
    retain_drop = max(reference_acc_r - acc_r, 0.0) / 100
    forget_gap = abs(acc_f - reference_acc_f) / 100
    aus = (1 - retain_drop) / (1 + forget_gap)
    rus_o = harmonic(1 - cka_f_o, cka_r_o)
    return {
        "drop_r": retain_drop * 100,
        "gap_f": forget_gap * 100,
        "AUS": aus,
        "RUS_o": rus_o,
        "final_score": harmonic(aus, rus_o),
    }


def _load_reference_cache(
    path: Path,
    refs: dict,
    expected_labels: np.ndarray,
    num_classes: int,
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != CACHE_KEYS:
            raise ValueError("M_o reference cache arrays do not match the v2 schema")
        for key, expected in (
            ("schema_version", SCHEMA_VERSION),
            ("phase", refs["phase"]),
            ("dataset_revision", refs["dataset_revision"]),
            ("split_name", refs["representation_split"]),
        ):
            value = archive[key]
            if value.ndim != 0 or value.item() != expected:
                raise ValueError(f"M_o reference cache {key} mismatch")
        labels = archive["labels"]
        if labels.dtype != np.dtype(np.int64) or labels.ndim != 1:
            raise ValueError("M_o reference cache labels must be a 1-D int64 array")
        if not np.array_equal(labels, expected_labels):
            raise ValueError("M_o reference cache labels/order do not match the manifest")
        correct, total = archive["correct"], archive["total"]
        for name, values in (("correct", correct), ("total", total)):
            if (
                values.dtype != np.dtype(np.float64)
                or values.shape != (num_classes,)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(
                    f"M_o reference cache {name} must be non-negative "
                    f"float64[{num_classes}] counts"
                )
        if (correct > total).any():
            raise ValueError("M_o reference cache correct exceeds total")
        expected_total = np.bincount(expected_labels, minlength=num_classes).astype(
            np.float64
        )
        if not np.array_equal(total, expected_total):
            raise ValueError("M_o reference cache totals do not match labels")
        features: dict[str, np.ndarray] = {}
        for depth in ALL_DEPTHS:
            values = archive[f"f_{depth}"]
            if (
                values.dtype != np.dtype(np.float32)
                or values.shape != (len(expected_labels), FEATURE_WIDTH)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"M_o reference cache f_{depth} must be finite "
                    f"float32[{len(expected_labels)},{FEATURE_WIDTH}]"
                )
            features[depth] = values
        return {
            "labels": labels,
            "feats": features,
        }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def score_submission(args: argparse.Namespace) -> dict:
    root, wnids, splits, refs = load_phase_inputs(
        args.split, args.refs, args.phase, args.image_root
    )
    device = _runtime_device(args.device)
    forget_labels = refs.get("forget_labels")
    if (
        not isinstance(forget_labels, list)
        or not forget_labels
        or any(isinstance(x, bool) or not isinstance(x, int) for x in forget_labels)
    ):
        raise ValueError("metric refs forget_labels must be a non-empty integer list")
    score_depth = refs.get("score_depth", DEFAULT_SCORE_DEPTH)
    if refs.get("depths") != ALL_DEPTHS or score_depth not in ALL_DEPTHS:
        raise ValueError("metric refs contain unsupported feature depths")
    reference_accuracy = refs.get("reference_accuracy")
    if not isinstance(reference_accuracy, dict):
        raise ValueError("metric refs reference_accuracy must be a mapping")

    model = load_submission_model(args.ckpt, device, len(wnids))
    representation_name = refs["representation_split"]
    accuracy_name = refs["accuracy_split"]
    representation_result = extract(
        model,
        splits[representation_name],
        root,
        device,
        capture_features=True,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if accuracy_name == representation_name:
        accuracy_result = representation_result
    else:
        accuracy_result = extract(
            model,
            splits[accuracy_name],
            root,
            device,
            capture_features=False,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    acc_f, acc_r = split_accuracy(accuracy_result, forget_labels)
    reference = _load_reference_cache(
        args.mo_cache, refs, representation_result["labels"], len(wnids)
    )
    forget_mask = np.isin(representation_result["labels"], forget_labels)
    cka_per_depth = {}
    for depth in ALL_DEPTHS:
        submitted = representation_result["feats"][depth]
        original = reference["feats"][depth]
        cka_per_depth[depth] = {
            "CKA_f_o": linear_cka(submitted[forget_mask], original[forget_mask]),
            "CKA_r_o": linear_cka(submitted[~forget_mask], original[~forget_mask]),
        }
    score_cka = cka_per_depth[score_depth]
    score = compute_score(
        acc_f=acc_f,
        acc_r=acc_r,
        reference_acc_f=float(reference_accuracy["acc_f"]),
        reference_acc_r=float(reference_accuracy["acc_r"]),
        cka_f_o=score_cka["CKA_f_o"],
        cka_r_o=score_cka["CKA_r_o"],
    )

    split_accuracy_report = {}
    for name, result in (
        (representation_name, representation_result),
        (accuracy_name, accuracy_result),
    ):
        if name not in split_accuracy_report:
            split_acc_f, split_acc_r = split_accuracy(result, forget_labels)
            split_accuracy_report[name] = {
                "Acc_f": split_acc_f,
                "Acc_r": split_acc_r,
            }

    report = {
        "schema_version": SCHEMA_VERSION,
        "score_version": SCORE_VERSION,
        "dataset_revision": refs["dataset_revision"],
        "phase": args.phase,
        "tag": args.tag,
        "accuracy_split": accuracy_name,
        "representation_split": representation_name,
        "score_depth": score_depth,
        "split_accuracy": split_accuracy_report,
        "accuracy_metric": {
            "Acc_f": acc_f,
            "Acc_r": acc_r,
            "reference_Acc_f": float(reference_accuracy["acc_f"]),
            "reference_Acc_r": float(reference_accuracy["acc_r"]),
            "drop_r": score["drop_r"],
            "gap_f": score["gap_f"],
            "AUS": score["AUS"],
        },
        "representation_metric": {
            "CKA_f_o": score_cka["CKA_f_o"],
            "CKA_r_o": score_cka["CKA_r_o"],
            "RUS_o": score["RUS_o"],
        },
        "cka_per_depth": cka_per_depth,
        "AUS": score["AUS"],
        "RUS_o": score["RUS_o"],
        "final_score": score["final_score"],
        "forget_wnids": refs.get("forget_wnids", []),
    }
    _write_json_atomic(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def prepare_reference(args: argparse.Namespace) -> None:
    root, wnids, splits, refs = load_phase_inputs(
        args.split, args.refs, args.phase, args.image_root
    )
    if args.output.exists() and not args.force:
        raise FileExistsError(f"reference cache already exists: {args.output}")
    device = _runtime_device(args.device)
    split_name = refs["representation_split"]
    model = load_trusted_reference_model(args.mo, device, len(wnids))
    result = extract(
        model,
        splits[split_name],
        root,
        device,
        capture_features=True,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
        np.savez(
            temporary_file,
            schema_version=np.asarray(SCHEMA_VERSION),
            phase=np.asarray(args.phase),
            dataset_revision=np.asarray(refs["dataset_revision"]),
            split_name=np.asarray(split_name),
            correct=result["correct"],
            total=result["total"],
            labels=result["labels"],
            **{f"f_{depth}": values for depth, values in result["feats"].items()},
        )
        temporary_file.flush()
        os.fchmod(temporary_file.fileno(), 0o444)
        os.fsync(temporary_file.fileno())
    try:
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {args.output}", flush=True)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase", required=True, choices=("validation", "test"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GRADER_BATCH_SIZE", "128")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("GRADER_NUM_WORKERS", "8")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    score_parser = commands.add_parser("score")
    _common_arguments(score_parser)
    score_parser.add_argument("--ckpt", type=Path, required=True)
    score_parser.add_argument("--mo-cache", type=Path, required=True)
    score_parser.add_argument("--tag", required=True)
    score_parser.add_argument("--report", type=Path, required=True)

    reference_parser = commands.add_parser("prepare-reference")
    _common_arguments(reference_parser)
    reference_parser.add_argument("--mo", type=Path, required=True)
    reference_parser.add_argument("--output", type=Path, required=True)
    reference_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "score":
        score_submission(args)
    else:
        prepare_reference(args)


if __name__ == "__main__":
    main()
