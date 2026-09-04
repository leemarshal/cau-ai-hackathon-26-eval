"""
ViT-B/16 (timm MAE) + local ImageNet wiring for the RUM codebase.

The init was switched from `augreg_in1k` to `mae`: augreg is supervised on all
1000 ImageNet labels, so it has already seen the forget classes WITH labels and
no honest "retrain" reference can exist on top of it. MAE pretraining is
label-free, so class knowledge enters only at fine-tuning time -- which is what
M_o / M_r / M_f are all defined relative to.

Three things this module exists to get right:

1. **Preprocessing follows the model, not a hardcoded guess.** The two
   checkpoints genuinely disagree, and `resolve_data_config` is what keeps this
   honest:

       augreg_in1k : mean = std = 0.5
       mae         : mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)

   Getting this wrong silently degrades every representation downstream.

2. **Normalization lives inside the model by default**, matching this repo's
   convention (see models/ResNet.py: `x = self.normalize(x)`), so unlearning
   methods, generate_mask.py and the CKA harness all see one interface. The MAE
   fine-tuning recipe needs timm's transform pipeline, which normalizes itself;
   pass `in_model_norm=False` there so the image is not normalized twice.
   `build_transforms` / `build_train_transform` return the flag they assume, and
   `assert_norm_consistent` checks a (model, transform) pair before training.

3. **MAE needs its fine-tuning recipe.** A ViT-B fine-tuned from MAE without
   layer-wise lr decay, RandAugment, mixup/cutmix, label smoothing and random
   erasing loses accuracy in 10%p units, not fractions of a percent -- which
   would collapse the dynamic range of every metric computed on M_o and M_r. The
   published ViT-B/16 recipe is mirrored in MAE_FT below; `build_param_groups`
   and `build_mixup` construct the two pieces that are not plain kwargs.

Local ImageNet layout (verified):
    train/n01440764/*.JPEG        1,281,167 images, 1000 wnid dirs
    val/0000 ... val/0999         50,000 images, already in sorted-wnid order
                                  (set.py remaps ILSVRC ids, so val/0000 == tench)

NOTE: `imagenet_loaders` below is a legacy generic ImageNet helper and is not
used by the v2 scorer. `score_unlearning.py` loads the phase-scoped immutable
manifest: public validation runs on student servers and the organizer bundle
contains only the private `test` split.
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DEFAULT_ROOT = "/data/hai_ssh/datasets/imagenet"
DEFAULT_INDEX_CACHE = "./es_imagenet_mae_1k/imagefolder_index.pt"

# ---- from-scratch (random init) recipe -------------------------------------
# DeiT (Touvron et al., arXiv:2012.12877) is the reference for training a ViT
# from random init: AdamW at lr = 5e-4 * batch/512, wd 0.05, cosine, RandAugment
# + mixup/cutmix + label smoothing + random erasing, drop-path 0.1.
#
# Two deliberate differences from MAE_FT:
#   layer_decay = 1.0  -- layer-wise lr decay exists to protect a PRETRAINED
#                         trunk. With random init there is nothing to protect and
#                         decaying early layers just starves them.
#   warmup      = 20   -- DeiT uses 5 epochs on 1.28M images. Here an epoch is
#                         11x smaller, so 5 epochs is ~700 steps at batch 512,
#                         far too short to stabilise AdamW from random init.
#
# Scale note: blr is per-256 in this file's convention, so blr 2.5e-4 gives
# DeiT's lr = 5e-4 at total batch 512.
SCRATCH = {
    "blr": 2.5e-4,
    "layer_decay": 1.0,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "drop_path": 0.1,
    "label_smoothing": 0.1,
    "mixup": 0.8,
    "cutmix": 1.0,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "reprob": 0.25,
    "warmup_epochs": 20,
    "min_lr": 1e-6,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "re_mode": "pixel",
    "re_count": 1,
    "color_jitter": None,
    "clip_grad": 1.0,      # from random init AdamW can spike; DeiT clips at 1.0
    "model_ema": False,
}

# MAE, not augreg -- see the module docstring for why.
DEFAULT_ARCH = "vit_base_patch16_224.mae"
# the competition model is 100-way (M_o on all 100, M_r on 90 of them but still
# with a 100-way head so the two are directly comparable).
DEFAULT_NUM_CLASSES = 100

# He et al., "Masked Autoencoders Are Scalable Vision Learners", ViT-B/16
# end-to-end fine-tuning recipe. These are NOT interchangeable with the short
# unlearning-run settings this file used to carry: M_o and M_r are full
# fine-tunes from a label-free init, which is exactly the regime where dropping
# layer decay or mixup costs double-digit accuracy.
MAE_FT = {
    # ViT-B uses blr 5e-4 + layer_decay 0.65; the 1e-3 + 0.75 pairing in the same
    # table is ViT-L/H. Mixing them up doubles the lr.
    "blr": 5e-4,            # base lr; actual lr = blr * total_batch / 256
    "layer_decay": 0.65,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "drop_path": 0.1,
    "label_smoothing": 0.1,
    "mixup": 0.8,
    "cutmix": 1.0,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "reprob": 0.25,         # random erasing
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "re_mode": "pixel",
    "re_count": 1,
    "color_jitter": None,   # off: RandAugment already covers it
    "clip_grad": None,
    "model_ema": False,
}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Normalize(nn.Module):
    """Same contract as utils.NormalizeByChannelMeanStd, but carrying the
    model's own statistics rather than the torchvision ImageNet ones."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

    def extra_repr(self):
        return f"mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()}"


class ViTWrapper(nn.Module):
    """timm ViT with in-model normalization and an explicit feature tap.

    `features(x)` returns phi = g(x; theta) -- the pre-logits CLS token that the
    classifier head consumes (768-d for ViT-B). This is the same vector
    es_class_ranking.py accumulated, so ES / CKA / probes all agree on what a
    representation is.
    """

    def __init__(self, arch=DEFAULT_ARCH, num_classes=DEFAULT_NUM_CLASSES,
                 pretrained=True, drop_path_rate=MAE_FT["drop_path"],
                 in_model_norm=True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=num_classes,
            drop_path_rate=drop_path_rate)
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        # the MAE checkpoint ships with num_classes=0, so timm builds a fresh
        # head; the published recipe initialises it small rather than with the
        # default trunc_normal_(std=0.02) used for the trunk.
        head = self.backbone.head
        if isinstance(head, nn.Linear) and head.out_features == num_classes:
            nn.init.trunc_normal_(head.weight, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        self.in_model_norm = in_model_norm
        self.normalize = Normalize(cfg["mean"], cfg["std"]) if in_model_norm \
            else nn.Identity()
        self.data_config = cfg
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone(self.normalize(x))

    def features(self, x):
        """Pre-logits representation (before the classifier head)."""
        z = self.backbone.forward_features(self.normalize(x))
        return self.backbone.forward_head(z, pre_logits=True)

    @property
    def head(self):
        return self.backbone.head


def vit_base_patch16_224(num_classes=DEFAULT_NUM_CLASSES, imagenet=True,
                         pretrained=True, **kw):
    """Factory with the signature models/__init__.py::model_dict expects."""
    return ViTWrapper(num_classes=num_classes, pretrained=pretrained, **kw)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_transforms(train, cfg=None):
    """Light pipeline that emits UN-normalized tensors (in_model_norm=True).

    This is the repo-convention interface: the model normalizes. Correct for
    every eval path and for short unlearning runs, and far too weak to fine-tune
    M_o / M_r from MAE -- use build_train_transform for that.
    """
    cfg = cfg or {"input_size": (3, 224, 224), "crop_pct": 0.9}
    size = cfg["input_size"][-1]
    bicubic = transforms.InterpolationMode.BICUBIC
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=bicubic),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),          # normalization happens in the model
        ])
    resize = int(round(size / cfg.get("crop_pct", 0.9)))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=bicubic),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


def build_train_transform(cfg, recipe=None):
    """MAE fine-tuning train pipeline: RandAugment + random erasing.

    timm's pipeline normalizes internally, so the model this feeds MUST be built
    with in_model_norm=False or every image is normalized twice. Returns
    (transform, in_model_norm) so the caller cannot forget.
    """
    import timm

    r = {**MAE_FT, **(recipe or {})}
    tf = timm.data.create_transform(
        input_size=cfg["input_size"][-1],
        is_training=True,
        auto_augment=r["auto_augment"],
        interpolation=cfg.get("interpolation", "bicubic"),
        re_prob=r["reprob"], re_mode="pixel", re_count=1,
        mean=cfg["mean"], std=cfg["std"],
    )
    return tf, False


def assert_norm_consistent(model, in_model_norm):
    """Guard against the one mistake this split invites: double normalization."""
    if bool(model.in_model_norm) != bool(in_model_norm):
        raise ValueError(
            f"model.in_model_norm={model.in_model_norm} but the transform "
            f"assumes in_model_norm={in_model_norm}. build_train_transform "
            f"normalizes itself -- construct the model with in_model_norm=False.")


# --------------------------------------------------------------------------- #
# MAE fine-tuning recipe pieces that are not plain kwargs
# --------------------------------------------------------------------------- #
def build_mixup(num_classes=DEFAULT_NUM_CLASSES, recipe=None):
    """mixup + cutmix collate op. Its output is SOFT targets, so the loss must be
    timm.loss.SoftTargetCrossEntropy -- label smoothing is folded in here, not
    applied a second time in the criterion."""
    import timm

    r = {**MAE_FT, **(recipe or {})}
    return timm.data.Mixup(
        mixup_alpha=r["mixup"], cutmix_alpha=r["cutmix"], prob=1.0,
        switch_prob=0.5, mode="batch",
        label_smoothing=r["label_smoothing"], num_classes=num_classes)


def build_param_groups(model, recipe=None):
    """Layer-wise lr decay groups. Dropping this is the single biggest way to
    lose accuracy fine-tuning from MAE."""
    from timm.optim.optim_factory import param_groups_layer_decay

    r = {**MAE_FT, **(recipe or {})}
    # the decay schedule is keyed off the ViT's own group_matcher, so it must see
    # the timm model, not the ViTWrapper around it
    net = model.backbone if isinstance(model, ViTWrapper) else model
    return param_groups_layer_decay(
        net, weight_decay=r["weight_decay"],
        no_weight_decay_list=net.no_weight_decay(),
        layer_decay=r["layer_decay"])


def scaled_lr(total_batch_size, recipe=None):
    """actual lr = blr * total_batch / 256 (linear scaling rule)."""
    r = {**MAE_FT, **(recipe or {})}
    return r["blr"] * total_batch_size / 256


def _imagefolder(root, transform, cache=None):
    """ImageFolder with an optional cached file index (scanning 1.28M files
    costs ~10s; the cache also guarantees the exact same ordering that
    es_class_ranking.py used)."""
    ds = ImageFolder(root, transform=transform)
    if cache and os.path.exists(cache):
        meta = torch.load(cache, weights_only=False)
        if meta["classes"] == ds.classes:
            ds.samples = meta["samples"]
            ds.targets = [s[1] for s in meta["samples"]]
            ds.class_to_idx = meta["class_to_idx"]
    return ds


def load_forget_classes(path):
    """Accepts FINAL_forget_100.json / forget_classes_*.json / a plain wnid list."""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        return list(d["wnid"])
    return list(d)


def imagenet_loaders(forget_wnids, root=DEFAULT_ROOT, batch_size=256, workers=16,
                     index_cache=DEFAULT_INDEX_CACHE, cfg=None, seed=1,
                     eval_only=False):
    """retain / forget train splits + val splits, keyed by class.

    Returns an OrderedDict with the keys the repo's unlearning methods expect
    (retain, forget, val, test) plus the class-wise val splits that class
    unlearning actually needs to be judged on.
    """
    from collections import OrderedDict

    train_tf = build_transforms(not eval_only, cfg)
    eval_tf = build_transforms(False, cfg)

    train = _imagefolder(os.path.join(root, "train"), train_tf, index_cache)
    val = _imagefolder(os.path.join(root, "val"), eval_tf)

    missing = [w for w in forget_wnids if w not in train.class_to_idx]
    if missing:
        raise KeyError(f"{len(missing)} forget wnids absent from {root}/train: "
                       f"{missing[:5]}")
    forget_ids = {train.class_to_idx[w] for w in forget_wnids}
    print(f"[data] forget {len(forget_ids)} classes / retain "
          f"{len(train.classes) - len(forget_ids)} classes")

    tt = torch.as_tensor(train.targets)
    f_mask = torch.isin(tt, torch.tensor(sorted(forget_ids)))
    f_idx = torch.nonzero(f_mask).flatten().tolist()
    r_idx = torch.nonzero(~f_mask).flatten().tolist()

    # val is in the same sorted-wnid order, so the same class ids apply
    vt = torch.as_tensor(val.targets)
    vf_mask = torch.isin(vt, torch.tensor(sorted(forget_ids)))
    vf_idx = torch.nonzero(vf_mask).flatten().tolist()
    vr_idx = torch.nonzero(~vf_mask).flatten().tolist()
    print(f"[data] train forget {len(f_idx):,} / retain {len(r_idx):,}   "
          f"val forget {len(vf_idx):,} / retain {len(vr_idx):,}")

    def mk(ds, idx, shuffle):
        g = torch.Generator().manual_seed(seed)
        return DataLoader(Subset(ds, idx), batch_size=batch_size,
                          shuffle=shuffle and not eval_only, num_workers=workers,
                          pin_memory=True, generator=g if shuffle else None,
                          persistent_workers=workers > 0)

    val_eval = _imagefolder(os.path.join(root, "train"), eval_tf, index_cache)
    return OrderedDict(
        retain=mk(train, r_idx, True),
        forget=mk(train, f_idx, True),
        retain_eval=mk(val_eval, r_idx, False),   # train images, eval transform
        forget_eval=mk(val_eval, f_idx, False),
        val=mk(val, vr_idx, False),               # held-out, retain classes
        test=mk(val, vf_idx, False),              # held-out, forget classes
        forget_ids=sorted(forget_ids),
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="smoke-test the ViT + ImageNet wiring")
    p.add_argument("--forget", default="./es_imagenet_mae/OLD_augreg_pool_100.json")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batches", type=int, default=4)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_base_patch16_224().eval().to(dev)
    print(f"[model] dim={model.num_features}  norm={model.normalize}")
    print(f"[model] cfg={model.data_config}")

    loaders = imagenet_loaders(load_forget_classes(args.forget), root=args.root,
                               batch_size=args.batch_size, workers=args.workers,
                               cfg=model.data_config, eval_only=True)

    for split in ["forget", "val", "test"]:
        correct = seen = 0
        feats = None
        with torch.no_grad():
            for i, (x, y) in enumerate(loaders[split]):
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                feats = model.features(x)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)
                if i + 1 >= args.batches:
                    break
        print(f"[{split:6s}] top-1 {100 * correct / seen:5.1f}%  ({seen} imgs)  "
              f"feat {tuple(feats.shape)}")
