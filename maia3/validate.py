"""Evaluate a trained Maia3 checkpoint on a parquet validation set.

Loads a checkpoint (weights + its embedded config), builds the model, runs a
single validation pass over a parquet file.

Usage:
    python -m maia3.validate --checkpoint runs/maia3/best.pt data/val.parquet
    python -m maia3.validate --checkpoint CKPT --val-parquet-path data/val.parquet \
        --val-batch-size 512 --device cuda
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from .config import (
    build_arg_parser,
    cfgs_from_args,
    load_checkpoint_config,
    load_yaml_config,
)
from .models import MAIA3Model
from .train import _build_loader, _validate
from .utils import seed_everything


def _build_parser() -> argparse.ArgumentParser:
    """Reuse the training parser (one flag per model/data/train field, plus
    --config) and add the eval-only arguments."""
    parser = build_arg_parser()
    parser.description = "Evaluate a Maia3 checkpoint on a parquet validation set."
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint (.pt) to evaluate; supplies the weights and the model "
        "architecture (its embedded config is layered in as defaults).",
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        default=None,
        help="Validation parquet file. Overrides data.val_parquet_path / "
        "--val-parquet-path when given.",
    )
    return parser


def _layered_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Layer config sources into the parser defaults before the final parse, so
    explicit CLI flags win:  checkpoint config  <  --config YAML  <  CLI flags."""
    pre_args, _ = parser.parse_known_args(argv)
    layered: dict[str, Any] = {}
    # Full embedded config first so the model section matches the weights.
    layered.update(load_checkpoint_config(pre_args.checkpoint))
    # Don't inherit the training-time validation cap: a validate run should cover
    # the whole file by default (val_steps=0). A --config or --val-steps still wins.
    layered.pop("val_steps", None)
    if pre_args.config is not None:
        layered.update(load_yaml_config(pre_args.config))
    if layered:
        parser.set_defaults(**layered)
    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    """Honor the requested device, falling back to CPU when CUDA is missing."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("note: CUDA unavailable, falling back to CPU")
        return "cpu"
    return device


@torch.no_grad()
def validate(argv: list[str] | None = None) -> dict[str, float]:
    parser = _build_parser()
    args = _layered_args(parser, argv)
    model_cfg, data_cfg, train_cfg = cfgs_from_args(args)

    val_path = args.parquet or data_cfg.val_parquet_path
    if not val_path:
        parser.error(
            "no validation parquet given (pass it positionally or via "
            "--val-parquet-path / data.val_parquet_path)"
        )

    seed_everything(train_cfg.seed)
    device = _resolve_device(train_cfg.device)

    model = MAIA3Model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step")
    print(
        f"loaded {args.checkpoint}"
        + (f" (step {step})" if step is not None else "")
        + f"\nevaluating on {val_path}"
    )

    loader = _build_loader(val_path, data_cfg, model_cfg, train=False)
    amp_enabled = train_cfg.use_amp and device.startswith("cuda")
    metrics = _validate(model, loader, device, amp_enabled, train_cfg)

    print("metrics:")
    for k in sorted(metrics):
        print(f"  {k:20s} {metrics[k]:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    validate(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
