"""Typed configuration dataclasses for Maia3.

`ModelConfig` mirrors the architecture / tokenization flags that `uci.py`'s
argparse builds today; `MAIA3Model.__init__` accepts any object with the same
attribute names, so a `ModelConfig` instance is a drop-in replacement for the
argparse `Namespace`. `DataConfig` and `TrainConfig` are training-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class ModelConfig:
    # Tokenization (must match the data pipeline)
    history: int = 8
    use_padding: bool = True
    include_time_info: bool = False

    # Transformer
    dim_emb: int = 128
    dim_vit: int = 192
    num_blocks: int = 8
    num_heads: int = 6
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    head_hid_dim: int = 192

    # GAB
    use_gab: bool = True
    gab_gen_size: int = 64
    gab_per_square_dim: int = 0
    gab_intermediate_dim: int = 64
    use_rms_norm: bool = True
    omit_qkv_biases: bool = True
    activation: str = "gelu"

    # Position-encoding alternatives
    use_relative_bias: bool = False
    use_absolute_pe: bool = False


@dataclass
class DataConfig:
    shard_dir: str = ""
    val_shard_dir: str | None = None
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    # Ply filtering
    min_ply: int = 0
    max_ply: int | None = None
    # Elo filtering (skip games with either player outside this range)
    min_elo: int | None = None
    max_elo: int | None = None


@dataclass
class TrainConfig:
    output_dir: str = "runs/maia3"
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    num_steps: int = 100_000
    warmup_steps: int = 1_000

    # Logging / checkpointing
    log_interval: int = 50
    val_interval: int = 0  # 0 = disabled
    ckpt_interval: int = 5_000
    resume_from: str | None = None


def _add_dataclass_args(parser: argparse.ArgumentParser, dc_cls: type, prefix: str = "") -> None:
    """Add one CLI flag per dataclass field. Booleans become --flag/--no-flag pairs."""
    for f in fields(dc_cls):
        flag = f"--{prefix}{f.name.replace('_', '-')}"
        kwargs: dict[str, Any] = {"default": f.default, "dest": f"{prefix.replace('-', '_')}{f.name}"}
        if f.type is bool or f.default is True or f.default is False:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            # Best-effort type coercion; argparse will surface bad values.
            kwargs["type"] = _coerce_type(f.type)
        parser.add_argument(flag, **kwargs)


def _coerce_type(annotation: Any):
    if annotation in (int, "int", "int | None"):
        return lambda v: None if v in (None, "None", "null") else int(v)
    if annotation in (float, "float"):
        return float
    if annotation in (str, "str", "str | None"):
        return lambda v: None if v in (None, "None", "null") else str(v)
    return str


def cfgs_from_args(args: argparse.Namespace) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """Pull a flat Namespace into the three dataclasses by name."""
    def pick(dc_cls):
        return dc_cls(**{f.name: getattr(args, f.name) for f in fields(dc_cls) if hasattr(args, f.name)})
    return pick(ModelConfig), pick(DataConfig), pick(TrainConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Argparse for `python -m maia3.train`. One flag per field across all three configs."""
    parser = argparse.ArgumentParser(
        description="Train a Maia3 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_grp = parser.add_argument_group("model")
    _add_dataclass_args(model_grp, ModelConfig)
    data_grp = parser.add_argument_group("data")
    _add_dataclass_args(data_grp, DataConfig)
    train_grp = parser.add_argument_group("train")
    _add_dataclass_args(train_grp, TrainConfig)
    return parser


def cfg_dict(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> dict:
    """Serializable snapshot of all three configs, for checkpoint metadata."""
    return {"model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)}
