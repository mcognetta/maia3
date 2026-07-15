"""Typed configuration dataclasses for Maia3.

`ModelConfig` mirrors the architecture / tokenization flags that `uci.py`'s
argparse builds today; `MAIA3Model.__init__` accepts any object with the same
attribute names, so a `ModelConfig` instance is a drop-in replacement for the
argparse `Namespace`. `DataConfig` and `TrainConfig` are training-only.

Architecture is selected entirely through the YAML config's `model:` section —
the per-size recipes live in `configs/maia3-*.yaml`. There are no built-in size
presets; the `ModelConfig` defaults below match the smallest (`3m`) size.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ModelConfig:
    # Tokenization (must match the data pipeline)
    history: int = 8

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
    # Parquet position datasets produced by `maia3-preprocess`. Game/position
    # filtering happens at preprocess time (see maia3/preprocess.py), so there
    # are no filter knobs here.
    #
    # Training data is one or more parquet shards. Use `parquet_path` for a
    # single file, or `parquet_paths` for a list of shards that are trained
    # through in order — one shard at a time, freed before the next — and one
    # full pass over the list counts as a single epoch. `parquet_paths` takes
    # precedence when both are set.
    parquet_path: str = ""
    parquet_paths: list[str] | None = None
    val_parquet_path: str | None = None
    batch_size: int = 256
    val_batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    def train_paths(self) -> list[str]:
        """Ordered list of training parquet shards. `parquet_paths` wins over
        the single `parquet_path`; returns ``[]`` when neither is set."""
        if self.parquet_paths:
            return list(self.parquet_paths)
        return [self.parquet_path] if self.parquet_path else []


@dataclass
class TrainConfig:
    output_dir: str = "runs/maia3"
    # When true (default), each run's artifacts (config.yaml + checkpoints) are
    # written to a fresh `output_dir/<timestamp>/` subdirectory so runs never
    # overwrite each other. Set false to write directly into `output_dir`.
    timestamp_runs: bool = True
    device: str = "cuda"
    seed: int = 42

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1  # micro-batches per optimizer step (effective batch = batch_size * this)

    # Schedule — linear warmup, then cosine anneal `lr` → `min_lr`. With
    # cosine_cycle_steps > 0 the cosine restarts ("refreshes") every that many
    # optimizer steps (CosineAnnealingWarmRestarts); 0 = a single cosine spanning
    # the whole run.
    num_steps: int = 100_000
    warmup_steps: int = 1_000
    min_lr: float = 0.0
    cosine_cycle_steps: int = 0

    # Mixed precision (GradScaler). The loss scale itself is not configurable:
    # on resume it is restored from the checkpoint, and on a fresh run it warms
    # up from GradScaler's default starting scale (backing off until stable).
    # These knobs only tune how that auto-calibration behaves.
    use_amp: bool = True
    amp_growth_factor: float = 2.0
    amp_growth_interval: int = 2000
    amp_backoff_factor: float = 0.5

    # Loss scales (one per model head; see compute_loss)
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Logging / checkpointing
    log_interval: int = 50
    val_interval: int = 0  # 0 = disabled
    val_steps: int = 0  # batches per validation pass; 0 = full val set
    ckpt_interval: int = 5_000
    # Checkpoint retention. keep_last_k > 0 keeps only that many most-recent
    # step_*.pt checkpoints (older ones pruned after each periodic save); 0 keeps
    # all. best.pt — the checkpoint with the best validation metric so far — is
    # written whenever validation improves and is never pruned, so it survives
    # regardless of keep_last_k. best_metric is the val metric key to monitor (as
    # produced by compute_loss, e.g. "loss/total", "acc/policy_top1"); best_mode
    # is "min" (lower is better, e.g. loss) or "max" (higher is better, e.g. acc).
    keep_last_k: int = 0
    best_metric: str = "loss/total"
    best_mode: str = "min"
    # Continue a run: restore weights + optimizer + LR schedule + step from this
    # checkpoint and pick up where it stopped. Its full embedded config is also
    # inherited (see apply_layered_defaults), so `--resume-from CKPT` alone works.
    resume_from: str | None = None

    # Weights & Biases. Logging is on only when use_wandb is true; the rest are
    # passed straight to wandb.init (wandb_mode: online | offline | disabled).
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"


def _add_dataclass_args(parser: argparse.ArgumentParser, dc_cls: type, prefix: str = "") -> None:
    """Add one CLI flag per dataclass field. Booleans become --flag/--no-flag pairs."""
    for f in fields(dc_cls):
        flag = f"--{prefix}{f.name.replace('_', '-')}"
        kwargs: dict[str, Any] = {"default": f.default, "dest": f"{prefix.replace('-', '_')}{f.name}"}
        if f.type is bool or f.default is True or f.default is False:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif str(f.type).startswith("list"):
            # e.g. `--parquet-paths a.parquet b.parquet`; element type is str.
            kwargs["nargs"] = "*"
            kwargs["type"] = str
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
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (e.g. one of configs/maia3-*.yaml, "
        "which select the model size). CLI flags override its values.",
    )
    model_grp = parser.add_argument_group("model")
    _add_dataclass_args(model_grp, ModelConfig)
    data_grp = parser.add_argument_group("data")
    _add_dataclass_args(data_grp, DataConfig)
    train_grp = parser.add_argument_group("train")
    _add_dataclass_args(train_grp, TrainConfig)
    return parser


def apply_layered_defaults(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse `argv`, layering config sources low-to-high priority into the parser
    defaults before the final parse so explicit CLI flags always win:

        dataclass defaults  <  checkpoint config  <  --config YAML  <  CLI flags

    The checkpoint config layered in comes from `--resume-from` (the full config,
    so the flag alone reconstructs that run). A --config or CLI flag still
    overrides it. The parser must define `--config` and `--resume-from`; each is
    skipped if unset. Used by `maia3.train`.
    """
    pre_args, _ = parser.parse_known_args(argv)
    layered: dict[str, Any] = {}
    # A checkpoint carries its own full config. Layer all of it first so
    # --resume-from alone rebuilds the exact model/data/train setup (and the
    # architecture matches the weights). An explicit --config or CLI flag wins.
    if getattr(pre_args, "resume_from", None):
        layered.update(load_checkpoint_config(pre_args.resume_from))
    if getattr(pre_args, "config", None) is not None:
        layered.update(load_yaml_config(pre_args.config))
    if layered:
        parser.set_defaults(**layered)
    return parser.parse_args(argv)


def cfg_dict(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> dict:
    """Serializable snapshot of all three configs, for checkpoint metadata."""
    return {"model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)}


def _yaml_safe(obj: Any) -> Any:
    """Coerce a config snapshot into types PyYAML's safe dumper handles — most
    notably turning tuples (e.g. `betas`) into lists."""
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    return obj


def save_yaml_config(
    path: str, model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig
) -> None:
    """Write the fully-resolved config (all three sections, every field) to a
    YAML file. The output is itself a valid input to `load_yaml_config` /
    `--config`, so a run can be reproduced from the file dropped next to its
    checkpoints. Re-reading it round-trips through the same section/field
    validation as any hand-written config."""
    import yaml

    with open(path, "w") as fh:
        yaml.safe_dump(
            _yaml_safe(cfg_dict(model_cfg, data_cfg, train_cfg)),
            fh,
            sort_keys=False,
            default_flow_style=False,
        )


#: Field names allowed in each top-level section of a YAML config file.
_SECTION_FIELDS = {
    "model": {f.name for f in fields(ModelConfig)},
    "data": {f.name for f in fields(DataConfig)},
    "train": {f.name for f in fields(TrainConfig)},
}


def load_yaml_config(path: str) -> dict[str, Any]:
    """Read a YAML config file into a flat ``{field: value}`` dict.

    The file has three optional sections — ``model``, ``data``, ``train`` —
    mirroring the dataclasses above. You only list the fields you want to
    change; everything else keeps its default. Example::

        model:
          dim_vit: 256
          num_heads: 8
        data:
          parquet_path: data/train.parquet
        train:
          lr: 2.0e-4

    Unknown sections or field names raise an error so typos fail loudly rather
    than being silently ignored.
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _flatten_sections(raw, path)


def _flatten_sections(raw: Any, source: str, strict: bool = True) -> dict[str, Any]:
    """Validate a ``{section: {field: value}}`` mapping against the dataclass
    fields and flatten it to ``{field: value}``. `source` names the origin for
    error messages.

    With ``strict`` (the default, for hand-written YAML) unknown sections or
    fields raise so typos fail loudly. With ``strict=False`` they are dropped
    with a note instead — used for configs embedded in checkpoints, which may
    come from a different code version and legitimately carry fields the current
    schema no longer defines (e.g. a removed ``amp_init_scale``)."""
    if not isinstance(raw, dict):
        raise ValueError(f"Config {source} must be a mapping of sections, got {type(raw).__name__}.")

    if unknown_sections := set(raw) - _SECTION_FIELDS.keys():
        if strict:
            raise ValueError(
                f"Unknown config section(s) {sorted(unknown_sections)} in {source}. "
                f"Expected: {sorted(_SECTION_FIELDS)}."
            )
        print(f"note: ignoring unknown section(s) {sorted(unknown_sections)} in {source}")

    flat: dict[str, Any] = {}
    for section, values in raw.items():
        if section not in _SECTION_FIELDS:
            continue
        values = values or {}
        if not isinstance(values, dict):
            raise ValueError(f"Section '{section}' in {source} must be a mapping of field: value.")
        if unknown_fields := set(values) - _SECTION_FIELDS[section]:
            if strict:
                raise ValueError(
                    f"Unknown field(s) {sorted(unknown_fields)} in section '{section}' of {source}. "
                    f"Valid fields: {sorted(_SECTION_FIELDS[section])}."
                )
            print(f"note: ignoring unknown field(s) {sorted(unknown_fields)} in section '{section}' of {source}")
            values = {k: v for k, v in values.items() if k in _SECTION_FIELDS[section]}
        flat.update(values)
    return flat


def load_checkpoint_config(
    path: str, sections: set[str] | None = None
) -> dict[str, Any]:
    """Read the config snapshot embedded in a training checkpoint (saved under
    the ``"config"`` key by `_save_checkpoint`) into the same flat
    ``{field: value}`` dict that `load_yaml_config` returns. With `sections`,
    only those top-level sections (e.g. ``{"model"}``) are kept — used by
    inference to rebuild just the architecture. Returns ``{}`` for older
    checkpoints that predate embedded configs."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt.get("config")
    if not raw:
        return {}
    if sections is not None:
        raw = {k: v for k, v in raw.items() if k in sections}
    # Lenient: a checkpoint may come from a different code version, so tolerate
    # (and drop) fields the current schema no longer defines.
    return _flatten_sections(raw, f"checkpoint {path}", strict=False)
