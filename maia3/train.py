from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader


from .config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    apply_layered_defaults,
    build_arg_parser,
    cfg_dict,
    cfgs_from_args,
    save_yaml_config,
)
from .dataset import ParquetPositionDataset, collate_fn
from .models import MAIA3Model
from .utils import seed_everything


def compute_loss(
    outputs: tuple, batch: dict, train_cfg: TrainConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    logits_move, logits_value = outputs

    masked_logits = logits_move.masked_fill(~batch["legal_mask"], float("-inf"))
    policy_loss = F.cross_entropy(masked_logits, batch["move_idx"])
    value_loss = F.cross_entropy(logits_value, batch["value_target"])

    total = (
        train_cfg.policy_loss_weight * policy_loss
        + train_cfg.value_loss_weight * value_loss
    )
    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/total": total.item(),
        "acc/policy_top1": _topk_accuracy(masked_logits, batch["move_idx"], 1),
        "acc/policy_top3": _topk_accuracy(masked_logits, batch["move_idx"], 3),
        "acc/value": _topk_accuracy(logits_value, batch["value_target"], 1),
    }
    return total, metrics


@torch.no_grad()
def _topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=-1).indices  # (B, k)
    correct = (topk == target.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    if train_cfg.cosine_cycle_steps > 0:
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=train_cfg.cosine_cycle_steps, eta_min=train_cfg.min_lr
        )
    else:
        cosine_steps = max(1, train_cfg.num_steps - train_cfg.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=train_cfg.min_lr
        )
    if train_cfg.warmup_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / train_cfg.warmup_steps,
        total_iters=train_cfg.warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[train_cfg.warmup_steps]
    )


def _build_loader(
    path: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    *,
    train: bool,
    persistent: bool = True,
) -> DataLoader:
    batch_size = data_cfg.batch_size if train else data_cfg.val_batch_size
    ds = ParquetPositionDataset(path, model_cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and data_cfg.shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=train,
        persistent_workers=persistent and data_cfg.num_workers > 0,
        # prefetch_factor is only valid with worker processes.
        prefetch_factor=4 if data_cfg.num_workers > 0 else None,  # TODO(marco): add this to the config
    )


def _total_rows(paths: list[str]) -> int:
    return sum(pq.read_metadata(p).num_rows for p in paths)


def _infinite(loader: DataLoader, counters: dict[str, int]):
    """Iterate over a single DataLoader infinitely. The training config determines the stopping point via cfg.num_steps."""
    while True:
        for batch in loader:
            counters["samples"] += batch["tokens"].shape[0]
            yield batch
        counters["epochs"] += 1


def _sharded_batches(
    paths: list[str],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    counters: dict[str, int],
    *,
    seed: int = 0,
):
    """
    Iterate over sharded data inifinitely (the config determines the stopping point during training).
    One epoch is a full pass over the entire shard, and the order we process shards is randomized each
    epoch.
    """


    rng = random.Random(seed)
    while True:
        order = list(paths)
        rng.shuffle(order)
        for path in order:
            # Non-persistent workers so each shard's loader tears down its
            # workers (and releases its in-memory table) before the next shard.
            loader = _build_loader(
                path, data_cfg, model_cfg, train=True, persistent=False
            )
            for batch in loader:
                counters["samples"] += batch["tokens"].shape[0]
                yield batch
            del loader
        counters["epochs"] += 1


def _move_to(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    seed_everything(train_cfg.seed)
    output_dir = Path(train_cfg.output_dir)
    if train_cfg.timestamp_runs:
        output_dir = output_dir / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {output_dir}")

    save_yaml_config(str(output_dir / "config.yaml"), model_cfg, data_cfg, train_cfg)

    device = train_cfg.device

    model = MAIA3Model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    scheduler = _build_scheduler(optimizer, train_cfg)

    amp_enabled = train_cfg.use_amp and device.startswith("cuda")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        growth_factor=train_cfg.amp_growth_factor,
        backoff_factor=train_cfg.amp_backoff_factor,
        growth_interval=train_cfg.amp_growth_interval,
    )

    start_step = 0
    if train_cfg.resume_from:
        # Full resume: continue optimizer, LR schedule, AMP scale, and step.
        ckpt = torch.load(
            train_cfg.resume_from, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {train_cfg.resume_from} @ step {start_step}")

    train_paths = data_cfg.train_paths()

    seen_samples = start_step * max(1, train_cfg.grad_accum_steps) * data_cfg.batch_size
    data_counters = {"samples": seen_samples, "epochs": 0}
    if len(train_paths) > 1:

        # The data can be sharded over many parquet files, so one epoch corresponds to a 
        # full pass over all shards. The shard we are processing is not preserved when
        # resuming model training.
        corpus_size = _total_rows(train_paths)
        batches = _sharded_batches(
            train_paths,
            data_cfg,
            model_cfg,
            data_counters,
            seed=train_cfg.seed,
        )
    else:
        # Single corpus: keep one persistent loader and cycle it.
        train_loader = _build_loader(train_paths[0], data_cfg, model_cfg, train=True)
        corpus_size = len(train_loader.dataset)
        batches = _infinite(train_loader, data_counters)

    val_loader = (
        _build_loader(data_cfg.val_parquet_path, data_cfg, model_cfg, train=False)
        if data_cfg.val_parquet_path
        else None
    )

    run = None
    if train_cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            name=train_cfg.wandb_run_name,
            mode=train_cfg.wandb_mode,
            config=cfg_dict(model_cfg, data_cfg, train_cfg),
        )

    model.train()
    log_t0 = time.time()
    log_accum = defaultdict(float)
    log_count = 0
    best_value: float | None = None

    accum = max(1, train_cfg.grad_accum_steps)
    for step in range(start_step + 1, train_cfg.num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        metrics: dict[str, float] = defaultdict(float)
        for _ in range(accum):
            batch = _move_to(next(batches), device)
            with autocast("cuda", enabled=amp_enabled):
                outputs = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"])
                loss, micro_metrics = compute_loss(outputs, batch, train_cfg)
            scaler.scale(loss / accum).backward()
            for k, v in micro_metrics.items():
                metrics[k] += v / accum

        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in metrics.items():
            log_accum[k] += v
        log_count += 1

        to_log: dict[str, float] = {}

        if step % train_cfg.log_interval == 0:
            dt = time.time() - log_t0
            sps = log_count * accum * data_cfg.batch_size / dt
            avg = {k: v / log_count for k, v in log_accum.items()}
            lr = optimizer.param_groups[0]["lr"]

            samples_seen = data_counters["samples"]
            epochs_seen = samples_seen / corpus_size if corpus_size else 0.0

            print(
                f"step {step:>7d} | loss {avg['loss/total']:.4f} "
                f"| top1 {avg['acc/policy_top1']:.3f} "
                f"| top3 {avg['acc/policy_top3']:.3f} "
                f"| epoch {epochs_seen:.2f} "
                f"| lr {lr:.2e} | {sps:.0f} samples/s"
            )

            to_log.update({f"train/{k}": v for k, v in avg.items()})
            to_log["train/lr"] = lr
            to_log["train/samples_per_s"] = sps
            to_log["data/samples_seen"] = samples_seen
            to_log["data/epochs"] = epochs_seen
            log_t0, log_accum, log_count = time.time(), defaultdict(float), 0

        if (
            val_loader is not None
            and train_cfg.val_interval > 0
            and step % train_cfg.val_interval == 0
        ):
            val_metrics = _validate(model, val_loader, device, amp_enabled, train_cfg)
            to_log.update({f"val/{k}": v for k, v in val_metrics.items()})

            # Track the best checkpoint according to the best metric config.
            monitored = val_metrics.get(train_cfg.best_metric)
            if monitored is not None and _is_better(
                monitored, best_value, train_cfg.best_mode
            ):
                best_value = monitored
                to_log["val/best"] = best_value
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    output_dir,
                    model_cfg,
                    data_cfg,
                    train_cfg,
                    name="best.pt",
                    extra={"best_metric": train_cfg.best_metric, "best_value": best_value},
                )

        if run is not None and to_log:
            run.log(to_log, step=step, commit=True)

        if step % train_cfg.ckpt_interval == 0 or step == train_cfg.num_steps:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                output_dir,
                model_cfg,
                data_cfg,
                train_cfg,
                prune=True,
            )

    if run is not None:
        run.finish()


@torch.no_grad()
def _validate(model, val_loader, device, amp_enabled, train_cfg) -> dict[str, float]:
    model.eval()
    accum = defaultdict(float)
    count = 0
    for batch in val_loader:
        if train_cfg.val_steps > 0 and count >= train_cfg.val_steps:
            break
        batch = _move_to(batch, device)
        with autocast("cuda", enabled=amp_enabled):
            outputs = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"])
            _, metrics = compute_loss(outputs, batch, train_cfg)
        for k, v in metrics.items():
            accum[k] += v
        count += 1
    avg = {k: v / max(1, count) for k, v in accum.items()}
    print(
        f"  val | loss {avg.get('loss/total', 0.0):.4f} "
        f"| top1 {avg.get('acc/policy_top1', 0.0):.3f} "
        f"| top3 {avg.get('acc/policy_top3', 0.0):.3f} "
        f"| {count} batches"
    )
    model.train()
    return avg


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return value > best if mode == "max" else value < best


def _prune_checkpoints(output_dir: Path, keep_last_k: int) -> None:
    """Delete all but the `keep_last_k` most-recent periodic checkpoints. Only
    touches the zero-padded `step_*.pt` files (so best.pt and config.yaml are
    safe); the zero-padding makes lexical sort match step order."""
    ckpts = sorted(output_dir.glob("step_*.pt"))
    for old in ckpts[:-keep_last_k]:
        old.unlink()
        print(f"  pruned {old}")


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    output_dir,
    model_cfg,
    data_cfg,
    train_cfg,
    *,
    name: str | None = None,
    extra: dict | None = None,
    prune: bool = False,
):
    """Write a checkpoint to `output_dir`. By default the file is the periodic
    `step_{step}.pt`; pass `name` to override (e.g. "best.pt"). `extra` adds keys
    to the saved payload. With `prune`, old periodic checkpoints beyond
    `train_cfg.keep_last_k` are deleted afterward (best.pt is never pruned)."""
    path = output_dir / (name or f"step_{step:07d}.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "config": cfg_dict(model_cfg, data_cfg, train_cfg),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  saved {path}")
    if prune and train_cfg.keep_last_k > 0:
        _prune_checkpoints(output_dir, train_cfg.keep_last_k)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = apply_layered_defaults(parser, argv)
    model_cfg, data_cfg, train_cfg = cfgs_from_args(args)
    if train_cfg.best_mode not in ("min", "max"):
        parser.error(f"--best-mode must be 'min' or 'max', got {train_cfg.best_mode!r}")
    if not data_cfg.train_paths():
        parser.error("set --parquet-path or --parquet-paths (data.parquet_path / data.parquet_paths)")
    train(model_cfg, data_cfg, train_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
