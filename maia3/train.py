from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

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
    """Weighted sum of the policy and value heads. Returns
    (total_loss, scalar_metrics_for_logging) — losses plus policy/value
    accuracy and policy top-3 accuracy, all over legal moves only."""
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
    """Fraction of rows whose target index is among the top-`k` logits."""
    topk = logits.topk(k, dim=-1).indices  # (B, k)
    correct = (topk == target.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup from 0 → `lr` over `warmup_steps`, then cosine anneal toward
    `min_lr`. Stepped once per optimizer step.

    `cosine_cycle_steps > 0` gives warm restarts: the cosine refreshes back to
    `lr` every `cosine_cycle_steps` steps (CosineAnnealingWarmRestarts, T_0=that),
    decoupled from `num_steps`. `0` gives a single cosine spanning the run."""
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
        prefetch_factor=4,  # TODO(marco): add this to the config
    )


def _total_rows(paths: list[str]) -> int:
    """Combined row count across parquet shards, read cheaply from the footers
    without loading any row data."""
    import pyarrow.parquet as pq

    return sum(pq.read_metadata(p).num_rows for p in paths)


def _infinite(loader: DataLoader, counters: dict[str, int]):
    """Yield batches forever, restarting at the end of the corpus. Records the
    cumulative sample count and the number of completed passes over the corpus
    in `counters` so the training loop can log dataset progress."""
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
):
    """Yield batches forever, cycling through `paths` (parquet shards) in order:
    one shard is loaded, iterated to exhaustion, then freed before the next is
    loaded, so only a single shard is resident in memory at a time. One full
    pass over all shards counts as one epoch. Records cumulative samples and
    completed epochs in `counters`, mirroring `_infinite`."""
    while True:
        for path in paths:
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

    # Drop a fully-resolved, re-runnable config next to the checkpoints so the
    # run is self-documenting (reproduce with `--config <run dir>/config.yaml`).
    # train_cfg keeps the base output_dir, so re-running from this file makes a
    # fresh timestamped run rather than reusing this one.
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

    start_step = 0
    if train_cfg.resume_from:
        ckpt = torch.load(
            train_cfg.resume_from, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {train_cfg.resume_from} @ step {start_step}")

    train_paths = data_cfg.train_paths()
    data_counters = {"samples": 0, "epochs": 0}
    if len(train_paths) > 1:
        # Multiple shards: stream through them one at a time (one full pass =
        # one epoch), keeping only one shard resident in memory.
        corpus_size = _total_rows(train_paths)
        batches = _sharded_batches(train_paths, data_cfg, model_cfg, data_counters)
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

    amp_enabled = train_cfg.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=train_cfg.amp_init_scale,
        growth_factor=train_cfg.amp_growth_factor,
        backoff_factor=train_cfg.amp_backoff_factor,
        growth_interval=train_cfg.amp_growth_interval,
    )

    model.train()
    log_t0 = time.time()
    log_accum = defaultdict(float)
    log_count = 0

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

        # Metrics logged on this step are gathered into one payload and
        # committed together. wandb defaults commit=False whenever an explicit
        # step is passed, and rejects a second commit at an already-committed
        # step, so train + val (which can land on the same step) must share a
        # single committed run.log call.
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

        if run is not None and to_log:
            run.log(to_log, step=step, commit=True)

        if step % train_cfg.ckpt_interval == 0 or step == train_cfg.num_steps:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                output_dir,
                model_cfg,
                data_cfg,
                train_cfg,
            )

    if run is not None:
        run.finish()


@torch.no_grad()
def _validate(model, val_loader, device, amp_enabled, train_cfg) -> dict[str, float]:
    """Run a validation pass and return loss + accuracy metrics averaged over the
    batches seen (capped at `val_steps` if set)."""
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


def _save_checkpoint(
    model, optimizer, scheduler, step, output_dir, model_cfg, data_cfg, train_cfg
):
    path = output_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": cfg_dict(model_cfg, data_cfg, train_cfg),
        },
        path,
    )
    print(f"  saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = apply_layered_defaults(parser, argv)
    model_cfg, data_cfg, train_cfg = cfgs_from_args(args)
    if not data_cfg.train_paths():
        parser.error("set --parquet-path or --parquet-paths (data.parquet_path / data.parquet_paths)")
    train(model_cfg, data_cfg, train_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
