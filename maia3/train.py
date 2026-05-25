"""Basic training loop for Maia3.

Reads game shards from disk, replays moves on a `chess.Board()` to tokenize on
the fly, and trains `MAIA3Model`. The on-the-fly tokenization is the design
choice that lets you add derived features (e.g. signals computed from the
board ± surrounding moves) without regenerating shards — those changes happen
in `Maia3Dataset.__getitem__` or `collate_fn`.

Shard format
------------
Each shard is a pickle file containing `list[dict]`, where each dict is one
game with at least:

    {
        "white_elo":  int,
        "black_elo":  int,
        "result":     int,           # 0=black wins, 1=draw, 2=white wins
        "moves":      list[str],     # UCI move strings, in play order
        "time_base":  float,         # seconds (0 if unknown)
        "time_inc":   float,         # seconds (0 if unknown)
        "clocks":     list[float] | None,   # per-ply clock remaining, optional
    }

Extra keys are ignored, so a preprocessor can stash anything else for future
use. Shards live in `data_cfg.shard_dir/*.pkl`.

Loss
----
`compute_loss` is a thin stub — cross-entropy on the policy head, masked to
legal moves. Replace it with the real multi-head loss once the model output
definitions are settled.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from .config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    build_arg_parser,
    cfg_dict,
    cfgs_from_args,
)
from .dataset import get_legal_moves_mask, tokenize_board
from .models import MAIA3Model
from .utils import get_all_possible_moves, mirror_move, seed_everything


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_shard(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def _list_shards(shard_dir: str | Path) -> list[Path]:
    shards = sorted(Path(shard_dir).glob("*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No *.pkl shards under {shard_dir}")
    return shards


def _game_passes_filters(game: dict, data_cfg: DataConfig) -> bool:
    if data_cfg.min_elo is not None and min(game["white_elo"], game["black_elo"]) < data_cfg.min_elo:
        return False
    if data_cfg.max_elo is not None and max(game["white_elo"], game["black_elo"]) > data_cfg.max_elo:
        return False
    return True


class Maia3Dataset(Dataset):
    """Map-style dataset over (game, ply) pairs.

    On `__getitem__`, replays moves up to the requested ply on a `chess.Board()`,
    tokenizes the last `model_cfg.history` positions, encodes the played move
    in the model's 4352-entry move vocabulary, and returns a flat dict of
    tensors.
    """

    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, split: str = "train"):
        shard_dir = data_cfg.shard_dir if split == "train" else (data_cfg.val_shard_dir or data_cfg.shard_dir)
        self.shard_paths = _list_shards(shard_dir)
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(all_moves)}
        self.vocab_size = len(all_moves)

        # Eagerly load all shards; for very large corpora swap this for lazy
        # per-worker loading with an offset table.
        self.games: list[dict] = []
        for p in self.shard_paths:
            self.games.extend(g for g in _load_shard(p) if _game_passes_filters(g, data_cfg))

        # Build a flat index of (game_idx, ply_idx). One sample per legal target ply.
        self.index: list[tuple[int, int]] = []
        for gi, g in enumerate(self.games):
            n = len(g["moves"])
            lo = data_cfg.min_ply
            hi = n if data_cfg.max_ply is None else min(n, data_cfg.max_ply)
            for pi in range(lo, hi):
                self.index.append((gi, pi))

    def __len__(self) -> int:
        return len(self.index)

    def _encode_move(self, board: chess.Board, move: chess.Move) -> int:
        # Mirror the move into side-to-move-perspective coords, matching the
        # mirrored board tokenization in `tokenize_board`.
        uci = move.uci() if board.turn == chess.WHITE else mirror_move(move.uci())
        return self.all_moves_dict[uci]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        game_idx, ply_idx = self.index[idx]
        game = self.games[game_idx]

        board = chess.Board()
        history_tokens: list[torch.Tensor] = []  # newest last
        for i, move_uci in enumerate(game["moves"][:ply_idx]):
            if i >= ply_idx - self.model_cfg.history + 1 and i >= 0:
                history_tokens.append(tokenize_board(board))
            board.push_uci(move_uci)
        history_tokens.append(tokenize_board(board))  # the position being predicted from
        history_tokens = history_tokens[-self.model_cfg.history:]

        # Left-pad with the earliest seen position when game is too short.
        if len(history_tokens) < self.model_cfg.history:
            pad = [history_tokens[0]] * (self.model_cfg.history - len(history_tokens))
            history_tokens = pad + history_tokens

        # (64, 12 * history)
        tokens = torch.cat(history_tokens, dim=1)

        # Trailing time-info channels — kept zero by default. The model slices
        # these off based on `include_time_info`; emit them so the shape is
        # always (64, 12*history + extras) and downstream losses can read them.
        if self.model_cfg.include_time_info:
            tokens = torch.cat([
                tokens,
                torch.zeros((64, 4), dtype=torch.float),  # base/inc/clk_before/clk_ponder
            ], dim=1)
        else:
            tokens = torch.cat([tokens, torch.zeros((64, 1), dtype=torch.float)], dim=1)

        played_move = chess.Move.from_uci(game["moves"][ply_idx])
        legal_mask = get_legal_moves_mask(board, self.all_moves_dict)
        move_idx = self._encode_move(board, played_move)

        # Side-to-move-perspective targets.
        if board.turn == chess.WHITE:
            self_elo, oppo_elo = game["white_elo"], game["black_elo"]
            value_target = game["result"]  # 0=loss, 1=draw, 2=win for white = STM
        else:
            self_elo, oppo_elo = game["black_elo"], game["white_elo"]
            value_target = 2 - game["result"]  # flip W/D/L

        clocks = game.get("clocks") or []
        ponder_target = float(clocks[ply_idx]) if ply_idx < len(clocks) else 0.0

        return {
            "tokens": tokens,
            "self_elo": torch.tensor(self_elo, dtype=torch.long),
            "oppo_elo": torch.tensor(oppo_elo, dtype=torch.long),
            "move_idx": torch.tensor(move_idx, dtype=torch.long),
            "legal_mask": legal_mask,
            "value_target": torch.tensor(value_target, dtype=torch.long),
            "ponder_target": torch.tensor(ponder_target, dtype=torch.float),
        }


def collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0]}


# ---------------------------------------------------------------------------
# Loss — STUB. Replace once model output definitions are settled.
# ---------------------------------------------------------------------------


def compute_loss(outputs: tuple, batch: dict, model_cfg: ModelConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (total_loss, scalar_metrics_for_logging)."""
    logits_move, logits_value, logits_ponder = outputs
    masked_logits = logits_move.masked_fill(~batch["legal_mask"], float("-inf"))
    policy_loss = F.cross_entropy(masked_logits, batch["move_idx"])
    metrics = {"loss/policy": policy_loss.item()}
    return policy_loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _lr_at(step: int, train_cfg: TrainConfig) -> float:
    if step < train_cfg.warmup_steps:
        return train_cfg.lr * (step + 1) / max(1, train_cfg.warmup_steps)
    progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.num_steps - train_cfg.warmup_steps)
    return train_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _make_loader(data_cfg: DataConfig, model_cfg: ModelConfig, split: str) -> DataLoader:
    ds = Maia3Dataset(data_cfg, model_cfg, split=split)
    return DataLoader(
        ds,
        batch_size=data_cfg.batch_size,
        shuffle=(split == "train") and data_cfg.shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
        persistent_workers=data_cfg.num_workers > 0,
    )


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _move_to(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    seed_everything(train_cfg.seed)
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = train_cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("info: CUDA unavailable, falling back to CPU")
        device = "cpu"

    model = MAIA3Model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )

    start_step = 0
    if train_cfg.resume_from:
        ckpt = torch.load(train_cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {train_cfg.resume_from} @ step {start_step}")

    train_loader = _make_loader(data_cfg, model_cfg, split="train")
    val_loader = _make_loader(data_cfg, model_cfg, split="val") if data_cfg.val_shard_dir else None
    batches = _infinite(train_loader)

    amp_enabled = train_cfg.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model.train()
    log_t0 = time.time()
    log_loss = 0.0
    log_count = 0

    for step in range(start_step, train_cfg.num_steps):
        for g in optimizer.param_groups:
            g["lr"] = _lr_at(step, train_cfg)

        batch = _move_to(next(batches), device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp_enabled):
            outputs = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"])
            loss, metrics = compute_loss(outputs, batch, model_cfg)

        scaler.scale(loss).backward()
        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        log_loss += loss.item()
        log_count += 1

        if (step + 1) % train_cfg.log_interval == 0:
            dt = time.time() - log_t0
            sps = log_count * data_cfg.batch_size / dt
            print(
                f"step {step+1:>7d} | loss {log_loss/log_count:.4f} "
                f"| lr {optimizer.param_groups[0]['lr']:.2e} | {sps:.0f} samples/s"
                + " | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
            log_t0, log_loss, log_count = time.time(), 0.0, 0

        if val_loader is not None and train_cfg.val_interval > 0 and (step + 1) % train_cfg.val_interval == 0:
            _validate(model, val_loader, device, amp_enabled, model_cfg)

        if (step + 1) % train_cfg.ckpt_interval == 0 or (step + 1) == train_cfg.num_steps:
            _save_checkpoint(model, optimizer, step + 1, output_dir, model_cfg, data_cfg, train_cfg)


@torch.no_grad()
def _validate(model, val_loader, device, amp_enabled, model_cfg):
    model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        batch = _move_to(batch, device)
        with autocast("cuda", enabled=amp_enabled):
            outputs = model(batch["tokens"], batch["self_elo"], batch["oppo_elo"])
            loss, _ = compute_loss(outputs, batch, model_cfg)
        total += loss.item()
        count += 1
    print(f"  val | loss {total/max(1, count):.4f} | {count} batches")
    model.train()


def _save_checkpoint(model, optimizer, step, output_dir, model_cfg, data_cfg, train_cfg):
    path = output_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": cfg_dict(model_cfg, data_cfg, train_cfg),
        },
        path,
    )
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    model_cfg, data_cfg, train_cfg = cfgs_from_args(args)
    if not data_cfg.shard_dir:
        parser.error("--shard-dir is required")
    train(model_cfg, data_cfg, train_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
