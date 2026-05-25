"""Quick end-to-end smoke test for Maia3.

Loads the smallest Maia3 model, asks it to pick a move from the starting
position at a couple of Elo levels, and prints the result. Downloads the
checkpoint on first run via Hugging Face; subsequent runs use the cache.

Usage:
    python scripts/smoke_test.py                 # uses maia3-5m
    python scripts/smoke_test.py --model 3m      # any built-in alias works
    python scripts/smoke_test.py --device cpu    # force CPU
"""

from __future__ import annotations

import argparse
import time

import chess
import torch

from maia3.model_registry import (
    apply_model_config,
    resolve_checkpoint_path,
    resolve_model_spec,
)
from maia3.uci import Maia3UCIEngine, parse_args


def build_cfg(model_alias: str, device: str | None, elo: int) -> argparse.Namespace:
    argv = ["--model", model_alias, "--elo", str(elo), "--temperature", "0"]
    if device is not None:
        argv += ["--device", device]
        if device == "cpu":
            argv += ["--no-use-amp"]
    return parse_args(argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="5m", help="Maia3 model alias (default: 5m)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument(
        "--elos",
        nargs="+",
        type=int,
        default=[1100, 1500, 2000],
        help="Elo levels to query (default: 1100 1500 2000)",
    )
    args = parser.parse_args()

    print(f"[1/4] Torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

    # Resolve and report the model spec before instantiating the engine.
    spec = resolve_model_spec(args.model)
    print(f"[2/4] Resolving {spec.display_name} ({spec.repo_id})")
    t0 = time.time()
    ckpt_path = resolve_checkpoint_path(spec)
    print(f"      checkpoint -> {ckpt_path}  ({time.time() - t0:.1f}s)")

    cfg = build_cfg(args.model, args.device, args.elos[0])
    print(f"[3/4] Loading model onto {cfg.device}")
    t0 = time.time()
    engine = Maia3UCIEngine(cfg)
    print(f"      loaded in {time.time() - t0:.1f}s")

    print(f"[4/4] Querying moves from the starting position at Elos {args.elos}")
    board = chess.Board()
    for elo in args.elos:
        engine.self_elo = elo
        engine.oppo_elo = elo
        engine.board = board.copy()
        engine._reset_history()
        move = engine.pick_move()
        san = board.san(move) if move else "<none>"
        print(f"      Elo {elo:>4}: {move.uci() if move else '----'}  ({san})")

    print("\nOK: Maia3 loaded and produced moves end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
