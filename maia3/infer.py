"""One-off inference for a trained Maia3 checkpoint.

Exposes a small programmatic API (`Maia3`) and a CLI (`maia3-infer`) for scoring
a single position — or a short history of positions — without a parquet dataset.

These models are history-aware: each example is the last `history` board
positions (newest last; see `maia3.preprocess`). Inference accepts the same
shape — pass either one FEN (the position to move from) or an ordered list of
FENs (oldest first, newest last). One FEN is the common case and is treated as a
length-1 history; the trainer's left-pad-by-repetition rule (see
`assemble_input_tokens`) then fills the window, so it is well-defined but mildly
out-of-distribution versus the real game history seen in training.

The API mirrors `simple-maia3-inference` (probs / logits / batch_*), adapted for
history input and this repo's PyTorch checkpoints. Inputs are tokenized with the
same code path as training (`tokenize_board` / `assemble_input_tokens` /
`get_legal_moves_mask`), so model inputs always match the trained model.

Usage:
    maia3-infer --checkpoint runs/maia3/best.pt \\
        "8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47" --elo-self 1569 --elo-oppo 1579

    # With history (oldest -> newest; the last is the position to move from):
    maia3-infer --checkpoint CKPT "<fen_t-2>" "<fen_t-1>" "<fen_now>" \\
        --elo-self 1500 --elo-oppo 1500
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import chess
import numpy as np
import torch

from .config import ModelConfig, load_checkpoint_config
from .dataset import (
    assemble_input_tokens,
    build_move_vocab,
    get_legal_moves_mask,
    tokenize_board,
)
from .models import MAIA3Model
from .utils import get_all_possible_moves, mirror_move

#: A position to evaluate: a single FEN, or an ordered history of FENs (oldest
#: first, newest last; the newest is the position the move is played from).
Position = str | list[str]


def _as_history(position: Position) -> list[str]:
    """Normalize a single FEN or a list of FENs into a history list (newest last)."""
    boards = [position] if isinstance(position, str) else list(position)
    if not boards:
        raise ValueError("position must be a FEN string or a non-empty list of FENs")
    return boards


def _resolve_device(device: str | None) -> str:
    """Pick a device, falling back to CPU when CUDA is requested but unavailable."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("note: CUDA unavailable, falling back to CPU")
        return "cpu"
    return device


def _model_cfg_from_checkpoint(path: str) -> ModelConfig:
    """Build the ModelConfig embedded in a checkpoint so the architecture matches
    the saved weights. Falls back to ModelConfig defaults for older checkpoints
    that predate embedded configs."""
    flat = load_checkpoint_config(path, sections={"model"})
    names = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in flat.items() if k in names})


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Maia3:
    """A loaded Maia3 checkpoint ready for one-off inference.

    Each public method takes a `position` (a FEN, or a list of FENs as history)
    and the two Elos. Batch variants take parallel lists.
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        """
        Args:
            checkpoint: Path to a `.pt` checkpoint (supplies weights + architecture).
            device:     Torch device (e.g. "cuda", "cpu"). Default: CUDA if available.
        """
        self.device = _resolve_device(device)
        self.model_cfg = _model_cfg_from_checkpoint(str(checkpoint))
        self.all_moves = get_all_possible_moves()  # index -> uci (STM coords)
        self.all_moves_dict = build_move_vocab()  # uci -> index

        model = MAIA3Model(self.model_cfg).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        self.step = ckpt.get("step")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _prep(self, position: Position) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Tokenize one position's history into model input.

        Returns (tokens (64, 12*history), legal_mask (n_moves,), black_to_move),
        all in the side-to-move perspective the model was trained on."""
        boards = [chess.Board(fen) for fen in _as_history(position)]
        tokens = assemble_input_tokens(
            [tokenize_board(b) for b in boards], self.model_cfg
        )
        legal_mask = get_legal_moves_mask(boards[-1], self.all_moves_dict)
        return tokens, legal_mask, boards[-1].turn == chess.BLACK

    @torch.no_grad()
    def _run(
        self,
        positions: list[Position],
        elo_selfs: list[float],
        elo_oppos: list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        """Shared preprocessing + forward pass for all public methods.

        Returns (logits_move (N, 4352), logits_value (N, 3), legal_masks (N, n_moves),
        black_flags). All move quantities are in side-to-move coordinates."""
        if not (len(positions) == len(elo_selfs) == len(elo_oppos)):
            raise ValueError("positions, elo_selfs and elo_oppos must be the same length")

        tokens_list, masks, black_flags = [], [], []
        for pos in positions:
            tokens, legal_mask, black = self._prep(pos)
            tokens_list.append(tokens)
            masks.append(legal_mask)
            black_flags.append(black)

        tokens = torch.stack(tokens_list).to(self.device)
        self_t = torch.tensor(elo_selfs, dtype=torch.float32, device=self.device)
        oppo_t = torch.tensor(elo_oppos, dtype=torch.float32, device=self.device)

        logits_move, logits_value = self.model(tokens, self_t, oppo_t)
        return (
            logits_move.float().cpu().numpy(),
            logits_value.float().cpu().numpy(),
            torch.stack(masks).cpu().numpy(),
            black_flags,
        )

    def _decode(
        self,
        logits_move: np.ndarray,
        logits_value: np.ndarray,
        legal_mask: np.ndarray,
        black_flag: bool,
    ) -> tuple[dict[str, float], tuple[float, float, float]]:
        """Turn one position's raw logits into (move_probs, ldw).

        Move logits are masked to legal moves, softmaxed, and decoded to UCI (with
        Black moves mirrored back to real-board coordinates). The value head is
        already in the side-to-move's perspective, so LDW = softmax(logits_value)
        directly — no Black flip (unlike simple-maia3-inference's ONNX export)."""
        ldw = _softmax(logits_value)
        ldw_probs = tuple(round(float(p), 4) for p in ldw)

        legal_idx = np.where(legal_mask > 0)[0]
        probs = _softmax(logits_move[legal_idx])
        move_probs = {}
        for i, idx in enumerate(legal_idx):
            uci = self.all_moves[int(idx)]
            if black_flag:
                uci = mirror_move(uci)
            move_probs[uci] = float(probs[i])
        move_probs = dict(sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True))
        return move_probs, ldw_probs

    # ------------------------------------------------------------------
    # Move probabilities + LDW
    # ------------------------------------------------------------------

    def probs(
        self, position: Position, elo_self: float, elo_oppo: float
    ) -> tuple[dict[str, float], tuple[float, float, float]]:
        """Single-position move probabilities and LDW.

        Returns (move_probs, ldw): a move -> probability dict (legal moves only,
        sorted descending) and the (loss, draw, win) tuple for the side to move."""
        return self.batch_probs([position], [elo_self], [elo_oppo])[0]

    def batch_probs(
        self,
        positions: list[Position],
        elo_selfs: list[float],
        elo_oppos: list[float],
    ) -> list[tuple[dict[str, float], tuple[float, float, float]]]:
        """Batched `probs`. Returns one (move_probs, ldw) tuple per input position."""
        lm, lv, masks, black = self._run(positions, elo_selfs, elo_oppos)
        return [self._decode(lm[i], lv[i], masks[i], black[i]) for i in range(len(positions))]

    # ------------------------------------------------------------------
    # Raw logits
    # ------------------------------------------------------------------

    def logits(
        self,
        position: Position,
        elo_self: float,
        elo_oppo: float,
        mask_move_logits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-position raw logits in side-to-move coordinates.

        Returns (logits_move (4352,), logits_value (3,)). With `mask_move_logits`
        (default), illegal-move logits are set to -inf."""
        lm, lv = self.batch_logits([position], [elo_self], [elo_oppo], mask_move_logits)
        return lm[0], lv[0]

    def batch_logits(
        self,
        positions: list[Position],
        elo_selfs: list[float],
        elo_oppos: list[float],
        mask_move_logits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched `logits`. Returns (logits_move (N, 4352), logits_value (N, 3))."""
        lm, lv, masks, _ = self._run(positions, elo_selfs, elo_oppos)
        if mask_move_logits:
            lm[masks == 0.0] = -np.inf
        return lm, lv


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _print_single(
    boards: list[str],
    elo_self: float,
    elo_oppo: float,
    move_probs: dict[str, float],
    ldw: tuple[float, float, float],
) -> None:
    fen = boards[-1]
    print(f"\nFEN:  {fen}")
    if len(boards) > 1:
        print(f"History: {len(boards)} positions (newest shown above)")
    print(f"Elo:  {elo_self:.0f} (self) vs {elo_oppo:.0f} (opponent)")
    print(f"LDW:  {ldw[0]:.4f} / {ldw[1]:.4f} / {ldw[2]:.4f}")
    print(f"Move probabilities ({len(move_probs)} legal moves):")
    board = chess.Board(fen)
    for move, prob in move_probs.items():
        bar = "#" * int(prob * 40)
        san = board.san(chess.Move.from_uci(move))
        print(f"  {move:6s} {san:6s} {prob:6.2%}  {bar}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="maia3-infer",
        description="Maia3 one-off move-probability inference for a single position "
        "or a short history of positions.",
    )
    parser.add_argument(
        "fens",
        nargs="+",
        metavar="FEN",
        help="One or more FEN strings, oldest first and newest last. The newest is "
        "the position to move from; earlier FENs are history context. A single FEN "
        "is the common case (history is filled by repetition).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint (.pt) to run; supplies the weights and architecture.",
    )
    parser.add_argument(
        "--elo-self", type=float, default=1500.0, metavar="ELO",
        help="Elo rating of the side to move (default: 1500).",
    )
    parser.add_argument(
        "--elo-oppo", type=float, default=1500.0, metavar="ELO",
        help="Elo rating of the opponent (default: 1500).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (e.g. cuda, cpu). Default: CUDA if available, else CPU.",
    )
    args = parser.parse_args(argv)

    maia = Maia3(args.checkpoint, device=args.device)
    move_probs, ldw = maia.probs(args.fens, args.elo_self, args.elo_oppo)
    _print_single(args.fens, args.elo_self, args.elo_oppo, move_probs, ldw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
