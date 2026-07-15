from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import torch
from torch.utils.data import Dataset

from .config import ModelConfig
from .utils import get_all_possible_moves, mirror_move

PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


def tokenize_board(board):

    tokens = torch.zeros((64, 12), dtype=torch.float)

    if board.turn == chess.BLACK:
        board = board.mirror()

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            mapped = PIECE_MAP[piece.piece_type]
            token = mapped + (6 if piece.color == chess.BLACK else 0)
            tokens[square][token - 1] = 1

    return tokens


def get_legal_moves_mask(board, all_moves_dict):

    legal_moves_mask = torch.zeros((len(all_moves_dict),), dtype=torch.bool)
    for legal_move in board.legal_moves:
        move_uci = (
            legal_move.uci()
            if board.turn == chess.WHITE
            else mirror_move(legal_move.uci())
        )
        if move_uci in all_moves_dict:
            legal_moves_mask[all_moves_dict[move_uci]] = 1

    return legal_moves_mask


def build_move_vocab() -> dict[str, int]:
    """The model's fixed move vocabulary as ``{uci: index}``."""
    return {m: i for i, m in enumerate(get_all_possible_moves())}


def assemble_input_tokens(
    history_tokens: list[torch.Tensor], model_cfg: ModelConfig
) -> torch.Tensor:
    """Concatenate a list of per-board (64, 12) tokenizations (newest last) into
    the model input. Keeps the most recent `history` boards, left-padding with the
    earliest available board when the game is too short."""
    window = list(history_tokens)[-model_cfg.history :]
    if len(window) < model_cfg.history:
        window = [window[0]] * (model_cfg.history - len(window)) + window

    return torch.cat(window, dim=1)  # (64, 12 * history)


def make_sample(
    history_boards: list[chess.Board],
    played_move: chess.Move,
    white_elo: int,
    black_elo: int,
    result: int,
    model_cfg: ModelConfig,
    all_moves_dict: dict[str, int],
) -> dict[str, Any]:
    """Build one training sample from `history_boards` (the last `history` board
    positions, newest last; the newest is the position the move is played from),
    the played move, and the game's scalar metadata. `result` is 0=black win,
    1=draw, 2=white win."""
    board = history_boards[-1]
    tokens = assemble_input_tokens(
        [tokenize_board(b) for b in history_boards], model_cfg
    )
    legal_mask = get_legal_moves_mask(board, all_moves_dict)

    # Mirror the move into side-to-move-perspective coords, matching the mirrored
    # board tokenization in `tokenize_board`.
    uci = (
        played_move.uci()
        if board.turn == chess.WHITE
        else mirror_move(played_move.uci())
    )
    move_idx = all_moves_dict[uci]

    if board.turn == chess.WHITE:
        self_elo, oppo_elo = white_elo, black_elo
        value_target = result  # 0=loss, 1=draw, 2=win for STM
    else:
        self_elo, oppo_elo = black_elo, white_elo
        value_target = 2 - result  # flip W/D/L

    return {
        "tokens": tokens,
        "self_elo": torch.tensor(self_elo, dtype=torch.long),
        "oppo_elo": torch.tensor(oppo_elo, dtype=torch.long),
        "move_idx": torch.tensor(move_idx, dtype=torch.long),
        "legal_mask": legal_mask,
        "value_target": torch.tensor(value_target, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------


#: Parquet file-level metadata key recording the history window used at
#: preprocess time (the number of board FENs stored per row).
HISTORY_META_KEY = b"maia3_history"


class ParquetPositionDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        model_cfg: ModelConfig,
        all_moves_dict: dict[str, int] | None = None,
    ):
        import pyarrow.parquet as pq

        self.path = str(path)
        self.model_cfg = model_cfg
        self.all_moves_dict = (
            all_moves_dict if all_moves_dict is not None else build_move_vocab()
        )
        # Cheap: read row count + metadata from the footer without loading data.
        meta = pq.read_metadata(self.path)
        self._num_rows = meta.num_rows
        self.prep_history = int(
            (meta.metadata or {}).get(HISTORY_META_KEY, model_cfg.history)
        )
        if model_cfg.history > self.prep_history:
            raise ValueError(
                f"{path} was preprocessed with history={self.prep_history}, but the model "
                f"config requests history={model_cfg.history}. Re-run maia3-preprocess with "
                f"a history >= {model_cfg.history}."
            )
        # Loaded once in the parent process so DataLoader workers share a single
        # decompressed copy via fork copy-on-write, rather than each worker
        # materializing its own (which OOMs with many workers / large files).
        # Arrow's buffers aren't mutated after load, so the COW pages stay shared.
        self._table = pq.read_table(self.path, memory_map=True)

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._table.slice(idx, 1).to_pylist()[0]

        history_boards = [chess.Board(fen) for fen in row["boards"]]
        played = chess.Move.from_uci(row["target_move"])
        return make_sample(
            history_boards,
            played,
            row["white_elo"],
            row["black_elo"],
            row["result"],
            self.model_cfg,
            self.all_moves_dict,
        )


def collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in samples[0]}
