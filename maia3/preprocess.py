from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import random
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .dataset import HISTORY_META_KEY

DEFAULT_HISTORY = 8

# TODO(marco): Make this part of the config.
# Only pick moves between begin + MIN_PLY and end + MAX_PLY_NEGATIVE_OFFSET
MIN_PLY = 10
MAX_PLY_NEGATIVE_OFFSET = 0

# TODO(marco): Make this part of the config.
MIN_TIME_REMAINING = 30

# TODO(marco): Make this part of the config.
# Cap the number of positions sampled from any single game, drawn (without
# replacement) from the positions that survive the keep_position filter. This
# limits how much any one game can dominate the dataset.
MAX_POSITIONS_PER_GAME = 32

# This comes from the paper (Appendix A)
BALANCE_CHUNK_SIZE = 20_000
BALANCE_PER_BIN_CAP = 10
BALANCE_BIN_LOW = 600
BALANCE_BIN_HIGH = 2600
BALANCE_BIN_WIDTH = 100


def keep_game(headers: chess.pgn.Headers) -> bool:
    """
    Filter games that are not our desired format (Rated Blitz, 900 to 2600 ELO).
    """
    result = _RESULT_TO_INT.get(headers.get("Result"))
    if result is None:
        return False
    white_elo = _parse_int(headers.get("WhiteElo", -1))
    black_elo = _parse_int(headers.get("BlackElo", -1))
    if not (900 <= white_elo <= 2600 and 900 <= black_elo <= 2600):
        return False
    return "Rated Blitz" in headers.get("Event", "")


def keep_position(ply: int, clock_remaining: int | None, game_length: int) -> bool:
    """
    For right now, this keeps the same filtering as the original Maia papers.
    Positions that are too close to the start/end of the game or with too little time
    are rejected.
    """
    if clock_remaining is not None and clock_remaining < MIN_TIME_REMAINING:
        return False
    # `ply` is a 0-based index (max is game_length - 1), so the last-ply offset is
    # applied relative to that: MAX_PLY_NEGATIVE_OFFSET = -10 trims the final 10 plies.
    return MIN_PLY <= ply <= game_length - 1 + MAX_PLY_NEGATIVE_OFFSET

def _num_bins(low: int, high: int, width: int) -> int:
    return (high - low) // width + 2


def _elo_bin(avg_elo: float, low: int, high: int, width: int) -> int:
    if avg_elo < low:
        return 0
    if avg_elo >= high:
        return _num_bins(low, high, width) - 1
    return 1 + int((avg_elo - low) // width)


def _rating_balancer() -> Callable[[chess.pgn.Headers], bool]:
    """
    An attempt at recreating the Maia3 paper's balanced dataset downsampler.

    TLDR: Collect 20k games (BALANCE_CHUNK_SIZE). Chunk the ELO range into bins of 
    width 100 (BALANCE_BIN_WIDTH) from 600 to 2600 (BALANCE_BIN_LOW/HIGH). 
    For each bin, accept at most 10 games (BALANCE_PER_BIN_CAP) per chunk. 
    Once the chunk is full or all bins are full, reset the counts and start a new chunk.
    """
    n_bins = _num_bins(BALANCE_BIN_LOW, BALANCE_BIN_HIGH, BALANCE_BIN_WIDTH)
    counts = [0] * n_bins
    full_bins = seen = 0

    def accept(headers: chess.pgn.Headers) -> bool:
        nonlocal full_bins, seen
        seen += 1
        accepted = False
        avg_elo = _avg_elo(headers)
        if avg_elo is not None:
            b = _elo_bin(avg_elo, BALANCE_BIN_LOW, BALANCE_BIN_HIGH, BALANCE_BIN_WIDTH)
            if counts[b] < BALANCE_PER_BIN_CAP:
                counts[b] += 1
                full_bins += counts[b] == BALANCE_PER_BIN_CAP
                accepted = True
        if seen >= BALANCE_CHUNK_SIZE or full_bins == n_bins:
            counts[:] = [0] * n_bins
            full_bins = seen = 0
        return accepted

    return accept


# ------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------

_RESULT_TO_INT = {"1-0": 2, "1/2-1/2": 1, "0-1": 0}


def _parse_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _avg_elo(headers: chess.pgn.Headers) -> float | None:
    white_elo = _parse_int(headers.get("WhiteElo"))
    black_elo = _parse_int(headers.get("BlackElo"))
    if white_elo is None or black_elo is None:
        return None
    return (white_elo + black_elo) / 2


@contextlib.contextmanager
def _open_pgn(input_path: str | Path) -> Iterator[io.TextIOBase]:
    """Open a .pgn or .pgn.zst file as a decoded text stream."""
    path = Path(input_path)
    if path.suffix == ".zst":
        import zstandard as zstd

        with path.open("rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            yield fh


def _iter_games(
    input_path: str | Path,
    header_filter: Callable[[chess.pgn.Headers], bool] = keep_game,
) -> Iterator[chess.pgn.Game]:

    class GameBuilder(chess.pgn.GameBuilder):
        def end_headers(self):
            if not header_filter(self.game.headers):
                return chess.pgn.SKIP  # stream past the movetext without parsing it
            return None

    with _open_pgn(input_path) as stream:
        while True:
            game = chess.pgn.read_game(stream, Visitor=GameBuilder)
            if game is None:  # end of file
                break
            yield game


def _game_rows(game: chess.pgn.Game, history: int) -> Iterator[dict]:
    """Yield one row dict per position-to-predict in a single game, each carrying
    the FENs of the last `history` board positions.

    Positions surviving the `keep_position` filter are capped at
    `MAX_POSITIONS_PER_GAME`; when a game has more, a random subset of that size
    is sampled (without replacement) so no single game dominates the dataset.

    Games rejected by `_iter_games`'s header filter arrive moveless, so the loop
    below simply yields nothing for them.
    """
    headers = game.headers
    result = _RESULT_TO_INT.get(headers.get("Result"))

    white_elo = _parse_int(headers.get("WhiteElo"))
    black_elo = _parse_int(headers.get("BlackElo"))

    board = game.board()
    recent: deque[str] = deque(maxlen=history)  # FENs of the most recent positions
    moves = list(game.mainline())
    game_len = len(moves)
    rows: list[dict] = []
    for ply, node in enumerate(moves):
        recent.append(board.fen())  # position the move at `ply` is played from
        if keep_position(ply, node.clock(), game_len):
            rows.append(
                {
                    "boards": list(recent),
                    "target_move": node.move.uci(),
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "result": result,
                }
            )
        board.push(node.move)

    if len(rows) > MAX_POSITIONS_PER_GAME:
        rows = random.sample(rows, MAX_POSITIONS_PER_GAME)
    yield from rows


# ------------------------------------------------------------------------------
# Parquet writing
# ------------------------------------------------------------------------------

_SCHEMA = pa.schema(
    [
        ("boards", pa.list_(pa.string())),
        ("target_move", pa.string()),
        ("white_elo", pa.int32()),
        ("black_elo", pa.int32()),
        ("result", pa.int8()),
    ]
)


def _batched(iterable: Iterator[dict], n: int) -> Iterator[list[dict]]:
    """Yield lists of up to `n` items (stdlib itertools.batched is 3.12+)."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _write_rows(
    writer: pq.ParquetWriter,
    games: Iterator[chess.pgn.Game],
    history: int,
    row_group_size: int,
    n_positions: int | None,
    bar: tqdm,
) -> tuple[int, int]:
    n_games = 0

    def rows() -> Iterator[dict]:
        nonlocal n_games
        for game in games:
            produced = False
            for row in _game_rows(game, history):
                produced = True
                bar.update(1)
                yield row
            n_games += produced

    stream = rows()
    if n_positions is not None:
        stream = itertools.islice(stream, n_positions)

    n_rows = 0
    for batch in _batched(stream, row_group_size):
        writer.write_table(pa.Table.from_pylist(batch, schema=_SCHEMA))
        n_rows += len(batch)
    return n_games, n_rows


def preprocess(
    history: int,
    input_path: str | Path,
    output_path: str | Path,
    row_group_size: int = 50_000,
    balance: bool = False,
    n_positions: int | None = None,
    progress: bool = True,
) -> Path:
    """Stream-convert a PGN export into a parquet position dataset, storing the
    last `history` board FENs per position.

    With `balance=True`, games are downsampled toward equal representation across
    rating bins after the `keep_game` header filter (see `_rating_balancer`).

    Both filters are header-only and applied at read time, so games that won't
    reach the dataset never have their (costly) movetext parsed.

    `n_positions`, if given, caps the output: preprocessing stops as soon as that
    many positions have been collected. Combined with `balance=True`, this yields
    a size-controlled balanced sample (handy for validation sets).
    """
    if n_positions is not None and n_positions <= 0:
        raise ValueError(f"n_positions must be positive, got {n_positions}.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    schema = _SCHEMA.with_metadata({HISTORY_META_KEY: str(history).encode()})

    header_filter = keep_game
    if balance:
        balancer = _rating_balancer()

        # keep_game first so its short-circuit keeps the balancer from seeing —
        # or counting toward its chunks — games that fail the header filter.
        def header_filter(headers: chess.pgn.Headers) -> bool:
            return keep_game(headers) and balancer(headers)

    # A bar toward n_positions when it's set; otherwise a plain position counter.
    bar = tqdm(total=n_positions, unit=" pos", disable=not progress)
    writer = pq.ParquetWriter(out, schema)
    try:
        games = _iter_games(input_path, header_filter)
        n_games, n_rows = _write_rows(
            writer, games, history, row_group_size, n_positions, bar
        )
    finally:
        bar.close()
        writer.close()

    if n_rows == 0:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"No positions produced from {input_path} (after keep_game / ply filters)."
        )

    balanced = " balanced" if balance else ""
    print(
        f"wrote {n_rows} positions from {n_games}{balanced} games "
        f"(history={history}) -> {out}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Lichess PGN export (.pgn.zst or .pgn) into a parquet position dataset.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .pgn.zst or .pgn file.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the .parquet dataset."
    )
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        help="Number of board positions stored per example (the history window "
        "baked into the parquet). Must be >= the trainer's history.",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=None,
        help="Stop after collecting this many positions (rows). Default: no limit.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample games toward equal representation across rating bins.",
    )
    args = parser.parse_args(argv)
    preprocess(
        args.history,
        args.input,
        args.output,
        balance=args.balance,
        n_positions=args.n_positions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
