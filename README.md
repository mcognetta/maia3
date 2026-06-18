# Maia-3 Training Reproduction

This is a training fork of the Maia3 repo, which included only inference (and with a slightly different model than what was described in their paper). 

This fork implements their base model types and the training objectives of the original paper. This serves two purposes:

1) To reproduce the results of the Maia team.
2) To give a baseline training repo for others to experiment off of.

Thus, this repo is not intended to be fully feature complete, but rather to define a basic training structure (model definitions, configs, the training loop, validation metrics, etc.) that is as close to the base Maia3 training setup as possible so that other training experiments can be built on top of it.

## Quick Start

Training is in two parts: preparing a dataset and training a model.


### Prepare a Dataset
Download a month of data from [Lichess](https://database.lichess.org/) and use maia3-preprocess to filter and extract valid positions and store in a parquet file. You should use different months for the training / validation parquet files, and you can use `--n-positions` to limit the size of the dataset. You can also use the `--balance` flag to roughly balance the number of positions per bin, according to the Maia3 data preparation note in Appendix A.1 of the paper. The last important parameter is `--history` which defines how many prior positions are given as input for the next move prediction. `--history 1` means I just predict the next move from teh current board, but `--history 8` means I have access to the previous 8 (padded if there are too few) positions.

**NOTE:** `--history` and the `history` parameter in the training config must be the same. If you want to train a new model with a different history, you must also generate new datasets.

```bash
maia3-preprocess --input lichess_train.pgn.zst --output data/train.parquet --balance --n-positions 100000000
maia3-preprocess --input lichess_val.pgn.zst   --output data/valid.parquet --balance --n-positions 1000000
```

### Train the Model

Define a yaml config (see the `configs/` directory) that outlines your model hyperparameters. We provide configs for the 3m, 5m, 23m, and 79m variants present in the Maia3 paper for you

Preprocess Lichess PGN exports into parquet datasets, then train:

```bash
maia3-preprocess --input lichess_train.pgn.zst --output data/train.parquet --balance --n-positions 100000000
maia3-preprocess --input lichess_val.pgn.zst   --output data/valid.parquet --balance --n-positions 1000000
maia3-train --config configs/maia3-5m.yaml
```

(`maia3-train` ≡ `python -m maia3.train`.) The config already points at
`data/train.parquet` and `data/valid.parquet` and writes checkpoints to
`runs/maia3-5m/`. Training uses `cuda`, falling back to CPU with a warning.

## Configuration

Everything is driven by a single YAML config (`--config`), with three optional
sections — `model`, `data`, `train` — mirroring the dataclasses in
[`maia3/config.py`](maia3/config.py). List only the fields you want to change;
the rest keep their defaults, and unknown fields fail loudly. Individual CLI
flags (one per field, e.g. `--lr`, `--batch-size`) override the config:

```bash
maia3-train --config configs/maia3-5m.yaml --lr 1e-4 --num-steps 300000
```

**Model size is just a config.** The ready-to-run `configs/maia3-*.yaml` each
fully specify an architecture plus the data + training recipe. The sizes share
everything except width and the GAB dimensions:

| Config | dim_vit | heads | head_hid_dim | gab_gen / per_square / intermediate |
| --- | --- | --- | --- | --- |
| [`maia3-3m.yaml`](configs/maia3-3m.yaml)   | 192  | 6  | 192  | 64 / 0 / 64 |
| [`maia3-5m.yaml`](configs/maia3-5m.yaml)   | 256  | 8  | 256  | 64 / 0 / 64 |
| [`maia3-23m.yaml`](configs/maia3-23m.yaml) | 512  | 16 | 512  | 128 / 32 / 128 |
| [`maia3-79m.yaml`](configs/maia3-79m.yaml) | 1024 | 32 | 1024 | 128 / 32 / 128 |

**Weights & Biases** is configured in the `train` section (`use_wandb`,
`wandb_project`, `wandb_mode: online | offline | disabled`, …). Run `wandb login`
once, or pass `--no-use-wandb` to skip it. Either way, loss and policy/value
accuracy print to the console.

## Data pipeline

`maia3-preprocess` reads a Lichess PGN export (`.pgn.zst`, a `.zip` of PGNs, or a
plain `.pgn`), filters games, and explodes each into one **row per
position-to-predict**, streamed to parquet:

```python
{
    "boards":      list[str],   # FENs of the last `history` positions (newest last)
    "target_move": str,         # the move played from the newest board (the label)
    "white_elo":   int,
    "black_elo":   int,
    "result":      int,         # 0 = black wins, 1 = draw, 2 = white wins
}
```

- **Filtering is yours to edit.** `keep_game` (and the `MIN_PLY` bounds) at the
  top of [`maia3/preprocess.py`](maia3/preprocess.py) is the single place that
  decides which games/positions enter the dataset. The default keeps Rated Blitz
  games with both Elos in 900–2600.
- **History is fixed at preprocess time** (`--history N`, default 8) and recorded
  in the parquet metadata. Each row stores exactly those boards as FENs. Training
  may use a *smaller* history but not a larger one.
- **Loading is lazy.** [`ParquetPositionDataset`](maia3/dataset.py) memory-maps
  the file and tokenizes one row at a time, so `shuffle=True` streams positions
  without holding tensors in RAM.
