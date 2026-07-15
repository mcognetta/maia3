# Maia-3 Training Reproduction

This is a training fork of the [Maia3 repo](https://github.com/CSSLab/maia3), which shipped inference only (and with a slightly different model than what was described in [the paper](https://arxiv.org/abs/2605.19091), since that included pondering). This fork implements the base model variants and training objectives from the paper, with two goals:

1. Reproduce the results of the Maia team.
2. Give others a baseline training repo to experiment on top of.

We include an inference pipeline, but nothing else (bots, UCI, etc.) to keep this repo minimial.

## Install

```bash
git clone https://github.com/mcognetta/maia3.git && cd maia3
uv venv && uv pip install -e .    # or: pip install -e .
```

## 1. Prepare data

The training data comes from the [Lichess database](https://database.lichess.org/). `maia3-preprocess` filters the games (Rated Blitz, both players 900–2600 Elo) and writes one parquet row per position to predict.

The bundled script reproduces the full training corpus: it downloads every month from 2023-01 through 2025-07, plus one extra held-out month (2025-08) for validation, and preprocesses them into the shards the configs expect:

```bash
cd data && ./download_and_rename_datasets.sh
```

NB: The data preparation takes a long time. Each month is 20+ GB when downloaded (compressed) and takes 20+ minutes to process. The final parquet file is relatively small for each month (a few 100 Mb / month). This script parallelizes the downloading and processing, so make sure you have enough hard drive space and cores to handle 4 months processing at once.

```bash
maia3-preprocess --input lichess_2024-01.pgn.zst --output data/train.parquet --history 7 --balance
maia3-preprocess --input lichess_2024-02.pgn.zst --output data/valid.parquet --history 7 --balance --n-positions 1000000
```

Important flags:

- `--history N` — how many past board positions each example carries. This is baked into the parquet; training can use a *smaller* history than the data was built with, but not a larger one.
- `--balance` — downsample games toward equal representation across rating bins (Appendix A.1 of the paper).
- `--n-positions M` — stop after M positions in case you want to limit the dataset size (e.g., for validation sets)

The Maia3 paper uses the [allie](https://huggingface.co/datasets/yimingzhang/allie-data/tree/main) dataset. This requires some postprocessing, which we will provide a script for soon.

## 2. Train

Training is based on configuration files in the config directory. These match the reported hyperparameters from the paper, except for the batch size (which we set to be much larger than the paper by default).

```bash
maia3-train --config configs/maia3-5m.yaml
```

Model size is just a config. Each `configs/maia3-*.yaml` fully specifies an architecture plus the data and training recipe; the sizes differ only in width and the GAB dimensions:

| Config | dim_vit | heads | head_hid_dim | gab_gen / per_square / intermediate |
| --- | --- | --- | --- | --- |
| [`maia3-3m.yaml`](configs/maia3-3m.yaml)   | 192  | 6  | 192  | 64 / 0 / 64 |
| [`maia3-5m.yaml`](configs/maia3-5m.yaml)   | 256  | 8  | 256  | 64 / 0 / 64 |
| [`maia3-23m.yaml`](configs/maia3-23m.yaml) | 512  | 16 | 512  | 128 / 32 / 128 |
| [`maia3-79m.yaml`](configs/maia3-79m.yaml) | 1024 | 32 | 1024 | 128 / 32 / 128 |

A config has three sections — `model`, `data`, `train` — mirroring the dataclasses in [`maia3/config.py`](maia3/config.py). List only the fields you want to change; unknown fields fail loudly. Every field also exists as a CLI flag, and flags override the config:

```bash
maia3-train --config configs/maia3-5m.yaml --lr 1e-4 --num-steps 300000
```

Each run writes to a fresh timestamped directory, e.g. `runs/maia3-5m/20260701-093000/`, containing the fully-resolved `config.yaml` (itself a valid `--config` input, so runs are reproducible) and periodic checkpoints. Checkpoints embed their config too, so an interrupted run can be continued with a single flag:

- `--resume-from CKPT` — continue an interrupted run exactly (weights, optimizer, LR schedule, step counter). The flag alone is enough; everything else is read from the checkpoint.

Training data can be a single parquet (`parquet_path`) or a list of shards (`parquet_paths`), which are trained through one at a time — only one shard is in memory at once, and a full pass over the list counts as one epoch.

We save checkpoints every `ckpt_interval` steps. To save space, the `keep_last_k` config field drops older checkpoints. However we always retain the best performing checkpoint (according to the `best_metric` field in the config) as `best.pt` in the output directory.

The provided configs enable Weights & Biases logging — run `wandb login` once, or set `use_wandb: false` in the `train` section to skip it. Loss and policy/value accuracy print to the console either way.

To evaluate a checkpoint on a validation parquet later:

```bash
maia3-validate --checkpoint runs/maia3-5m/<run>/best.pt data/valid.parquet
```

## 3. Inference

`maia3-infer` runs inference from a checkpoint on a given position.

```bash
maia3-infer --checkpoint runs/maia3-5m/<run>/best.pt \
    "8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47" --elo-self 1569 --elo-oppo 1579

FEN:  8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47
Elo:  1569 (self) vs 1579 (opponent)
LDW:  0.3403 / 0.1344 / 0.5254
Move probabilities (17 legal moves):
  f5g6   Kg6    15.49%  ######
  f5e5   Ke5    12.00%  ####
  f5e6   Ke6    11.58%  ####
  g5g4   g4+    11.46%  ####
  f5f6   Kf6     9.41%  ###
  c4d3   Bd3     7.91%  ###
```

It prints the win/draw/loss estimate and a probability for every legal move. Since the models are history-aware, you can also pass several FENs (oldest first) and the last one is the position to move from.

### Python API

We have a similar Python API to the [simple-maia3-inference](https://github.com/mcognetta/simple-maia3-inference) repo.

```python
from maia3.infer import Maia3

maia = Maia3("runs/maia3-5m/<run>/best.pt")
move_probs, ldw = maia.probs(
    "8/8/7B/1p3kpp/p1b5/2P2KP1/1P6/8 b - - 3 47", 1569, 1579
)
```

`move_probs` is a `{uci: probability}` dict over the legal moves, sorted best-first; `ldw` is the (loss, draw, win) tuple for the side to move. There are also `logits` and batched variants — see [`maia3/infer.py`](maia3/infer.py).

## License

AGPL-3.0, same as the upstream repo — see [LICENSE](LICENSE).
