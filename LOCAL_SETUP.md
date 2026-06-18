# Local Setup (this machine)

Notes for running Maia3 here. The upstream [README.md](README.md) covers the
project itself; this file is the workstation-specific quickstart.

## Environment

- Workstation GPU: NVIDIA RTX PRO 5000 Blackwell (48 GB), driver supports CUDA up to 12.8
- Python: 3.12 (via `uv`)
- Venv lives at [.venv/](.venv/) (prompt name `maia3`)

The default PyTorch wheel on PyPI is built for CUDA 13, which is too new for
the installed driver. We pin a CUDA 12.8 build of Torch instead — see the
install steps below.

## One-time install

```bash
# from the repo root
uv venv --python 3.12 --prompt maia3 .venv
uv pip install --python .venv/bin/python -e .

# replace the default (cu13) torch with a CUDA 12.8 build that matches the driver
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  --reinstall torch
```

If the CUDA-specific reinstall is skipped, everything still runs — it just
falls back to CPU silently with a "driver too old" warning.

## Activating the venv

```bash
source .venv/bin/activate     # then `maia3-uci ...`, `python ...`, etc.
```

Or invoke binaries directly without activating:

```bash
.venv/bin/maia3-uci --list-models
.venv/bin/python scripts/smoke_test.py
```

## Smoke test

[scripts/smoke_test.py](scripts/smoke_test.py) loads the 5M model, picks a
move from the starting position at three Elo levels, and prints the result.
First run downloads ~20 MB from Hugging Face; subsequent runs are cached at
`~/.cache/huggingface/`.

```bash
.venv/bin/python scripts/smoke_test.py
```

Expected tail:

```
[1/4] Torch 2.11.0+cu128 | CUDA available: True
[2/4] Resolving Maia3 5M (UofTCSSLab/Maia3-5M)
[3/4] Loading model onto cuda
[4/4] Querying moves from the starting position at Elos [1100, 1500, 2000]
      Elo 1100: e2e4  (e4)
      ...
OK: Maia3 loaded and produced moves end-to-end.
```

Useful flags:

```bash
.venv/bin/python scripts/smoke_test.py --model 23m            # bigger model
.venv/bin/python scripts/smoke_test.py --device cpu           # force CPU
.venv/bin/python scripts/smoke_test.py --elos 800 1600 2400   # custom Elos
```

## Running Maia3 as a UCI engine

```bash
.venv/bin/maia3-uci --model maia3-5m
```

Try one move from the command line:

```bash
printf 'uci\nucinewgame\nposition startpos\ngo nodes 1\nquit\n' \
  | .venv/bin/maia3-uci --model 5m
```

Built-in aliases: `3m`, `5m`, `23m`, `79m`. See the upstream README for the
full architecture flag list.

## Training

Training entry point: [maia3/train.py](maia3/train.py). Config dataclasses
live in [maia3/config.py](maia3/config.py) — `ModelConfig` is shared with
inference (`uci.py`'s argparse builds a Namespace with the same field names),
`DataConfig` and `TrainConfig` are training-only.

```bash
.venv/bin/python -m maia3.train --shard-dir /path/to/shards \
  --output-dir runs/test --num-steps 1000
```

`python -m maia3.train --help` lists every flag, grouped by config dataclass.

### Shard format

Each shard is a pickle file containing `list[dict]`; one entry per game:

```python
{
    "white_elo": int,
    "black_elo": int,
    "result":    int,            # 0=black wins, 1=draw, 2=white wins
    "moves":     list[str],      # UCI strings, in play order
}
```

Shards store raw game data, **not pre-tokenized tensors**. Tokenization
happens in [Maia3Dataset.__getitem__](maia3/train.py) on the fly by replaying
moves on `chess.Board()`. This means new derived features (e.g. signals
computed from board ± surrounding moves) can be added directly in
`__getitem__` or `collate_fn` with zero shard regen. Extra keys in each game
dict are ignored, so a preprocessor can stash future-use fields safely.

### Loss

[compute_loss](maia3/train.py) is a placeholder — masked cross-entropy on the
policy head only. Replace it once the model output definitions are settled;
the per-batch dict already carries `move_idx`, `value_target`, and
`legal_mask` so the multi-head loss can read what it needs without further
plumbing.

## Troubleshooting

- **`CUDA available: False` with a "driver too old" warning** — the cu13 wheel
  got reinstalled. Re-run the cu128 reinstall command from the install step.
- **`Multiple checkpoint files found in ...`** — pass
  `--checkpoint-filename <file>` to pick one.
- **Hugging Face rate-limit warnings** — set `HF_TOKEN` in the environment
  to authenticate; not required for the public Maia3 weights.
