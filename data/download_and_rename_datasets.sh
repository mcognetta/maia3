#!/usr/bin/env bash
set -euo pipefail

# Download monthly Lichess dumps and preprocess them into the parquet shards
# the training configs expect:
#
#   lichess_parquet/train_YYYY-MM.parquet   one per month in START_YM..END_YM
#   lichess_parquet/val_YYYY-MM.parquet     the month after END_YM (or VAL_YM),
#                                           capped at VAL_POSITIONS positions
#
# Validation uses its own month so it never overlaps the training data.
# All knobs are env vars, e.g.: JOBS=4 VAL_POSITIONS=500000 ./download_and_rename_datasets.sh

# Inclusive month range of training data, as YYYYMM. Defaults: Jan 2023 .. Jul 2025.
START_YM="${START_YM:-202301}"
END_YM="${END_YM:-202507}"
JOBS="${JOBS:-3}"

# Validation month (YYYYMM; default: the month after END_YM) and its size cap.
VAL_POSITIONS="${VAL_POSITIONS:-1000000}"

# History window baked into the parquet (must be >= the trainer's history).
HISTORY="${HISTORY:-7}"

WORKDIR="${WORKDIR:-./lichess_raw}"
OUTDIR="${OUTDIR:-./lichess_parquet}"
LOGDIR="${LOGDIR:-./logs}"

mkdir -p "$WORKDIR" "$OUTDIR" "$LOGDIR"

next_month() {
    local ym="$1"  # YYYYMM
    local y=$((10#$ym / 100)) m=$((10#$ym % 100))
    if ((m == 12)); then y=$((y + 1)); m=1; else m=$((m + 1)); fi
    printf '%04d%02d' "$y" "$m"
}

VAL_YM="${VAL_YM:-$(next_month "$END_YM")}"

run_month() {
    local kind="$1"  # train | val
    local ym="$2"    # e.g. 2024-01

    local base="lichess_db_standard_rated_${ym}"
    local url="https://database.lichess.org/standard/${base}.pgn.zst"

    local input="${WORKDIR}/${base}.pgn.zst"
    local partial="${input}.part"
    local output="${OUTDIR}/${kind}_${ym}.parquet"
    local tmp_output="${OUTDIR}/${kind}_${ym}.partial.parquet"
    local log="${LOGDIR}/${kind}_${ym}.log"

    (
        set -euo pipefail

        echo "[$kind $ym] starting"

        if [[ -s "$output" ]]; then
            echo "[$kind $ym] output already exists, skipping: $output"
            exit 0
        fi

        rm -f "$tmp_output"

        if [[ ! -s "$input" ]]; then
            echo "[$kind $ym] downloading $url"
            curl \
                --fail \
                --location \
                --retry 5 \
                --retry-delay 10 \
                --continue-at - \
                --output "$partial" \
                "$url"

            mv "$partial" "$input"
        else
            echo "[$kind $ym] input already exists, reusing: $input"
        fi

        # The validation month is capped so it stays a manageable size.
        local extra_args=()
        if [[ "$kind" == "val" ]]; then
            extra_args+=(--n-positions "$VAL_POSITIONS")
        fi

        echo "[$kind $ym] preprocessing"
        maia3-preprocess \
            --input "$input" \
            --output "$tmp_output" \
            --history "$HISTORY" \
            --balance \
            "${extra_args[@]}"

        mv "$tmp_output" "$output"

        echo "[$kind $ym] removing raw input"
        rm -f "$input" "$partial"

        echo "[$kind $ym] done: $output"
    ) >"$log" 2>&1
}

failed=0

# Jobs are "kind YYYY-MM" pairs: every training month plus the validation month.
jobs_list=()
for y in $(seq "${START_YM:0:4}" "${END_YM:0:4}"); do
    for m in $(seq -w 1 12); do
        ym_num=$((10#$y * 100 + 10#$m))
        if ((ym_num >= 10#$START_YM && ym_num <= 10#$END_YM)); then
            jobs_list+=("train ${y}-${m}")
        fi
    done
done
jobs_list+=("val ${VAL_YM:0:4}-${VAL_YM:4:2}")

# Continuous job pool: keep up to JOBS months running at once, starting a new
# one as soon as any finishes (rather than waiting for a whole batch). A failing
# month is recorded but does not stop the others; the script exits non-zero at
# the end if any month failed.
declare -A pid_job   # background pid -> "kind YYYY-MM"
total=${#jobs_list[@]}
next=0
running=0   # tracked separately: reading an empty assoc array under `set -u` errors

while ((next < total || running > 0)); do
    # Top the pool up to JOBS while months remain.
    while ((running < JOBS && next < total)); do
        job="${jobs_list[next]}"
        run_month $job &
        pid_job["$!"]="$job"
        next=$((next + 1))
        running=$((running + 1))
        echo "Started $job (running ${running}/$JOBS, ${next}/${total} launched)"
    done

    # Block until any one job finishes, then reap exactly that one.
    finished_pid=""
    if wait -n -p finished_pid; then
        status=0
    else
        status=$?
    fi
    [[ -z "$finished_pid" ]] && continue   # spurious wake / no tracked child

    job="${pid_job[$finished_pid]}"
    unset 'pid_job[$finished_pid]'
    running=$((running - 1))
    if ((status != 0)); then
        echo "[$job] FAILED (exit $status). Check $LOGDIR/${job/ /_}.log"
        failed=1
    else
        echo "[$job] finished"
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "One or more months failed. Check logs in $LOGDIR."
    exit 1
fi

echo "All done."
