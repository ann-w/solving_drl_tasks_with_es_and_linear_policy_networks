#!/bin/bash

# Rerun MLP 64x64 experiments with revised per-environment hyperparameters
# Strategies: lm-ma-es, sep-cma-es
# Uses tuned sigma0 and lambda from mlp_hyperparameters_revised/overleaf_table.md
#
# Optimised for AMD Ryzen 9 3950X (16 cores / 32 threads, 62 GB RAM):
#   - 3 concurrent experiments × 8 AsyncVectorEnv workers = 24 subprocesses + 3 parents
#     (workers are mostly pipe-blocked; real CPU load is lighter than 27 busy cores)
#   - Both strategies interleaved (not sequential) to avoid idle tail
#   - Experiments sorted longest-first for better bin-packing
#   - Failed jobs logged and summarised at end

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/es/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/annie/miniconda3/envs/solving-drl-with-es-py310/bin/python"

SEEDS=(43 44 45 46)
STRATEGIES=("lm-ma-es" "sep-cma-es")
DATA_DIR="es/data/mlp_experiments_revised"
LOG_DIR="es/logs"
MAX_PARALLEL_ENVS=8    # Env subprocesses per experiment (via multiprocessing)
MAX_JOBS=3             # Concurrent experiments  (3 × 8 = 24 env subprocesses + 3 parents)
BASE_ARGS="--normalized --mlp --max_parallel $MAX_PARALLEL_ENVS --data_dir $DATA_DIR --break_timesteps"
FAIL_LOG="$LOG_DIR/revised_failures.log"

# ── Per-environment hyperparameters (sorted longest-first) ───────────────────
# Format: ENV|SIGMA0|LAMBDA|MAX_TIMESTEPS
#
# Longest experiments first → short jobs fill gaps at the end (bin-packing).
EXPERIMENTS=(
    # Atari (20M timesteps each)
    "Atlantis-v5|0.10|128|20000000"
    "BeamRider-v5|0.10|128|20000000"
    "Pong-v5|0.10|128|20000000"
    "CrazyClimber-v5|0.10|128|20000000"
    "Enduro-v5|0.10|128|20000000"
    "Qbert-v5|0.10|128|20000000"
    "Seaquest-v5|0.10|128|20000000"
    "Boxing-v5|0.10|128|20000000"
    "SpaceInvaders-v5|0.10|128|20000000"
    # High-dim MuJoCo (10M)
    "Ant-v4|0.08|128|10000000"
    "Humanoid-v4|0.03|128|10000000"
    # Medium MuJoCo
    "HalfCheetah-v4|0.05|128|3000000"
    "Walker2d-v4|0.05|128|2000000"
    "BipedalWalker-v3|0.05|64|2000000"
    "Hopper-v4|0.08|64|1000000"
    # Fast
    "Swimmer-v4|0.10|64|500000"
    "CartPole-v1|0.20|64|500000"
    "Acrobot-v1|0.15|64|500000"
    "Pendulum-v1|0.20|64|500000"
    "LunarLander-v2|0.15|64|500000"
)

# ── Helper: parse experiment spec ────────────────────────────────────────────
get_env()           { echo "$1" | cut -d'|' -f1; }
get_sigma()         { echo "$1" | cut -d'|' -f2; }
get_lambda()        { echo "$1" | cut -d'|' -f3; }
get_max_timesteps() { echo "$1" | cut -d'|' -f4; }

# ── Run a single experiment (with retry + failure logging) ───────────────────
run_experiment() {
    local strategy=$1
    local env=$2
    local seed=$3
    local sigma0=$4
    local lamb=$5
    local max_ts=$6
    local logfile="$LOG_DIR/revised_${strategy}_${env}_seed${seed}.log"

    local tries=0
    local max_tries=2

    echo "[$(date '+%H:%M:%S')] START  $strategy | $env | seed=$seed | σ₀=$sigma0 λ=$lamb max_ts=$max_ts"

    while (( tries < max_tries )); do
        tries=$((tries + 1))
        if $PYTHON es/main.py \
            --strategy "$strategy" \
            --env_name "$env" \
            --seed "$seed" \
            --sigma0 "$sigma0" \
            --lamb "$lamb" \
            --max_train_timesteps "$max_ts" \
            $BASE_ARGS \
            > "$logfile" 2>&1; then
            echo "[$(date '+%H:%M:%S')] DONE   $strategy | $env | seed=$seed"
            return 0
        fi
        echo "[$(date '+%H:%M:%S')] Attempt $tries/$max_tries failed for $strategy | $env | seed=$seed"
    done

    echo "[$(date '+%H:%M:%S')] FAILED $strategy | $env | seed=$seed after $max_tries attempts (see $logfile)"
    echo "$strategy | $env | seed=$seed | attempts=$max_tries | $(date)" >> "$FAIL_LOG"
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────────────
ENV_COUNT=${#EXPERIMENTS[@]}
SEED_COUNT=${#SEEDS[@]}
STRAT_COUNT=${#STRATEGIES[@]}
TOTAL_EXPS=$((ENV_COUNT * STRAT_COUNT * SEED_COUNT))

echo "=============================================="
echo "Revised MLP 64×64 Experiments"
echo "Architecture: 2 hidden layers, 64 units each"
echo "Environments: $ENV_COUNT"
echo "Seeds: ${SEEDS[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo "Parallelization: $MAX_JOBS jobs × $MAX_PARALLEL_ENVS env workers = $((MAX_JOBS * MAX_PARALLEL_ENVS)) threads"
echo "Data directory: $DATA_DIR"
echo "=============================================="
echo ""
echo "Per-environment hyperparameters:"
printf "  %-20s  σ₀     λ    max_timesteps\n" "Environment"
printf "  %-20s  -----  ---  -------------\n" "-------------------"
for spec in "${EXPERIMENTS[@]}"; do
    printf "  %-20s  %-5s  %-3s  %s\n" \
        "$(get_env "$spec")" "$(get_sigma "$spec")" "$(get_lambda "$spec")" "$(get_max_timesteps "$spec")"
done
echo ""
echo "Total experiments: $ENV_COUNT envs × $STRAT_COUNT strategies × $SEED_COUNT seeds = $TOTAL_EXPS"
echo "Logs: $LOG_DIR/revised_<strategy>_<env>_seed<seed>.log"
echo ""

# Create directories
mkdir -p "$LOG_DIR" "$DATA_DIR"
> "$FAIL_LOG"  # Clear previous failure log

# ── Launch all experiments (both strategies interleaved) ─────────────────────
START_TIME=$SECONDS
job_count=0

for spec in "${EXPERIMENTS[@]}"; do
    env=$(get_env "$spec")
    sigma0=$(get_sigma "$spec")
    lamb=$(get_lambda "$spec")
    max_ts=$(get_max_timesteps "$spec")

    for strategy in "${STRATEGIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$strategy" "$env" "$seed" "$sigma0" "$lamb" "$max_ts" &
            job_count=$((job_count + 1))

            # Throttle: wait for a slot when at capacity
            if ((job_count >= MAX_JOBS)); then
                if ! wait -n; then
                    # A job failed; failure already logged by run_experiment
                    :
                fi
                job_count=$((job_count - 1))
            fi
        done
    done
done

# Wait for remaining jobs (non-fatal; failures already logged)
if ! wait; then :; fi

ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "=============================================="
echo "All $TOTAL_EXPS experiments completed!"
echo "Wall time: ${HOURS}h ${MINS}m"
echo "Results: $DATA_DIR"
echo "Logs:    $LOG_DIR/revised_*.log"
if [[ -s "$FAIL_LOG" ]]; then
    NFAIL=$(wc -l < "$FAIL_LOG")
    echo ""
    echo "⚠  $NFAIL experiment(s) FAILED — see $FAIL_LOG:"
    cat "$FAIL_LOG"
fi
echo "=============================================="
