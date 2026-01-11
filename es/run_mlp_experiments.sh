#!/bin/bash

# Run MLP 64x64 experiments with LM-MA-ES and sep-CMA-ES
# This script tests whether larger network architectures (matching PPO's 64x64)
# can be effectively optimized by evolution strategies

# Note: Don't use set -e with background jobs and wait -n

# Configuration
# Seeds 42 already done, run seeds 43-46 for 4 more runs (5 total)
# SEEDS=(42 43 44 45 46)
SEEDS=(43 44 45 46)
BASE_ARGS="--normalized --mlp --max_parallel 7 --data_dir data/mlp_experiments --break_timesteps"
DATA_DIR="data/mlp_experiments"
MAX_JOBS=4  # Run 4 experiments in parallel (4 jobs × 7 workers = 28 threads)

# Classic Control environments
CLASSIC_CONTROL=(
    "CartPole-v1"
    "Acrobot-v1"
    "LunarLander-v2"
    "Pendulum-v1"
    "BipedalWalker-v3"
)

# MuJoCo environments
MUJOCO=(
    "Swimmer-v4"
    "Hopper-v4"
    "HalfCheetah-v4"
    "Walker2d-v4"
    "Ant-v4"
    "Humanoid-v4"
)

# Atari environments (RAM-based)
ATARI=(
    "Pong-v5"
    "Boxing-v5"
    "Assault-v5"
    "Atlantis-v5"
    "BeamRider-v5"
    "CrazyClimber-v5"
    "Enduro-v5"
)

# All environments
ALL_ENVS=("${CLASSIC_CONTROL[@]}" "${MUJOCO[@]}" "${ATARI[@]}")

# Function to run experiment
run_experiment() {
    local strategy=$1
    local env=$2
    local seed=$3
    
    echo "[$(date '+%H:%M:%S')] Starting $strategy on $env (seed=$seed)"
    
    python main.py \
        --strategy "$strategy" \
        --env_name "$env" \
        --seed "$seed" \
        $BASE_ARGS \
        > "logs/${strategy}_${env}_seed${seed}.log" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Completed $strategy on $env (seed=$seed)"
}

# Function to run all experiments for a strategy in parallel
run_all_for_strategy() {
    local strategy=$1
    
    echo "########################################"
    echo "# Starting experiments for: $strategy"
    echo "# Seeds: ${SEEDS[*]}"
    echo "# Running $MAX_JOBS jobs in parallel"
    echo "########################################"
    echo ""
    
    # Create logs directory
    mkdir -p logs
    
    # Run all environments and seeds in parallel with job control
    local job_count=0
    
    for seed in "${SEEDS[@]}"; do
        for env in "${ALL_ENVS[@]}"; do
            run_experiment "$strategy" "$env" "$seed" &
            ((job_count++))
            
            # Wait if we've reached max parallel jobs
            if ((job_count >= MAX_JOBS)); then
                wait -n  # Wait for any job to finish
                ((job_count--))
            fi
        done
    done
    
    # Wait for remaining jobs
    wait
    
    echo ""
    echo "########################################"
    echo "# Completed all experiments for: $strategy"
    echo "########################################"
    echo ""
}

# Main execution
echo "==============================================" 
echo "MLP 64x64 Experiments - Additional Runs"
echo "Architecture: 2 hidden layers, 64 units each"
echo "Seeds: ${SEEDS[*]} (seed 42 already completed)"
echo "Parallelization: $MAX_JOBS jobs × 7 workers = 28 threads"
echo "=============================================="
echo ""

# Print parameter counts
echo "Expected parameter counts with MLP 64x64:"
echo "  Classic Control: ~4,400 - 6,000 params"
echo "  MuJoCo:          ~4,800 - 29,400 params"
echo "  Atari:           ~12,600 - 13,600 params"
echo ""
echo "Total experiments: ${#ALL_ENVS[@]} environments × 2 strategies × ${#SEEDS[@]} seeds = $((${#ALL_ENVS[@]} * 2 * ${#SEEDS[@]}))"
echo "Logs will be saved to: logs/<strategy>_<env>_seed<seed>.log"
echo ""

# Create logs directory
mkdir -p logs

# Run LM-MA-ES first (designed for large-scale optimization)
run_all_for_strategy "lm-ma-es"

# Run sep-CMA-ES second (O(n) complexity, diagonal covariance)
run_all_for_strategy "sep-cma-es"

echo "=============================================="
echo "All MLP 64x64 additional runs completed!"
echo "Total runs per env/method: 5 (seed 42 + seeds 43-46)"
echo "Check logs/ directory for individual experiment outputs"
echo "=============================================="
