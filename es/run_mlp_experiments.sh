#!/bin/bash

# Run MLP 64x64 experiments with LM-MA-ES and sep-CMA-ES
# This script tests whether larger network architectures (matching PPO's 64x64)
# can be effectively optimized by evolution strategies

set -e  # Exit on error

# Configuration
COMMON_ARGS="--normalized --seed 42 --mlp"
DATA_DIR="data/mlp_experiments"

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
    
    echo "========================================"
    echo "Running $strategy on $env with MLP 64x64"
    echo "========================================"
    
    python main.py \
        --strategy "$strategy" \
        --env_name "$env" \
        $COMMON_ARGS
    
    echo "Completed $strategy on $env"
    echo ""
}

# Function to run all experiments for a strategy
run_all_for_strategy() {
    local strategy=$1
    
    echo "########################################"
    echo "# Starting experiments for: $strategy"
    echo "########################################"
    echo ""
    
    # Classic Control
    echo ">>> Classic Control environments"
    for env in "${CLASSIC_CONTROL[@]}"; do
        run_experiment "$strategy" "$env"
    done
    
    # MuJoCo
    echo ">>> MuJoCo environments"
    for env in "${MUJOCO[@]}"; do
        run_experiment "$strategy" "$env"
    done
    
    # Atari
    echo ">>> Atari environments"
    for env in "${ATARI[@]}"; do
        run_experiment "$strategy" "$env"
    done
    
    echo "########################################"
    echo "# Completed all experiments for: $strategy"
    echo "########################################"
    echo ""
}

# Main execution
echo "=============================================="
echo "MLP 64x64 Experiments"
echo "Architecture: 2 hidden layers, 64 units each"
echo "=============================================="
echo ""

# Print parameter counts
echo "Expected parameter counts with MLP 64x64:"
echo "  CartPole-v1:     ~4,610 params"
echo "  Acrobot-v1:      ~4,803 params"
echo "  LunarLander-v2:  ~5,060 params"
echo "  Pendulum-v1:     ~4,417 params"
echo "  BipedalWalker-v3:~6,084 params"
echo "  Swimmer-v4:      ~4,866 params"
echo "  Hopper-v4:       ~5,059 params"
echo "  HalfCheetah-v4:  ~5,574 params"
echo "  Walker2d-v4:     ~5,574 params"
echo "  Ant-v4:          ~6,344 params"
echo "  Humanoid-v4:     ~29,248 params"
echo ""

# Run LM-MA-ES first (designed for large-scale optimization)
run_all_for_strategy "lm-ma-es"

# Run sep-CMA-ES second (O(n) complexity, diagonal covariance)
run_all_for_strategy "sep-cma-es"

echo "=============================================="
echo "All MLP 64x64 experiments completed!"
echo "=============================================="
