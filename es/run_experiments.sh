#!/bin/bash

# Experiment runner script for ES methods on RL environments
# Usage: ./run_experiments.sh [--dry-run] [--subset CATEGORY]

set -e

# Parse arguments
DRY_RUN=false
SUBSET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Strategies
STRATEGIES=(
    "csa"
    "cma-es"
    "sep-cma-es"
    "ars"
    "ars-v2"
)

# Atari environments
ATARI=(
    "Atlantis-v5"
    "BeamRider-v5"
    "Boxing-v5"
    "Pong-v5"
    "CrazyClimber-v5"
    "Enduro-v5"
    "SpaceInvaders-v5"
)

# MuJoCo environments
MUJOCO=(
    "Hopper-v4"
    "Walker2d-v4"
    "HalfCheetah-v4"
    "Ant-v4"
    "Swimmer-v4"
    "Humanoid-v4"
)

# Classic Control environments
CLASSIC_CONTROL=(
    "CartPole-v1"
    "Acrobot-v1"
    "Pendulum-v1"
    "BipedalWalker-v3"
    "LunarLander-v2"
)

# Select environments based on subset argument
case $SUBSET in
    "atari")
        ENVIRONMENTS=("${ATARI[@]}")
        ;;
    "mujoco")
        ENVIRONMENTS=("${MUJOCO[@]}")
        ;;
    "classic")
        ENVIRONMENTS=("${CLASSIC_CONTROL[@]}")
        ;;
    *)
        # All environments
        ENVIRONMENTS=("${CLASSIC_CONTROL[@]}" "${MUJOCO[@]}" "${ATARI[@]}")
        ;;
esac

# Common arguments
COMMON_ARGS="--normalized --seed 42 --reward_shaping false --data_dir data/no_reward_shaping"

# Counter for progress
TOTAL=$((${#ENVIRONMENTS[@]} * ${#STRATEGIES[@]}))
CURRENT=0

echo "Running $TOTAL experiments..."
echo "Strategies: ${STRATEGIES[*]}"
echo "Environments: ${ENVIRONMENTS[*]}"
echo ""

for ENV in "${ENVIRONMENTS[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "[$CURRENT/$TOTAL] Running $STRATEGY on $ENV"
        
        CMD="python main.py --env_name $ENV --strategy $STRATEGY $COMMON_ARGS"
        
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] $CMD"
        else
            echo "  $CMD"
            $CMD
        fi
        
        echo ""
    done
done

echo "All experiments completed!"
