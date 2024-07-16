#!/bin/bash

# Associative array for timesteps for each MuJoCo environment
declare -A mujoco=(
    ["Hopper-v4"]=1000000
    ["Swimmer-v4"]=500000
    ["HalfCheetah-v4"]=3000000
    ["Walker2d-v4"]=2000000
    ["Ant-v4"]=10000000
    ["Humanoid-v4"]=10000000
)

# Associative arrays for hyperparameters
declare -A stepsize=(
    ["Swimmer-v4"]=0.02
    ["Hopper-v4"]=0.01
    ["HalfCheetah-v4"]=0.02
    ["Walker2d-v4"]=0.03
    ["Ant-v4"]=0.015
    ["Humanoid-v4"]=0.02
)

declare -A std_exploration_noise=(
    ["Swimmer-v4"]=0.01
    ["Hopper-v4"]=0.025
    ["HalfCheetah-v4"]=0.03
    ["Walker2d-v4"]=0.025
    ["Ant-v4"]=0.025
    ["Humanoid-v4"]=0.0075
)

declare -A number_of_directions=(
    ["Swimmer-v4"]=1
    ["Hopper-v4"]=8
    ["HalfCheetah-v4"]=32
    ["Walker2d-v4"]=40
    ["Ant-v4"]=60
    ["Humanoid-v4"]=230
)

declare -A number_of_top_performing_directions=(
    ["Swimmer-v4"]=1
    ["Hopper-v4"]=4
    ["HalfCheetah-v4"]=4
    ["Walker2d-v4"]=30
    ["Ant-v4"]=20
    ["Humanoid-v4"]=230
)

# Predefined seeds
seeds=(140 320 690 817 304)

# Semaphore mechanism to control concurrent jobs
MAX_PARALLEL=2
echo 0 > num_active_tasks

# Function to train a model given environment, timesteps, and hyperparameters.
train_model () {
    local env_id=$1
    local timesteps=$2
    local step=$3
    local noise=$4
    local directions=$5
    local top_directions=$6
    for SEED_NUM in {0..4}
    do
        SEED=${seeds[$SEED_NUM]}
        echo "Starting training for $env_id with seed $SEED, timesteps $timesteps, step $step, noise $noise, directions $directions, top_directions $top_directions"
        local log_file="logs/${env_id}_seed_${SEED}.log"
        mkdir -p logs
        (nohup python custom_ars.py --env $env_id -n $timesteps --n-eval-envs 10 --seed $SEED --learning_rate $step --delta_std $noise --n_delta $directions --n_top $top_directions > "$log_file" 2>&1; echo $(( $(cat num_active_tasks) - 1 )) > num_active_tasks) &
        sleep 1

        echo $(( $(cat num_active_tasks) + 1 )) > num_active_tasks

        while [ $(cat num_active_tasks) -ge $MAX_PARALLEL ]; do
            sleep 1
            jobs > /dev/null
        done
    done
}

# Iterate over mujoco environments and pass hyperparameters
for env in "${!mujoco[@]}"
do
    train_model $env ${mujoco[$env]} ${stepsize[$env]} ${std_exploration_noise[$env]} ${number_of_directions[$env]} ${number_of_top_performing_directions[$env]}
done

wait