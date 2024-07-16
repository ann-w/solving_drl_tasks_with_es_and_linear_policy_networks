# Associative array for Atari environments and their respective timesteps.
declare -A atari=(
    ["ALE/Atlantis-ram-v5"]=20000000
    ["ALE/BeamRider-ram-v5"]=20000000
    ["ALE/CrazyClimber-ram-v5"]=20000000
    ["ALE/Enduro-ram-v5"]=20000000
    ["ALE/Qbert-ram-v5"]=20000000
    ["ALE/Seaquest-ram-v5"]=20000000
    ["ALE/Pong-ram-v5"]=20000000
    ["ALE/Boxing-ram-v5"]=20000000
    ["ALE/SpaceInvaders-ram-v5"]=20000000
)

# Predefined seeds
seeds=(140 320 690 817 304)

# Semaphore mechanism to control concurrent jobs
MAX_PARALLEL=2
echo 0 > num_active_tasks

# Function to train a model given environment and timesteps.
train_model () {
    local env_id=$1
    local timesteps=$2
    for SEED_NUM in {0..4}  # Iterate over the indices of the seeds array
    do
        SEED=${seeds[$SEED_NUM]}  # Use predefined seed for each training session.
        echo "Starting training for $env_id with seed $SEED and timesteps $timesteps"
        # Construct log file name
        local log_file="logs/${env_id}_seed_${SEED}.log"
        # Ensure the entire directory path for the log file exists
        mkdir -p "$(dirname "$log_file")"
        # Start training in the background with the specified algorithm, environment, and seed.
        # Redirect both stdout and stderr to the log file.
        (nohup python custom_ars.py --env $env_id -n $timesteps --n-eval-envs 10 --seed $SEED > "$log_file" 2>&1; echo $(( $(cat num_active_tasks) - 1 )) > num_active_tasks) &        
        sleep 1  # Pause for a moment before starting the next training session to reduce load.

        # Increment active_tasks
        echo $(( $(cat num_active_tasks) + 1 )) > num_active_tasks

        # If the number of active tasks is equal to MAX_PARALLEL, wait
        while [ $(cat num_active_tasks) -ge $MAX_PARALLEL ]; do
            sleep 1  # Sleep for a moment to prevent constant loop checking
            # Check if any background job has finished
            jobs > /dev/null
        done
    done
}

# Iterate over atari  environments
for env in "${!atari[@]}"
do
    train_model $env ${atari[$env]}
done

wait  # Wait for all background processes to finish.