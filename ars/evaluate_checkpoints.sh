#!/bin/bash

# Define the path to the evaluate_ars.py script
EVALUATE_SCRIPT_PATH="evaluate_ars.py"

# Define the method, number of episodes, and number of processes
METHOD="best"
N_EPISODES=10
N_PROCESSES=20

# Iterate over all directories under logs/CustomARS/
for GAME_DIR in ./logs/CustomARS/*; do
    for TIMESTAMP_DIR in "$GAME_DIR"/*; do
        if [ -d "$TIMESTAMP_DIR" ]; then
            # Check if 'rewards.csv' exists in the directory
            if [ ! -f "$TIMESTAMP_DIR/rewards.csv" ]; then
                echo "Evaluating weights in $TIMESTAMP_DIR"
                python $EVALUATE_SCRIPT_PATH "$TIMESTAMP_DIR" --method $METHOD --n_episodes $N_EPISODES --n_processes $N_PROCESSES
            else
                echo "Skipping $TIMESTAMP_DIR because 'rewards.csv' exists"
            fi
        fi
    done
done

