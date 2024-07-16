# ARS Baselines

This folder contains the scripts to run and evaluate the baselines for ARS. We use the ARS implementation in stable baselines3. In `custom_ars.py` we extend the ARS class by adding a functionality to save the best and mean weights, which can later be used for evaluation.

## Installation

Create a virtual environment and then activate it.
```
conda create -n sb3 python=3.11
```

```
conda activate sb3
```

Install libraries:

```
pip install -r requirements.txt
```

## Run experiments

The folder contains several bash files. Make them executable.
```
chmod +x *.sh
```

To train and save the weights, run the following:

For Atari
```
run_experiments_atari.sh
```

For MuJoCo
```
run_experiments_mujoco.sh
```

For Classic control
```
run_experiments_classic_control.sh
```

To evaluate the weights, do
```
evaluate_checkpoints.sh
```