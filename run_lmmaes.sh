#!/usr/bin/env bash
set -euo pipefail

# Search space definitions ---------------------------------------------------
ENVS=(
  "CartPole-v1"
  "Acrobot-v1"
  "Pendulum-v1"
  "LunarLander-v2"
  "BipedalWalker-v3"
  "Swimmer-v4"
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Ant-v4"
  "Humanoid-v4"
  "SpaceInvaders-v5"
  "Atlantis-v5"
  "Assault-v5"
  "BeamRider-v5"
  "Breakout-v5"
  "Boxing-v5"
  "Pong-v5"
  "CrazyClimber-v5"
  "Enduro-v5"
  "Qbert-v5"
  "SpaceInvaders-v5"
)

# Seeds: [x * 12 for x in range(0, 10)]
SEEDS=()
for x in {0..9}; do
  SEEDS+=("$((x * 12))")
done

# Only LM-MA-ES is supported by this launcher for now
STRATEGY="lm-ma-es"
SIGMAS=(5.0 2.0 0.5 0.1 0.05 0.01 1)
LAMBDAS=(4 default "n/2")

# Orchestration knobs --------------------------------------------------------
MAX_PARALLEL=${MAX_PARALLEL:-10}
RUN_DIR=${RUN_DIR:-"$(pwd)/run"}
MUJOCO_GL=${MUJOCO_GL:-egl}
PYTHON_BIN=${PYTHON_BIN:-python}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_DIR="${ROOT_DIR}/es"

mkdir -p "$RUN_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TS_DIR="${RUN_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_TS_DIR"
SCRIPT_LOG="${RUN_TS_DIR}/lmmaes_launcher_${TIMESTAMP}.log"
# Mirror stdout/stderr so the run is inspectable after completion.
exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "Logging launcher output to ${SCRIPT_LOG}"
echo "Running strategy: ${STRATEGY}"
echo "Configured ${#ENVS[@]} environments, ${#SIGMAS[@]} sigmas, ${#LAMBDAS[@]} lambda labels, ${#SEEDS[@]} seeds"

# Make sure we import the local rl_es sources rather than an installed release.
export PYTHONPATH="${ES_DIR}/src:${PYTHONPATH:-}"

load_guard() {
  while [[ $(jobs -rp | wc -l | tr -d ' ') -ge ${MAX_PARALLEL} ]]; do
    sleep 5
  done
}

resolve_lambda() {
  local env_name="$1"
  local method_label="$2"
  "${PYTHON_BIN}" - <<PY
from rl_es.setting import ENVIRONMENTS
from rl_es.algorithms import init_lambda
env = ENVIRONMENTS["${env_name}"]
n = env.action_size * env.state_size
label = "${method_label}"
try:
    method = int(label)
except ValueError:
    method = label
print(init_lambda(n, method))
PY
}

job_id=0
for env in "${ENVS[@]}"; do
  env_dir="${RUN_TS_DIR}/${env}"
  mkdir -p "$env_dir"
  mapfile -t RESOLVED_LAMBDAS < <(
    for method in "${LAMBDAS[@]}"; do
      resolve_lambda "$env" "$method"
    done
  )

  env_generations=$(( ${#SIGMAS[@]} * ${#RESOLVED_LAMBDAS[@]} * ${#SEEDS[@]} ))
  echo "Preparing ${env_generations} generations for ${env} (${#RESOLVED_LAMBDAS[@]} lambda choices)"

  for sigma in "${SIGMAS[@]}"; do
    for lambda_val in "${RESOLVED_LAMBDAS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        load_guard
        ((++job_id))
        log_file="${env_dir}/${env//\//_}_${sigma}_${lambda_val}_${seed}.log"
        (
          cd "$ES_DIR"
          MUJOCO_GL="$MUJOCO_GL" "${PYTHON_BIN}" main.py \
            --env_name "$env" \
            --strategy "$STRATEGY" \
            --seed "$seed" \
            --normalized \
            --sigma0 "$sigma" \
            --lamb "$lambda_val" \
            --break_timesteps
        ) &>"$log_file" &
        echo "[job ${job_id}] $env sigma=${sigma} lambda=${lambda_val} seed=${seed} -> ${log_file}"
      done
    done
  done

done

wait
echo "All ${job_id} jobs have finished. Logs are in ${RUN_DIR}."
