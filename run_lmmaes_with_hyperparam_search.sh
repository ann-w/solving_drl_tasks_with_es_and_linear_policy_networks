#!/usr/bin/env bash
set -euo pipefail

# Search space definitions ---------------------------------------------------
declare -a ENVS=()
declare -A MAX_TIMESTEPS=()

while read -r env steps; do
  [[ -z "$env" ]] && continue
  ENVS+=("$env")
  MAX_TIMESTEPS["$env"]="$steps"
done <<'EOF'
CartPole-v1 500000
Acrobot-v1 500000
Pendulum-v1 500000
LunarLander-v2 500000
BipedalWalker-v3 2000000
Swimmer-v4 500000
HalfCheetah-v4 3000000
Hopper-v4 1000000
Walker2d-v4 2000000
Ant-v4 10000000
Humanoid-v4 10000000
SpaceInvaders-v5 10000000
Atlantis-v5 10000000
Assault-v5 10000000
BeamRider-v5 10000000
Breakout-v5 10000000
Boxing-v5 10000000
Pong-v5 10000000
CrazyClimber-v5 10000000
Enduro-v5 10000000
Qbert-v5 10000000
Seaquest-v5 10000000
EOF

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

resolve_max_timesteps() {
  local env="$1"
  if [[ -n "${MAX_TIMESTEPS[$env]+set}" ]]; then
    echo "${MAX_TIMESTEPS[$env]}"
  else
    echo 10000000
  fi
}

job_id=0
for env in "${ENVS[@]}"; do
  env_dir="${RUN_TS_DIR}/${env}"
  mkdir -p "$env_dir"
  env_timesteps=$(resolve_max_timesteps "$env")

  mapfile -t RESOLVED_LAMBDAS < <(
    for method in "${LAMBDAS[@]}"; do
      resolve_lambda "$env" "$method"
    done
  )

  env_generations=$(( ${#SIGMAS[@]} * ${#RESOLVED_LAMBDAS[@]} * ${#SEEDS[@]} ))
  echo "Preparing ${env_generations} jobs for ${env} (${#RESOLVED_LAMBDAS[@]} lambda choices)"
  echo "We run ${env_generations} jobs for environment ${env} with strategy ${STRATEGY}" >> "${SCRIPT_LOG}"
  
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
            --break_timesteps \
            --max_train_timesteps "$env_timesteps"
        ) &>"$log_file" &
        echo "[job ${job_id}] $env sigma=${sigma} lambda=${lambda_val} seed=${seed} -> ${log_file}"
      done
    done
  done

done

wait
echo "All ${job_id} jobs have finished. Logs are in ${RUN_DIR}."
