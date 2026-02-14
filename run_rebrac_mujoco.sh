#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-10 50 100 500}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ENVS="${ENVS:-halfcheetah-medium-v2 halfcheetah-medium-expert-v2 halfcheetah-medium-replay-v2 hopper-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2 walker2d-medium-v2 walker2d-medium-expert-v2 walker2d-medium-replay-v2}"
NUM_ACTORS="${NUM_ACTORS:-6}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_ROOT="${LOG_ROOT:-logs/pogo_rebrac}"

export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

config_for_env() {
  local env_name="$1"
  local family task task_key config_path

  family="${env_name%%-*}"          # e.g. halfcheetah / hopper / walker2d
  task="${env_name#${family}-}"     # e.g. medium-v2 / medium-replay-v2 / expert-v2
  task_key="${task//-/_}"           # e.g. medium_v2 / medium_replay_v2 / expert_v2

  config_path="configs/offline/pogo_multi/${family}/${task_key}_rebrac.yaml"
  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found for env=${env_name}: ${config_path}" >&2
    echo "Expected pattern: configs/offline/pogo_multi/<family>/<task>_rebrac.yaml" >&2
    exit 1
  fi
  echo "${config_path}"
}

build_w2_weights() {
  local weight="$1"
  local num_actors="$2"
  local n i out

  n=$((num_actors - 1))
  if (( n < 1 )); then
    echo "NUM_ACTORS must be >= 2, got ${num_actors}" >&2
    exit 1
  fi

  out="["
  for ((i = 1; i <= n; i++)); do
    out+="${weight}"
    if (( i < n )); then
      out+=", "
    fi
  done
  out+="]"
  echo "${out}"
}

count_items() { wc -w <<<"$1" | tr -d ' '; }
num_envs="$(count_items "${ENVS}")"
num_seeds="$(count_items "${SEEDS}")"
num_weights="$(count_items "${WEIGHTS}")"
total_runs=$((num_envs * num_seeds * num_weights))

echo "=========================================="
echo "POGO Multi-Actor ReBRAC (JAX) 훈련 시작"
echo "총 ${num_envs}개 환경 x ${num_seeds}개 시드 x ${num_weights}개 weight = ${total_runs}개 실험"
echo "=========================================="
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ENVS=${ENVS}"
echo "SEEDS=${SEEDS}"
echo "WEIGHTS=${WEIGHTS}"
echo "NUM_ACTORS=${NUM_ACTORS}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "LOG_ROOT=${LOG_ROOT}"
echo ""

mkdir -p "${LOG_ROOT}"

echo "환경 검사: ${PYTHON_BIN} 필수 모듈 확인"
if ! "${PYTHON_BIN}" - <<'PY'
import importlib
required = ["jax", "flax", "optax", "chex", "gym", "d4rl", "pyrallis", "wandb"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("Missing modules:", ", ".join(missing))
    raise SystemExit(1)
print("필수 모듈 확인 완료")
PY
then
  echo "오류: 현재 PYTHON_BIN 환경에 필수 패키지가 없습니다."
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "예시: PYTHON_BIN=/home/choi/miniconda3/envs/offrl/bin/python"
  exit 1
fi

for env_name in ${ENVS}; do
  config_path="$(config_for_env "${env_name}")"
  env_clean="${env_name//-/_}"

  actor_type="$(grep -A 20 '^actor_configs:' "${config_path}" 2>/dev/null | grep 'type:' | head -1 | sed 's/.*type:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//')"
  actor_type="${actor_type:-deterministic}"

  for seed in ${SEEDS}; do
    for w in ${WEIGHTS}; do
      w_float="${w}.0"
      w2_weights="$(build_w2_weights "${w_float}" "${NUM_ACTORS}")"
      run_log_dir="${LOG_ROOT}/${env_clean}/w${w}/seed${seed}"
      mkdir -p "${run_log_dir}"
      ts="$(date +%Y%m%d_%H%M%S)"
      log_file="${run_log_dir}/${env_clean}_rebrac_${actor_type}_${NUM_ACTORS}_${w}_${seed}_${ts}.log"

      echo "=========================================="
      echo "환경: ${env_name}"
      echo "시드: ${seed}"
      echo "W2 weight: ${w_float}"
      echo "W2 list: ${w2_weights}"
      echo "Config: ${config_path}"
      echo "Log: ${log_file}"
      echo "시작: $(date)"
      echo "=========================================="

      if "${PYTHON_BIN}" -u -m algorithms.offline.pogo_multi_jax \
        --config_path "${config_path}" \
        --seed "${seed}" \
        --dataset_name "${env_name}" \
        --num_actors "${NUM_ACTORS}" \
        --w2_weights "${w2_weights}" \
        --eval_every "${EVAL_EVERY}" \
        --name "rebrac-jax-${env_name}-actor${NUM_ACTORS}-w${w}-seed${seed}" \
        ${EXTRA_ARGS} 2>&1 | tee "${log_file}"; then
        echo ""
        echo "완료: $(date)" | tee -a "${log_file}"
        echo "✅ ${env_name} (seed ${seed}, w ${w}) 완료"
      else
        exit_code=$?
        echo ""
        echo "종료: $(date) (exit=${exit_code})" | tee -a "${log_file}"
        echo "❌ ${env_name} (seed ${seed}, w ${w}) 실패 (exit code: ${exit_code})"
      fi

      echo ""
    done
  done
done

echo "=========================================="
echo "모든 POGO Multi-Actor ReBRAC 훈련 완료!"
echo "=========================================="
