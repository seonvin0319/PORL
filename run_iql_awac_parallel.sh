#!/bin/bash
# IQL, AWAC 병렬 실행 (동시에 각각 9개 환경 순차 학습)
# IQL 끝나면 별도로 종료 가능 (AWAC PID 참고)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_common.sh"

cd "$SCRIPT_DIR"

# conda 활성화
if [ -n "$(which conda)" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate off_rl_gpu 2>/dev/null || conda activate base
fi

ENVIRONMENTS=(
    "halfcheetah-medium-v2"
    "halfcheetah-medium-expert-v2"
    "halfcheetah-medium-replay-v2"
    "hopper-medium-v2"
    "hopper-medium-expert-v2"
    "hopper-medium-replay-v2"
    "walker2d-medium-v2"
    "walker2d-medium-expert-v2"
    "walker2d-medium-replay-v2"
)

SEED=0

run_algorithm() {
    local ALG="$1"
    local CONFIG="$2"

    for env in "${ENVIRONMENTS[@]}"; do
        env_clean=$(echo "$env" | sed 's/-/_/g')
        log_dir="logs/${ALG}/${env_clean}/seed${SEED}"
        mkdir -p "$log_dir"
        log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"

        TEMP_CONFIG="/tmp/${ALG}_pogo_${env_clean}_seed${SEED}.yaml"
        cp "$CONFIG" "$TEMP_CONFIG"
        sed -i "s/env: .*/env: ${env}/" "$TEMP_CONFIG"
        sed -i "s/seed: .*/seed: ${SEED}/" "$TEMP_CONFIG"
        if [[ "$env" == *"expert"* ]]; then
            sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$TEMP_CONFIG"
        else
            sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$TEMP_CONFIG"
        fi

        echo "[$ALG] >>> 환경: $env | Log: $log_file"
        {
            echo "=========================================="
            echo "$ALG | 환경: $env | 시작: $(date)"
            echo "=========================================="
            python -u -m algorithms.offline.pogo_multi_main \
                --config_path "$TEMP_CONFIG" \
                --env "$env" \
                --seed $SEED
            echo ""
            echo "완료: $(date)"
        } 2>&1 | tee "$log_file"
        rm -f "$TEMP_CONFIG"
    done
}

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "IQL + AWAC 병렬 실행 (seed $SEED)"
echo "=========================================="
echo ""

# IQL 백그라운드 (함수는 현재 셸에서 상속)
(run_algorithm "iql" "configs/offline/iql_pogo_base.yaml") > "$LOG_DIR/parallel_iql_${TS}.out" 2>&1 &
IQL_PID=$!

# AWAC 백그라운드
(run_algorithm "awac" "configs/offline/awac_pogo_base.yaml") > "$LOG_DIR/parallel_awac_${TS}.out" 2>&1 &
AWAC_PID=$!

echo "IQL PID: $IQL_PID"
echo "AWAC PID: $AWAC_PID"
echo ""
echo "로그:"
echo "  IQL:  tail -f $LOG_DIR/parallel_iql_${TS}.out"
echo "  AWAC: tail -f $LOG_DIR/parallel_awac_${TS}.out"
echo ""
echo "IQL 끝나면 AWAC만 종료: kill $AWAC_PID"
echo "둘 다 종료: kill $IQL_PID $AWAC_PID"
echo "=========================================="

wait
