#!/bin/bash
# IQL 환경별 seed 0 순차 실행
# D4RL MuJoCo: halfcheetah, hopper, walker2d (medium, medium-expert, medium-replay)
# expert: w2_weights [100, 100], 나머지: [10, 10]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_common.sh"

cd "$SCRIPT_DIR"

# conda 활성화 (d4rl, pytorch 등 필요)
if [ -n "$(which conda)" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate off_rl_gpu 2>/dev/null || conda activate base
fi

ENVIRONMENTS=(
    # "halfcheetah-medium-v2"
    # "halfcheetah-medium-expert-v2"
    # "halfcheetah-medium-replay-v2"
    "hopper-medium-v2"
    # "hopper-medium-expert-v2"
    # "hopper-medium-replay-v2"
    # "walker2d-medium-v2"
    # "walker2d-medium-expert-v2"
    # "walker2d-medium-replay-v2"
)

SEED=0
BASE_CONFIG="configs/offline/iql_pogo_base.yaml"

echo "=========================================="
echo "IQL 전체 환경 순차 학습 (seed $SEED)"
echo "expert: w2_weights [100, 100], 나머지: [10, 10]"
echo "=========================================="

for env in "${ENVIRONMENTS[@]}"; do
    env_clean=$(echo "$env" | sed 's/-/_/g')
    log_dir="logs/iql/${env_clean}/seed${SEED}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"

    TEMP_CONFIG="/tmp/iql_pogo_${env_clean}_seed${SEED}.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"
    sed -i "s/env: .*/env: ${env}/" "$TEMP_CONFIG"
    sed -i "s/seed: .*/seed: ${SEED}/" "$TEMP_CONFIG"
    if [[ "$env" == *"expert"* ]]; then
        sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$TEMP_CONFIG"
    else
        sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$TEMP_CONFIG"
    fi

    echo ""
    echo ">>> IQL | 환경: $env"
    echo "    Log: $log_file"
    {
        echo "=========================================="
        echo "IQL | 환경: $env | 시작: $(date)"
        echo "=========================================="
        python -u -m algorithms.offline.pogo_multi_main \
            --config_path "$TEMP_CONFIG" \
            --env "$env" \
            --seed $SEED
        echo ""
        echo "완료: $(date)"
    } 2>&1 | tee "$log_file"
    rm -f "$TEMP_CONFIG"
    echo "    완료: $env"
done

echo ""
echo "=========================================="
echo "모든 IQL 학습 완료"
echo "=========================================="
