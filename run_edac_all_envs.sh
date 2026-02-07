#!/bin/bash
# POGO Multi-Actor EDAC 모든 환경 훈련 스크립트

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/env_common.sh" ]; then
    source "$SCRIPT_DIR/env_common.sh"
fi

cd "$SCRIPT_DIR"

# conda 활성화 (offrl 또는 off_rl_gpu)
if [ -n "$(which conda)" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate offrl 2>/dev/null || conda activate off_rl_gpu
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

# POGO base config 사용
BASE_CONFIG="configs/offline/edac_pogo_base.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config not found: $BASE_CONFIG"
    exit 1
fi

echo "=========================================="
echo "POGO Multi-Actor EDAC 훈련 시작"
echo "총 ${#ENVIRONMENTS[@]}개 환경"
echo "=========================================="
echo ""

LOG_DIR="logs/pogo_edac"
mkdir -p "$LOG_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    env_clean=$(echo "$env" | sed 's/-/_/g')
    log_dir="${LOG_DIR}/${env_clean}/seed${SEED}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"
    
    # POGO config 파일 찾기 (환경별)
    # 예: halfcheetah-medium-v2 -> configs/offline/pogo_multi/halfcheetah/medium_v2_edac.yaml
    env_dir=$(echo "$env" | cut -d'-' -f1)
    env_task=$(echo "$env" | cut -d'-' -f2- | sed 's/-/_/g')
    config_file="configs/offline/pogo_multi/${env_dir}/${env_task}_edac.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "Warning: Config file not found: $config_file, using base config..."
        config_file="$BASE_CONFIG"
    fi
    
    # 임시 config 파일 생성 (seed 설정)
    TEMP_CONFIG="/tmp/pogo_edac_${env_clean}_seed${SEED}.yaml"
    cp "$config_file" "$TEMP_CONFIG"
    sed -i "s/^env:.*/env: ${env}/" "$TEMP_CONFIG"
    sed -i "s/^seed:.*/seed: ${SEED}/" "$TEMP_CONFIG"
    if [[ "$env" == *"expert"* ]]; then
        sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$TEMP_CONFIG"
    else
        sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$TEMP_CONFIG"
    fi
    
    echo "=========================================="
    echo "환경: $env"
    echo "Config: $config_file"
    echo "Log: $log_file"
    echo "시작: $(date)"
    echo "=========================================="
    
    {
        python -u -m algorithms.offline.pogo_multi_main \
            --config_path "$TEMP_CONFIG" \
            --env "$env" \
            --seed $SEED
        echo ""
        echo "완료: $(date)"
    } 2>&1 | tee "$log_file"
    
    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✅ $env 완료"
    else
        echo "❌ $env 실패 (exit code: $exit_code)"
    fi
    rm -f "$TEMP_CONFIG"
    echo ""
done

echo "=========================================="
echo "모든 POGO Multi-Actor EDAC 훈련 완료!"
echo "=========================================="
