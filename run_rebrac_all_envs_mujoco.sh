#!/bin/bash
# POGO Multi-Actor ReBRAC 모든 환경 훈련 스크립트 (JAX)

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

SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "POGO Multi-Actor ReBRAC 훈련 시작"
echo "총 ${#ENVIRONMENTS[@]}개 환경 × ${#SEEDS[@]}개 시드 = $((${#ENVIRONMENTS[@]} * ${#SEEDS[@]}))개 실험"
echo "=========================================="
echo ""

LOG_DIR="logs/pogo_rebrac"
mkdir -p "$LOG_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    env_clean=$(echo "$env" | sed 's/-/_/g')
    
    # POGO config 파일 찾기 (환경별)
    # 예: halfcheetah-medium-v2 -> configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml
    env_dir=$(echo "$env" | cut -d'-' -f1)
    env_task=$(echo "$env" | cut -d'-' -f2- | sed 's/-/_/g')
    config_file="configs/offline/pogo_multi/${env_dir}/${env_task}_rebrac.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        echo "해당 환경의 config 파일을 먼저 생성해주세요."
        exit 1
    fi
    
    for SEED in "${SEEDS[@]}"; do
        log_dir="${LOG_DIR}/${env_clean}/seed${SEED}"
        mkdir -p "$log_dir"
        log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"
        
        # 임시 config 파일 생성 (seed, dataset_name, w2_weights 설정)
        TEMP_CONFIG="/tmp/pogo_rebrac_${env_clean}_seed${SEED}.yaml"
        cp "$config_file" "$TEMP_CONFIG"
        
        # dataset_name 설정
        sed -i "s/^dataset_name:.*/dataset_name: ${env}/" "$TEMP_CONFIG"
        
        # train_seed 필드가 있으면 교체, 없으면 추가 (None/null 값도 처리)
        if grep -q "^train_seed:" "$TEMP_CONFIG"; then
            sed -i "s/^train_seed:.*/train_seed: ${SEED}/" "$TEMP_CONFIG"
        else
            # dataset_name 다음에 train_seed 추가
            sed -i "/^dataset_name:/a train_seed: ${SEED}" "$TEMP_CONFIG"
        fi
        
        # None이나 null 값이 있으면 제거하고 다시 설정
        sed -i "s/^train_seed:.*null.*/train_seed: ${SEED}/" "$TEMP_CONFIG"
        sed -i "s/^train_seed:.*None.*/train_seed: ${SEED}/" "$TEMP_CONFIG"
        
        # w2_weights 설정
        if [[ "$env" == *"expert"* ]]; then
            sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$TEMP_CONFIG"
        else
            sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$TEMP_CONFIG"
        fi
        
        echo "=========================================="
        echo "환경: $env"
        echo "시드: $SEED"
        echo "Config: $config_file"
        echo "Log: $log_file"
        echo "시작: $(date)"
        echo "=========================================="
        
        {
            python -u -m algorithms.offline.pogo_multi_jax \
                --config_path "$TEMP_CONFIG" \
                --no_wandb
            echo ""
            echo "완료: $(date)"
        } 2>&1 | tee "$log_file"
        
        exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            echo "✅ $env (seed $SEED) 완료"
        else
            echo "❌ $env (seed $SEED) 실패 (exit code: $exit_code)"
        fi
        rm -f "$TEMP_CONFIG"
        echo ""
    done
done

echo "=========================================="
echo "모든 POGO Multi-Actor ReBRAC 훈련 완료!"
echo "=========================================="
