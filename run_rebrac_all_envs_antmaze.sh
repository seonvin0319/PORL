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
    "antmaze-umaze-v2"
    "antmaze-umaze-diverse-v2"
    "antmaze-medium-play-v2"
    "antmaze-medium-diverse-v2"
    "antmaze-large-play-v2"
    "antmaze-large-diverse-v2"
)

SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "POGO Multi-Actor ReBRAC AntMaze 훈련 시작"
echo "총 ${#ENVIRONMENTS[@]}개 환경 × ${#SEEDS[@]}개 시드 = $((${#ENVIRONMENTS[@]} * ${#SEEDS[@]}))개 실험"
echo "=========================================="
echo ""

LOG_DIR="logs/pogo_rebrac"
mkdir -p "$LOG_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    env_clean=$(echo "$env" | sed 's/-/_/g')
    
    # POGO config 파일 찾기 (환경별)
    # 예: antmaze-umaze-v2 -> configs/offline/pogo_multi/antmaze/umaze_v2_rebrac.yaml
    # 예: antmaze-medium-play-v2 -> configs/offline/pogo_multi/antmaze/medium_play_v2_rebrac.yaml
    if [[ "$env" == antmaze-* ]]; then
        env_dir="antmaze"
        # antmaze-umaze-v2 -> umaze_v2
        # antmaze-umaze-diverse-v2 -> umaze_diverse_v2
        # antmaze-medium-play-v2 -> medium_play_v2
        # antmaze-umaze-v2 -> umaze-v2 -> umaze_v2
        env_task=$(echo "$env" | sed 's/^antmaze-//' | sed 's/-/_/g')
        config_file="configs/offline/pogo_multi/${env_dir}/${env_task}_rebrac.yaml"
    else
        # MuJoCo 환경 (기존 로직)
        env_dir=$(echo "$env" | cut -d'-' -f1)
        env_task=$(echo "$env" | cut -d'-' -f2- | sed 's/-/_/g')
        config_file="configs/offline/pogo_multi/${env_dir}/${env_task}_rebrac.yaml"
    fi
    
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
        # antmaze 환경은 w2_weights를 기본값으로 유지 (config 파일에 설정된 값 사용)
        if [[ "$env" == antmaze-* ]]; then
            # antmaze는 config 파일의 w2_weights를 그대로 사용
            :
        elif [[ "$env" == *"expert"* ]]; then
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
echo "모든 POGO Multi-Actor ReBRAC AntMaze 훈련 완료!"
echo "=========================================="
