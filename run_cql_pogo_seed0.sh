#!/bin/bash
# CQL 전체 환경 순차 실행 (seed 0)
# cql_pogo_base.yaml 사용
# w2_weights: expert는 [100.0, 100.0], 나머지는 [10.0, 10.0]

set -e

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl

cd /home/svcho/PORL
source $(conda info --base)/etc/profile.d/conda.sh
conda activate off_rl_gpu

# 환경 순서: hc-m, hc-me, hc-mr, h-m, h-me, h-mr, w-m, w-me, w-mr
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

ALGORITHM="cql"
SEED=0
CONFIG_PATH="configs/offline/cql_pogo_base.yaml"

echo "=========================================="
echo "CQL 전체 환경 순차 학습 (TanhGaussian Policy)"
echo "Seed: $SEED | Config: $CONFIG_PATH"
echo "Expert: w2_weights=[100.0, 100.0], 나머지: w2_weights=[10.0, 10.0]"
echo "=========================================="

mkdir -p logs/cql

for env in "${ENVIRONMENTS[@]}"; do
    # 환경 이름 정리 (로그 디렉토리용)
    env_name_clean=$(echo "$env" | sed 's/-/_/g')
    log_dir="logs/cql/${env_name_clean}/seed${SEED}"
    mkdir -p "$log_dir"
    
    log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo ">>> 환경: $env"
    echo "    Log: $log_file"
    
    # w2_weights 설정: expert 환경은 [100.0, 100.0], 나머지는 [10.0, 10.0]
    # 임시 config 파일 생성
    TEMP_CONFIG="/tmp/cql_pogo_${env//-/_}_seed${SEED}.yaml"
    cp "$CONFIG_PATH" "$TEMP_CONFIG"
    
    if [[ "$env" == *"expert"* ]]; then
        sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$TEMP_CONFIG"
        echo "    [expert] w2_weights: [100.0, 100.0]"
    else
        sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$TEMP_CONFIG"
    fi
    
    {
        echo "=========================================="
        echo "환경: $env"
        echo "시작 시간: $(date)"
        echo "=========================================="
        
        python -u -m algorithms.offline.pogo_multi_main \
            --config_path "$TEMP_CONFIG" \
            --algorithm "$ALGORITHM" \
            --env "$env" \
            --seed $SEED
        
        rm -f "$TEMP_CONFIG"
        
        echo ""
        echo "=========================================="
        echo "환경: $env 완료"
        echo "종료 시간: $(date)"
        echo "=========================================="
    } > "$log_file" 2>&1
    
    echo "    완료: $env"
done

echo ""
echo "=========================================="
echo "모든 CQL 학습 완료"
echo "=========================================="
