#!/bin/bash
# IQL 전체 환경 병렬 실행 (seed 0)
# GPU 메모리 고려하여 최대 3개씩 병렬 실행
# iql_pogo_base.yaml 사용, actor_configs: gaussian 설정됨
# w2_weights: expert는 [100.0, 100.0], 나머지는 [10.0, 10.0] (config에서 설정)

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

ALGORITHM="iql"
SEED=0
CONFIG_PATH="configs/offline/iql_pogo_base.yaml"
MAX_PARALLEL=3  # GPU 메모리 고려하여 최대 3개씩 병렬 실행

echo "=========================================="
echo "IQL 전체 환경 병렬 학습 (Gaussian Policy)"
echo "Seed: $SEED | Config: $CONFIG_PATH"
echo "최대 병렬 실행 수: $MAX_PARALLEL"
echo "Expert: w2_weights=[100.0, 100.0], 나머지: w2_weights=[10.0, 10.0]"
echo "=========================================="

mkdir -p logs/iql

# 병렬 실행을 위한 함수
run_environment() {
    local env=$1
    local env_name_clean=$(echo "$env" | sed 's/-/_/g')
    local log_dir="logs/iql/${env_name_clean}/seed${SEED}"
    mkdir -p "$log_dir"
    
    local log_file="${log_dir}/$(date +%Y%m%d_%H%M%S).log"
    
    # w2_weights 설정: expert 환경은 [100.0, 100.0], 나머지는 [10.0, 10.0]
    # 임시 config 파일 생성
    local temp_config="/tmp/iql_pogo_${env_name_clean}_seed${SEED}_$$.yaml"
    cp "$CONFIG_PATH" "$temp_config"
    
    if [[ "$env" == *"expert"* ]]; then
        sed -i "s/w2_weights: \[.*\]/w2_weights: [100.0, 100.0]/" "$temp_config"
    else
        sed -i "s/w2_weights: \[.*\]/w2_weights: [10.0, 10.0]/" "$temp_config"
    fi
    
    echo "[$(date +%H:%M:%S)] 시작: $env"
    
    {
        echo "=========================================="
        echo "환경: $env"
        echo "시작 시간: $(date)"
        echo "=========================================="
        
        python -u -m algorithms.offline.pogo_multi_main \
            --config_path "$temp_config" \
            --algorithm "$ALGORITHM" \
            --env "$env" \
            --seed $SEED
        
        rm -f "$temp_config"
        
        echo ""
        echo "=========================================="
        echo "환경: $env 완료"
        echo "종료 시간: $(date)"
        echo "=========================================="
    } > "$log_file" 2>&1
    
    echo "[$(date +%H:%M:%S)] 완료: $env"
}

# 병렬 실행 관리
PIDS=()
for env in "${ENVIRONMENTS[@]}"; do
    # 최대 병렬 수에 도달하면 대기
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        # 완료된 프로세스 확인
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                # 프로세스 완료
                wait "${PIDS[$i]}"
                unset PIDS[$i]
            fi
        done
        # 배열 재인덱싱
        PIDS=("${PIDS[@]}")
        sleep 1
    done
    
    # 새 프로세스 시작
    run_environment "$env" &
    PIDS+=($!)
done

# 남은 프로세스들 대기
echo ""
echo "모든 환경 실행 시작됨. 완료 대기 중..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=========================================="
echo "모든 IQL 학습 완료"
echo "=========================================="
