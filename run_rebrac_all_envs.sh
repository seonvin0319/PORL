#!/bin/bash
# ReBRAC 전체 환경 순차 실행 (JAX)
# w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)
# 일반: [10.0, 10.0], expert: [100.0, 100.0]

set -e

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
# export WANDB_MODE=disabled  # wandb 사용 시 주석 처리

cd /home/choi/PORL
source $(conda info --base)/etc/profile.d/conda.sh
conda activate offrl

ALGORITHM="rebrac"
SEED=0

echo "=========================================="
echo "ReBRAC 전체 환경 순차 학습 (JAX)"
echo "Seed: $SEED"
echo "순서: halfcheetah → hopper → walker2d → antmaze"
echo "각 도메인: medium → medium-replay → medium-expert"
echo "=========================================="

# 환경 순서 지정: halfcheetah → hopper → walker2d → antmaze
# 각 도메인 내에서: medium → medium-replay → medium-expert 순서
CONFIG_FILES=""

# 1. halfcheetah: medium → medium-replay → medium-expert
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/halfcheetah/medium_replay_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/halfcheetah/medium_expert_v2_rebrac.yaml"

# 2. hopper: medium → medium-replay → medium-expert
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/hopper/medium_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/hopper/medium_replay_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/hopper/medium_expert_v2_rebrac.yaml"

# 3. walker2d: medium → medium-replay → medium-expert
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/walker2d/medium_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/walker2d/medium_replay_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/walker2d/medium_expert_v2_rebrac.yaml"

# 4. antmaze: umaze → medium-play → medium-diverse (antmaze는 medium/replay/expert 구조가 없음)
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/antmaze/umaze_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/antmaze/medium_play_v2_rebrac.yaml"
CONFIG_FILES="$CONFIG_FILES configs/offline/pogo_multi/antmaze/medium_diverse_v2_rebrac.yaml"

TOTAL=$(echo "$CONFIG_FILES" | wc -l)
CURRENT=0

for config_path in $CONFIG_FILES; do
    CURRENT=$((CURRENT + 1))
    
    # config 파일에서 dataset_name 추출
    dataset_name=$(grep "^dataset_name:" "$config_path" | awk '{print $2}')
    domain=$(echo "$config_path" | cut -d'/' -f4)
    task=$(basename "$config_path" | sed 's/_rebrac.yaml//')
    env_dir="${domain}_${task}"
    
    echo ""
    echo "[$CURRENT/$TOTAL] >>> 환경: $dataset_name"
    echo "    Config: $config_path"
    
    log_dir="results/${ALGORITHM}/${env_dir}/seed_${SEED}/logs"
    checkpoint_dir="results/${ALGORITHM}/${env_dir}/seed_${SEED}/checkpoints"
    mkdir -p "$log_dir" "$checkpoint_dir"
    
    log_file="${log_dir}/${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log"
    echo "    Log: $log_file"
    
    # JAX 버전 실행
    python -u -m algorithms.offline.pogo_multi_jax \
        --config_path "$config_path" \
        > "$log_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[실패] $dataset_name - 로그: $log_file"
        exit 1
    fi
    echo "    완료: $dataset_name"
done

echo ""
echo "=========================================="
echo "모든 ReBRAC 학습 완료"
echo "=========================================="
