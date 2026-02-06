#!/bin/bash
# IQL 전체 환경 순차 실행 (seed 0)
# w2_weights: actor0=0, actor1/2만 적용 (일반 [0,10,10], expert [0,100,100])

set -e

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export WANDB_MODE=disabled

cd /home/offrl/CORL
source $(conda info --base)/etc/profile.d/conda.sh
conda activate offrl

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
MAX_TIMESTEPS=1000000
EVAL_FREQ=5000
N_EPISODES=10
DEVICE="cuda"

# actor0=0, actor1/2만 w2/sinkhorn 적용
W2_WEIGHTS_BASE="[0.0, 10.0, 10.0]"
W2_WEIGHTS_EXPERT="[0.0, 100.0, 100.0]"

echo "=========================================="
echo "IQL 전체 환경 순차 학습"
echo "Seed: $SEED | Base: $W2_WEIGHTS_BASE | Expert: $W2_WEIGHTS_EXPERT"
echo "=========================================="

for env in "${ENVIRONMENTS[@]}"; do
    domain=$(echo "$env" | cut -d'-' -f1)
    task=$(echo "$env" | cut -d'-' -f2- | sed 's/-/_/g')
    env_dir=$(echo "$env" | sed 's/-/_/g')

    echo ""
    echo ">>> 환경: $env"

    if [[ "$env" == *"expert"* ]]; then
        W2_WEIGHTS_CURRENT="$W2_WEIGHTS_EXPERT"
        echo "    [expert] weights: $W2_WEIGHTS_CURRENT"
    else
        W2_WEIGHTS_CURRENT="$W2_WEIGHTS_BASE"
    fi

    config_path="configs/offline/pogo_multi/${domain}/${task}_${ALGORITHM}.yaml"

    if [ ! -f "$config_path" ]; then
        mkdir -p "configs/offline/pogo_multi/${domain}/"
        base_config="configs/offline/pogo_multi/halfcheetah/medium_v2_${ALGORITHM}.yaml"
        if [ ! -f "$base_config" ]; then
            echo "[에러] 기본 config 없음: $base_config"
            exit 1
        fi
        cp "$base_config" "$config_path"
        sed -i "s/env: halfcheetah-medium-v2/env: ${env}/" "$config_path"
    fi

    sed -i "s/w2_weights: \[.*\]/w2_weights: ${W2_WEIGHTS_CURRENT}/" "$config_path"

    log_dir="results/${ALGORITHM}/${env_dir}/seed_${SEED}/logs"
    checkpoint_dir="results/${ALGORITHM}/${env_dir}/seed_${SEED}/checkpoints"
    mkdir -p "$log_dir" "$checkpoint_dir"

    log_file="${log_dir}/${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log"
    echo "    Log: $log_file"

    nohup python -u -m algorithms.offline.pogo_multi_main \
        --config_path "$config_path" \
        --max_timesteps $MAX_TIMESTEPS \
        --eval_freq $EVAL_FREQ \
        --n_episodes $N_EPISODES \
        --device $DEVICE \
        --seed $SEED \
        > "$log_file" 2>&1 &

    TRAIN_PID=$!
    wait $TRAIN_PID

    if [ $? -ne 0 ]; then
        echo "[실패] $env - 로그: $log_file"
        exit 1
    fi
    echo "    완료: $env"
done

echo ""
echo "=========================================="
echo "모든 IQL 학습 완료"
echo "=========================================="
