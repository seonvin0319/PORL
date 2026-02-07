#!/bin/bash
# 단일 환경 ReBRAC 실행 스크립트

set -e

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl

cd /home/choi/PORL
source $(conda info --base)/etc/profile.d/conda.sh
conda activate offrl

# wandb API key 확인 및 설정
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f ~/.config/wandb/settings ]; then
        WANDB_API_KEY=$(grep -i "api_key" ~/.config/wandb/settings | grep -v "^#" | cut -d'=' -f2 | tr -d ' ' | head -1)
    fi
    
    if [ -z "$WANDB_API_KEY" ]; then
        echo "⚠️  WANDB_API_KEY가 설정되지 않았습니다."
        exit 1
    fi
fi

export WANDB_API_KEY

# 사용법 확인
if [ $# -lt 1 ]; then
    echo "사용법: $0 <config_path> [seed]"
    echo "예시: $0 configs/offline/pogo_multi/halfcheetah/medium_v2_rebrac.yaml 0"
    exit 1
fi

CONFIG_PATH=$1
SEED=${2:-0}

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config 파일을 찾을 수 없습니다: $CONFIG_PATH"
    exit 1
fi

# dataset_name 추출
dataset_name=$(grep "^dataset_name:" "$CONFIG_PATH" | awk '{print $2}')
domain=$(echo "$CONFIG_PATH" | cut -d'/' -f4)
task=$(basename "$CONFIG_PATH" | sed 's/_rebrac.yaml//')
env_dir="${domain}_${task}"

echo "=========================================="
echo "ReBRAC 단일 환경 실행"
echo "=========================================="
echo "환경: $dataset_name"
echo "Config: $CONFIG_PATH"
echo "Seed: $SEED"
echo "=========================================="

log_dir="results/rebrac/${env_dir}/seed_${SEED}/logs"
checkpoint_dir="results/rebrac/${env_dir}/seed_${SEED}/checkpoints"
mkdir -p "$log_dir" "$checkpoint_dir"

log_file="${log_dir}/rebrac_$(date +%Y%m%d_%H%M%S).log"
echo "로그 파일: $log_file"

# 실행
python -u -m algorithms.offline.pogo_multi_jax \
    --config_path "$CONFIG_PATH" \
    --train_seed $SEED \
    > "$log_file" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 완료: $dataset_name"
else
    echo "❌ 실패: $dataset_name - 로그: $log_file"
    exit 1
fi
