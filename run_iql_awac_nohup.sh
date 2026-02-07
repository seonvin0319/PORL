#!/bin/bash
# nohup으로 IQL + AWAC 학습 실행 (wandb: config 기본 활성화)
# 터미널 종료 후에도 백그라운드에서 계속 실행됨

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_common.sh"
cd "$SCRIPT_DIR"
if [ -n "$(which conda 2>/dev/null)" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate off_rl_gpu 2>/dev/null || conda activate base
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
NOHUP_LOG="${LOG_DIR}/nohup_$(date +%Y%m%d_%H%M%S).out"

echo "=========================================="
echo "IQL + AWAC nohup 실행"
echo "=========================================="
echo "로그: $NOHUP_LOG"
echo ""

nohup ./run_iql_awac_all_envs_seed0.sh > "$NOHUP_LOG" 2>&1 &
PID=$!

echo "PID: $PID"
echo ""
echo "모니터링: tail -f $NOHUP_LOG"
echo "프로세스 확인: ps -p $PID"
echo "=========================================="
