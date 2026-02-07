#!/bin/bash
# 공통 환경변수 (run 스크립트에서 source로 사용)
# wandb: config use_wandb 기본 true. 비활성화: --no_wandb 옵션

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
