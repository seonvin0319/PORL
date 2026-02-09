#!/usr/bin/env python3
"""
원래 rebrac config 파일들을 읽어서 pogo_multi rebrac config로 변환
- 원래 config의 모든 하이퍼파라미터 유지
- w2 관련 설정만 추가
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

# W2 관련 기본 설정
W2_DEFAULT_CONFIG = {
    "w2_weights": [10.0, 10.0],
    "num_actors": 3,
    "actor_configs": [
        {"type": "deterministic"},  # Actor0
        {"type": "gaussian"},        # Actor1
        {"type": "gaussian"},        # Actor2
    ],
    "sinkhorn_K": 4,
    "sinkhorn_blur": 0.05,
    "sinkhorn_backend": "auto",
}

def load_yaml(filepath: Path) -> Dict[str, Any]:
    """YAML 파일 로드 (간단한 파서)"""
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # key: value 형식 파싱
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 리스트 파싱 (예: [10.0, 10.0])
                if value.startswith('[') and value.endswith(']'):
                    value = [float(x.strip()) for x in value[1:-1].split(',')]
                # 불린 파싱
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # 숫자 파싱
                else:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # 문자열로 유지
                
                config[key] = value
    return config

def save_yaml(filepath: Path, data: Dict[str, Any]):
    """YAML 파일 저장 (주석 포함)"""
    with open(filepath, 'w') as f:
        f.write("# POGO Multi-Actor with ReBRAC algorithm (JAX)\n")
        f.write("# ReBRAC의 구조를 그대로 사용하되, Actor만 multi-actor\n")
        f.write("# 원래 rebrac config에서 w2 설정만 추가됨\n\n")
        
        # dataset_name, train_seed, eval_seed 먼저
        if "dataset_name" in data:
            f.write(f"dataset_name: {data['dataset_name']}\n")
        if "train_seed" in data:
            f.write(f"train_seed: {data['train_seed']}\n")
        if "eval_seed" in data:
            f.write(f"eval_seed: {data['eval_seed']}\n")
        f.write("\n")
        
        # POGO Multi-Actor 설정
        f.write("# POGO Multi-Actor 설정\n")
        f.write("# w2_weights: Actor1부터의 가중치 리스트 (Actor0는 W2 penalty 없음)\n")
        f.write(f"w2_weights: {data.get('w2_weights', W2_DEFAULT_CONFIG['w2_weights'])}\n")
        f.write(f"num_actors: {data.get('num_actors', W2_DEFAULT_CONFIG['num_actors'])}\n")
        f.write("# actor_configs: 각 actor의 타입 지정\n")
        f.write("#   - \"gaussian\": Gaussian policy (mean에 tanh 적용된 상태에서 샘플링, closed form W2 사용)\n")
        f.write("#   - \"tanh_gaussian\": TanhGaussian policy (unbounded Gaussian에서 샘플링 후 tanh 적용, Sinkhorn 사용)\n")
        f.write("#   - \"stochastic\": Stochastic policy (Sinkhorn distance 사용)\n")
        f.write("#   - \"deterministic\": Deterministic policy (L2 distance 사용)\n")
        f.write("actor_configs:\n")
        for actor_config in data.get('actor_configs', W2_DEFAULT_CONFIG['actor_configs']):
            f.write(f"  - type: {actor_config['type']}\n")
        f.write("\n")
        
        # Sinkhorn 설정
        f.write("# Sinkhorn 설정 (Actor1+용)\n")
        f.write(f"sinkhorn_K: {data.get('sinkhorn_K', W2_DEFAULT_CONFIG['sinkhorn_K'])}\n")
        f.write(f"sinkhorn_blur: {data.get('sinkhorn_blur', W2_DEFAULT_CONFIG['sinkhorn_blur'])}\n")
        f.write(f"sinkhorn_backend: \"{data.get('sinkhorn_backend', W2_DEFAULT_CONFIG['sinkhorn_backend'])}\"\n")
        f.write("\n")
        
        # ReBRAC 파라미터 (원래 config에서 가져온 값들)
        f.write("# ReBRAC 파라미터 (원래 config 값 유지)\n")
        rebrac_params = [
            "actor_learning_rate", "critic_learning_rate", "hidden_dim",
            "actor_n_hiddens", "critic_n_hiddens", "gamma", "tau",
            "actor_bc_coef", "critic_bc_coef", "actor_ln", "critic_ln",
            "policy_noise", "noise_clip", "policy_freq", "normalize_q",
        ]
        for param in rebrac_params:
            if param in data:
                value = data[param]
                if isinstance(value, bool):
                    f.write(f"{param}: {str(value).lower()}\n")
                else:
                    f.write(f"{param}: {value}\n")
        f.write("\n")
        
        # Training params
        f.write("# Training params\n")
        training_params = ["batch_size", "num_epochs", "num_updates_on_epoch", 
                          "normalize_reward", "normalize_states"]
        for param in training_params:
            if param in data:
                value = data[param]
                if isinstance(value, bool):
                    f.write(f"{param}: {str(value).lower()}\n")
                else:
                    f.write(f"{param}: {value}\n")
        f.write("\n")
        
        # Evaluation params
        f.write("# Evaluation params\n")
        if "eval_episodes" in data:
            f.write(f"eval_episodes: {data['eval_episodes']}\n")
        # eval_freq는 BaseConfig에 이미 있으므로 명시하지 않아도 됨 (기본값: 5000)
        # 필요시 config 파일에서 직접 설정 가능
        f.write("\n")
        
        # Wandb (project는 PORL로, group과 name은 pogo-multi-rebrac로 설정)
        f.write("# Wandb\n")
        f.write("project: PORL\n")
        f.write("group: pogo-multi-rebrac\n")
        dataset_name = data.get("dataset_name", "").replace("-", "_")
        f.write(f"name: pogo-multi-rebrac-{dataset_name}\n")

def convert_config(original_path: Path, target_path: Path):
    """원래 config를 읽어서 pogo_multi config로 변환"""
    print(f"Converting: {original_path} -> {target_path}")
    
    # 원래 config 로드
    original_config = load_yaml(original_path)
    
    # W2 설정 추가
    converted_config = original_config.copy()
    converted_config.update(W2_DEFAULT_CONFIG)
    
    # target 디렉토리 생성
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 변환된 config 저장
    save_yaml(target_path, converted_config)
    print(f"  ✓ Saved: {target_path}")

def main():
    base_dir = Path(__file__).parent
    rebrac_dir = base_dir / "configs" / "offline" / "rebrac"
    pogo_multi_dir = base_dir / "configs" / "offline" / "pogo_multi"
    
    # 모든 rebrac config 파일 찾기
    rebrac_configs = list(rebrac_dir.rglob("*.yaml"))
    
    print(f"Found {len(rebrac_configs)} rebrac config files")
    print("=" * 60)
    
    for rebrac_config_path in rebrac_configs:
        # 상대 경로 계산
        relative_path = rebrac_config_path.relative_to(rebrac_dir)
        
        # 파일명 변환: {task}.yaml -> {task}_rebrac.yaml
        # 예: medium_v2.yaml -> medium_v2_rebrac.yaml
        task_name = relative_path.stem
        new_filename = f"{task_name}_rebrac.yaml"
        
        # pogo_multi 경로 생성
        pogo_multi_config_path = pogo_multi_dir / relative_path.parent / new_filename
        
        # 변환 실행
        convert_config(rebrac_config_path, pogo_multi_config_path)
    
    print("=" * 60)
    print(f"✓ All {len(rebrac_configs)} config files converted!")

if __name__ == "__main__":
    main()
