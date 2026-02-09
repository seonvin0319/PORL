#!/usr/bin/env python3
"""FQL 하이퍼파라미터 업데이트 스크립트
- lr = 3e-4 (actor_learning_rate, critic_learning_rate)
- batch_size = 256
- hidden_dim = 512
- actor_n_hiddens = 4
- critic_n_hiddens = 4
- tau = 0.005 (이미 맞는지 확인)
"""

import os
import yaml
from pathlib import Path

# 업데이트할 값들
UPDATES = {
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "batch_size": 256,
    "hidden_dim": 512,
    "actor_n_hiddens": 4,
    "critic_n_hiddens": 4,
    "tau": 0.005,
}

def update_config_file(config_path: Path):
    """단일 config 파일 업데이트"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    updated = False
    for key, value in UPDATES.items():
        if key in data:
            old_value = data[key]
            if old_value != value:
                data[key] = value
                updated = True
                print(f"  {key}: {old_value} -> {value}")
        else:
            data[key] = value
            updated = True
            print(f"  {key}: (추가) -> {value}")
    
    if updated:
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    return False

def main():
    """모든 FQL config 파일 업데이트"""
    config_dir = Path("configs/offline/pogo_multi")
    
    # 모든 fql.yaml 파일 찾기
    fql_configs = list(config_dir.rglob("*_fql.yaml"))
    
    print(f"총 {len(fql_configs)}개의 FQL config 파일을 찾았습니다.")
    print(f"업데이트 내용:")
    for key, value in UPDATES.items():
        print(f"  {key}: {value}")
    print()
    
    updated_count = 0
    for config_path in sorted(fql_configs):
        print(f"업데이트 중: {config_path}")
        if update_config_file(config_path):
            updated_count += 1
            print(f"  ✓ 업데이트 완료")
        else:
            print(f"  - 변경사항 없음")
        print()
    
    print(f"총 {updated_count}개 파일이 업데이트되었습니다.")

if __name__ == "__main__":
    main()
