# MLP Building Utilities and Initialization
# 레이어 생성 + init + seed 생성기 공통화

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def build_mlp_layers(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 2,
    output_dim: Optional[int] = None,
    layernorm: bool = False,
    activation: type = nn.ReLU,
):
    """MLP 레이어 빌딩 함수.
    
    - Policy용 (output_dim=None): base Linear layers만 반환. activation은 _forward_base에서 F.relu로 명시.
    - Critic용 (output_dim 설정): Sequential용 레이어 리스트 반환. activation은 nn.Module 클래스로 받음.
    
    Args:
        input_dim: 입력 차원
        hidden_dim: Hidden layer 차원
        n_hiddens: Hidden layer 개수
        output_dim: 출력 차원 (None이면 policy용 base layers만)
        layernorm: LayerNorm 사용 여부 (critic용)
        activation: nn.Module 클래스 (기본 nn.ReLU). F.relu 같은 함수는 사용 불가.
    
    Returns:
        output_dim이 None이면: (base_layers, head_input_dim)
        output_dim이 있으면: 완전한 레이어 리스트 (activation, layernorm 포함)
    """
    layers = []
    current_dim = input_dim

    for _ in range(n_hiddens):
        layers.append(nn.Linear(current_dim, hidden_dim))
        current_dim = hidden_dim

    if output_dim is None:
        return layers, current_dim

    # Critic용: Sequential에 넣을 레이어 (activation은 nn.Module 클래스로 인스턴스화)
    result_layers = []
    for layer in layers:
        result_layers.append(layer)
        result_layers.append(activation())
        if layernorm:
            result_layers.append(nn.LayerNorm(hidden_dim))

    result_layers.append(nn.Linear(hidden_dim, output_dim))
    return result_layers


def build_mlp(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 3,
    output_dim: int = 1,
    layernorm: bool = False,
    activation: type = nn.ReLU,
) -> list:
    """MLP 레이어 빌딩 유틸리티 (Critic 및 기타 네트워크용)
    build_mlp_layers를 사용하여 구현
    
    Args:
        input_dim: 입력 차원
        hidden_dim: Hidden layer 차원
        n_hiddens: Hidden layer 개수
        output_dim: 출력 차원 (기본값: 1)
        layernorm: LayerNorm 사용 여부
        activation: 활성화 함수 클래스 (기본값: nn.ReLU)
    
    Returns:
        레이어 리스트
    """
    return build_mlp_layers(
        input_dim, hidden_dim, n_hiddens, output_dim, layernorm, activation
    )


def create_generator(device: str, seed: int) -> torch.Generator:
    """Seed 생성기 공통화 - 모든 actor에서 사용"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def init_module_weights(
    module: torch.nn.Sequential,
    init_type: str = "xavier",
    orthogonal_gain: float = 1.0,
    last_layer_gain: float = 1e-2,
) -> None:
    """초기화 정책 중앙화 - orthogonal/xavier/kaiming 선택을 mlp.py에서만 처리
    
    Args:
        module: 초기화할 Sequential 모듈
        init_type: "orthogonal", "xavier", "kaiming" 중 선택
        orthogonal_gain: Orthogonal init의 gain (내부 레이어)
        last_layer_gain: 마지막 레이어의 gain
    """
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    
    for i, layer in enumerate(linear_layers):
        if i == len(linear_layers) - 1:
            # Last layer
            if init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=last_layer_gain)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=last_layer_gain)
            nn.init.constant_(layer.bias, 0.0)
        else:
            # Inner layers
            if init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=orthogonal_gain)
                nn.init.constant_(layer.bias, 0.0)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0.1)
            else:  # xavier
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    """Extend tensor along dimension and repeat"""
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)
