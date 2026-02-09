# POGO Policy implementations for JAX/Flax
# Policy Protocol: sample_actions, deterministic_actions, get_mean_std (for Gaussian)
# GaussianMLP, TanhGaussianMLP, StochasticMLP, DeterministicMLP, FQLFlowPolicy

from typing import Tuple, Optional, Callable

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax import linen as nn

from .rebrac import pytorch_init, uniform_init, identity


# ============================================================================
# 공통 유틸리티 함수
# ============================================================================

def _get_act(name: Optional[str]) -> Callable:
    """Activation 함수 이름을 함수로 변환"""
    if name is None or name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unknown activation: {name}")

def build_mlp_layers(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_hiddens: int,
    layernorm: bool = False,
    final_activation=None,
    activation=nn.relu,
) -> list:
    """공통 MLP 레이어 빌딩 함수
    
    Args:
        input_dim: 입력 차원 (kernel init scale 계산용, 실제 Dense shape 추론은 Flax가 자동 수행)
        hidden_dim: Hidden layer 차원
        output_dim: 출력 차원
        n_hiddens: Hidden layer 개수
        layernorm: LayerNorm 적용 여부
        final_activation: 마지막 레이어 activation (None이면 적용 안 함)
        activation: Hidden layer activation (기본값: nn.relu, FQL에서는 nn.gelu 사용)
    
    Returns:
        레이어 리스트
    """
    s_d, h_d = input_dim, hidden_dim
    layers = [
        nn.Dense(
            hidden_dim,
            kernel_init=pytorch_init(s_d),
            bias_init=nn.initializers.constant(0.1),
        ),
        activation,
        nn.LayerNorm() if layernorm else identity,
    ]
    for _ in range(n_hiddens - 1):
        layers += [
            nn.Dense(
                hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            activation,
            nn.LayerNorm() if layernorm else identity,
        ]
    layers += [
        nn.Dense(
            output_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        ),
    ]
    if final_activation is not None:
        layers.append(final_activation)
    return layers


# ============================================================================
# FQL Flow Policy 관련 클래스들
# ============================================================================

class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.
    
    DeterministicMLP와 유사한 구조를 사용하지만,
    입력이 (observations, actions, times)이고 출력이 velocity vector입니다.
    
    Attributes:
        hidden_dim: Hidden layer dimension.
        action_dim: Action dimension.
        n_hiddens: Number of hidden layers.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        activation: Activation function name ("relu" or "gelu", 기본값: "relu")
    """
    hidden_dim: int = 256
    action_dim: int = 1
    n_hiddens: int = 2
    layer_norm: bool = False
    encoder: nn.Module = None
    activation: str = "relu"  # "relu" or "gelu"

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).
        
        Args:
            observations: Observations [B, D] or [B*K, D] (flattened)
            actions: Actions [B, A] or [B*K, A] (flattened)
            times: Times (optional) [B, 1] or [B*K, 1] (flattened)
            is_encoded: Whether the observations are already encoded.
        
        Returns:
            velocity: [B, A] or [B*K, A] (same shape as actions)
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        # Activation 함수 선택 (모듈 필드에서 가져옴)
        if self.activation == "gelu":
            act = nn.gelu
        elif self.activation == "relu":
            act = nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # 직접 레이어 구성
        h = inputs
        for _ in range(self.n_hiddens):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h.shape[-1]),
                bias_init=nn.initializers.constant(0.1),
            )(h)
            if self.layer_norm:
                h = nn.LayerNorm()(h)
            h = act(h)
        
        # Output layer (velocity, activation 없음)
        v = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(h)
        
        return v


class FQLFlowPolicy(nn.Module):
    """FQL Flow Policy: actor_bc_flow (multi-step) + actor_onestep_flow (one-step)
    
    FQL의 원래 구조를 그대로 사용:
    - actor_bc_flow: multi-step flow matching (times를 받아서 velocity field 예측)
    - actor_onestep_flow: one-step policy (빠른 샘플링)
    """
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 2
    encoder: nn.Module = None
    
    # Policy 타입 속성 (클래스 변수)
    is_gaussian: bool = False
    is_stochastic: bool = True
    
    def setup(self):
        """Setup actor_bc_flow and actor_onestep_flow"""
        # actor_bc_flow: multi-step flow matching (times 필요)
        self.actor_bc_flow = ActorVectorField(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            n_hiddens=self.n_hiddens,
            layer_norm=self.layernorm,
            encoder=self.encoder,
        )
        
        # actor_onestep_flow: one-step policy (times=None)
        self.actor_onestep_flow = ActorVectorField(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            n_hiddens=self.n_hiddens,
            layer_norm=self.layernorm,
            encoder=self.encoder,
        )
    
    def __call__(self, state: jax.Array, noise: jax.Array, times: jax.Array = None, use_onestep: bool = True) -> jax.Array:
        """
        Forward pass: actor_onestep_flow 또는 actor_bc_flow 사용
        
        Args:
            state: [B, state_dim] or [B*K, state_dim] (flattened)
            noise: [B, action_dim] or [B*K, action_dim] (flattened)
            times: None (one-step flow) or [B, 1] or [B*K, 1] (for actor_bc_flow)
            use_onestep: If True, use actor_onestep_flow; else use actor_bc_flow
        
        Returns:
            action: [B, action_dim] or [B*K, action_dim] (same shape as noise)
        """
        # For initialization: if times is provided, also call actor_bc_flow to init it
        if times is not None:
            # Initialize actor_bc_flow by calling it (even though we don't use the result)
            _ = self.actor_bc_flow(state, noise, times, is_encoded=False)
        
        if use_onestep:
            # Use actor_onestep_flow for forward pass
            return self.actor_onestep_flow(state, noise, times=None, is_encoded=False)
        else:
            # Use actor_bc_flow (requires times)
            if times is None:
                raise ValueError("actor_bc_flow requires times parameter")
            return self.actor_bc_flow(state, noise, times, is_encoded=False)
    
    def compute_flow_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        noise: jax.Array,
        flow_steps: int = 10,
    ) -> jax.Array:
        """Compute actions from BC flow using Euler method (multi-step)
        
        FQL의 compute_flow_actions와 동일한 구조
        
        Args:
            params: Full FQLFlowPolicy parameters
            state: [B, state_dim]
            noise: [B, action_dim]
            flow_steps: Number of Euler steps
        
        Returns:
            actions: [B, action_dim]
        """
        actions = noise
        # Euler method
        for i in range(flow_steps):
            t = jnp.full((*state.shape[:-1], 1), i / flow_steps)
            # Use full module apply with use_onestep=False to call actor_bc_flow
            vels = self.apply(
                params,
                state,
                actions,
                times=t,
                use_onestep=False,
            )
            actions = actions + vels / flow_steps
        
        actions = jnp.clip(actions, -1, 1) * self.max_action
        return actions
    
    def sample_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
    ) -> jax.Array:
        """Sample K actions using actor_onestep_flow
        
        Args:
            params: Full FQLFlowPolicy parameters
            state: [B, state_dim]
            key: PRNG key
            K: Number of samples per state
        
        Returns:
            actions: [B, K, action_dim]
        """
        B = state.shape[0]
        state_dim = state.shape[-1]
        noise = jax.random.normal(key, (B, K, self.action_dim))
        
        # Tile state to [B, K, state_dim] and flatten to [B*K, state_dim]
        state_tiled = jnp.expand_dims(state, axis=1)  # [B, 1, state_dim]
        state_tiled = jnp.tile(state_tiled, (1, K, 1))  # [B, K, state_dim]
        state_flat = state_tiled.reshape(B * K, state_dim)  # [B*K, state_dim]
        noise_flat = noise.reshape(B * K, self.action_dim)  # [B*K, action_dim]
        
        # Use full module apply with use_onestep=True to call actor_onestep_flow
        actions_flat = self.apply(
            params,
            state_flat,
            noise_flat,
            times=None,
            use_onestep=True,
        )
        
        # Reshape back to [B, K, action_dim]
        actions = actions_flat.reshape(B, K, self.action_dim)
        actions = jnp.clip(actions, -1, 1) * self.max_action
        return actions
    
    def deterministic_actions(self, params: FrozenDict, state: jax.Array) -> jax.Array:
        """Deterministic action: use z=0 noise
        
        Args:
            params: Full FQLFlowPolicy parameters
            state: [B, state_dim]
        
        Returns:
            actions: [B, action_dim]
        """
        noise = jnp.zeros((state.shape[0], self.action_dim), dtype=state.dtype)
        
        # Use full module apply with use_onestep=True to call actor_onestep_flow
        actions = self.apply(
            params,
            state,
            noise,
            times=None,
            use_onestep=True,
        )
        return jnp.clip(actions, -1, 1) * self.max_action


# ============================================================================
# 일반 POGO Policy 클래스들
# ============================================================================

def build_base_network(
    input_dim: int,
    hidden_dim: int,
    n_hiddens: int,
    layernorm: bool = False,
) -> nn.Module:
    """Base network builder for Gaussian/TanhGaussian policies
    
    Args:
        input_dim: Input dimension (kernel init scale 계산용)
        hidden_dim: Hidden layer dimension
        n_hiddens: Number of hidden layers
        layernorm: Whether to apply layer normalization
    
    Returns:
        Base network Sequential module (output: hidden_dim)
    """
    layers = []
    # First layer
    layers.append(
        nn.Dense(
            hidden_dim,
            kernel_init=pytorch_init(input_dim),
            bias_init=nn.initializers.constant(0.1),
        )
    )
    layers.append(nn.relu)
    if layernorm:
        layers.append(nn.LayerNorm())
    
    # Hidden layers
    for _ in range(n_hiddens - 1):
        layers.append(
            nn.Dense(
                hidden_dim,
                kernel_init=pytorch_init(hidden_dim),
                bias_init=nn.initializers.constant(0.1),
            )
        )
        layers.append(nn.relu)
        if layernorm:
            layers.append(nn.LayerNorm())
    
    return nn.Sequential(layers)


class GaussianMLP(nn.Module):
    """POGO Gaussian Actor: mean에 tanh가 적용된 상태에서 샘플링
    mean = tanh(...) * max_action (bounded mean)
    그 mean, std로 Gaussian 샘플링
    Closed form W2 distance 사용 가능
    """
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    
    # Policy 타입 속성 (클래스 변수)
    is_gaussian: bool = True
    is_stochastic: bool = True

    @nn.compact
    def __call__(self, state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Forward pass: state -> (mean, log_std)
        Returns:
            mean: [B, action_dim] (tanh applied, bounded)
            log_std: [B, action_dim]
        """
        # Base network 생성 (별도 함수 사용)
        base_net = build_base_network(
            input_dim=state.shape[-1],
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            layernorm=self.layernorm,
        )
        
        # Mean head (with tanh)
        mean_head = nn.Sequential([
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
            nn.tanh,
        ])
        
        # Log std head (learnable parameter)
        log_std = self.param(
            'log_std',
            nn.initializers.zeros,
            (self.action_dim,),
        )
        log_std = jnp.broadcast_to(log_std, (state.shape[0], self.action_dim))
        
        x = base_net(state)
        mean = mean_head(x) * self.max_action
        
        # Clamp log_std
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def deterministic_actions(self, params: FrozenDict, state: jax.Array) -> jax.Array:
        """Deterministic action: mean"""
        mean, _ = self.apply(params, state)
        return mean

    def sample_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
    ) -> jax.Array:
        """Sample K actions: [B, K, action_dim]
        mean에 tanh가 적용된 상태에서 샘플링
        """
        mean, log_std = self.apply(params, state)  # [B, action_dim]
        std = jnp.exp(log_std)  # [B, action_dim]
        
        B = state.shape[0]
        # Sample noise: [B, K, action_dim]
        noise = jax.random.normal(key, (B, K, self.action_dim))
        
        # Expand mean and std: [B, K, action_dim]
        mean_expanded = jnp.expand_dims(mean, axis=1)  # [B, 1, action_dim]
        mean_expanded = jnp.tile(mean_expanded, (1, K, 1))  # [B, K, action_dim]
        std_expanded = jnp.expand_dims(std, axis=1)  # [B, 1, action_dim]
        std_expanded = jnp.tile(std_expanded, (1, K, 1))  # [B, K, action_dim]
        
        # Sample actions (mean은 이미 bounded)
        actions = mean_expanded + std_expanded * noise
        actions = jnp.clip(actions, -self.max_action, self.max_action)
        return actions
    
    def get_mean_std(
        self,
        params: FrozenDict,
        state: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Get mean and std: [B, action_dim]"""
        mean, log_std = self.apply(params, state)
        std = jnp.exp(log_std)
        return mean, std


class TanhGaussianMLP(nn.Module):
    """POGO TanhGaussian Actor: unbounded Gaussian에서 샘플링 후 tanh 적용
    mean = ... (unbounded)
    mean, std로 Gaussian 샘플링 (unbounded space)
    그 다음 tanh를 적용하여 bounded로 만듦
    Closed form W2 사용 불가 (Sinkhorn 사용)
    """
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    
    # Policy 타입 속성 (클래스 변수)
    is_gaussian: bool = False
    is_stochastic: bool = True

    @nn.compact
    def __call__(self, state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Forward pass: state -> (mean, log_std)
        Returns:
            mean: [B, action_dim] (unbounded)
            log_std: [B, action_dim]
        """
        # Base network 생성 (별도 함수 사용)
        base_net = build_base_network(
            input_dim=state.shape[-1],
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            layernorm=self.layernorm,
        )
        
        # Mean head (no tanh)
        mean_head = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )
        
        # Log std head (learnable parameter)
        log_std = self.param(
            'log_std',
            nn.initializers.zeros,
            (self.action_dim,),
        )
        log_std = jnp.broadcast_to(log_std, (state.shape[0], self.action_dim))
        
        x = base_net(state)
        mean = mean_head(x)  # Unbounded
        
        # Clamp log_std
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def deterministic_actions(self, params: FrozenDict, state: jax.Array) -> jax.Array:
        """Deterministic action: tanh(mean) * max_action"""
        mean, _ = self.apply(params, state)
        return jnp.tanh(mean) * self.max_action

    def sample_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
    ) -> jax.Array:
        """Sample K actions: [B, K, action_dim]
        unbounded Gaussian에서 샘플링 후 tanh 적용
        """
        mean, log_std = self.apply(params, state)  # [B, action_dim]
        std = jnp.exp(log_std)  # [B, action_dim]
        
        B = state.shape[0]
        # Sample noise: [B, K, action_dim]
        noise = jax.random.normal(key, (B, K, self.action_dim))
        
        # Expand mean and std: [B, K, action_dim]
        mean_expanded = jnp.expand_dims(mean, axis=1)  # [B, 1, action_dim]
        mean_expanded = jnp.tile(mean_expanded, (1, K, 1))  # [B, K, action_dim]
        std_expanded = jnp.expand_dims(std, axis=1)  # [B, 1, action_dim]
        std_expanded = jnp.tile(std_expanded, (1, K, 1))  # [B, K, action_dim]
        
        # Sample actions from unbounded Gaussian
        actions_unbounded = mean_expanded + std_expanded * noise
        
        # Apply tanh to make bounded
        actions = jnp.tanh(actions_unbounded) * self.max_action
        return actions
    
    def get_mean_std(
        self,
        params: FrozenDict,
        state: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Get mean and std: [B, action_dim]"""
        mean, log_std = self.apply(params, state)
        std = jnp.exp(log_std)
        return mean, std


class StochasticMLP(nn.Module):
    """POGO Stochastic Actor: state + z -> action
    
    일반적인 stochastic policy
    FQL의 flow policy로도 사용 가능 (FQLFlowPolicy 대신 사용)
    """
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 2
    activation: str = "relu"  # "relu" or "gelu"
    
    # Policy 타입 속성 (클래스 변수)
    is_gaussian: bool = False
    is_stochastic: bool = True

    @nn.compact
    def __call__(self, state: jax.Array, z: jax.Array) -> jax.Array:
        """
        Forward pass: state + z -> action
        z must be provided (for deterministic, use z=0)
        
        Args:
            state: State array [B, state_dim]
            z: Noise array [B, action_dim]
        """
        # Activation 함수 선택 (모듈 필드에서 가져옴)
        if self.activation == "gelu":
            act = nn.gelu
        elif self.activation == "relu":
            act = nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Concatenate state and z
        h = jnp.concatenate([state, z], axis=-1)
        
        # 직접 레이어 구성
        for _ in range(self.n_hiddens):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h.shape[-1]),
                bias_init=nn.initializers.constant(0.1),
            )(h)
            if self.layernorm:
                h = nn.LayerNorm()(h)
            h = act(h)
        
        # Output layer (tanh activation)
        a = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(h)
        a = jnp.tanh(a) * self.max_action
        return a

    def deterministic_actions(self, params: FrozenDict, state: jax.Array) -> jax.Array:
        """Deterministic action: z = 0"""
        z = jnp.zeros((state.shape[0], self.action_dim), dtype=state.dtype)
        return self.apply(params, state, z)

    def sample_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
    ) -> jax.Array:
        """Sample K actions: [B, K, action_dim]"""
        B = state.shape[0]
        # Sample z: [B, K, action_dim]
        z = jax.random.normal(key, (B, K, self.action_dim))
        
        # Expand state: [B, K, state_dim]
        state_expanded = jnp.expand_dims(state, axis=1)  # [B, 1, state_dim]
        state_expanded = jnp.tile(state_expanded, (1, K, 1))  # [B, K, state_dim]
        
        # Flatten for batch processing: [B*K, state_dim], [B*K, action_dim]
        state_flat = state_expanded.reshape(B * K, -1)
        z_flat = z.reshape(B * K, self.action_dim)
        
        # Forward pass: [B*K, action_dim]
        actions_flat = self.apply(params, state_flat, z_flat)
        
        # Reshape back: [B, K, action_dim]
        actions = actions_flat.reshape(B, K, self.action_dim)
        return actions


class DeterministicMLP(nn.Module):
    """POGO Deterministic Actor: state -> action"""
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 2
    
    # Policy 타입 속성 (클래스 변수)
    is_gaussian: bool = False
    is_stochastic: bool = False

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        # 공통 MLP 레이어 빌딩 함수 사용
        layers = build_mlp_layers(
            input_dim=state.shape[-1],
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            n_hiddens=self.n_hiddens,
            layernorm=self.layernorm,
            final_activation=nn.tanh,
        )
        net = nn.Sequential(layers)
        actions = net(state) * self.max_action
        return actions

    def deterministic_actions(self, params: FrozenDict, state: jax.Array) -> jax.Array:
        """Deterministic action (same as forward)"""
        return self.apply(params, state)

    def sample_actions(
        self,
        params: FrozenDict,
        state: jax.Array,
        key: jax.random.PRNGKey,
        K: int = 1,
    ) -> jax.Array:
        """Sample K actions: deterministic, so just repeat"""
        actions = self.apply(params, state)  # [B, action_dim]
        actions = jnp.expand_dims(actions, axis=1)  # [B, 1, action_dim]
        actions = jnp.tile(actions, (1, K, 1))  # [B, K, action_dim]
        return actions
