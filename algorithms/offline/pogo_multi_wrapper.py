# POGO Multi-Actor Wrapper
# 각 알고리즘의 구조를 그대로 사용하되, Actor만 multi-actor로 교체
# - Critic, V function 등은 각 알고리즘의 구조 그대로 사용
# - Actor만 multi-actor (Actor0: W2, Actor1+: Sinkhorn)

import copy
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from .pogo_policies import BaseActor, DeterministicMLP, Policy, StochasticMLP

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


def _per_state_sinkhorn(
    policy: Policy,
    ref: Policy,
    states: torch.Tensor,
    K: int = 4,
    blur: float = 0.05,
    p: int = 2,
    backend: str = "tensorized",
    sinkhorn_loss=None,
    seed: Optional[int] = None,
):
    if sinkhorn_loss is None:
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend)
    a = policy.sample_actions(states, K=K, seed=None if seed is None else seed + 0)
    with torch.no_grad():
        b_detached = ref.sample_actions(states, K=K, seed=None if seed is None else seed + 10000).detach()
    b = b_detached
    loss = sinkhorn_loss(a, b)
    return loss.mean() if loss.dim() > 0 else loss


class MultiActorWrapper:
    """
    기존 알고리즘의 trainer를 감싸서 multi-actor 지원 추가
    
    사용법:
        base_trainer = ImplicitQLearning(...)  # 기존 IQL trainer
        multi_trainer = MultiActorWrapper(
            base_trainer=base_trainer,
            w2_weights=[10.0, 10.0, 10.0],
            actor_configs=[{}, {}, {}],
            ...
        )
        # base_trainer.train() 대신 multi_trainer.train() 사용
    """

    def __init__(
        self,
        base_trainer: Any,  # IQL, CQL, TD3_BC 등의 trainer
        w2_weights: List[float],
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor_configs: Optional[List[dict]] = None,
        sinkhorn_K: int = 4,
        sinkhorn_blur: float = 0.05,
        sinkhorn_backend: str = "tensorized",
        seed: Optional[int] = None,
        lr: float = 3e-4,
    ):
        self.base_trainer = base_trainer
        self.num_actors = len(w2_weights)
        self.w2_weights = w2_weights
        self.seed = seed
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = base_trainer.device

        if actor_configs is None:
            actor_configs = [{} for _ in range(self.num_actors)]
        assert len(actor_configs) == self.num_actors

        # Multi-actor 생성
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []
        self.actor_is_stochastic = []

        for i in range(self.num_actors):
            config = actor_configs[i]
            deterministic = config.get("deterministic", False)
            if deterministic:
                actor_cls = DeterministicMLP
                is_stochastic = False
            else:
                actor_cls = StochasticMLP
                is_stochastic = True
            actor = actor_cls(state_dim, action_dim, max_action).to(self.device)
            actor_target = copy.deepcopy(actor)
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=lr))
            self.actor_is_stochastic.append(is_stochastic)

        # Base trainer의 actor를 첫 번째 actor로 교체 (호환성)
        self.base_trainer.actor = self.actors[0]
        self.base_trainer.actor_optimizer = self.actor_optimizers[0]

        # Sinkhorn loss
        self._sinkhorn_loss = SamplesLoss(
            loss="sinkhorn", p=2, blur=sinkhorn_blur, backend=sinkhorn_backend
        )
        self.sinkhorn_K = sinkhorn_K
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_backend = sinkhorn_backend

        # Base trainer의 total_it 추적
        self.total_it = 0

    def _get_base_actor_loss_fn(self, trainer_type: str):
        """각 알고리즘의 actor loss 계산 함수 반환"""
        if hasattr(self.base_trainer, "_update_policy"):
            # IQL, AWAC 등
            return self.base_trainer._update_policy
        elif hasattr(self.base_trainer, "train"):
            # CQL, TD3_BC 등은 train 내부에서 actor loss 계산
            return None
        return None

    def _compute_actor_loss_iql_style(
        self, i: int, observations, actions, adv, log_dict
    ):
        """IQL 스타일의 actor loss (exp_adv * bc_loss)"""
        actor_i = self.actors[i]
        w2_weight_i = self.w2_weights[i]

        # Base IQL의 advantage 기반 weight
        exp_adv = torch.exp(self.base_trainer.beta * adv.detach()).clamp(max=100.0)

        if i == 0:
            # Actor0: W2 to dataset
            policy_out = actor_i(observations)
            if isinstance(policy_out, torch.distributions.Distribution):
                bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            elif torch.is_tensor(policy_out):
                bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
            else:
                raise NotImplementedError

            # IQL style: exp_adv * bc_loss + w2 distance
            w2_i = ((policy_out if torch.is_tensor(policy_out) else policy_out.mean) - actions) ** 2
            w2_i = w2_i.sum(dim=1).mean()
            actor_loss_i = torch.mean(exp_adv * bc_losses) + w2_weight_i * w2_i
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = self.actors[i - 1]
            pi_i = actor_i.sample_actions(observations, K=1, seed=self.total_it * 1000 + i)[:, 0, :]

            is_i_stoch = self.actor_is_stochastic[i]
            is_ref_stoch = self.actor_is_stochastic[i - 1]

            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i,
                    ref_actor,
                    observations,
                    K=self.sinkhorn_K,
                    blur=self.sinkhorn_blur,
                    p=2,
                    backend=self.sinkhorn_backend,
                    sinkhorn_loss=self._sinkhorn_loss,
                    seed=self.total_it * 1000 + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(observations)
                w2_i = ((pi_i - ref_action) ** 2).mean()

            # Q 값 계산 (base trainer의 Q 사용)
            if hasattr(self.base_trainer, "qf"):
                Q_i = self.base_trainer.qf(observations, pi_i)
            elif hasattr(self.base_trainer, "critic_1"):
                Q_i = self.base_trainer.critic_1(observations, pi_i)
            else:
                raise NotImplementedError("Cannot find Q function")

            actor_loss_i = -Q_i.mean() + w2_weight_i * w2_i

        return actor_loss_i, w2_i if i > 0 else None

    def _compute_actor_loss_td3_style(
        self, i: int, observations, actions, log_dict
    ):
        """TD3_BC 스타일의 actor loss (-lambda * Q + BC)"""
        actor_i = self.actors[i]
        w2_weight_i = self.w2_weights[i]

        pi_i = actor_i.sample_actions(observations, K=1, seed=self.total_it * 1000 + i)[:, 0, :]

        # Q 값 계산
        if hasattr(self.base_trainer, "critic_1"):
            q = self.base_trainer.critic_1(observations, pi_i)
            lmbda = self.base_trainer.alpha / q.abs().mean().detach()
        else:
            raise NotImplementedError("TD3_BC style requires critic_1")

        if i == 0:
            # Actor0: W2 to dataset
            w2_i = ((pi_i - actions) ** 2).mean()
            actor_loss_i = -lmbda * q.mean() + w2_weight_i * w2_i
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = self.actors[i - 1]
            is_i_stoch = self.actor_is_stochastic[i]
            is_ref_stoch = self.actor_is_stochastic[i - 1]

            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i,
                    ref_actor,
                    observations,
                    K=self.sinkhorn_K,
                    blur=self.sinkhorn_blur,
                    p=2,
                    backend=self.sinkhorn_backend,
                    sinkhorn_loss=self._sinkhorn_loss,
                    seed=self.total_it * 1000 + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(observations)
                w2_i = ((pi_i - ref_action) ** 2).mean()

            actor_loss_i = -lmbda * q.mean() + w2_weight_i * w2_i

        return actor_loss_i, w2_i if i > 0 else None

    def _compute_actor_loss_cql_style(
        self, i: int, observations, actions, log_dict
    ):
        """CQL 스타일의 actor loss"""
        actor_i = self.actors[i]
        w2_weight_i = self.w2_weights[i]

        new_actions, log_pi = actor_i(observations)

        # CQL의 alpha 계산
        if hasattr(self.base_trainer, "_alpha_and_alpha_loss"):
            alpha, _ = self.base_trainer._alpha_and_alpha_loss(observations, log_pi)
        else:
            alpha = self.base_trainer.alpha if hasattr(self.base_trainer, "alpha") else 1.0

        if i == 0:
            # Actor0: W2 to dataset
            w2_i = ((new_actions - actions) ** 2).mean()
            # CQL policy loss + W2
            if hasattr(self.base_trainer, "_policy_loss"):
                policy_loss_base = self.base_trainer._policy_loss(
                    observations, actions, new_actions, alpha, log_pi
                )
            else:
                # Fallback: Q 기반
                q1 = self.base_trainer.critic_1(observations, new_actions)
                policy_loss_base = (alpha * log_pi - q1).mean()
            actor_loss_i = policy_loss_base + w2_weight_i * w2_i
        else:
            # Actor1+: Sinkhorn to previous actor
            ref_actor = self.actors[i - 1]
            is_i_stoch = self.actor_is_stochastic[i]
            is_ref_stoch = self.actor_is_stochastic[i - 1]

            if is_i_stoch and is_ref_stoch:
                w2_i = _per_state_sinkhorn(
                    actor_i,
                    ref_actor,
                    observations,
                    K=self.sinkhorn_K,
                    blur=self.sinkhorn_blur,
                    p=2,
                    backend=self.sinkhorn_backend,
                    sinkhorn_loss=self._sinkhorn_loss,
                    seed=self.total_it * 1000 + 100 + i,
                )
            else:
                with torch.no_grad():
                    ref_action = ref_actor.deterministic_actions(observations)
                w2_i = ((new_actions - ref_action) ** 2).mean()

            # CQL policy loss + Sinkhorn
            if hasattr(self.base_trainer, "_policy_loss"):
                policy_loss_base = self.base_trainer._policy_loss(
                    observations, actions, new_actions, alpha, log_pi
                )
            else:
                q1 = self.base_trainer.critic_1(observations, new_actions)
                policy_loss_base = (alpha * log_pi - q1).mean()
            actor_loss_i = policy_loss_base + w2_weight_i * w2_i

        return actor_loss_i, w2_i if i > 0 else None

    def train(self, batch) -> Dict[str, float]:
        """Base trainer의 train을 호출하되, actor 부분만 multi-actor로 교체"""
        self.total_it += 1

        # Base trainer의 train 호출 (critic, V 등 업데이트)
        # 하지만 actor 업데이트는 우리가 직접 처리
        log_dict = {}

        # 알고리즘 타입 감지
        trainer_type = type(self.base_trainer).__name__

        if "IQL" in trainer_type or "ImplicitQLearning" in trainer_type:
            # IQL: V, Q 업데이트는 base trainer가, Actor는 우리가
            observations, actions, rewards, next_observations, dones = batch

            # Base trainer의 V, Q 업데이트
            with torch.no_grad():
                next_v = self.base_trainer.vf(next_observations)
            adv = self.base_trainer._update_v(observations, actions, log_dict)
            rewards = rewards.squeeze(dim=-1)
            dones = dones.squeeze(dim=-1)
            self.base_trainer._update_q(next_v, observations, actions, rewards, dones, log_dict)

            # Multi-actor 업데이트
            for i in range(self.num_actors):
                actor_loss_i, w2_i = self._compute_actor_loss_iql_style(i, observations, actions, adv, log_dict)
                opt = self.actor_optimizers[i]
                opt.zero_grad()
                actor_loss_i.backward()
                opt.step()

                log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
                if w2_i is not None:
                    log_dict[f"w2_{i}_distance"] = float(w2_i.item())

            # Target update
            for actor, actor_target in zip(self.actors, self.actor_targets):
                for p, tp in zip(actor.parameters(), actor_target.parameters()):
                    tp.data.copy_(self.base_trainer.tau * p.data + (1 - self.base_trainer.tau) * tp.data)

        elif "TD3" in trainer_type or "TD3_BC" in trainer_type:
            # TD3_BC: Critic 업데이트는 base trainer가, Actor는 우리가
            state, action, reward, next_state, done = batch
            not_done = 1.0 - done

            # Base trainer의 critic 업데이트 (내부 메서드 직접 호출)
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.base_trainer.policy_noise).clamp(
                    -self.base_trainer.noise_clip, self.base_trainer.noise_clip
                )
                next_action = (self.actor_targets[0](next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )
                target_q1 = self.base_trainer.critic_1_target(next_state, next_action)
                target_q2 = self.base_trainer.critic_2_target(next_state, next_action)
                target_q = reward + not_done * self.base_trainer.discount * torch.min(target_q1, target_q2)

            current_q1 = self.base_trainer.critic_1(state, action)
            current_q2 = self.base_trainer.critic_2(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.base_trainer.critic_1_optimizer.zero_grad()
            self.base_trainer.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.base_trainer.critic_1_optimizer.step()
            self.base_trainer.critic_2_optimizer.step()
            log_dict["critic_loss"] = float(critic_loss.item())

            # Multi-actor 업데이트 (policy_freq마다)
            if self.total_it % self.base_trainer.policy_freq == 0:
                for i in range(self.num_actors):
                    actor_loss_i, w2_i = self._compute_actor_loss_td3_style(i, state, action, log_dict)
                    opt = self.actor_optimizers[i]
                    opt.zero_grad()
                    actor_loss_i.backward()
                    opt.step()

                    log_dict[f"actor_{i}_loss"] = float(actor_loss_i.item())
                    if w2_i is not None:
                        log_dict[f"w2_{i}_distance"] = float(w2_i.item())

                # Target update
                for actor, actor_target in zip(self.actors, self.actor_targets):
                    for p, tp in zip(actor.parameters(), actor_target.parameters()):
                        tp.data.copy_(self.base_trainer.tau * p.data + (1 - self.base_trainer.tau) * tp.data)
                for p, tp in zip(self.base_trainer.critic_1.parameters(), self.base_trainer.critic_1_target.parameters()):
                    tp.data.copy_(self.base_trainer.tau * p.data + (1 - self.base_trainer.tau) * tp.data)
                for p, tp in zip(self.base_trainer.critic_2.parameters(), self.base_trainer.critic_2_target.parameters()):
                    tp.data.copy_(self.base_trainer.tau * p.data + (1 - self.base_trainer.tau) * tp.data)

        elif "CQL" in trainer_type or "ContinuousCQL" in trainer_type:
            # CQL: Q, alpha 업데이트는 base trainer가, Actor는 우리가
            observations, actions, rewards, next_observations, dones = batch

            # Base trainer의 Q loss 계산 (내부 메서드 사용)
            # 실제로는 base_trainer.train()을 호출하되 actor 부분만 override하는 게 나을 수도...
            # 일단 간단하게 구현
            raise NotImplementedError("CQL multi-actor는 아직 구현 중")

        else:
            raise NotImplementedError(f"Unsupported trainer type: {trainer_type}")

        return log_dict

    def select_action(
        self,
        state,
        deterministic: bool = True,
        actor_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Action 선택 (base trainer의 act와 호환)"""
        if actor_idx is None:
            actor_idx = self.num_actors - 1
        actor = self.actors[actor_idx]
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        if deterministic:
            action = actor.deterministic_actions(state_tensor)
        else:
            action = actor.sample_actions(state_tensor, K=1, seed=seed)[:, 0, :]
        return action.detach().cpu().numpy().flatten()

    def act(self, state, device: str = "cpu"):
        """Base trainer와 호환되는 act 메서드"""
        return self.select_action(state, deterministic=True, actor_idx=self.num_actors - 1)

    @property
    def actor(self):
        """Base trainer 호환성: 첫 번째 actor 반환"""
        return self.actors[0]
