"""Proprioceptive Motion Context Sharing skrl MAPPO model.

이 파일은 motion-context task용 actor/critic network와 custom MAPPO update를 담고 있다.
이 task에서는 learned latent mode를 학습하지 않고, 환경이 직접 계산한
communication message z=[signed EE motion 3D, proprioceptive motion context 3D]를 공유한다.

최종 communication baseline:
    none:
        own_obs만 보는 MLP baseline이다. trainer/play 코드를 단순하게 유지하기 위해
        CLI alias no_intent와 연결된다.

    motion_context:
        actor branch는 agent-specific MLP로 유지하고, 상대 팔의 6D motion context
        descriptor만 partner message encoder를 통해 policy에 주입한다.

파일 구조:
    1. Small helpers
       - MLP.
    2. MotionContextPolicyModel
       - actor branch와 partner motion-context encoder.
    3. CriticValueModel
       - centralized critic value network.
    4. MotionContextMAPPO
       - skrl MAPPO update without auxiliary communication losses.
    5. build_motion_context_mappo_agent
       - env/cfg에서 dimension을 읽고 left/right model과 memory를 생성.
"""

from __future__ import annotations

import copy
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR


def _build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """actor, critic, auxiliary head에서 쓰는 작은 ELU MLP를 만든다."""

    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_sizes]
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ELU())
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class MotionContextPolicyModel(GaussianMixin, Model):
    """Policy with partner z=[signed EE motion, proprioceptive motion context]."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes: list[int],
        own_observation_dim: int,
        motion_intent_dim: int,
        motion_context_dim: int,
        communication_mode: str = "motion_context",
        agent_id: str = "",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        partner_intent_embed_dim: int = 32,
        communication_feature_dim: int | None = None,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.communication_mode = str(communication_mode)
        self.agent_id = str(agent_id)
        self.own_observation_dim = int(own_observation_dim)
        self.motion_intent_dim = int(motion_intent_dim)
        self.motion_context_dim = int(motion_context_dim)
        default_message_dim = self.motion_intent_dim + self.motion_context_dim
        self.communication_feature_dim = int(communication_feature_dim if communication_feature_dim is not None else default_message_dim)
        self.intent_feature_dim = self.communication_feature_dim
        # Every ablation keeps the same fixed-width actor slot. ``none``
        # nevertheless bypasses the partner encoder entirely.
        self.uses_context = self.communication_mode != "none" and self.communication_feature_dim > 0
        self.partner_intent_embed_dim = max(int(partner_intent_embed_dim), 1)

        # actor branch는 agent-specific이다. partner message, contact sensor,
        # reward, success, object height는 이 encoder에 들어가지 않는다.
        # 이름은 기존 checkpoint 호환성을 위해 own_backbone으로 유지한다.
        self.own_backbone = _build_mlp(self.own_observation_dim, hidden_sizes, hidden_sizes[-1])
        self.actor_feature_dim = int(hidden_sizes[-1])
        self.policy_feature_dim = self.actor_feature_dim
        if self.num_actions < 2:
            raise ValueError(f"Motion-context policy expects arm actions plus gripper action, got {self.num_actions}")
        self.partner_intent_encoder = None
        if self.uses_context:
            self.partner_intent_encoder = nn.Sequential(
                nn.Linear(self.intent_feature_dim, self.partner_intent_embed_dim),
                nn.ELU(),
                nn.Linear(self.partner_intent_embed_dim, self.partner_intent_embed_dim),
                nn.ELU(),
            )
            self._zero_partner_intent_encoder_biases()
        self.actor_action_input_dim = self.policy_feature_dim
        if self.uses_context:
            self.actor_action_input_dim += self.partner_intent_embed_dim
        self.arm_action_head = nn.Linear(self.actor_action_input_dim, self.num_actions - 1)
        # gripper timing은 local proprioception에 강하게 의존하므로 arm action과 head를 분리한다.
        # own_obs layout: target_delta_b=22:25, closure=26:27, ee_lin_vel_b=27:30.
        self.gripper_head_input_dim = self.actor_action_input_dim + 7
        self.gripper_action_head = nn.Sequential(
            nn.Linear(self.gripper_head_input_dim, hidden_sizes[-1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[-1], 1),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std), device=self.device)
        )

    def _zero_partner_intent_encoder_biases(self) -> None:
        """zero partner intent가 zero embedding으로 유지되게 한다."""

        if self.partner_intent_encoder is None:
            return
        with torch.no_grad():
            for module in self.partner_intent_encoder:
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()

    def _split_states(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """최종 dimension 규약에 따라 actor input을 own_obs와 partner intent로 나눈다."""

        own = states[..., : self.own_observation_dim]
        partner = states[..., self.own_observation_dim : self.own_observation_dim + self.intent_feature_dim]
        if partner.shape[-1] != self.intent_feature_dim:
            partner = torch.zeros((*states.shape[:-1], self.intent_feature_dim), device=states.device, dtype=states.dtype)
        return own, partner

    def _format_motion_intent(self, motion_intent, hidden: torch.Tensor) -> torch.Tensor:
        """deterministic m_t를 검증하고 policy batch 크기에 맞게 broadcast한다."""

        if motion_intent is None:
            return torch.zeros((hidden.shape[0], self.motion_intent_dim), device=hidden.device, dtype=hidden.dtype)
        motion = motion_intent.to(device=hidden.device, dtype=hidden.dtype)
        if motion.dim() == 3:
            motion = motion[:, -1]
        if motion.dim() != 2:
            motion = motion.reshape(-1, self.motion_intent_dim)
        if motion.shape[0] != hidden.shape[0]:
            if hidden.shape[0] % motion.shape[0] != 0:
                return torch.zeros((hidden.shape[0], self.motion_intent_dim), device=hidden.device, dtype=hidden.dtype)
            motion = motion.repeat_interleave(hidden.shape[0] // motion.shape[0], dim=0)
        if motion.shape[-1] != self.motion_intent_dim:
            raise ValueError(f"motion_i must have shape [..., {self.motion_intent_dim}], got {tuple(motion.shape)}")
        return torch.clamp(motion, -1.0, 1.0)

    def compute(self, inputs, role=""):
        """Gaussian mean action과 motion-context metadata를 계산한다.

        actor graph의 핵심 branch:
        1. own_obs -> own_backbone (local representation)
        2. fixed communication slot -> partner_intent_encoder
        3. action head는 local representation + encoded partner context를 본다.
        """

        states = inputs["states"]
        if states.dim() != 2:
            raise ValueError(f"Paper feed-forward policy expects states [batch, obs], got {tuple(states.shape)}")

        own_states, partner_intent = self._split_states(states)
        hidden = self.own_backbone(own_states)
        if self.uses_context:
            actor_partner_intent = partner_intent.detach()
        else:
            actor_partner_intent = torch.zeros(
                (hidden.shape[0], self.communication_feature_dim),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        motion_intent = self._format_motion_intent(inputs.get("motion_intent"), hidden)
        if self.uses_context and actor_partner_intent.shape[-1] >= self.motion_intent_dim + self.motion_context_dim:
            motion_context = actor_partner_intent[
                ...,
                self.motion_intent_dim : self.motion_intent_dim + self.motion_context_dim,
            ]
        else:
            motion_context = torch.zeros((hidden.shape[0], self.motion_context_dim), device=hidden.device, dtype=hidden.dtype)
        shared_intent = actor_partner_intent
        if motion_intent.shape[-1] != self.motion_intent_dim:
            raise RuntimeError("Invalid motion intent dimension")
        if shared_intent.shape[-1] != self.communication_feature_dim:
            raise RuntimeError("Invalid shared communication dimension")
        if actor_partner_intent.shape[-1] != self.communication_feature_dim:
            raise RuntimeError("Invalid partner communication dimension")

        # partner message는 환경이 계산한 상대 팔의 communication descriptor이며,
        # actor graph로 역전파하지 않는다.
        if self.uses_context:
            if self.partner_intent_encoder is None:
                raise RuntimeError("share_intent requires partner_intent_encoder")
            partner_intent_feat = self.partner_intent_encoder(actor_partner_intent)
            actor_input = torch.cat([hidden, partner_intent_feat], dim=-1)
        else:
            partner_intent_feat = torch.zeros((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
            actor_input = hidden
        if actor_input.shape[-1] != self.actor_action_input_dim:
            raise RuntimeError("Invalid actor input dimension")
        own_for_gripper = own_states
        if own_for_gripper.shape[0] != actor_input.shape[0]:
            raise RuntimeError(
                f"Invalid gripper feature batch: own={own_for_gripper.shape[0]}, actor={actor_input.shape[0]}"
            )
        if own_for_gripper.shape[-1] < 30:
            raise RuntimeError(f"Expected paper own_obs >= 30D, got {own_for_gripper.shape[-1]}")

        # 최종 own_obs layout:
        # 0:3   ee_pos_b
        # 3:7   ee_quat_b
        # 7:8   gripper_pos
        # 8:15  joint_pos
        # 15:22 joint_vel
        # 22:25 target_delta_b
        # 25:26 target_quat_error
        # 26:27 closure
        # 27:30 ee_lin_vel_b
        target_delta_b = own_for_gripper[..., 22:25]
        closure = own_for_gripper[..., 26:27]
        ee_lin_vel_b = own_for_gripper[..., 27:30]
        gripper_parts = [actor_input, closure, target_delta_b, ee_lin_vel_b]
        gripper_features = torch.cat(gripper_parts, dim=-1)
        if gripper_features.shape[-1] != self.gripper_head_input_dim:
            raise RuntimeError("Invalid gripper head input dimension")
        arm_actions = self.arm_action_head(actor_input)
        gripper_action = self.gripper_action_head(gripper_features)
        mean_actions = torch.cat([arm_actions, gripper_action], dim=-1)
        if mean_actions.shape[-1] != self.num_actions:
            raise RuntimeError("Invalid action dimension after separate gripper head")
        outputs = {
            "mean_actions": mean_actions,
            "intent": shared_intent.reshape(hidden.shape[0], 1, self.intent_feature_dim),
            "motion_intent": motion_intent,
            "motion_context": motion_context,
            "communication_message": shared_intent.reshape(hidden.shape[0], 1, self.communication_feature_dim),
            "partner_message_norm": actor_partner_intent.detach().norm(dim=-1).mean(),
            "partner_message_feat_norm": partner_intent_feat.detach().norm(dim=-1).mean(),
            "partner_intent_norm": actor_partner_intent.detach().norm(dim=-1).mean(),
            "partner_intent_feat_norm": partner_intent_feat.detach().norm(dim=-1).mean(),
            "partner_motion_context_mean": motion_context.detach().mean(dim=0),
            "communication_mode": self.communication_mode,
        }
        return mean_actions, self.log_std_parameter, outputs


class CriticValueModel(DeterministicMixin, Model):
    """task_logic.critic_state()가 만든 centralized critic state를 보는 value network."""

    def __init__(self, observation_space, action_space, device, hidden_sizes: list[int], clip_actions: bool = False):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
        )
        self.value_net = _build_mlp(self.num_observations, hidden_sizes, 1)

    def compute(self, inputs, role=""):
        value = self.value_net(inputs["states"])
        return value, {}


class MotionContextMAPPO(MAPPO):
    """Paper MAPPO variant using directly computed proprioceptive context."""

    def __init__(self, *args, **kwargs):
        raw_models = kwargs.get("models", None)
        if raw_models is None and len(args) >= 2:
            raw_models = args[1]
        self._value_models = {}
        if isinstance(raw_models, dict):
            for uid, modules in raw_models.items():
                if isinstance(modules, dict) and "value" in modules:
                    self._value_models[uid] = modules["value"]

        super().__init__(*args, **kwargs)
        self._motion_context_state_provider = None
        self._motion_context_state_loader = None

        if not self._value_models:
            for uid in self.possible_agents:
                candidate = None
                try:
                    candidate = self.models[uid]["value"]
                except Exception:
                    pass
                if candidate is None:
                    try:
                        candidate = self.values[uid]
                    except Exception:
                        pass
                if hasattr(candidate, "act"):
                    self._value_models[uid] = candidate

        self._current_motion_intents = {uid: None for uid in self.possible_agents}
        self._context_debug = {uid: self._empty_context_debug() for uid in self.possible_agents}

    @staticmethod
    def _motion_context_sidecar_path(path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_motion_context{ext or '.pt'}"

    def set_motion_context_state_provider(self, provider) -> None:
        """Register a callback returning env-side motion-context normalization state."""

        self._motion_context_state_provider = provider

    def set_motion_context_state_loader(self, loader) -> None:
        """Register a callback restoring env-side motion-context normalization state."""

        self._motion_context_state_loader = loader

    def _save_motion_context_state(self, checkpoint_path: str) -> None:
        if not callable(self._motion_context_state_provider):
            return
        try:
            state = self._motion_context_state_provider()
        except Exception as exc:
            print(f"[WARN] Failed to collect motion-context state for checkpoint: {exc}")
            return
        if not state:
            return
        sidecar_path = self._motion_context_sidecar_path(checkpoint_path)
        try:
            torch.save({"motion_context": state}, sidecar_path)
        except Exception as exc:
            print(f"[WARN] Failed to save motion-context state: path={sidecar_path} error={exc}")

    def _load_motion_context_state(self, checkpoint_path: str) -> None:
        if not callable(self._motion_context_state_loader):
            return
        sidecar_path = self._motion_context_sidecar_path(checkpoint_path)
        if not os.path.exists(sidecar_path):
            print(f"[WARN] Motion-context sidecar not found: {sidecar_path}")
            return
        try:
            data = torch.load(sidecar_path, map_location=self.device)
            state = data.get("motion_context", data) if isinstance(data, dict) else data
            self._motion_context_state_loader(state)
            print(f"[INFO] Loaded motion-context state from: {sidecar_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load motion-context state: path={sidecar_path} error={exc}")

    def save(self, path: str) -> None:
        super().save(path)
        self._save_motion_context_state(path)

    def load(self, path: str) -> None:
        super().load(path)
        self._load_motion_context_state(path)

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        best_pending = bool(
            self.checkpoint_best_modules.get("modules")
            and not self.checkpoint_best_modules.get("saved", True)
        )
        super().write_checkpoint(timestep, timesteps)

        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        if timestep is not None:
            self._save_motion_context_state(os.path.join(checkpoint_dir, f"agent_{timestep}.pt"))
        if best_pending:
            self._save_motion_context_state(os.path.join(checkpoint_dir, "best_agent.pt"))

    @staticmethod
    def _empty_context_debug():
        return {
            "motion": [0.0, 0.0, 0.0],
            "motion_context": [0.0, 0.0, 1.0],
        }

    def set_current_motion_intents(self, motion_intents: dict[str, torch.Tensor]) -> None:
        """policy action 선택 전에 env에서 deterministic m_t를 받아 저장한다."""

        self._current_motion_intents = {
            uid: motion_intents.get(uid).detach() if motion_intents.get(uid) is not None else None
            for uid in self.possible_agents
        }

    def get_motion_context_debug(self):
        """trainer log/trace metadata에 쓸 최신 context statistics를 반환한다."""

        return {uid: dict(values) for uid, values in self._context_debug.items()}

    def get_intent_warmup_timesteps(self) -> int:
        return 0

    def load_intent_heads(self, path: str, strict: bool = False) -> None:
        """Checkpoint 파일에서 motion-context policy module을 로드한다."""

        checkpoint = torch.load(path, map_location=self.device)
        modules = checkpoint.get(
            "motion_context_modules",
            checkpoint.get(
                "paper_intent_modules",
                checkpoint.get("motion_mode_modules", checkpoint.get("timing_intent_modules", checkpoint)),
            ),
        )
        for uid in self.possible_agents:
            state_dict = modules.get(uid) if isinstance(modules, dict) else None
            if state_dict is not None:
                self.policies[uid].load_state_dict(state_dict, strict=strict)

    def save_intent_heads(self, path: str, metadata: dict | None = None, metrics: dict | None = None) -> None:
        """Offline inspection 또는 transfer용 policy module을 저장한다."""

        torch.save(
            {
                "metadata": metadata or {},
                "metrics": metrics or {},
                "motion_context_modules": {uid: self.policies[uid].state_dict() for uid in self.possible_agents},
            },
            path,
        )

    def freeze_intent_heads(self) -> None:
        """다른 policy parameter는 학습 가능하게 두고 partner-context encoder만 freeze한다."""

        for uid in self.possible_agents:
            for name in ("partner_intent_encoder",):
                module = getattr(self.policies[uid], name, None)
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad_(False)

    def _preprocess_policy_states(self, uid: str, states: torch.Tensor, train: bool = False) -> torch.Tensor:
        """선택적 sequence shape를 보존하면서 skrl state normalization을 적용한다."""

        if states.dim() == 3:
            batch_size, sequence_length, obs_dim = states.shape
            flat_states = states.reshape(batch_size * sequence_length, obs_dim)
            flat_states = self._state_preprocessor[uid](flat_states, train=train)
            return flat_states.reshape(batch_size, sequence_length, obs_dim)
        return self._state_preprocessor[uid](states, train=train)

    def _update_context_debug(self, uid: str, outputs: dict[str, torch.Tensor], losses: dict[str, float] | None = None) -> None:
        """console log와 JSON trace metadata에 쓰는 agent별 context stats를 갱신한다."""

        del losses
        with torch.no_grad():
            motion = outputs.get("motion_intent")
            context = outputs.get("motion_context")
            motion_mean = [0.0, 0.0, 0.0]
            if isinstance(motion, torch.Tensor) and motion.numel() > 0:
                motion_mean = [float(v) for v in motion.detach().float().mean(dim=0).cpu().tolist()]
            context_mean = [0.0, 0.0, 1.0]
            if isinstance(context, torch.Tensor) and context.numel() > 0:
                context_mean = [float(v) for v in context.detach().float().mean(dim=0).cpu().tolist()]
            previous = self._context_debug.get(uid, self._empty_context_debug())
            policy = self.policies[uid]
            partner_intent_norm = outputs.get("partner_intent_norm")
            partner_intent_feat_norm = outputs.get("partner_intent_feat_norm")
            partner_context_mean = outputs.get("partner_motion_context_mean")
            partner_message_norm = outputs.get("partner_message_norm", partner_intent_norm)
            partner_message_feat_norm = outputs.get("partner_message_feat_norm", partner_intent_feat_norm)
            self._context_debug[uid] = {
                "motion": motion_mean,
                "motion_context": context_mean,
                "partner_message_embed_dim": int(getattr(policy, "partner_intent_embed_dim", 0)),
                "partner_intent_embed_dim": int(getattr(policy, "partner_intent_embed_dim", 0)),
                "actor_action_input_dim": int(getattr(policy, "actor_action_input_dim", 0)),
                "gripper_head_input_dim": int(getattr(policy, "gripper_head_input_dim", 0)),
                "partner_message_norm": float(partner_message_norm.detach().float().cpu())
                if isinstance(partner_message_norm, torch.Tensor)
                else float(previous.get("partner_message_norm", previous.get("partner_intent_norm", 0.0))),
                "partner_message_feat_norm": float(partner_message_feat_norm.detach().float().cpu())
                if isinstance(partner_message_feat_norm, torch.Tensor)
                else float(previous.get("partner_message_feat_norm", previous.get("partner_intent_feat_norm", 0.0))),
                "partner_intent_norm": float(partner_intent_norm.detach().float().cpu())
                if isinstance(partner_intent_norm, torch.Tensor)
                else float(previous.get("partner_intent_norm", 0.0)),
                "partner_intent_feat_norm": float(partner_intent_feat_norm.detach().float().cpu())
                if isinstance(partner_intent_feat_norm, torch.Tensor)
                else float(previous.get("partner_intent_feat_norm", 0.0)),
                "partner_motion_context": [float(v) for v in partner_context_mean.detach().float().cpu().tolist()]
                if isinstance(partner_context_mean, torch.Tensor) and partner_context_mean.numel() > 0
                else previous.get("partner_motion_context", []),
            }

    def act(self, states, timestep: int, timesteps: int):
        """양쪽 policy를 실행하고 memory/trainer hook에 필요한 출력을 cache한다."""

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            data = []
            for uid in self.possible_agents:
                model_inputs = {
                    "states": self._preprocess_policy_states(uid, states[uid]),
                    "motion_intent": self._current_motion_intents.get(uid),
                }
                outputs = self.policies[uid].act(model_inputs, role="policy")
                self._update_context_debug(uid, outputs[2])
                data.append(outputs)

            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}
            self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ) -> None:
        """MAPPO transition tensors and the final shared state for bootstrapping."""

        super(MAPPO, self).record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memories:
            shared_states = infos["shared_states"]
            self._current_shared_next_states = infos["shared_next_states"]

            for uid in self.possible_agents:
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    values, _, _ = self._value_models[uid].act(
                        {"states": self._shared_state_preprocessor[uid](shared_states)}, role="value"
                    )
                    values = self._value_preprocessor[uid](values, inverse=True)

                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

                samples = {
                    "states": states[uid],
                    "actions": actions[uid],
                    "rewards": rewards[uid],
                    "terminated": terminated[uid],
                    "truncated": truncated[uid],
                    "log_prob": self._current_log_prob[uid],
                    "values": values,
                    "shared_states": shared_states,
                }
                self.memories[uid].add_samples(**samples)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Policy/value loss만 사용하는 motion-context MAPPO update."""

        def compute_gae(
            rewards,
            dones,
            values,
            bootstrap_values,
            discount_factor=0.99,
            lambda_coefficient=0.95,
        ):
            """skrl memory tensor를 이용한 generalized advantage estimation."""

            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else bootstrap_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            return returns, advantages

        sample_names = list(self._tensors_names)
        for name in ("terminated", "truncated"):
            if name not in sample_names:
                sample_names.append(name)
        update_data = {}
        stats = {}
        for uid in self.possible_agents:
            policy = self.policies[uid]
            value_model = self._value_models[uid]
            memory = self.memories[uid]

            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                value_model.train(False)
                last_values, _, _ = value_model.act(
                    {"states": self._shared_state_preprocessor[uid](self._current_shared_next_states.float())},
                    role="value",
                )
                value_model.train(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            stored_values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=stored_values,
                bootstrap_values=last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            memory.set_tensor_by_name("values", self._value_preprocessor[uid](stored_values, train=True))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)
            update_data[uid] = {
                "policy": policy,
                "value_model": value_model,
                "sampled_batches": memory.sample_all(names=sample_names, mini_batches=self._mini_batches[uid]),
            }
            stats[uid] = {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "value_loss": 0.0,
                "count": 0,
                "kl": [],
            }

        max_epochs = max(int(self._learning_epochs[uid]) for uid in self.possible_agents)
        max_batches = max(len(update_data[uid]["sampled_batches"]) for uid in self.possible_agents)
        for epoch in range(max_epochs):
            epoch_kl = {uid: [] for uid in self.possible_agents}
            skip_uid = {uid: False for uid in self.possible_agents}

            for batch_index in range(max_batches):
                active_uids = [
                    uid
                    for uid in self.possible_agents
                    if epoch < int(self._learning_epochs[uid])
                    and batch_index < len(update_data[uid]["sampled_batches"])
                    and not skip_uid[uid]
                ]
                if not active_uids:
                    continue

                # 어떤 backward도 시작하기 전에 모든 optimizer grad를 비운다.
                for uid in active_uids:
                    self.optimizers[uid].zero_grad()

                batch_results = {}

                for uid in active_uids:
                    policy = update_data[uid]["policy"]
                    value_model = update_data[uid]["value_model"]
                    batch = update_data[uid]["sampled_batches"][batch_index]
                    sampled = {name: value for name, value in zip(sample_names, batch)}

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        sampled_states = self._preprocess_policy_states(uid, sampled["states"], train=not epoch)
                        sampled_shared_states = self._shared_state_preprocessor[uid](
                            sampled["shared_states"], train=not epoch
                        )
                        _, next_log_prob, _ = policy.act(
                            {
                                "states": sampled_states,
                                "taken_actions": sampled["actions"],
                            },
                            role="policy",
                        )

                        with torch.no_grad():
                            ratio = next_log_prob - sampled["log_prob"]
                            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                            epoch_kl[uid].append(kl_divergence)

                        if self._kl_threshold[uid] and kl_divergence > self._kl_threshold[uid]:
                            skip_uid[uid] = True
                            continue

                        entropy_loss = (
                            -self._entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                            if self._entropy_loss_scale[uid]
                            else 0
                        )
                        ratio = torch.exp(next_log_prob - sampled["log_prob"])
                        surrogate = sampled["advantages"] * ratio
                        surrogate_clipped = sampled["advantages"] * torch.clip(
                            ratio, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid]
                        )
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                        predicted_values, _, _ = value_model.act({"states": sampled_shared_states}, role="value")
                        if self._clip_predicted_values:
                            predicted_values = sampled["values"] + torch.clip(
                                predicted_values - sampled["values"],
                                min=-self._value_clip[uid],
                                max=self._value_clip[uid],
                            )
                        value_loss = self._value_loss_scale[uid] * F.mse_loss(sampled["returns"], predicted_values)
                        total_loss = policy_loss + entropy_loss + value_loss

                    batch_results[uid] = {
                        "total_loss": total_loss,
                        "policy": policy,
                        "value_model": value_model,
                    }

                    uid_stats = stats[uid]
                    uid_stats["policy_loss"] += policy_loss.item()
                    uid_stats["value_loss"] += value_loss.item()
                    uid_stats["count"] += 1
                    if self._entropy_loss_scale[uid]:
                        uid_stats["entropy_loss"] += entropy_loss.item()

                if not batch_results:
                    continue

                stepped_uids = list(batch_results.keys())
                total_loss = sum(result["total_loss"] for result in batch_results.values())

                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    for uid in stepped_uids:
                        policy = batch_results[uid]["policy"]
                        value_model = batch_results[uid]["value_model"]
                        policy.reduce_parameters()
                        if policy is not value_model:
                            value_model.reduce_parameters()

                if not stepped_uids:
                    continue

                for uid in stepped_uids:
                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.optimizers[uid])
                        policy = update_data[uid]["policy"]
                        value_model = update_data[uid]["value_model"]
                        params = list(policy.parameters()) if policy is value_model else list(
                            itertools.chain(policy.parameters(), value_model.parameters())
                        )
                        nn.utils.clip_grad_norm_(params, self._grad_norm_clip[uid])
                    self.scaler.step(self.optimizers[uid])

                self.scaler.update()

            for uid in self.possible_agents:
                if epoch < int(self._learning_epochs[uid]) and self._learning_rate_scheduler[uid]:
                    if isinstance(self.schedulers[uid], KLAdaptiveLR):
                        if epoch_kl[uid]:
                            kl = torch.stack(epoch_kl[uid]).mean()
                            if config.torch.is_distributed:
                                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                                kl /= config.torch.world_size
                            self.schedulers[uid].step(kl.item())
                    else:
                        self.schedulers[uid].step()

        for uid in self.possible_agents:
            uid_stats = stats[uid]
            denom = max(uid_stats["count"], 1)
            if self.tracking_data:
                self.track_data(f"Loss / Policy loss ({uid})", uid_stats["policy_loss"] / denom)
                self.track_data(f"Loss / Value loss ({uid})", uid_stats["value_loss"] / denom)
                if self._entropy_loss_scale[uid]:
                    self.track_data(f"Loss / Entropy loss ({uid})", uid_stats["entropy_loss"] / denom)

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])


def _resolve_component(name):
    """압축 YAML 문자열을 skrl component class로 해석한다."""

    if name in (None, "", False):
        return None
    if name == "RunningStandardScaler":
        return RunningStandardScaler
    if name == "KLAdaptiveLR":
        return KLAdaptiveLR
    raise ValueError(f"Unsupported skrl component reference: {name}")


def _reward_shaper_from_scale(scale):
    """skrl용 선택적 reward scaling callback을 반환한다."""

    if scale == 1.0:
        return None

    def _reward_shaper(rewards, *args, **kwargs):
        return rewards * scale

    return _reward_shaper


def _unwrap_env(env):
    """skrl/gym wrapper 아래의 실제 Isaac Lab task env를 찾는다."""

    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "intent_horizon") and hasattr(current, "possible_agents"):
            return current
        next_env = getattr(current, "_env", None)
        if next_env is None:
            next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return env


def build_motion_context_mappo_agent(env, experiment_cfg: dict) -> MotionContextMAPPO:
    """Policy, critic, memory, motion-context MAPPO agent를 구성한다."""

    cfg = copy.deepcopy(experiment_cfg)
    task_env = _unwrap_env(env)
    observation_spaces = task_env.observation_spaces
    action_spaces = task_env.action_spaces
    state_spaces = task_env.state_spaces
    possible_agents = task_env.possible_agents
    hidden_sizes = cfg["models"]["policy"]["network"][0]["layers"]
    communication_mode = str(getattr(task_env, "communication_mode", "motion_context"))
    intent_horizon = int(getattr(task_env, "intent_horizon", 1))
    if intent_horizon != 1:
        raise RuntimeError("Motion-context communication expects intent_horizon=1.")
    motion_intent_dim = int(getattr(task_env, "motion_intent_dim", cfg["models"]["policy"].get("motion_intent_dim", 3)))
    motion_context_dim = int(getattr(task_env, "motion_context_dim", cfg["models"]["policy"].get("motion_context_dim", 3)))
    communication_feature_dim = int(getattr(task_env, "communication_feature_dim", getattr(task_env, "intent_feature_dim", 0)))
    intent_feature_dim = communication_feature_dim
    own_observation_dim = int(
        getattr(task_env, "own_observation_dim", observation_spaces[possible_agents[0]].shape[0] - intent_feature_dim)
    )
    partner_intent_embed_dim = int(
        cfg["models"]["policy"].get(
            "partner_message_embed_dim",
            cfg["models"]["policy"].get("partner_intent_embed_dim", cfg["agent"].get("partner_intent_embed_dim", 32)),
        )
    )
    expected_actor_dim = own_observation_dim + intent_feature_dim
    actor_action_input_dim = hidden_sizes[-1]
    if communication_mode != "none" and communication_feature_dim > 0:
        actor_action_input_dim += partner_intent_embed_dim
    gripper_head_input_dim = actor_action_input_dim + 7
    for agent_id in possible_agents:
        obs_dim = observation_spaces[agent_id].shape[0] if hasattr(observation_spaces[agent_id], "shape") else int(observation_spaces[agent_id])
        if obs_dim != expected_actor_dim:
            raise RuntimeError(f"{communication_mode} actor observation dim must be {expected_actor_dim}, got {obs_dim}")

    print(f"[INFO] communication_mode={communication_mode}")
    print(f"[INFO] own_obs_dim={own_observation_dim}")
    print(f"[INFO] communication_feature_dim={communication_feature_dim}")
    print(f"[INFO] motion_dim={motion_intent_dim}")
    print(f"[INFO] context_dim={motion_context_dim}")
    print(f"[INFO] actor_input_dim={expected_actor_dim}")
    print(f"[INFO] actor_action_input_dim={actor_action_input_dim}")
    print(f"[INFO] gripper_head_input_dim={gripper_head_input_dim}")
    print("[INFO] discrete_context_classifier=False")
    print("[INFO] recurrent_intent_encoder=False")
    print(f"[INFO] partner_message_embed_dim={partner_intent_embed_dim}")
    print("[INFO] actor_backbone=agent_specific")
    print("[INFO] action_heads=agent_specific")
    print("[INFO] proposed z=[base_frame_signed_EE_motion_3D, linear_activity, angular_activity, action_smoothness]")
    print("[INFO] communication_mode=none is the no-share baseline.")

    models = {}
    memories = {}
    for agent_id in possible_agents:
        models[agent_id] = {
            "policy": MotionContextPolicyModel(
                observation_space=observation_spaces[agent_id],
                action_space=action_spaces[agent_id],
                device=env.device,
                hidden_sizes=hidden_sizes,
                own_observation_dim=own_observation_dim,
                motion_intent_dim=motion_intent_dim,
                motion_context_dim=motion_context_dim,
                communication_mode=communication_mode,
                agent_id=agent_id,
                partner_intent_embed_dim=partner_intent_embed_dim,
                communication_feature_dim=communication_feature_dim,
                clip_actions=cfg["models"]["policy"]["clip_actions"],
                clip_log_std=cfg["models"]["policy"]["clip_log_std"],
                min_log_std=cfg["models"]["policy"]["min_log_std"],
                max_log_std=cfg["models"]["policy"]["max_log_std"],
                initial_log_std=cfg["models"]["policy"]["initial_log_std"],
            ),
            "value": CriticValueModel(
                observation_space=state_spaces[agent_id],
                action_space=action_spaces[agent_id],
                device=env.device,
                hidden_sizes=cfg["models"]["value"]["network"][0]["layers"],
                clip_actions=cfg["models"]["value"]["clip_actions"],
            ),
        }
        memories[agent_id] = RandomMemory(
            memory_size=cfg["memory"]["memory_size"] if cfg["memory"]["memory_size"] > 0 else cfg["agent"]["rollouts"],
            num_envs=env.num_envs,
            device=env.device,
        )

    agent_cfg = copy.deepcopy(cfg["agent"])
    agent_cfg["learning_rate_scheduler"] = _resolve_component(agent_cfg.get("learning_rate_scheduler"))
    agent_cfg["state_preprocessor"] = _resolve_component(agent_cfg.get("state_preprocessor"))
    agent_cfg["shared_state_preprocessor"] = _resolve_component(agent_cfg.get("shared_state_preprocessor"))
    agent_cfg["value_preprocessor"] = _resolve_component(agent_cfg.get("value_preprocessor"))
    agent_cfg["rewards_shaper"] = _reward_shaper_from_scale(agent_cfg.pop("rewards_shaper_scale", 1.0))
    agent_cfg["state_preprocessor_kwargs"] = {
        agent_id: {"size": observation_spaces[agent_id], "device": env.device} for agent_id in possible_agents
    }
    agent_cfg["shared_state_preprocessor_kwargs"] = {
        agent_id: {"size": state_spaces[agent_id], "device": env.device} for agent_id in possible_agents
    }
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    return MotionContextMAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        shared_observation_spaces=state_spaces,
        device=env.device,
        cfg=agent_cfg,
    )
