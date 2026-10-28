# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO must score sampled Gaussian commands before physical action clipping."""

import math
import runpy
from pathlib import Path

import gymnasium as gym
import pytest
import torch
import yaml

from isaaclab_tasks.direct.openarm_motion_context.models.bounded_execution_mean import install_mean_bound

ROOT = Path(__file__).resolve().parents[3]
GaussianMixin = runpy.run_path(str(ROOT / "compat/skrl_1_4_3/skrl/models/torch/gaussian.py"))["GaussianMixin"]


class KnownGaussian(GaussianMixin):
    """Use the pinned runtime's real sampler with an analytically known density."""

    def __init__(self, clip_actions):
        self.device = torch.device("cpu")
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        super().__init__(clip_actions=clip_actions)

    def compute(self, inputs, role=""):
        return torch.full((8192, 1), 4.0), torch.tensor([math.log(0.2)]), {}


@pytest.mark.parametrize("task", ["lift", "peg_in_hole"])
def test_training_defaults_preserve_the_gaussian_sample_for_likelihoods(task):
    config = yaml.safe_load(
        (
            ROOT / f"source/isaaclab_tasks/isaaclab_tasks/direct/openarm_motion_context/train/configs/{task}_mappo.yaml"
        ).read_text()
    )
    model = KnownGaussian(config["models"]["policy"]["clip_actions"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        commands, log_prob, _ = model.act({})
    # A true distribution's score has zero expectation. Clipping these samples
    # before evaluating the Gaussian produces a score of -75 instead.
    score_mean = ((commands - 4.0) / 0.2**2).mean()
    assert abs(float(score_mean)) < 0.25
    torch.testing.assert_close(log_prob, torch.distributions.Normal(4.0, 0.2).log_prob(commands))
    executed = commands.clamp(-1.0, 1.0)
    assert torch.all(executed == 1.0)
    assert torch.all(commands > 1.0)


@pytest.mark.parametrize("arm_only", [True, False])
def test_bounded_means_keep_raw_sampling_and_gripper_switches(arm_only):
    """Bound drifting means without changing PPO's density or finger hysteresis."""

    class ControlGaussian(torch.nn.Module, GaussianMixin):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.device = torch.device("cpu")
            self.num_actions = 7 if arm_only else 8
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.num_actions,))
            GaussianMixin.__init__(self, clip_actions=False)
            self.arm_action_head = torch.nn.Linear(3, 7)

        def compute(self, inputs, role=""):
            return inputs["states"], torch.zeros(self.num_actions), {}

    model = ControlGaussian()
    parameters = sum(p.numel() for p in model.parameters())
    install_mean_bound(model)
    assert sum(p.numel() for p in model.parameters()) == parameters
    states = torch.full((8192, model.num_actions), 20.0)
    states[:, :3] = torch.tensor([-20.0, 0.0, 20.0])
    if not arm_only:
        switches = torch.tensor([-20.0, -0.05, -0.01, 0.0, 0.01, 0.05, 20.0])
        states[:, -1] = switches[torch.arange(len(states)) % len(switches)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        commands, log_prob, outputs = model.act({"states": states})
    mean = outputs["mean_actions"]
    assert mean[:, :7].abs().max() < 1.0
    torch.testing.assert_close(mean[:, :3], torch.tensor([-20 / 21, 0.0, 20 / 21]).expand(len(states), -1))
    torch.testing.assert_close(log_prob, torch.distributions.Normal(mean, 1.0).log_prob(commands).sum(-1, keepdim=True))
    assert (commands.abs() > 1.0).any()  # only execution, not sampling, is clipped
    if not arm_only:
        torch.testing.assert_close(mean[:, -1], states[:, -1])

        def switch(values):
            return (values > 0.05).int() - (values < -0.05).int()

        assert torch.equal(switch(states[:, -1]), switch(mean[:, -1]))
        retain = states[:, -1].abs() <= 0.05
        torch.testing.assert_close(mean[retain, -1], states[retain, -1])
    incompatible = ControlGaussian()
    install_mean_bound(incompatible)
    old_switch_density = model.state_dict()
    old_switch_density["_bounded_execution_mean_version"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="version mismatch"):
        incompatible.load_state_dict(old_switch_density)
