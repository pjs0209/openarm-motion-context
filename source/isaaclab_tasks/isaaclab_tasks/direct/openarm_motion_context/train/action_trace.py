"""Read-only capture of raw policy means for physical-clipping diagnostics."""

import torch


def record_policy_mean_actions(trace, outputs, env_index):
    """Record detached pre-clamp means, never mutate policy outputs/actions.

    Empty rows explicitly represent unavailable legacy/mock outputs. These are
    latent Gaussian means, not stochastic executed actions or physical joints.
    """
    for side, agent in (("left", "left_arm"), ("right", "right_arm")):
        means = outputs.get(agent, {}).get("mean_actions")
        if isinstance(means, torch.Tensor):
            if means.ndim != 2 or not 0 <= env_index < means.shape[0] or not torch.isfinite(means[env_index]).all():
                raise ValueError("Policy mean-action trace requires finite [environments, actions] means")
            row = means[env_index].detach().float().cpu().tolist()
        else:
            row = []
        trace[f"{side}_policy_mean_action"].append(row)
