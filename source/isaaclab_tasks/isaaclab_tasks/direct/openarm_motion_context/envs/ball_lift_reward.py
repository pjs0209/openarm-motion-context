"""Bounded height credit for the opt-in heavy-ball reward pilot."""

import math

import torch


def ball_lift_bootstrap(height_progress, tilt_deg, inside, scale, tilt_free, tilt_bad):
    """Reward actual elevation without multiplying by TCP grasp readiness.

    This is state-based shaping, not potential-based shaping: it can change
    the optimal policy. It never rewards height-zero camping or a lost ball.
    The same tilt attenuation as Lift is retained.
    """
    if not all(math.isfinite(float(x)) for x in (scale, tilt_free, tilt_bad)):
        raise ValueError("Bootstrap parameters must be finite")
    if scale < 0 or tilt_bad <= tilt_free:
        raise ValueError("Invalid bootstrap scale or tilt thresholds")
    if height_progress.shape != tilt_deg.shape or inside.shape != height_progress.shape:
        raise ValueError("Bootstrap tensors must have matching shapes")
    valid = torch.isfinite(height_progress) & torch.isfinite(tilt_deg) & inside.bool()
    height = torch.nan_to_num(height_progress, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    tilt = torch.nan_to_num(tilt_deg, nan=tilt_bad, posinf=tilt_bad, neginf=tilt_bad)
    penalty = ((tilt - tilt_free) / (tilt_bad - tilt_free)).clamp(0, 1).square()
    quality = (1 - penalty).clamp(0, 1).square()
    return torch.where(valid, scale * height * quality, torch.zeros_like(height))


def ball_height_upright_bridge(
    height_progress, tilt_deg, inside, grasp_gate, height_scale, upright_scale, upright_decay_deg
):
    """Keep elevation credit at high tilt while rewarding upright lifting.

    Both terms require an in-box ball and bilateral grasp readiness. The
    height term does not vanish above the legacy 20-degree tilt cutoff.
    """
    if not all(math.isfinite(float(x)) for x in (height_scale, upright_scale, upright_decay_deg)):
        raise ValueError("Bridge parameters must be finite")
    if height_scale < 0 or upright_scale < 0 or upright_decay_deg <= 0:
        raise ValueError("Invalid bridge scales or tilt decay")
    if not (height_progress.shape == tilt_deg.shape == inside.shape == grasp_gate.shape):
        raise ValueError("Bridge tensors must have matching shapes")
    valid = torch.isfinite(height_progress) & torch.isfinite(tilt_deg) & torch.isfinite(grasp_gate) & inside.bool()
    height = torch.nan_to_num(height_progress, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    gate = torch.nan_to_num(grasp_gate, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    tilt = torch.nan_to_num(tilt_deg, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    common = torch.where(valid, height * gate, torch.zeros_like(height))
    return height_scale * common, upright_scale * common * torch.exp(-tilt / upright_decay_deg)


def bilateral_closure_gate(left_closure, right_closure, threshold):
    """Soft AND of measured closures; zero for either fully open gripper.

    This is a kinematic proxy, not a contact/force-based grasp certificate.
    Nonfinite measurements receive no extra height credit.
    """
    if not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("Closure threshold must be positive and finite")
    if left_closure.shape != right_closure.shape:
        raise ValueError("Bilateral closure shapes must match")
    valid = torch.isfinite(left_closure) & torch.isfinite(right_closure)
    left = torch.nan_to_num(left_closure, nan=0.0, posinf=0.0, neginf=0.0)
    right = torch.nan_to_num(right_closure, nan=0.0, posinf=0.0, neginf=0.0)
    gate = torch.minimum((left / threshold).clamp(0, 1), (right / threshold).clamp(0, 1))
    return torch.where(valid, gate, torch.zeros_like(gate))
