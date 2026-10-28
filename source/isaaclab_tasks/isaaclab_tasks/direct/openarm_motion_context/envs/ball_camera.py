"""RGB-D/segmentation sphere-position estimator; no simulator object pose input.

Simulation segmentation acts as an oracle image detector. Replacing it with a
real RGB detector is required for deployment. Pixel depth and camera intrinsics
recover the visible sphere surface, then the known radius approximates center.
"""

import ast
import math

import torch


def estimate_ball_camera_position(depth, mask, intrinsics, radius=0.04, min_pixels=9):
    """Return ROS-camera [N,3] sphere centers and a detection-valid [N] flag.

    Depth is axial distance_to_image_plane, not Euclidean range. Estimate uses
    the mask centroid's 3x3 depth neighborhood and compensates visible surface
    distance by the known visual radius. Partial occlusion can bias this proxy.
    Missing/invalid detections return zero plus valid=False; never a GT fallback.
    """
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if depth.ndim != 3 or depth.shape != mask.shape or intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError("Expected matching [N,H,W] depth/mask and [N,3,3] intrinsics")
    if not math.isfinite(radius) or radius <= 0 or min_pixels < 1:
        raise ValueError("Invalid sphere radius or minimum detection area")
    if not torch.isfinite(intrinsics).all() or (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("Invalid camera focal lengths/intrinsics")
    n, height, width = depth.shape
    rows, columns = torch.meshgrid(
        torch.arange(height, device=depth.device), torch.arange(width, device=depth.device), indexing="ij"
    )
    valid_pixels = mask.bool() & torch.isfinite(depth) & (depth > 0)
    counts = valid_pixels.sum(dim=(1, 2))
    u = (valid_pixels * columns).sum(dim=(1, 2)) / counts.clamp_min(1)
    v = (valid_pixels * rows).sum(dim=(1, 2)) / counts.clamp_min(1)
    neighborhood = valid_pixels & ((columns - u[:, None, None].round()).abs() <= 1)
    neighborhood &= (rows - v[:, None, None].round()).abs() <= 1
    samples = neighborhood.sum(dim=(1, 2))
    distance = torch.where(neighborhood, depth, torch.zeros_like(depth)).sum(dim=(1, 2)) / samples.clamp_min(1)
    rays = torch.stack(
        (
            (u - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0],
            (v - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1],
            torch.ones_like(u),
        ),
        dim=1,
    )
    surface = rays * distance[:, None]
    centers = surface + radius * torch.nn.functional.normalize(rays, dim=1)
    valid = (counts >= min_pixels) & (samples > 0) & torch.isfinite(centers).all(dim=1)
    return torch.where(valid[:, None], centers, torch.zeros_like(centers)), valid


def ball_semantic_ids(info):
    """Resolve numeric IDs or native6.2 packed RGBA label keys to uint32."""
    labels = info.get("idToLabels", {})
    ids = []
    for key, values in labels.items():
        if not isinstance(values, dict) or values.get("class") != "ball":
            continue
        try:
            semantic_id = int(key)
        except (ValueError, TypeError):
            rgba = ast.literal_eval(key)
            if (
                not isinstance(rgba, tuple)
                or len(rgba) != 4
                or any(type(channel) is not int or not 0 <= channel <= 255 for channel in rgba)
            ):
                raise ValueError(f"Invalid segmentation label key: {key!r}")
            semantic_id = sum(channel << (8 * index) for index, channel in enumerate(rgba))
        ids.append(semantic_id & 0xFFFFFFFF)
    return ids


def ball_semantic_mask(segmentation, info):
    """Select ball-labelled pixels without consulting any simulator pose."""
    mask = torch.zeros_like(segmentation, dtype=torch.bool)
    # RTX exposes packed uint32 colors via signed int32 ProxyArray buffers.
    unsigned_pixels = segmentation.to(torch.int64) & 0xFFFFFFFF
    for semantic_id in ball_semantic_ids(info):
        mask |= unsigned_pixels == semantic_id
    return mask
