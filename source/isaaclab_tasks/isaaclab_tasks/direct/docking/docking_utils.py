"""Utility helpers dedicated to the docking environment."""

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils


def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly spaced keypoints along a unit line centered at the origin."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Compute critically damped derivative gains."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def wrap_yaw(angle):
    """Wrap yaw to avoid discontinuities around 360 degrees."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def set_friction(asset, value, num_envs):
    """Update material friction parameters for the given articulation."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value
    materials[..., 1] = value
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)


def set_body_inertias(robot, num_envs):
    """Adjust inertias to match older armature assumptions."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(num_envs))


def get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device):
    """Return pose offset between held asset frame and its geometric base."""
    if task_name != "dock":
        raise NotImplementedError(f"Docking utilities only support the 'dock' task, got '{task_name}'.")
    held_base_pos_local = torch.zeros((num_envs, 3), device=device)
    held_base_pos_local[:, 2] = 0.0
    return held_base_pos_local


def get_held_base_pose(held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Combine world pose with local offset to obtain held base pose."""
    held_base_pos_local = get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Target pose for the held asset base relative to the fixed asset."""
    if task_name != "dock":
        raise NotImplementedError(f"Docking utilities only support the 'dock' task, got '{task_name}'.")
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x, a, b):
    """Bounded reward function used for keypoint shaping."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict, obs_order):
    """Stack observations according to the provided order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors
