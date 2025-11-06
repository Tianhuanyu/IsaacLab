"""Control utilities for the docking environment, adapted from the factory control module."""

import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
):
    """Compute Franka joint torques to move the fingertip towards the target pose."""
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    arm_mass_matrix_task = torch.inverse(jacobian @ arm_mass_matrix_inv @ jacobian_T)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (2 * math.pi) - math.pi
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between the target and current fingertip pose."""
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    if jacobian_type == "geometric":
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[:, 0]
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)
    else:
        raise ValueError(f"Unsupported jacobian_type: {jacobian_type}")

    if rot_error_type == "quat":
        return pos_error, quat_error
    if rot_error_type == "axis_angle":
        return pos_error, axis_angle_error
    raise ValueError(f"Unsupported rot_error_type: {rot_error_type}")


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Solve for joint-space increments corresponding to a desired task-space delta."""
    if ik_method == "pinv":
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)
    if ik_method == "trans":
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = jacobian_T @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)
    if ik_method == "dls":
        lambda_val = 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)
    if ik_method == "svd":
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = torch.where(S > 1.0e-5, 1.0 / S, torch.zeros_like(S))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
            @ torch.diag_embed(S_inv)
            @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)
    raise ValueError(f"Unsupported IK method: {ik_method}")


def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
):
    """Interpret PD gains as task-space gains and apply them to the pose error."""
    task_wrench = torch.zeros_like(delta_fingertip_pose)
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    ang_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * ang_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )

    return task_wrench
