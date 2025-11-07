from __future__ import annotations

import isaacsim.core.utils.torch as torch_utils
import torch

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .admittance_control import AdmittanceController
from .admittance_env_cfg import AdmittancePegInsertEnvCfg


class AdmittanceEnv(FactoryEnv):
    """Factory peg-insert environment that drives the robot via admittance control."""

    cfg: AdmittancePegInsertEnvCfg

    def _init_tensors(self):
        super()._init_tensors()
        self.admittance_controller = AdmittanceController(
            ctrl_cfg=self.cfg.ctrl,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.admittance_controller.reset(env_ids)

    # ——最小修改：从观测/状态里“尽力”提取交互力/矩；找不到则返回 0——
    def _get_measured_wrench(self) -> torch.Tensor:
        # 1) 直接查常见张量属性
        direct_names = [
            "ee_wrench_wcs", "ee_wrench_ics",
            "fingertip_wrench_wcs", "fingertip_wrench_ics",
            "measured_ee_wrench",
            "net_contact_wrench", "ee_contact_wrench",
        ]
        for name in direct_names:
            w = getattr(self, name, None)
            if isinstance(w, torch.Tensor) and w.shape[-1] == 6:
                return w

        # 2) 从“更丰富的 critic state / 内部字典”里找
        def _extract_from_dict(d):
            if not isinstance(d, dict):
                return None
            # 优先找 6D wrench
            for key in [
                "ee_wrench_wcs", "ee_wrench_ics",
                "fingertip_wrench_wcs", "fingertip_wrench_ics",
                "contact_wrench_wcs", "contact_wrench_ics",
                "net_contact_wrench", "ee_contact_wrench",
            ]:
                t = d.get(key, None)
                if isinstance(t, torch.Tensor) and t.shape[-1] == 6:
                    return t
            # 次之：只有力 3D 的，拼成 [F, 0]
            for key in [
                "ee_force_wcs", "ee_force_ics",
                "fingertip_force_wcs", "fingertip_force_ics",
                "net_contact_force", "contact_force",
            ]:
                t = d.get(key, None)
                if isinstance(t, torch.Tensor) and t.shape[-1] == 3:
                    zeros = torch.zeros_like(t)
                    return torch.cat([t, zeros], dim=-1)
            return None

        for dict_name in [
            "_factory_state_dict", "_factory_obs_state_dict",
            "state_dict", "obs_dict",
            "_state_dict", "_obs_dict",
        ]:
            d = getattr(self, dict_name, None)
            w = _extract_from_dict(d)
            if isinstance(w, torch.Tensor):
                return w

        # 3) 兜底：没有力传感就返回 0（只走策略名义增量）
        return torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        admittance_actions = self.actions.clone()
        if self.cfg_task.unidirectional_rot:
            admittance_actions[:, 5] = -(admittance_actions[:, 5] + 1.0) * 0.5

        # 传入观测中的交互力/矩（若没有则为 0）
        measured_wrench_6 = self._get_measured_wrench()

        pos_offset, rot_offset = self.admittance_controller.step(
            admittance_actions, dt=self.physics_dt, measured_wrench_6=measured_wrench_6
        )

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_offset

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        if self.cfg_task.unidirectional_rot:
            rot_offset[:, 2] = torch.clamp(rot_offset[:, 2], max=0.0)

        angle = torch.norm(rot_offset, p=2, dim=-1)
        axis = rot_offset / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            identity_quat,
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )
