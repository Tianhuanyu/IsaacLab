from __future__ import annotations

import torch
import carb

from .docking_base_env import DockingBaseEnv
from .docking_env_cfg import DockingEnvCfg
from .docking_tasks_cfg import DockTask


class DockingEnv(DockingBaseEnv):
    """Docking task where the trocar is a rigid body held in place by a Dynamic Control attractor."""

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        if not hasattr(cfg.task, "fixed_asset"):
            old_task = cfg.task
            dock_task = DockTask()
            if hasattr(old_task, "to_dict"):
                dock_task.from_dict(old_task.to_dict())
            cfg.task = dock_task
        cfg.task_name = "dock"

        self._dc_module = None
        self._dc_interface = None
        self._dc_warning_emitted = False
        self._hole_attractors: list | None = None
        self._hole_target_pos: torch.Tensor | None = None
        self._hole_target_quat: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)

    # ---------------------------------------------------------------------
    # Scene setup / teardown
    # ---------------------------------------------------------------------
    def __del__(self):
        self._destroy_hole_attractors()
        try:
            super().__del__()
        except AttributeError:
            pass

    def _setup_scene(self):
        super()._setup_scene()
        self._setup_hole_attractors()
        self._update_hole_attractor_targets()

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        super()._reset_idx(env_ids)
        self._update_hole_attractor_targets(env_ids)
        if getattr(self.cfg, "debug_hole_motion", False):
            self._debug_hole_pose_error(env_ids)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _get_env_ids_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        env_ids = env_ids if isinstance(env_ids, (list, tuple)) else [int(env_ids)]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _acquire_dynamic_control_interface(self):
        if self._dc_interface is not None:
            return self._dc_interface
        try:
            from omni.isaac.dynamic_control import _dynamic_control as dynamic_control
        except Exception as exc:  # noqa: BLE001
            if not self._dc_warning_emitted:
                carb.log_warn(
                    f"DockingEnv: Dynamic Control interface unavailable ({exc}). "
                    "Trocar attractors will be disabled."
                )
                self._dc_warning_emitted = True
            self._dc_module = None
            self._dc_interface = None
            return None

        self._dc_module = dynamic_control
        try:
            self._dc_interface = dynamic_control.acquire_dynamic_control_interface()
        except Exception as exc:  # noqa: BLE001
            if not self._dc_warning_emitted:
                carb.log_warn(
                    f"DockingEnv: Failed to acquire Dynamic Control interface ({exc}). "
                    "Trocar attractors will be disabled."
                )
                self._dc_warning_emitted = True
            self._dc_interface = None
        return self._dc_interface

    def _get_attractor_gains(self):
        cfg = getattr(self.cfg.task, "hole_impedance", None)
        if cfg is None:
            return 2.0e4, 4.0e2, 1.0e6
        trans_cfg = getattr(cfg, "transZ", None)
        if trans_cfg is None:
            return 2.0e4, 4.0e2, 1.0e6
        return (
            float(getattr(trans_cfg, "stiffness", 2.0e4)),
            float(getattr(trans_cfg, "damping", 4.0e2)),
            float(getattr(trans_cfg, "max_force", 1.0e6)),
        )

    def _setup_hole_attractors(self):
        dc = self._acquire_dynamic_control_interface()
        env_paths = getattr(self.scene, "env_prim_paths", [])
        self._hole_attractors = [None] * len(env_paths)
        if dc is None or not env_paths:
            return

        stiffness, damping, force_limit = self._get_attractor_gains()
        axes = getattr(self._dc_module, "AXIS_ALL", None)
        if axes is None:
            axes = (
                getattr(self._dc_module, "AXIS_ALL_TRANSLATION", 0)
                | getattr(self._dc_module, "AXIS_ALL_ROTATION", 0)
            )

        invalid_handle = getattr(self._dc_module, "INVALID_HANDLE", None)

        for env_id, env_path in enumerate(env_paths):
            body_path = f"{env_path}/FixedAsset"
            try:
                body_handle = dc.get_rigid_body(body_path)
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"DockingEnv: Failed to acquire rigid body '{body_path}' ({exc}).")
                continue

            if body_handle in (None, invalid_handle):
                carb.log_warn(f"DockingEnv: Rigid body '{body_path}' unavailable; skipping attractor setup.")
                continue

            props = self._dc_module.AttractorProperties()
            props.body = body_handle
            props.axes = axes
            props.stiffness = stiffness
            props.damping = damping
            props.force_limit = force_limit
            props.offset.p = (0.0, 0.0, 0.0)
            props.offset.r = (0.0, 0.0, 0.0, 1.0)

            target_pos, target_quat = self._get_initial_target_pose(env_id)
            props.target.p = tuple(target_pos)
            props.target.r = self._quat_wxyz_to_xyzw(target_quat)

            try:
                handle = dc.create_rigid_body_attractor(props)
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"DockingEnv: Failed to create attractor for '{body_path}' ({exc}).")
                continue

            if handle in (None, invalid_handle):
                carb.log_warn(f"DockingEnv: Invalid attractor handle for '{body_path}'.")
                continue

            self._hole_attractors[env_id] = handle

    def _get_initial_target_pose(self, env_id: int):
        data = getattr(self._fixed_asset, "data", None)
        if data is None or getattr(data, "default_root_state", None) is None:
            return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        default_state = data.default_root_state
        env_state = default_state[min(env_id, default_state.shape[0] - 1)]
        pos = (env_state[0:3] + self.scene.env_origins[env_id]).tolist()
        quat = env_state[3:7].tolist()
        return pos, quat

    def _update_hole_attractor_targets(self, env_ids: torch.Tensor | None = None):
        dc = self._acquire_dynamic_control_interface()
        if dc is None or not self._hole_attractors:
            return

        env_ids_t = self._get_env_ids_tensor(env_ids)
        invalid_handle = getattr(self._dc_module, "INVALID_HANDLE", None)

        if self._hole_target_pos is None or self._hole_target_pos.shape[0] != self.num_envs:
            self._hole_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._hole_target_quat = torch.zeros((self.num_envs, 4), device=self.device)

        for env_id in env_ids_t.tolist():
            if env_id >= len(self._hole_attractors):
                continue
            handle = self._hole_attractors[env_id]
            if handle in (None, invalid_handle):
                continue

            try:
                pos = self._fixed_asset.data.root_pos_w[env_id].detach().cpu().tolist()
                quat = self._fixed_asset.data.root_quat_w[env_id].detach().cpu().tolist()
            except Exception:
                continue

            target = self._dc_module.Transform()
            target.p = (float(pos[0]), float(pos[1]), float(pos[2]))
            target.r = self._quat_wxyz_to_xyzw(quat)

            try:
                dc.set_attractor_target(handle, target)
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"DockingEnv: Failed to update attractor target for env {env_id}: {exc}")
                continue

            self._hole_target_pos[env_id] = torch.tensor(pos, device=self.device)
            self._hole_target_quat[env_id] = torch.tensor(quat, device=self.device)

    def _destroy_hole_attractors(self):
        dc = self._dc_interface
        if dc is None or not self._hole_attractors:
            return
        invalid_handle = getattr(self._dc_module, "INVALID_HANDLE", None)
        for handle in self._hole_attractors:
            if handle in (None, invalid_handle):
                continue
            try:
                dc.destroy_rigid_body_attractor(handle)
            except Exception:
                pass
        self._hole_attractors = []

    def _debug_hole_pose_error(self, env_ids: torch.Tensor | None = None):
        if self._hole_target_pos is None or self._hole_target_quat is None:
            return
        env_ids_t = self._get_env_ids_tensor(env_ids)
        curr_pos = self._fixed_asset.data.root_pos_w[env_ids_t]
        curr_quat = self._fixed_asset.data.root_quat_w[env_ids_t]
        tgt_pos = self._hole_target_pos[env_ids_t]
        tgt_quat = self._hole_target_quat[env_ids_t]
        delta_pos = (curr_pos - tgt_pos).norm(dim=-1).mean().item()
        dot = torch.sum(curr_quat * tgt_quat, dim=-1).abs().clamp(max=1.0)
        delta_quat = (1.0 - dot).mean().item()
        print(f"[Docking Debug] attractor delta_pos={delta_pos:.3e} delta_quat={delta_quat:.3e}")

    def _quat_wxyz_to_xyzw(self, quat):
        if len(quat) != 4:
            return (0.0, 0.0, 0.0, 1.0)
        return (float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))
