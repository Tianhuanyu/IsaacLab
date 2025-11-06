"""Docking environment mirroring the Factory peg-insertion setup."""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from .docking_env_cfg import DockingEnvCfg


class DockingEnv(FactoryEnv):
    """Identical to the standard Factory peg-insertion environment with hole force logging."""

    cfg: DockingEnvCfg

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._hole_rest_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        self._hole_log_counter = 0
        self._cache_hole_rest_wrench()

    def _cache_hole_rest_wrench(self, env_ids: torch.Tensor | None = None):
        """Store the baseline wrench on the hole's root link for the given environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[env_ids, 0].to(self.device)
        self._hole_rest_wrench[env_ids] = wrench

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._cache_hole_rest_wrench(env_ids)
        self._hole_log_counter = 0

    def _apply_action(self):
        self._log_hole_contact()
        super()._apply_action()

    def _log_hole_contact(self):
        """Print the interaction force/torque on the hole at a low frequency."""
        if not hasattr(self._fixed_asset.data, "body_incoming_joint_wrench_b"):
            return

        wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[:, 0].to(self.device)
        delta = wrench - self._hole_rest_wrench

        if self._hole_log_counter % 60 == 0 and self.num_envs > 0:
            hole_force = delta[0, :3].detach().cpu().tolist()
            hole_torque = delta[0, 3:].detach().cpu().tolist()
            print(f"[Docking Debug] hole_force={hole_force}, hole_torque={hole_torque}")

        self._hole_log_counter += 1
