"""Admittance-controlled factory tasks."""

import gymnasium as gym

from isaaclab_tasks.direct.factory import agents as factory_agents

from .admittance_env import AdmittanceEnv
from .admittance_env_cfg import AdmittancePegInsertEnvCfg

gym.register(
    id="Isaac-Admittance-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.admittance:AdmittanceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AdmittancePegInsertEnvCfg,
        "rl_games_cfg_entry_point": f"{factory_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = ["AdmittanceEnv", "AdmittancePegInsertEnvCfg"]
