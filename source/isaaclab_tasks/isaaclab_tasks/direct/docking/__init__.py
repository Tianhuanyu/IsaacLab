# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Docking direct workflow environment registration.
"""

import gymnasium as gym

from . import agents as docking_agents

from .docking_env import DockingEnv
from .docking_env_cfg import DockingEnvCfg

__all__ = ["DockingEnv", "DockingEnvCfg"]

gym.register(
    id="Isaac-Docking-Direct-v0",
    entry_point="isaaclab_tasks.direct.docking:DockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DockingEnvCfg,
        "rl_games_cfg_entry_point": f"{docking_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
