"""
Docking environment configuration mirroring the Factory peg-insertion task,
with contact sensors activated on FixedAsset & HeldAsset spawners.
"""

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import PegInsert  # 提供 fixed/held 资产定义


def _activate_contact_sensors_on_task_assets(task_obj):
    """Turn on contact sensors on the spawn cfg of fixed/held assets if available."""
    for attr_name in ("fixed_asset", "held_asset"):
        if hasattr(task_obj, attr_name):
            art_cfg = getattr(task_obj, attr_name)
            spawn = getattr(art_cfg, "spawn", None)
            if spawn is not None:
                setattr(spawn, "activate_contact_sensors", True)
    return task_obj


# 基于 PegInsert 构造任务对象，并打开资产的 contact sensors
_DOCKING_TASK = _activate_contact_sensors_on_task_assets(PegInsert())


class DockingEnvCfg(FactoryTaskPegInsertCfg):
    """Peg-insertion environment cfg with asset contact sensors enabled by default.

    Attributes
    ----------
    use_articulated_hole : bool
        If True, replaces the hole's single D6 compliance joint with a 6×1-DoF
        articulated chain (Px, Py, Pz, Rx, Ry, Rz) so it can be vectorized and driven
        as part of an articulation. If False (default), use the original D6 joint.
    """
    task = _DOCKING_TASK

    # 新增：是否用 6×R/P articulation 近似 6D 柔性
    use_articulated_hole: bool = True
    # Debug 开关：打印孔的受力与位移信息
    debug_hole_motion: bool = True
