# admittance_env_cfg.py
"""Configuration objects for admittance-controlled factory tasks."""

from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg, FactoryTaskPegInsertCfg


@configclass
class AdmittanceCtrlCfg(CtrlCfg):
    """Extends the factory controller configuration with admittance parameters."""

    admittance_mass: list[float] = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    admittance_damping: list[float] = [5000.0, 5000.0, 5000.0, 500.0, 500.0, 500.0]
    admittance_stiffness: list[float] = [2000.0, 2000.0, 2000.0, 200.0, 200.0, 200.0]
    admittance_max_velocity: list[float] = [0.25, 0.25, 0.2, 1.5, 1.5, 1.5]
    admittance_max_displacement: list[float] = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
    admittance_wrench_scale: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


@configclass
class AdmittancePegInsertEnvCfg(FactoryTaskPegInsertCfg):
    """Peg insertion environment with admittance control."""

    ctrl: AdmittanceCtrlCfg = AdmittanceCtrlCfg()
