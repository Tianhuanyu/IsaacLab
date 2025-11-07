from __future__ import annotations
import torch

class AdmittanceController:
    """
    将策略 6D 动作视作名义位姿增量 (Δx, Δθ)，内部用二阶 M-B-K 跟踪器做平滑与限幅：
      x_ref[k+1] = x_ref[k] + Δx_nom
      M a = W - B v - K (x - x_ref)   （W 为外部扳手，单位 [N, N·m]）
    输出为本步应施加的偏移：Δx_out = x[k+1] - x[k]，Δθ_out = θ[k+1] - θ[k]
    ——接口兼容：step(actions, dt, measured_wrench_6=None) -> (pos_offset, rot_offset)
    """

    def __init__(self, ctrl_cfg, num_envs: int, device: torch.device):
        self.cfg = ctrl_cfg
        self.num_envs = num_envs
        self.device = device
        self.dtype = torch.float32

        # ---------- helpers ----------
        def _vec3_from_any(x, default=None):
            if x is None:
                x = default
            if isinstance(x, (list, tuple)):
                t = torch.tensor(x, dtype=self.dtype, device=self.device)
                if t.numel() == 3:
                    return t
                if t.numel() == 1:
                    return t.repeat(3)
                return torch.full((3,), float(t.flatten()[0]), dtype=self.dtype, device=self.device)
            elif x is not None:
                return torch.full((3,), float(x), dtype=self.dtype, device=self.device)
            else:
                return torch.zeros(3, dtype=self.dtype, device=self.device)

        def _nonzero_or_none(t: torch.Tensor | None):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return None if torch.allclose(t, torch.zeros_like(t)) else t
            try:
                v = float(t)
                return None if abs(v) < 1e-12 else t
            except Exception:
                return t

        # ---------- resolve per-step thresholds (pos/rot) ----------
        pos_thr = _nonzero_or_none(_vec3_from_any(getattr(self.cfg, "pos_threshold", None)))
        rot_thr = _nonzero_or_none(_vec3_from_any(getattr(self.cfg, "rot_threshold", None)))

        if pos_thr is None:
            pos_thr = _nonzero_or_none(_vec3_from_any(getattr(self.cfg, "pos_action_scale", None)))
        if rot_thr is None:
            rot_thr = _nonzero_or_none(_vec3_from_any(getattr(self.cfg, "rot_action_scale", None)))

        if pos_thr is None:
            bounds = getattr(self.cfg, "pos_action_bounds", None)
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                pos_thr = _vec3_from_any(min(float(bounds[0]), float(bounds[1])))
        if rot_thr is None:
            rot_thr = _vec3_from_any(0.10)
        if pos_thr is None:
            pos_thr = _vec3_from_any(0.005)

        self._thr6 = torch.cat([pos_thr, rot_thr], dim=0)  # [6]

        # ---------- admittance 6D params ----------
        self.M = torch.tensor(getattr(self.cfg, "admittance_mass",       [1,1,1, 0.1,0.1,0.1]),
                              dtype=self.dtype, device=self.device)
        self.B = torch.tensor(getattr(self.cfg, "admittance_damping",    [5000,5000,5000, 500,500,500]),
                              dtype=self.dtype, device=self.device)
        self.K = torch.tensor(getattr(self.cfg, "admittance_stiffness",  [2000,2000,2000, 200,200,200]),
                              dtype=self.dtype, device=self.device)

        self.vmax = torch.tensor(getattr(self.cfg, "admittance_max_velocity",     [0.25,0.25,0.2, 1.5,1.5,1.5]),
                                 dtype=self.dtype, device=self.device)
        self.xmax = torch.tensor(getattr(self.cfg, "admittance_max_displacement", [0.05,0.05,0.05, 0.5,0.5,0.5]),
                                 dtype=self.dtype, device=self.device)

        # 外部扳手缩放（逐通道）
        self.Ws = torch.tensor(getattr(self.cfg, "admittance_wrench_scale", [1,1,1, 1,1,1]),
                               dtype=self.dtype, device=self.device)

        # 状态
        self.x_ref = torch.zeros((num_envs, 6), dtype=self.dtype, device=self.device)
        self.x     = torch.zeros((num_envs, 6), dtype=self.dtype, device=self.device)
        self.v     = torch.zeros((num_envs, 6), dtype=self.dtype, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.x_ref[env_ids] = 0.0
        self.x[env_ids]     = 0.0
        self.v[env_ids]     = 0.0

    def step(self, actions: torch.Tensor, dt: float, measured_wrench_6: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        actions: [N,6] 策略名义增量（前3平移、后3轴角）
        measured_wrench_6: [N,6] 外部扳手（Fx,Fy,Fz, Tx,Ty,Tz），可为 None
        返回: (pos_offset[N,3], rot_offset[N,3])
        """
        # 0) 外部扳手
        if measured_wrench_6 is None:
            W = torch.zeros((self.num_envs, 6), dtype=self.dtype, device=self.device)
        else:
            if measured_wrench_6.shape[-1] != 6:
                raise ValueError(f"measured_wrench_6 must have shape [N,6], got {measured_wrench_6.shape}")
            W = measured_wrench_6.to(self.dtype).to(self.device)
        W = W * self.Ws.unsqueeze(0)  # 逐通道缩放

        # 1) 名义位姿增量（按阈值缩放）
        delta_nom = actions * self._thr6  # [N,6]

        # 2) 参考位姿步进
        self.x_ref = self.x_ref + delta_nom

        # 3) M-B-K 显式积分（加入外部扳手）
        a = (W - self.B.unsqueeze(0) * self.v - self.K.unsqueeze(0) * (self.x - self.x_ref)) / self.M.unsqueeze(0)

        v_new = self.v + a * dt
        v_new = torch.clamp(v_new, min=-self.vmax.unsqueeze(0), max=self.vmax.unsqueeze(0))

        x_prev = self.x
        x_new  = self.x + v_new * dt

        err = x_new - self.x_ref
        err = torch.clamp(err, min=-self.xmax.unsqueeze(0), max=self.xmax.unsqueeze(0))
        x_new = self.x_ref + err

        self.v = v_new
        self.x = x_new

        delta_out = x_new - x_prev
        pos_offset = delta_out[:, :3]
        rot_offset = delta_out[:, 3:]
        return pos_offset, rot_offset
