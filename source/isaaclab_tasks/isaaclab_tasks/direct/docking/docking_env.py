"""Docking environment mirroring the Factory peg-insertion setup."""

from __future__ import annotations

import carb
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

import isaacsim.core.utils.torch as torch_utils

from isaaclab.sim import utils as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from .docking_env_cfg import DockingEnvCfg


class DockingEnv(FactoryEnv):
    """Identical to the standard Factory peg-insertion environment with hole force logging."""

    cfg: DockingEnvCfg
    _HOLE_COMPLIANCE_DRIVES: dict[str, dict[str, float]] = {
        "transX": {"stiffness": 2000.0, "damping": 80.0, "max_force": 1.0e6},
        "transY": {"stiffness": 2000.0, "damping": 80.0, "max_force": 1.0e6},
        "transZ": {"stiffness": 3000.0, "damping": 100.0, "max_force": 1.0e6},
        "rotX": {"stiffness": 40.0, "damping": 1.2, "max_force": 1.0e6},
        "rotY": {"stiffness": 40.0, "damping": 1.2, "max_force": 1.0e6},
        "rotZ": {"stiffness": 60.0, "damping": 2.0, "max_force": 1.0e6},
    }
    _HOLE_COMPLIANCE_LIMITS: dict[str, tuple[float, float]] = {
        "transX": (-0.004, 0.004),
        "transY": (-0.004, 0.004),
        "transZ": (-0.002, 0.002),
        "rotX": (-0.0872665, 0.0872665),
        "rotY": (-0.0872665, 0.0872665),
        "rotZ": (-0.0698132, 0.0698132),
    }

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        self._hole_joint_debug_info: list[dict[str, Sdf.Path]] = []
        self._debug_pose_warning_emitted = False
        self._hole_contact_sensor: ContactSensor | None = None
        self._force_debug_has_warned = False
        self._hole_body_index: int | None = None
        self._hole_body_name: str | None = None
        self._hole_index_warning_emitted = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._hole_rest_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        self._hole_log_counter = 0
        self._hole_rest_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._hole_rest_quat = torch.zeros((self.num_envs, 4), device=self.device)
        print(f"DockingEnv: fixed body names = {self._fixed_asset.body_names}")
        self._cache_hole_rest_wrench()

    def _setup_scene(self):
        # 父类会创建资产并准备克隆；随后 SimulationContext.reset() 会初始化传感器
        super()._setup_scene()
        # 关键修复：在 reset 之前就创建并注册传感器，避免运行时懒加载造成 _timestamp 缺失
        self._setup_hole_contact_sensor()
        # 其余自定义构造：根据配置选择 D6 柔性 或 6×R/P articulation
        if getattr(self.cfg, "use_articulated_hole", False):
            self._create_hole_articulated_chain()
        else:
            self._create_hole_compliance_joints()

    def _cache_hole_rest_wrench(self, env_ids: torch.Tensor | None = None):
        """Store the baseline wrench on the hole's root link for the given environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self._hole_body_index is None:
            self._resolve_hole_body_index()
            if self._hole_body_index is None:
                if not self._hole_index_warning_emitted:
                    self._hole_index_warning_emitted = True
                    carb.log_warn("DockingEnv: hole body index unresolved; skipping rest wrench caching.")
                return

        wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[env_ids, self._hole_body_index].to(self.device)
        self._hole_rest_wrench[env_ids] = wrench

        # 也缓存初始位姿，便于调试“孔被拉跑了”的问题
        self._hole_rest_pos[env_ids] = self._fixed_asset.data.body_state_w[env_ids, self._hole_body_index, :3]
        self._hole_rest_quat[env_ids] = self._fixed_asset.data.body_state_w[env_ids, self._hole_body_index, 3:7]

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        super()._reset_idx(env_ids)
        self._cache_hole_rest_wrench(env_ids)

    def _apply_action(self):
        # 直接沿用父类动作逻辑
        super()._apply_action()
        self._log_hole_contact()

    def _log_hole_contact(self):
        """Print the interaction force/torque on the hole at a low frequency."""
        hole_force_tensor: torch.Tensor | None = None
        hole_torque_tensor: torch.Tensor | None = None
        if self._hole_body_index is None:
            self._resolve_hole_body_index()

        # print("111")
        if self._hole_contact_sensor is not None:
            if not self._hole_contact_sensor.is_initialized:
                return
            net_forces = self._hole_contact_sensor.data.net_forces_w
            if net_forces is None:
                return
            if net_forces.shape[1] == 0:
                if not self._force_debug_has_warned:
                    self._force_debug_has_warned = True
                    carb.log_warn(
                        "DockingEnv: Hole contact sensor is not attached to any rigid bodies. "
                        "Check the prim_path in DockingEnvCfg.hole_contact_sensor."
                    )
                return
            if self._hole_body_index is not None and self._hole_body_index < net_forces.shape[1]:
                hole_force_tensor = net_forces[:, self._hole_body_index]
            else:
                hole_force_tensor = net_forces.sum(dim=1)
            hole_torque_tensor = torch.zeros_like(hole_force_tensor)
        else:
            # 兜底：没有传感器就用 incoming_joint_wrench 差值
            if self._hole_body_index is None or not hasattr(self._fixed_asset.data, "body_incoming_joint_wrench_b"):
                return
            wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[:, self._hole_body_index].to(self.device)
            delta = wrench - self._hole_rest_wrench
            hole_force_tensor = delta[:, :3]
            hole_torque_tensor = delta[:, 3:]
        # print("222")    

        if hole_force_tensor is None or hole_torque_tensor is None:
            return
        debug_hole_motion = getattr(self.cfg, "debug_hole_motion", False)
        log_interval = 10 if debug_hole_motion else 100

        self._hole_log_counter += 1
        if (self._hole_log_counter % log_interval) != 0:
            return
        mean_force = hole_force_tensor.abs().mean(dim=0).tolist()
        mean_torque = hole_torque_tensor.abs().mean(dim=0).tolist()
        print(f"[Docking] |F|_mean = {mean_force}, |Tau|_mean = {mean_torque}")

        # 位姿漂移告警（可选）
        if self._hole_body_index is None:
            return
        curr_pos = self._fixed_asset.data.body_state_w[:, self._hole_body_index, :3]
        curr_quat = self._fixed_asset.data.body_state_w[:, self._hole_body_index, 3:7]
        delta_pos = (curr_pos - self._hole_rest_pos).norm(dim=-1).mean().item()
        # 四元数差异简化评估：1 - |dot|
        dot = torch.sum(curr_quat * self._hole_rest_quat, dim=-1).abs().clamp(max=1.0)
        delta_quat = (1.0 - dot).mean().item()
        if debug_hole_motion:
            mean_force_mag = hole_force_tensor.norm(dim=-1).mean().item()
            mean_torque_mag = hole_torque_tensor.norm(dim=-1).mean().item()
            displaced = delta_pos > 1e-5 or delta_quat > 1e-4
            print(
                f"[Docking Debug] hole_displaced={displaced} delta_pos={delta_pos:.3e} "
                f"delta_quat={delta_quat:.3e} |F|={mean_force_mag:.3e} |Tau|={mean_torque_mag:.3e}"
            )
        if delta_pos > 1e-5 or delta_quat > 1e-4:
            carb.log_warn(f"[Docking Debug] hole_delta_pos={delta_pos:.3e}, hole_delta_quat={delta_quat:.3e}")

    def _create_hole_compliance_joints(self):
        stage = getattr(self.scene, "stage", None)
        env_paths = getattr(self.scene, "env_prim_paths", [])
        if stage is None or not env_paths:
            carb.log_warn("DockingEnv: Scene stage or environment paths unavailable; skipping compliance joint setup.")
            return

        self._hole_joint_debug_info = []
        for env_path in env_paths:
            hole_body_prim = self._resolve_hole_body_prim(stage, env_path)
            if hole_body_prim is None:
                continue
            #（冗余但安全）确保孔身体也允许接触上报
            PhysxSchema.PhysxContactReportAPI.Apply(hole_body_prim)

            anchor_path = Sdf.Path(f"{env_path}/HoleComplianceAnchor")
            anchor_prim = stage.GetPrimAtPath(anchor_path)
            if not anchor_prim.IsValid():
                anchor_prim = stage.DefinePrim(anchor_path, "Xform")

            self._align_anchor_to_hole(stage, anchor_prim, hole_body_prim, env_path)

            anchor_rigid = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
            anchor_rigid.CreateKinematicEnabledAttr(True)

            hole_rigid = UsdPhysics.RigidBodyAPI.Apply(hole_body_prim)
            hole_rigid.CreateKinematicEnabledAttr(False)

            # D6 关节：把孔刚体与锚点通过弹簧-阻尼驱动连接
            joint_path = Sdf.Path(f"{env_path}/HoleComplianceD6")
            joint = self._define_or_get_d6_joint(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([anchor_path])
            joint.CreateBody1Rel().SetTargets([hole_body_prim.GetPath()])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

            # 配置 Drives & Limits
            self._configure_joint_drives(joint.GetPrim())
            self._configure_joint_limits(joint.GetPrim())

            self._hole_joint_debug_info.append({
                "anchor": anchor_path, "hole": hole_body_prim.GetPath(), "env": Sdf.Path(env_path)
            })

        print("DockingEnv: compliance D6 joints created for all envs.")

    def _resolve_hole_body_prim(self, stage, env_path: str):
        fixed_asset_path = f"{env_path}/FixedAsset"
        fixed_prim = stage.GetPrimAtPath(fixed_asset_path)
        if not fixed_prim.IsValid():
            matches = sim_utils.find_matching_prims(self._fixed_asset.cfg.prim_path, stage=stage)
            for prim in matches:
                if prim.GetPath().pathString.startswith(env_path):
                    fixed_prim = prim
                    break
        if not fixed_prim.IsValid():
            carb.log_warn(f"DockingEnv: Fixed asset prim not found for environment '{env_path}'.")
            return None

        body_prims = sim_utils.get_all_matching_child_prims(
            fixed_prim.GetPath(),
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=True,
        )
        if not body_prims:
            carb.log_warn(f"DockingEnv: No rigid bodies found under '{fixed_asset_path}'.")
            return None
        # 默认取第一个刚体（原 Factory Peg 的孔根 body）
        hole_body_prim = body_prims[0]
        if self._hole_body_name is None:
            self._hole_body_name = hole_body_prim.GetName()
        # Index resolution deferred until articulation data available
        return hole_body_prim

    def _resolve_hole_body_index(self):
        """Resolve the index of the hole body within the fixed asset articulation data."""
        if self._hole_body_index is not None:
            return
        if self._hole_body_name is None:
            if not self._hole_index_warning_emitted:
                self._hole_index_warning_emitted = True
                carb.log_warn(
                    "DockingEnv: hole body name unresolved; hole force logging will be skipped until resolved."
                )
            return
        body_names = getattr(self._fixed_asset.data, "body_names", None)
        if body_names is None or len(body_names) == 0:
            return
        try:
            self._hole_body_index = list(body_names).index(self._hole_body_name)
            carb.log_info(
                f"DockingEnv: hole body resolved to '{self._hole_body_name}' (index {self._hole_body_index})."
            )
        except ValueError:
            if not self._debug_pose_warning_emitted:
                self._debug_pose_warning_emitted = True
                carb.log_warn(
                    f"DockingEnv: hole body name '{self._hole_body_name}' not found in articulation data; "
                    "hole metrics disabled."
                )

    def _align_anchor_to_hole(self, stage, anchor_prim, hole_body_prim, env_path: str):
        xform_cache = UsdGeom.XformCache()
        hole_world = xform_cache.GetLocalToWorldTransform(hole_body_prim)
        env_prim = stage.GetPrimAtPath(env_path)
        env_world = xform_cache.GetLocalToWorldTransform(env_prim) if env_prim.IsValid() else Gf.Matrix4d(1.0)
        local_transform = env_world.GetInverse() * hole_world

        anchor_xform = UsdGeom.Xformable(anchor_prim)
        anchor_xform.ClearXformOpOrder()
        anchor_xform.AddTransformOp().Set(local_transform)

    def _define_or_get_d6_joint(self, stage, joint_path: Sdf.Path):
        if hasattr(UsdPhysics, "D6Joint"):
            joint = UsdPhysics.D6Joint.Define(stage, joint_path)
            joint_prim = joint.GetPrim()
        else:
            joint = UsdPhysics.Joint.Define(stage, joint_path)
            joint_prim = joint.GetPrim()
            desired = getattr(UsdPhysics.Tokens, "jointTypePhysicsD6", "physicsD6")
            joint_type_attr = joint_prim.GetAttribute("physics:jointType")
            if not joint_type_attr or not joint_type_attr.IsValid():
                joint_type_attr = joint_prim.CreateAttribute("physics:jointType", Sdf.ValueTypeNames.Token)
            joint_type_attr.Set(desired)
            if hasattr(PhysxSchema, "PhysxD6JointAPI"):
                PhysxSchema.PhysxD6JointAPI.Apply(joint_prim)
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateBreakForceAttr().Set(1.0e12)
        joint.CreateBreakTorqueAttr().Set(1.0e12)
        return joint

    def _configure_joint_drives(self, joint_prim):
        for axis, params in self._HOLE_COMPLIANCE_DRIVES.items():
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive.CreateStiffnessAttr(params["stiffness"])
            drive.CreateDampingAttr(params["damping"])
            drive.CreateMaxForceAttr(params["max_force"])
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateTypeAttr(getattr(UsdPhysics.Tokens, "force", "force"))

    def _configure_joint_limits(self, joint_prim):
        for axis, (low, high) in self._HOLE_COMPLIANCE_LIMITS.items():
            limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit.CreateLowAttr().Set(low)
            limit.CreateHighAttr().Set(high)
            if hasattr(PhysxSchema, "PhysxLimitAPI"):
                physx_limit = PhysxSchema.PhysxLimitAPI.Apply(joint_prim, axis)
                if physx_limit:
                    physx_limit.CreateStiffnessAttr().Set(0.0)
                    physx_limit.CreateDampingAttr().Set(0.0)
                    physx_limit.CreateRestitutionAttr().Set(0.0)
                    physx_limit.CreateBounceThresholdAttr().Set(0.0)

    def _setup_hole_contact_sensor(self):
        # 提前（在 reset 前）创建并注册传感器
        if self._hole_contact_sensor is not None:
            return

        # 用 body_names 生成匹配表达式更精确；如果需要也可改成 '/World/envs/env_.*/FixedAsset/.*'
        body_names = getattr(self._fixed_asset, "body_names", [])
        if not body_names:
            carb.log_warn("DockingEnv: Fixed asset has no bodies; hole contact sensor not created.")
            return

        # 生成匹配模式，仅匹配固定资产的刚体
        prim_path = self._fixed_asset.cfg.prim_path
        try:
            sensor_cfg = ContactSensorCfg(
                prim_path=prim_path,
                history_length=1,
                aggregation_mode="sum",
                track_pose=True,
                debug_vis=False,
            )
            self._hole_contact_sensor = ContactSensor(sensor_cfg)
            # 注册到 scene，让后续 sim.reset() 的传感器初始化流程接管
            self.scene.sensors["hole_contact_sensor"] = self._hole_contact_sensor
            print(f"DockingEnv: Hole contact sensor configured for '{prim_path}'.")
        except Exception as exc:
            self._hole_contact_sensor = None
            carb.log_warn(f"DockingEnv: Failed to initialize hole contact sensor: {exc}")

    def _enable_hole_contact_reporting(self):
        stage = getattr(self.scene, "stage", None)
        if stage is None:
            carb.log_warn("DockingEnv: Unable to enable contact reporting; scene stage unavailable.")
            return

        prims = sim_utils.find_matching_prims(self._fixed_asset.cfg.prim_path, stage=stage)
        if not prims:
            carb.log_warn(
                f"DockingEnv: No prims matched '{self._fixed_asset.cfg.prim_path}' when enabling contact reporting."
            )
            return

        for prim in prims:
            body_prims = sim_utils.get_all_matching_child_prims(
                prim.GetPath(),
                predicate=lambda child: child.HasAPI(UsdPhysics.RigidBodyAPI),
                traverse_instance_prims=True,
            )
            for body_prim in body_prims:
                PhysxSchema.PhysxContactReportAPI.Apply(body_prim)

    
    def _create_hole_articulated_chain(self):
        """
        Build a 6-DoF chain (Px, Py, Pz, Rx, Ry, Rz) between a kinematic anchor and the hole body.
        This version avoids creating an ArticulationRoot and excludes all joints from articulation,
        to prevent Isaac Lab from expecting actuators. It also assigns valid mass/inertia to the
        intermediate links to silence PhysX warnings.
        """
        stage = getattr(self.scene, "stage", None)
        env_paths = getattr(self.scene, "env_prim_paths", [])
        if stage is None or not env_paths:
            carb.log_warn("DockingEnv: Scene stage or environment paths unavailable; skipping articulated hole setup.")
            return

        def _apply_small_mass_inertia(prim, mass=0.01, Idiag=(1e-5, 1e-5, 1e-5), disable_gravity=True):
            # Ensure positive mass/inertia without adding colliders
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(mass)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*Idiag))
            if hasattr(PhysxSchema, "PhysxRigidBodyAPI") and disable_gravity:
                rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb_api.CreateDisableGravityAttr(True)

        self._hole_joint_debug_info = []
        for env_path in env_paths:
            hole_body_prim = self._resolve_hole_body_prim(stage, env_path)
            if hole_body_prim is None:
                continue

            # Allow contact reporting on the true hole body
            PhysxSchema.PhysxContactReportAPI.Apply(hole_body_prim)

            # 1) Root anchor (kinematic) -- NO ArticulationRootAPI
            anchor_path = Sdf.Path(f"{env_path}/HoleArticulationRoot")
            anchor_prim = stage.GetPrimAtPath(anchor_path)
            if not anchor_prim.IsValid():
                anchor_prim = stage.DefinePrim(anchor_path, "Xform")

            self._align_anchor_to_hole(stage, anchor_prim, hole_body_prim, env_path)

            UsdPhysics.RigidBodyAPI.Apply(anchor_prim).CreateKinematicEnabledAttr(True)
            # Intentionally do NOT apply ArticulationRootAPI to avoid actuator expectations

            # 2) Create 6 intermediate links and joints: Px, Py, Pz, Rx, Ry, Rz
            axes = [("Px","x"), ("Py","y"), ("Pz","z"), ("Rx","x"), ("Ry","y"), ("Rz","z")]
            parent_path = anchor_path

            for name, axis in axes:
                link_path = Sdf.Path(f"{env_path}/HoleChain_{name}")
                link_prim = stage.GetPrimAtPath(link_path)
                if not link_prim.IsValid():
                    link_prim = stage.DefinePrim(link_path, "Xform")

                # Light rigid body without colliders; assign small positive mass/inertia
                link_rb = UsdPhysics.RigidBodyAPI.Apply(link_prim)
                link_rb.CreateKinematicEnabledAttr(False)
                _apply_small_mass_inertia(link_prim)

                # Create joint between parent and this link
                jpath = Sdf.Path(f"{env_path}/J_{name}")
                if name.startswith("P"):
                    joint = UsdPhysics.PrismaticJoint.Define(stage, jpath)
                    drive_axis = getattr(UsdPhysics.Tokens, "linear")
                else:
                    joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
                    drive_axis = getattr(UsdPhysics.Tokens, "angular")

                joint.CreateBody0Rel().SetTargets([parent_path])
                joint.CreateBody1Rel().SetTargets([link_path])
                joint.CreateAxisAttr(getattr(UsdPhysics.Tokens, axis))
                joint.CreateJointEnabledAttr().Set(True)
                joint.CreateBreakForceAttr().Set(1.0e12)
                joint.CreateBreakTorqueAttr().Set(1.0e12)
                # Key: exclude these joints from articulation so Lab won't require actuators
                joint.CreateExcludeFromArticulationAttr().Set(True)

                # Default zero local frames
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

                # Map D6 drive params to this 1-DoF joint
                keymap = {"Px":"transX","Py":"transY","Pz":"transZ","Rx":"rotX","Ry":"rotY","Rz":"rotZ"}
                pkey = keymap[name]
                params = self._HOLE_COMPLIANCE_DRIVES[pkey]

                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_axis)
                drive.CreateStiffnessAttr(params["stiffness"])
                drive.CreateDampingAttr(params["damping"])
                drive.CreateMaxForceAttr(params["max_force"])
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateTargetVelocityAttr(0.0)
                drive.CreateTypeAttr(getattr(UsdPhysics.Tokens, "force", "force"))

                # Limits
                low, high = self._HOLE_COMPLIANCE_LIMITS[pkey]
                lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), drive_axis)
                lim.CreateLowAttr().Set(low)
                lim.CreateHighAttr().Set(high)
                if hasattr(PhysxSchema, "PhysxLimitAPI"):
                    physx_lim = PhysxSchema.PhysxLimitAPI.Apply(joint.GetPrim(), drive_axis)
                    physx_lim.CreateStiffnessAttr().Set(0.0)
                    physx_lim.CreateDampingAttr().Set(0.0)
                    physx_lim.CreateRestitutionAttr().Set(0.0)
                    physx_lim.CreateBounceThresholdAttr().Set(0.0)

                parent_path = link_path

            # 3) Attach hole body to last link with a FixedJoint (no extra DoF)
            fixed_path = Sdf.Path(f"{env_path}/J_Final_Fixed")
            fj = UsdPhysics.FixedJoint.Define(stage, fixed_path)
            fj.CreateBody0Rel().SetTargets([parent_path])
            fj.CreateBody1Rel().SetTargets([hole_body_prim.GetPath()])
            fj.CreateJointEnabledAttr().Set(True)
            fj.CreateBreakForceAttr().Set(1.0e12)
            fj.CreateBreakTorqueAttr().Set(1.0e12)
            fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            fj.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

            self._hole_joint_debug_info.append({
                "anchor": anchor_path, "hole": hole_body_prim.GetPath(), "env": Sdf.Path(env_path)
            })

        print("DockingEnv: built 6-DoF hole chain without articulation root (Px/Py/Pz + Rx/Ry/Rz + Fixed).")
