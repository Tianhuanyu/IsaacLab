from __future__ import annotations

import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

from isaaclab.sim import utils as sim_utils
from .docking_base_env import DockingBaseEnv
from .docking_env_cfg import DockingEnvCfg
from .docking_tasks_cfg import DockTask


class DockingEnv(DockingBaseEnv):
    """Docking env (multi-env) that logs hole interaction using joint-wrench deltas
    and attaches the single Hole8mm to a compliant 6-DoF chain instead of fixing it to world.
    """

    cfg: DockingEnvCfg

    # Compliance parameters for the hole flex chain.
    _HOLE_COMPLIANCE_DRIVES = {
        "transX": {"stiffness": 2000.0, "damping": 80.0, "max_force": 1.0e6},
        "transY": {"stiffness": 2000.0, "damping": 80.0, "max_force": 1.0e6},
        "transZ": {"stiffness": 3000.0, "damping": 100.0, "max_force": 1.0e6},
        "rotX": {"stiffness": 40.0, "damping": 1.2, "max_force": 1.0e6},
        "rotY": {"stiffness": 40.0, "damping": 1.2, "max_force": 1.0e6},
        "rotZ": {"stiffness": 60.0, "damping": 2.0, "max_force": 1.0e6},
    }
    _HOLE_COMPLIANCE_LIMITS = {
        "transX": (-0.004, 0.004),
        "transY": (-0.004, 0.004),
        "transZ": (-0.002, 0.002),
        "rotX": (-0.0872665, 0.0872665),
        "rotY": (-0.0872665, 0.0872665),
        "rotZ": (-0.0698132, 0.0698132),
    }

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        # Ensure docking-specific task config is active before parent setup.
        if not hasattr(cfg.task, "fixed_asset"):
            old_task = cfg.task
            dock_task = DockTask()
            if hasattr(old_task, "to_dict"):
                dock_task.from_dict(old_task.to_dict())
            cfg.task = dock_task
        cfg.task_name = "dock"

        # Hole identification / bookkeeping
        self._hole_body_index: int | None = None
        self._hole_body_name: str | None = None
        self._hole_index_warning_emitted = False
        self._hole_body_names_missing_logged = False
        self._disabled_hole_constraints_envs: set[str] = set()
        self._debug_pose_warning_emitted = False

        # Logging
        self._hole_log_counter = 0
        self._hole_force_zero_logged = False

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Per-env baselines for wrench & pose (for delta computation)
        self._hole_rest_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        self._hole_rest_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._hole_rest_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Cache baselines now that scene and buffers exist
        self._cache_hole_rest_wrench()
        print(f"DockingEnv Debug: fixed asset body names = {self._fixed_asset.body_names}")

    # -------------------------------------------------------------------------------------
    # Scene setup
    # -------------------------------------------------------------------------------------
    def _setup_scene(self):
        # Parent spawns assets & clones envs
        super()._setup_scene()
        # Build the 6-DoF compliant chain on the hole (per env)
        self._create_hole_articulated_chain()

    # -------------------------------------------------------------------------------------
    # Hole body resolution helpers
    # -------------------------------------------------------------------------------------
    def _resolve_hole_body_prim(self, stage, env_path: str):
        """Find the first rigid body under '{env_path}/FixedAsset' and store its name."""
        fixed_asset_path = f"{env_path}/FixedAsset"
        fixed_prim = stage.GetPrimAtPath(fixed_asset_path)
        if not fixed_prim.IsValid():
            print(f"DockingEnv: FixedAsset prim not found at '{fixed_asset_path}'.")
            return None

        body_prims = sim_utils.get_all_matching_child_prims(
            fixed_prim.GetPath(),
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=True,
        )
        if not body_prims:
            if not self._hole_body_names_missing_logged:
                self._hole_body_names_missing_logged = True
                print(f"DockingEnv: No rigid bodies found under '{fixed_asset_path}'.")
            return None

        preferred = None
        for prim in body_prims:
            path = prim.GetPath().pathString
            if path.endswith("/forge_hole_8mm"):
                preferred = prim
                break
        hole_body_prim = preferred if preferred is not None else body_prims[0]
        if self._hole_body_name is None:
            self._hole_body_name = hole_body_prim.GetName()
            print(
                f"DockingEnv Debug: candidate hole body prim '{hole_body_prim.GetPath()}', "
                f"name '{self._hole_body_name}'."
            )
        return hole_body_prim

    def _resolve_hole_body_index(self):
        """Find the index of the hole body within fixed-asset articulation data."""
        if self._hole_body_index is not None:
            return
        if self._hole_body_name is None:
            if not self._hole_index_warning_emitted:
                self._hole_index_warning_emitted = True
                print("DockingEnv: hole body name unresolved; will retry when data is ready.")
            return
        body_names = getattr(self._fixed_asset.data, "body_names", None)
        if body_names is None or len(body_names) == 0:
            if not self._hole_body_names_missing_logged:
                self._hole_body_names_missing_logged = True
                print("DockingEnv Debug: fixed asset data.body_names unavailable; cannot resolve hole index yet.")
            return
        try:
            self._hole_body_index = list(body_names).index(self._hole_body_name)
            print(f"DockingEnv: hole body resolved to '{self._hole_body_name}' (index {self._hole_body_index}).")
        except ValueError:
            if not self._debug_pose_warning_emitted:
                self._debug_pose_warning_emitted = True
                print(
                    f"DockingEnv: hole body name '{self._hole_body_name}' not in articulation data; "
                    f"available = {list(body_names)}"
                )

    # -------------------------------------------------------------------------------------
    # Build the 6-DoF compliant chain: Px, Py, Pz, Rx, Ry, Rz, then Fixed to hole body
    # -------------------------------------------------------------------------------------
    def _create_hole_articulated_chain(self):
        stage = getattr(self.scene, "stage", None)
        env_paths = getattr(self.scene, "env_prim_paths", [])
        if stage is None or not env_paths:
            print("DockingEnv: Scene stage or environment paths unavailable; skipping articulated hole setup.")
            return

        def _apply_small_mass_inertia(prim, mass=0.01, Idiag=(1e-5, 1e-5, 1e-5)):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(mass)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*Idiag))
            if hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

        for env_path in env_paths:
            hole_body_prim = self._resolve_hole_body_prim(stage, env_path)
            if hole_body_prim is None:
                print(f"DockingEnv: cannot resolve hole body under '{env_path}/FixedAsset'; skipping chain.")
                continue
            self._disable_original_hole_constraints(stage, env_path)

            chain_root_path = Sdf.Path(f"{env_path}/HoleFlex")
            if stage.GetPrimAtPath(chain_root_path).IsValid():
                stage.RemovePrim(chain_root_path)
            stage.DefinePrim(chain_root_path, "Xform")

            # Kinematic anchor colocated with the original hole (in env-local coords)
            anchor_path = chain_root_path.AppendChild("Anchor")
            anchor_prim = stage.DefinePrim(anchor_path, "Xform")
            UsdPhysics.RigidBodyAPI.Apply(anchor_prim).CreateKinematicEnabledAttr(True)

            xform_cache = UsdGeom.XformCache()
            hole_world = xform_cache.GetLocalToWorldTransform(hole_body_prim)
            env_prim = stage.GetPrimAtPath(env_path)
            env_world = xform_cache.GetLocalToWorldTransform(env_prim) if env_prim.IsValid() else Gf.Matrix4d(1.0)
            local_transform = env_world.GetInverse() * hole_world
            UsdGeom.Xformable(anchor_prim).ClearXformOpOrder()
            UsdGeom.Xformable(anchor_prim).AddTransformOp().Set(local_transform)

            axes = [("Px","x"), ("Py","y"), ("Pz","z"), ("Rx","x"), ("Ry","y"), ("Rz","z")]
            parent_path = anchor_path

            for name, axis in axes:
                link_path = chain_root_path.AppendChild(f"Link_{name}")
                link_prim = stage.GetPrimAtPath(link_path)
                if not link_prim.IsValid():
                    link_prim = stage.DefinePrim(link_path, "Xform")

                link_rb = UsdPhysics.RigidBodyAPI.Apply(link_prim)
                link_rb.CreateKinematicEnabledAttr(False)
                _apply_small_mass_inertia(link_prim)

                jpath = chain_root_path.AppendChild(f"J_{name}")
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
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

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

                low, high = self._HOLE_COMPLIANCE_LIMITS[pkey]
                lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), drive_axis)
                lim.CreateLowAttr().Set(low)
                lim.CreateHighAttr().Set(high)

                parent_path = link_path

            # Final fixed joint from the last link to the actual hole rigid body
            fixed_path = chain_root_path.AppendChild("J_Final_Fixed")
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

        print("DockingEnv: built 6-DoF hole chain without articulation root (Px/Py/Pz + Rx/Ry/Rz + Fixed).")

    def _disable_original_hole_constraints(self, stage, env_path: str):
        """Disable fixed joints shipped with the hole asset so it can move under the compliance chain."""
        if env_path in self._disabled_hole_constraints_envs:
            return
        fixed_root = stage.GetPrimAtPath(f"{env_path}/FixedAsset")
        if not fixed_root.IsValid():
            return

        joint_prims = sim_utils.get_all_matching_child_prims(
            fixed_root.GetPath(),
            predicate=lambda prim: prim.GetTypeName().endswith("Joint"),
            traverse_instance_prims=True,
        )
        disabled = False
        for joint_prim in joint_prims:
            name = joint_prim.GetName()
            # 保留我们自己创建的链和任何以 J_ 开头的临时关节命名
            if name.startswith("J_") or "HoleChain" in name:
                continue
            stage.RemovePrim(joint_prim.GetPath())
            disabled = True
            print(f"DockingEnv Debug: removed original joint '{joint_prim.GetPath()}'.")

        if disabled:
            self._disabled_hole_constraints_envs.add(env_path)

    # -------------------------------------------------------------------------------------
    # Rest wrench caching & resets
    # -------------------------------------------------------------------------------------
    def _cache_hole_rest_wrench(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if self._hole_body_index is None:
            self._resolve_hole_body_index()
            if self._hole_body_index is None:
                # Data not ready yet; will try again later
                return

        wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[env_ids, self._hole_body_index].to(self.device)
        self._hole_rest_wrench[env_ids] = wrench
        self._hole_rest_pos[env_ids] = self._fixed_asset.data.body_state_w[env_ids, self._hole_body_index, :3]
        self._hole_rest_quat[env_ids] = self._fixed_asset.data.body_state_w[env_ids, self._hole_body_index, 3:7]

    # --- new: helper to (re)remove baked constraints for specific envs -------------------
    def _disable_constraints_for_env_ids(self, env_ids):
        stage = getattr(self.scene, "stage", None)
        env_paths = getattr(self.scene, "env_prim_paths", [])
        if stage is None or not env_paths:
            return

        if env_ids is None:
            indices = range(len(env_paths))
        elif isinstance(env_ids, torch.Tensor):
            indices = env_ids.tolist()
        else:
            indices = list(env_ids) if isinstance(env_ids, (list, tuple)) else [int(env_ids)]

        for idx in indices:
            if 0 <= idx < len(env_paths):
                self._disable_original_hole_constraints(stage, env_paths[idx])

    # --- override: make sure after factory reset we still have a flexible hole -----------
    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        super()._reset_idx(env_ids)
        # Factory's flows may revive baked joints; immediately undo and realign.
        self._disable_constraints_for_env_ids(env_ids)
        self._realign_hole_flex_chain(env_ids)
        self._cache_hole_rest_wrench(env_ids)

    # Keep FixedAsset free while still resetting held asset, then realign chain
    def _set_assets_to_default_pose(self, env_ids):
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        # 不调用 self._fixed_asset.reset()，避免恢复 USD 自带关节
        self._realign_hole_flex_chain(env_ids)

    # --- override: after Factory randomization, re-disable + realign + re-cache ----------
    def randomize_initial_state(self, env_ids):
        # Run Factory's randomization first (this re-writes root pose and calls fixed_asset.reset()).
        super().randomize_initial_state(env_ids)
        # Then enforce flexible-hole wiring.
        self._disable_constraints_for_env_ids(env_ids)
        self._realign_hole_flex_chain(env_ids)
        self._cache_hole_rest_wrench(env_ids)

    # --- keep anchor colocated with the actual hole body (env-local) ---------------------
    def _realign_hole_flex_chain(self, env_ids):
        stage = getattr(self.scene, "stage", None)
        if stage is None:
            return
        env_paths = getattr(self.scene, "env_prim_paths", [])
        if env_ids is None:
            env_indices = list(range(len(env_paths)))
        elif isinstance(env_ids, torch.Tensor):
            env_indices = env_ids.tolist()
        else:
            env_indices = list(env_ids) if isinstance(env_ids, (list, tuple)) else [int(env_ids)]
        for idx in env_indices:
            if idx >= len(env_paths):
                continue
            env_path = env_paths[idx]
            hole_body_prim = self._resolve_hole_body_prim(stage, env_path)
            if hole_body_prim is None:
                continue
            chain_root_path = Sdf.Path(f"{env_path}/HoleFlex")
            anchor_path = chain_root_path.AppendChild("Anchor")
            anchor_prim = stage.GetPrimAtPath(anchor_path)
            if not anchor_prim.IsValid():
                continue

            xform_cache = UsdGeom.XformCache()
            hole_world = xform_cache.GetLocalToWorldTransform(hole_body_prim)
            env_prim = stage.GetPrimAtPath(env_path)
            env_world = xform_cache.GetLocalToWorldTransform(env_prim) if env_prim.IsValid() else Gf.Matrix4d(1.0)
            local_transform = env_world.GetInverse() * hole_world
            anchor_xform = UsdGeom.Xformable(anchor_prim)
            anchor_xform.ClearXformOpOrder()
            anchor_xform.AddTransformOp().Set(local_transform)

            # reset intermediate links to identity (avoid drift)
            for suffix in ["Px", "Py", "Pz", "Rx", "Ry", "Rz"]:
                link_path = chain_root_path.AppendChild(f"Link_{suffix}")
                link_prim = stage.GetPrimAtPath(link_path)
                if not link_prim.IsValid():
                    continue
                link_xform = UsdGeom.Xformable(link_prim)
                link_xform.ClearXformOpOrder()
                link_xform.AddTransformOp().Set(Gf.Matrix4d(1.0))

    # -------------------------------------------------------------------------------------
    # Control step & logging
    # -------------------------------------------------------------------------------------
    def _apply_action(self):
        super()._apply_action()
        self._log_hole_contact()

    def _log_hole_contact(self):
        if self._hole_body_index is None:
            self._resolve_hole_body_index()
        if self._hole_body_index is None:
            return

        wrench = self._fixed_asset.data.body_incoming_joint_wrench_b[:, self._hole_body_index].to(self.device)
        delta = wrench - self._hole_rest_wrench  # [num_envs, 6]
        hole_force = delta[:, :3]
        hole_torque = delta[:, 3:]

        # Print at low rate
        debug_hole_motion = getattr(self.cfg, "debug_hole_motion", False)
        log_interval = 10 if debug_hole_motion else 100
        self._hole_log_counter += 1
        if (self._hole_log_counter % log_interval) != 0:
            return

        if not self._hole_force_zero_logged:
            max_force = float(hole_force.abs().max().item())
            max_torque = float(hole_torque.abs().max().item())
            if max_force < 1e-8 and max_torque < 1e-8:
                self._hole_force_zero_logged = True
                print("DockingEnv Debug: hole force/torque magnitudes remain near zero; check chain attachment.")

        mean_force = hole_force.abs().mean(dim=0).tolist()
        mean_torque = hole_torque.abs().mean(dim=0).tolist()
        print(f"[Docking(JointOnly)] |F|_mean = {mean_force}, |Tau|_mean = {mean_torque}")

        # Optional pose diagnostics
        curr_pos = self._fixed_asset.data.body_state_w[:, self._hole_body_index, :3]
        curr_quat = self._fixed_asset.data.body_state_w[:, self._hole_body_index, 3:7]
        delta_pos = (curr_pos - self._hole_rest_pos).norm(dim=-1).mean().item()
        dot = torch.sum(curr_quat * self._hole_rest_quat, dim=-1).abs().clamp(max=1.0)
        delta_quat = (1.0 - dot).mean().item()
        if debug_hole_motion:
            mean_force_mag = hole_force.norm(dim=-1).mean().item()
            mean_torque_mag = hole_torque.norm(dim=-1).mean().item()
            displaced = delta_pos > 1e-5 or delta_quat > 1e-4
            print(
                f"[Docking Debug] hole_displaced={displaced} delta_pos={delta_pos:.3e} "
                f"delta_quat={delta_quat:.3e} |F|={mean_force_mag:.3e} |Tau|={mean_torque_mag:.3e}"
            )
        if delta_pos > 1e-5 or delta_quat > 1e-4:
            print(f"[Docking Debug] hole_delta_pos={delta_pos:.3e}, hole_delta_quat={delta_quat:.3e}")
