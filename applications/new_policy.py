# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController


class SpotFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1


class SpotArmFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        walking_policy_path: str = None,
        balance_policy_path: str = None,
        arm_balance_policy_path: str = None,
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load hybrid RL policies.
        """
        super().__init__(name=name, prim_path=prim_path, root_path=root_path, usd_path=usd_path, position=position, orientation=orientation)

        self.load_hybrid_policies(walking_policy_path, balance_policy_path, arm_balance_policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0

    def load_hybrid_policies(self, walking_policy_path: str, balance_policy_path: str, arm_balance_policy_path: str = None, policy_params_path: str = None) -> None:
        """Loads both walking JIT policy and balance state_dict policy."""
        import torch
        import io
        import omni
        from rsl_rl.models.mlp_model import MLPModel
        from isaacsim.robot.policy.examples.controllers.config_loader import parse_env_config, get_physics_properties
        
        self.policy_env_params = parse_env_config(policy_params_path)
        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)
        
        # 1. Load Walking Policy (TorchScript JIT or State Dict)
        try:
            file_content = omni.client.read_file(walking_policy_path)[2]
            file = io.BytesIO(memoryview(file_content).tobytes())
            self.walking_policy = torch.jit.load(file)
            self.walking_policy.eval()
            self._walking_is_jit = True
        except Exception:
            checkpoint = torch.load(walking_policy_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict", checkpoint))
            dummy_obs = {"policy": torch.zeros(1, 69)}
            obs_groups = {"actor": ["policy"]}
            self.walking_policy = MLPModel(
                obs=dummy_obs,
                obs_groups=obs_groups,
                obs_set="actor",
                output_dim=19,
                hidden_dims=[512, 256, 128],
                activation="elu",
                obs_normalization=False
            )
            self.walking_policy.load_state_dict(state_dict, strict=False)
            self.walking_policy.eval()
            self._walking_is_jit = False

        # 2. Load Balance Policy (State Dict)
        checkpoint = torch.load(balance_policy_path, map_location="cpu", weights_only=False)
        if "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        else:
            state_dict = checkpoint

        dummy_obs = {"policy": torch.zeros(1, 69)}
        obs_groups = {"actor": ["policy"]}
        self.balance_policy = MLPModel(
            obs=dummy_obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=19,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False
        )
        self.balance_policy.load_state_dict(state_dict, strict=False)
        self.balance_policy.eval()
        
        # 3. Load Arm Balance Policy (76-dim)
        if arm_balance_policy_path:
            arm_checkpoint = torch.load(arm_balance_policy_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in arm_checkpoint:
                arm_state_dict = arm_checkpoint["model_state_dict"]
            elif "actor_state_dict" in arm_checkpoint:
                arm_state_dict = arm_checkpoint["actor_state_dict"]
            else:
                arm_state_dict = arm_checkpoint
                
            dummy_obs_arm = {"policy": torch.zeros(1, 76)}
            self.arm_balance_policy = MLPModel(
                obs=dummy_obs_arm,
                obs_groups={"actor": ["policy"]},
                obs_set="actor",
                output_dim=19,
                hidden_dims=[512, 256, 128],
                activation="elu",
                obs_normalization=False
            )
            self.arm_balance_policy.load_state_dict(arm_state_dict, strict=False)
            self.arm_balance_policy.eval()
        
        self.use_balance_policy = False
        self.balance_timer = 0.0
        self.extend_arm_mode = False
        self.use_arm_balance_policy = False
        self.target_arm_angles = None

    def trigger_balance_mode(self, duration=3.0):
        self.use_balance_policy = True
        self.balance_timer = duration

    def _compute_action(self, obs):
        """Override to run through hybrid policies"""
        import torch
        with torch.no_grad():
            if getattr(self, "use_arm_balance_policy", False) and hasattr(self, "arm_balance_policy") and self.target_arm_angles is not None:
                obs_76 = np.zeros(76)
                obs_76[:69] = obs
                obs_76[69:76] = self.target_arm_angles
                obs_tensor = torch.FloatTensor(obs_76).unsqueeze(0)
                obs_dict = {"policy": obs_tensor}
                action = self.arm_balance_policy(obs_dict).numpy()[0]
            elif self.balance_timer > 0 or getattr(self, "extend_arm_mode", False):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                obs_dict = {"policy": obs_tensor}
                action = self.balance_policy(obs_dict).numpy()[0]
                if self.balance_timer > 0:
                    self.balance_timer -= self._dt * self._decimation
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                self.use_balance_policy = False
                if getattr(self, "_walking_is_jit", True):
                    action = self.walking_policy(obs_tensor).detach().view(-1).numpy()
                else:
                    action = self.walking_policy({"policy": obs_tensor}).numpy()[0]
                
        return action

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(69)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:31] = current_joint_pos - self.default_pos
        obs[31:50] = current_joint_vel
        # Previous Action
        obs[50:69] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            
            # 부드러운 팔 뻗기 보간(Interpolation) 로직 (현재는 사용하지 않으므로 주석 처리)
            # if not hasattr(self, "arm_extend_factor"):
            #     self.arm_extend_factor = 0.0
            #     
            # target_factor = 1.0 if getattr(self, "extend_arm_mode", False) else 0.0
            # 
            # # 팔을 뻗을 때는 로봇이 완전히 멈췄을 때(선속도 < 0.15)만 펴지기 시작하도록 합니다.
            # if target_factor > 0.5:
            #     lin_vel = self.robot.get_linear_velocity()
            #     speed = np.linalg.norm(lin_vel[:2]) if lin_vel is not None else 0.0
            #     if speed < 0.15:
            #         self.arm_extend_factor += (target_factor - self.arm_extend_factor) * 0.01
            # else:
            #     # 접을 때는 즉시 접기 시작합니다.
            #     self.arm_extend_factor += (target_factor - self.arm_extend_factor) * 0.02
            # 
            # if self.arm_extend_factor > 0.01:
            #     if not hasattr(self, "_arm_indices"):
            #         self._arm_indices = {}
            #         dofs = self.robot.dof_names if hasattr(self.robot, "dof_names") else self.robot._articulation_view.joint_names
            #         for i, name in enumerate(dofs):
            #             if "arm" in name:
            #                 self._arm_indices[name] = i
            #                 
            #     # 팔을 앞으로 뻗는 자세 적용 (어깨 -0.5, 팔꿈치 1.0)
            #     if "arm0_sh1" in self._arm_indices:
            #         idx = self._arm_indices["arm0_sh1"]
            #         target_action = (-0.5 - self.default_pos[idx]) / self._action_scale
            #         self.action[idx] = (1 - self.arm_extend_factor) * self.action[idx] + self.arm_extend_factor * target_action
            #     if "arm0_el0" in self._arm_indices:
            #         idx = self._arm_indices["arm0_el0"]
            #         target_action = (1.0 - self.default_pos[idx]) / self._action_scale
            #         self.action[idx] = (1 - self.arm_extend_factor) * self.action[idx] + self.arm_extend_factor * target_action

            self._previous_action = self.action.copy()

        target_positions = self.default_pos + (self.action * self._action_scale)
        
        # 기본 Stiffness/Damping 저장 (복구용)
        if not hasattr(self, "default_stiffness"):
            try:
                self.default_stiffness = self.robot._articulation_view.get_stiffnesses()[0].copy()
                self.default_damping = self.robot._articulation_view.get_dampings()[0].copy()
            except Exception:
                pass # 만약 지원하지 않는다면 None으로 유지
                
        target_stiffness = None
        target_damping = None
        
        # 1. 레그 프리즈 기능 (다리 얼리기)
        if hasattr(self, "override_leg_freeze") and self.override_leg_freeze:
            if not hasattr(self, "_leg_indices_list"):
                dofs = self.robot.dof_names if hasattr(self.robot, "dof_names") else self.robot._articulation_view.joint_names
                self._leg_indices_list = [i for i, name in enumerate(dofs) if "arm" not in name and "f1x" not in name]
                
            # 프리즈가 시작되는 순간의 현재 다리 관절 각도를 캡처하여 저장 (갑자기 튀는 현상 방지)
            if not getattr(self, "_leg_frozen", False):
                self._leg_frozen = True
                self._frozen_leg_pos = self.robot.get_joint_positions().copy()
            
            for idx in self._leg_indices_list:
                # default_pos가 아닌 멈춘 순간의 각도로 고정
                target_positions[idx] = self._frozen_leg_pos[idx]
                
            if hasattr(self, "default_stiffness") and self.default_stiffness is not None:
                target_stiffness = self.default_stiffness.copy()
                target_damping = self.default_damping.copy()
                for idx in self._leg_indices_list:
                    target_stiffness[idx] = 2000.0
                    target_damping[idx] = 100.0
        else:
            self._leg_frozen = False
            # 원상복구 (freeze 해제 시 원래 강도로 복구)
            if hasattr(self, "override_leg_freeze") and not self.override_leg_freeze:
                if hasattr(self, "default_stiffness") and self.default_stiffness is not None:
                    target_stiffness = self.default_stiffness.copy()
                    target_damping = self.default_damping.copy()
        
        # 2. 오버라이드 기능: 외부(spot_warehouse)에서 특정 모션을 위해 팔/그리퍼 각도를 강제 지정할 경우 적용
        if hasattr(self, "override_arm_angles") and self.override_arm_angles is not None:
            if not hasattr(self, "_arm_indices_list"):
                dofs = self.robot.dof_names if hasattr(self.robot, "dof_names") else self.robot._articulation_view.joint_names
                self._arm_indices_list = [i for i, name in enumerate(dofs) if "arm0" in name and "f1x" not in name]
                self._grip_idx = next((i for i, name in enumerate(dofs) if "f1x" in name), None)
            
            if target_stiffness is None and hasattr(self, "default_stiffness") and self.default_stiffness is not None:
                target_stiffness = self.default_stiffness.copy()
                target_damping = self.default_damping.copy()
                
            for i, idx in enumerate(self._arm_indices_list):
                if i < len(self.override_arm_angles):
                    target_positions[idx] = self.override_arm_angles[i]
                    if target_stiffness is not None:
                        target_stiffness[idx] = 5000.0
                        target_damping[idx] = 250.0
            if hasattr(self, "override_grip_angle") and self._grip_idx is not None:
                target_positions[self._grip_idx] = self.override_grip_angle
                if target_stiffness is not None:
                    target_stiffness[self._grip_idx] = 5000.0
                    target_damping[self._grip_idx] = 250.0

        if target_stiffness is not None:
            action = ArticulationAction(joint_positions=target_positions, joint_stiffnesses=target_stiffness, joint_dampings=target_damping)
        else:
            action = ArticulationAction(joint_positions=target_positions)
        self.robot.apply_action(action)

        self._policy_counter += 1
