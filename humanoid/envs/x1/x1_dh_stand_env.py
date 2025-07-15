# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from humanoid.utils.math import wrap_to_pi


import torch
from enum import Enum
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  Terrain


class GaitState(Enum):
    """Enumeration for gait states to handle walk-to-stand transitions"""
    WALKING = "walking"
    TRANSITIONING = "transitioning" 
    STANDING = "standing"

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class X1DHStandEnv(LeggedRobot):
    '''
    X1DHStandEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = self.cfg.rewards.feet_to_ankle_distance
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        
        # V2.2: Add transition state management for smooth walk-to-stand transitions
        # State encoding: 0=WALKING, 1=TRANSITIONING, 2=STANDING
        self.gait_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)  # Start in WALKING
        self.transition_timer = torch.zeros(self.num_envs, device=self.device)
        self.transition_duration = getattr(cfg.commands, 'transition_duration', 1.0)  # Default 1 second
        self.previous_command_norm = torch.zeros(self.num_envs, device=self.device)      


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)  #angular vel xyz

        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_gait_state(self):
        """V2.5: Update gait state for smooth walk-to-stand transitions with better triggering"""
        current_command_norm = torch.norm(self.commands[:, :3], dim=1)
        stand_command = current_command_norm <= self.cfg.commands.stand_com_threshold
        walk_command = current_command_norm > self.cfg.commands.stand_com_threshold
        
        # State transitions
        walking_mask = self.gait_state == 0  # WALKING
        transitioning_mask = self.gait_state == 1  # TRANSITIONING
        standing_mask = self.gait_state == 2  # STANDING
        
        # Transition from WALKING to TRANSITIONING when stand command is given
        start_transition = walking_mask & stand_command
        self.gait_state[start_transition] = 1  # TRANSITIONING
        self.transition_timer[start_transition] = 0.0
        
        # Transition from STANDING to WALKING when walk command is given
        start_walking = standing_mask & walk_command
        self.gait_state[start_walking] = 0  # WALKING
        self.transition_timer[start_walking] = 0.0
        
        # Update transition timer
        self.transition_timer[transitioning_mask] += self.dt
        
        # Complete transition from TRANSITIONING to STANDING
        transition_complete = transitioning_mask & (self.transition_timer >= self.transition_duration)
        # Also check if both feet are in contact for stable transition
        both_feet_contact = self._check_both_feet_stable()
        complete_transition = transition_complete & both_feet_contact
        self.gait_state[complete_transition] = 2  # STANDING
        
        # V2.5: 添加gait_state分布调试输出
        if self.common_step_counter % 100 == 0:  # 每100步输出一次
            gait_counts = torch.bincount(self.gait_state, minlength=3)
            print(f"[DEBUG] Step {self.common_step_counter}: Gait state distribution - WALKING: {gait_counts[0]}, TRANSITIONING: {gait_counts[1]}, STANDING: {gait_counts[2]}")
        
        self.previous_command_norm = current_command_norm.clone()

    def _check_both_feet_stable(self):
        """V2.2: Check if both feet are in stable contact with ground"""
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        both_feet_stable = torch.all(foot_contact, dim=1)
        return both_feet_stable
    
    def _get_transition_progress(self):
        """V2.2: Get transition progress for smooth interpolation"""
        transitioning_mask = self.gait_state == 1
        progress = torch.zeros(self.num_envs, device=self.device)
        progress[transitioning_mask] = torch.clamp(
            self.transition_timer[transitioning_mask] / self.transition_duration, 0.0, 1.0
        )
        return progress
    
    def _get_coordinated_foot_transition(self):
        """V2.2: Get coordinated foot landing sequence during transitions"""
        transition_progress = self._get_transition_progress()
        transitioning_mask = self.gait_state == 1
        
        # Phase-based foot priority: swing foot lands first, then stance foot coordinates
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        # Determine which foot is currently swinging (should land first)
        left_swing = sin_pos <= 0  # Left foot in swing when sin_pos <= 0
        right_swing = sin_pos > 0   # Right foot in swing when sin_pos > 0
        
        # Foot landing timing - swing foot gets priority
        swing_foot_progress = torch.clamp(transition_progress * 2.0, 0.0, 1.0)  # Complete in first half
        stance_foot_progress = torch.clamp((transition_progress - 0.5) * 2.0, 0.0, 1.0)  # Start after swing
        
        # Initialize progress for both feet
        left_foot_progress = torch.zeros(self.num_envs, device=self.device)
        right_foot_progress = torch.zeros(self.num_envs, device=self.device)
        
        # Apply coordinated timing only during transition
        transitioning_envs = transitioning_mask
        
        # Left foot timing
        left_foot_progress[transitioning_envs & left_swing] = swing_foot_progress[transitioning_envs & left_swing]
        left_foot_progress[transitioning_envs & right_swing] = stance_foot_progress[transitioning_envs & right_swing]
        
        # Right foot timing  
        right_foot_progress[transitioning_envs & right_swing] = swing_foot_progress[transitioning_envs & right_swing]
        right_foot_progress[transitioning_envs & left_swing] = stance_foot_progress[transitioning_envs & left_swing]
        
        # For non-transitioning environments, use standard values
        left_foot_progress[~transitioning_mask] = 1.0
        right_foot_progress[~transitioning_mask] = 1.0
        
        return left_foot_progress, right_foot_progress

    def _detect_ankle_roll_oscillation(self):
        """V2.3: Detect ankle roll oscillation during transitions"""
        # Ankle roll joint indices: left=5, right=11
        left_ankle_roll_pos = self.dof_pos[:, 5]   # left_ankle_roll_joint
        right_ankle_roll_pos = self.dof_pos[:, 11] # right_ankle_roll_joint
        left_ankle_roll_vel = self.dof_vel[:, 5]
        right_ankle_roll_vel = self.dof_vel[:, 11]
        
        # Detect high-frequency oscillation (high angular velocity)
        vel_threshold = getattr(self.cfg.control, 'ankle_oscillation_velocity_threshold', 2.0)
        left_oscillation = torch.abs(left_ankle_roll_vel) > vel_threshold
        right_oscillation = torch.abs(right_ankle_roll_vel) > vel_threshold
        
        # Detect position deviation from neutral (large roll angles)
        pos_threshold = getattr(self.cfg.control, 'ankle_oscillation_position_threshold', 0.1)
        left_deviation = torch.abs(left_ankle_roll_pos) > pos_threshold
        right_deviation = torch.abs(right_ankle_roll_pos) > pos_threshold
        
        # Combined ankle disturbance detection
        ankle_disturbance = left_oscillation | right_oscillation | left_deviation | right_deviation
        
        return ankle_disturbance, left_oscillation, right_oscillation, left_deviation, right_deviation

    def _update_ankle_control_during_transition(self):
        """V2.3: Dynamically adjust ankle control parameters during transitions"""
        transitioning_mask = self.gait_state == 1  # TRANSITIONING state
        
        if torch.any(transitioning_mask):
            # Get multipliers from config
            stiffness_mult = getattr(self.cfg.control, 'transition_stiffness_multiplier', 1.5)
            damping_mult = getattr(self.cfg.control, 'transition_damping_multiplier', 3.0)
            
            # Ankle roll joint indices
            left_ankle_roll_idx = 5
            right_ankle_roll_idx = 11
            
            # Store original values if not already stored
            if not hasattr(self, '_original_ankle_stiffness'):
                self._original_ankle_stiffness = 60.0  # From config
                self._original_ankle_damping = 2.0     # From config
            
            # Calculate enhanced parameters
            enhanced_stiffness = self._original_ankle_stiffness * stiffness_mult
            enhanced_damping = self._original_ankle_damping * damping_mult
            
            # Apply enhanced parameters to transitioning environments
            # Note: This would require access to the underlying control system
            # For now, we'll modify the reference positions more aggressively
            return enhanced_stiffness, enhanced_damping
        
        return None, None

    def _suppress_ankle_roll_oscillation(self):
        """V2.3: Actively suppress ankle roll oscillation"""
        ankle_disturbance, left_osc, right_osc, left_dev, right_dev = self._detect_ankle_roll_oscillation()
        
        if torch.any(ankle_disturbance):
            # Target ankle roll position (neutral)
            target_ankle_roll = 0.0
            
            # Apply aggressive correction to disturbed ankles
            # Left ankle correction
            needs_left_correction = left_osc | left_dev
            if torch.any(needs_left_correction):
                self.ref_dof_pos[needs_left_correction, 5] = target_ankle_roll
            
            # Right ankle correction
            needs_right_correction = right_osc | right_dev
            if torch.any(needs_right_correction):
                self.ref_dof_pos[needs_right_correction, 11] = target_ankle_roll
            
            return ankle_disturbance
        
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _compute_balance_ankle_correction(self):
        """V2.3: Compute ankle roll correction based on center of mass balance"""
        # Get center of mass position
        com_position = self.root_states[:, :3]  # [x, y, z]
        
        # Get foot positions 
        foot_positions = self.rigid_state[:, self.feet_indices, :3]  # [left_foot, right_foot]
        
        # Calculate foot center (midpoint between feet)
        foot_center = torch.mean(foot_positions, dim=1)  # Average of left and right foot positions
        
        # Calculate lateral (Y-axis) offset of COM from foot center
        lateral_offset = com_position[:, 1] - foot_center[:, 1]
        
        # Convert lateral offset to ankle roll correction (small angle approximation)
        # Negative feedback: if COM shifts right (+Y), roll ankles left (-angle) to compensate
        ankle_roll_correction = -lateral_offset * 0.5  # Scaling factor for correction strength
        
        return ankle_roll_correction

    def _apply_transition_ankle_control(self):
        """V2.3: Apply specialized ankle control during transitions"""
        transitioning_mask = self.gait_state == 1
        
        if torch.any(transitioning_mask):
            # Step 1: Apply dynamic parameter enhancement
            enhanced_stiffness, enhanced_damping = self._update_ankle_control_during_transition()
            
            # Step 2: Suppress any detected oscillations
            oscillation_suppressed = self._suppress_ankle_roll_oscillation()
            
            # Step 3: Apply balance-based correction for transitioning robots
            ankle_correction = self._compute_balance_ankle_correction()
            
            # Apply balance correction to transitioning environments (only if not being suppressed)
            non_suppressed_transition = transitioning_mask & (~oscillation_suppressed)
            if torch.any(non_suppressed_transition):
                self.ref_dof_pos[non_suppressed_transition, 5] += ankle_correction[non_suppressed_transition]   # left ankle
                self.ref_dof_pos[non_suppressed_transition, 11] += ankle_correction[non_suppressed_transition]  # right ankle

    def  _get_phase(self):
        """V2.2: Enhanced phase calculation with smooth transition support"""
        cycle_time = self.cfg.rewards.cycle_time
        
        # Update gait state first
        self._update_gait_state()
        
        if self.cfg.commands.sw_switch:
            # V2.2: Use gait state instead of immediate command check
            walking_mask = self.gait_state == 0  # WALKING
            transitioning_mask = self.gait_state == 1  # TRANSITIONING
            standing_mask = self.gait_state == 2  # STANDING
            
            # For standing robots, phase is 0
            self.phase_length_buf[standing_mask] = 0
            
            # For transitioning robots, gradually reduce phase
            transition_progress = self._get_transition_progress()
            transitioning_phase_factor = 1.0 - transition_progress
            
            # Calculate base phase for walking and transitioning robots
            base_phase = self.phase_length_buf * self.dt / cycle_time + self.gait_start
            
            # Apply phase calculation
            phase = torch.zeros_like(base_phase)
            phase[walking_mask] = base_phase[walking_mask]
            phase[transitioning_mask] = base_phase[transitioning_mask] * transitioning_phase_factor[transitioning_mask]
            phase[standing_mask] = 0.0
        else:
            phase = self.episode_length_buf * self.dt / cycle_time + self.gait_start

        return phase

    def _get_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Add double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        # stand mask == 1 means stand leg 
        return stance_mask

    def generate_gait_time(self,envs):
        if len(envs) == 0:
            return

        # rand sample 
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                            gait_time_range[1],
                                            (len(envs), 1),device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor,dim=1,keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:,1:] = scaled_tensor[:,:-1].clone()
        scaled_tensor[:,0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor,dim=1).int()
     
    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
            
    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
    
    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self,env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)
        
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

    def compute_ref_state(self):
        """V2.2: Enhanced reference state computation with coordinated foot transitions"""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        
        # V2.2: Get coordinated foot transition progress
        left_foot_progress, right_foot_progress = self._get_coordinated_foot_transition()
        transitioning_mask = self.gait_state == 1
        standing_mask = self.gait_state == 2
        
        # Original swing logic for walking
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0
        
        # Left leg joints (0-5)
        left_swing_deltas = torch.zeros((self.num_envs, 6), device=self.device)
        left_swing_deltas[:, 0] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[0]
        left_swing_deltas[:, 1] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[1]
        left_swing_deltas[:, 2] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[2]
        left_swing_deltas[:, 3] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[3]
        left_swing_deltas[:, 4] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[4]
        left_swing_deltas[:, 5] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[5]
        
        # Right leg joints (6-11)
        right_swing_deltas = torch.zeros((self.num_envs, 6), device=self.device)
        right_swing_deltas[:, 0] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[6]
        right_swing_deltas[:, 1] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[7]
        right_swing_deltas[:, 2] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[8]
        right_swing_deltas[:, 3] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[9]
        right_swing_deltas[:, 4] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[10]
        right_swing_deltas[:, 5] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[11]
        
        # V2.2: Apply coordinated transition smoothing
        # For transitioning robots, interpolate towards default position based on foot progress
        if torch.any(transitioning_mask):
            # Smooth transition to default position
            left_transition_factor = 1.0 - left_foot_progress[transitioning_mask]
            right_transition_factor = 1.0 - right_foot_progress[transitioning_mask]
            
            left_swing_deltas[transitioning_mask] *= left_transition_factor.unsqueeze(1)
            right_swing_deltas[transitioning_mask] *= right_transition_factor.unsqueeze(1)
        
        # Set joint positions
        self.ref_dof_pos[:, 0:6] = left_swing_deltas
        self.ref_dof_pos[:, 6:12] = right_swing_deltas

        # Original double support logic (but respect transition states)
        double_support_mask = torch.abs(sin_pos) < 0.1
        walking_double_support = double_support_mask & (self.gait_state == 0)
        self.ref_dof_pos[walking_double_support] = 0.
        
        # V2.2: Standing robots always have zero deltas
        self.ref_dof_pos[standing_mask] = 0.
        
        # V2.3: Apply ankle roll stability control during transitions
        self._apply_transition_ankle_control()
        
        # if use_ref_actions=True, action += ref_action
        self.ref_action = 2 * self.ref_dof_pos
        
        # self.ref_dof_pos set ref dof pos for swing leg, ref_dof_pos=0 for stance leg
        self.ref_dof_pos += self.default_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: self.cfg.env.num_commands] = 0.  # commands
        noise_vec[self.cfg.env.num_commands: self.cfg.env.num_commands+self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[self.cfg.env.num_commands+self.num_actions: self.cfg.env.num_commands+2*self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[self.cfg.env.num_commands+2*self.num_actions: self.cfg.env.num_commands+3*self.num_actions] = 0.  # previous actions
        noise_vec[self.cfg.env.num_commands+3*self.num_actions: self.cfg.env.num_commands+3*self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[self.cfg.env.num_commands+3*self.num_actions + 3: self.cfg.env.num_commands+3*self.num_actions + 6] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec



    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        return super().step(actions)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        # critic no lag
        diff = self.dof_pos - self.ref_dof_pos
        # 73
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 10.,  # 1 # sum of all fix link mass
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)
        
        # random add dof_pos and dof_vel same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]  
        # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]
                
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        # imu lag, including rpy and omega
        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]
        
        # obs q and dq
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel  

        # 47
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12
            dq,  # 12
            self.actions,   # 12
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.env.num_single_obs == 48:
            stand_command = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) <= self.cfg.commands.stand_com_threshold)
            obs_buf = torch.cat((obs_buf, stand_command),dim=1)
            
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)
        
        if self.add_noise:  
            # add obs noise
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)
        
        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        
        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)*0.5
        
        # V2.2: Reset transition state variables
        self.gait_state[env_ids] = 0  # Reset to WALKING
        self.transition_timer[env_ids] = 0.0
        self.previous_command_norm[env_ids] = 0.0
        
        #resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # V2.9 Fix: 修复子类方法重写冲突，强制显示所有奖励
            reward_value = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = reward_value
            self.episode_sums[key][env_ids] = 0.
            
            # V2.9: 强制调试输出踝关节相关奖励
            if key in ['ankle_roll_stability', 'foot_coordination_during_transition', 'smooth_transition']:
                print(f"[V2.9 DEBUG] {key}: {reward_value.item():.6f}")
            
        # V2.9: 输出episode_sums统计信息
        if len(env_ids) > 0:  # 只在有环境重置时输出
            all_keys = list(self.episode_sums.keys())
            print(f"[V2.9 INFO] Total rewards in episode_sums: {len(all_keys)}")
            ankle_keys = [k for k in all_keys if 'ankle' in k or 'transition' in k or 'smooth' in k]
            if ankle_keys:
                print(f"[V2.9 INFO] Ankle-related rewards found: {ankle_keys}")
            else:
                print(f"[V2.9 WARNING] No ankle-related rewards in episode_sums")
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)
        
        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait) ,dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5
        
        # V2.2: Initialize transition state buffers 
        # Note: These are also initialized in __init__, but we ensure they exist here
        if not hasattr(self, 'gait_state'):
            self.gait_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        if not hasattr(self, 'transition_timer'):
            self.transition_timer = torch.zeros(self.num_envs, device=self.device)
        if not hasattr(self, 'previous_command_norm'):
            self.previous_command_norm = torch.zeros(self.num_envs, device=self.device)

# ================================================ Rewards ================================================== #
    def _reward_ref_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0
        return r
    
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact conditions.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, [1,2,5]]
        right_yaw_roll = joint_diff[:, [7,8,11]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_stance_mask()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        r = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        return r
    
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r 
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_to_ankle_distance
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_stance_mask()

        # feet height should larger than target feet height at the peak
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height) * (self.feet_height < self.cfg.rewards.target_feet_height_max)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.1 * absolute_command  # V2.11: 从1.2降为1.1，更严格的速度控制
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high - V2.11: 增加过速惩罚，从0.改为-0.8
        reward[speed_too_high] = -0.8
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_ankle_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        ankle_idx = [4,5,10,11]
        return torch.sum(torch.square(self.torques[:,ankle_idx]), dim=1)
    
    def _reward_feet_rotation(self):
        feet_euler_xyz = self.feet_euler_xyz
        rotation = torch.sum(torch.square(feet_euler_xyz[:,:,:2]),dim=[1,2])
        # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
        r = torch.exp(-rotation*15)
        return r

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_smooth_transition(self):
        """V2.2: Reward smooth walk-to-stand transitions and coordinated foot behavior"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        transitioning_mask = self.gait_state == 1
        if not torch.any(transitioning_mask):
            return reward
            
        # Get transition progress and foot coordination
        transition_progress = self._get_transition_progress()
        left_foot_progress, right_foot_progress = self._get_coordinated_foot_transition()
        
        # Reward coordinated foot landing (no simultaneous foot movements)
        foot_progress_diff = torch.abs(left_foot_progress - right_foot_progress)
        coordination_bonus = torch.exp(-foot_progress_diff * 5.0)  # Higher reward for coordinated timing
        
        # Reward stable transition progress (smooth, no oscillations)
        transition_smoothness = 1.0 - torch.abs(transition_progress - 0.5) * 2.0  # Peak at 50% progress
        
        # Bonus for maintaining balance during transition
        both_feet_stable = self._check_both_feet_stable()
        balance_bonus = torch.where(both_feet_stable, 1.2, 0.8)
        
        # Combine rewards for transitioning environments
        total_reward = (coordination_bonus + transition_smoothness) * balance_bonus
        reward[transitioning_mask] = total_reward[transitioning_mask]
        
        return reward
    
    def _reward_foot_coordination_during_transition(self):
        """V2.2: Penalize foot disturbances and simultaneous foot movements during transitions"""
        penalty = torch.zeros(self.num_envs, device=self.device)
        
        transitioning_mask = self.gait_state == 1
        if not torch.any(transitioning_mask):
            return penalty
            
        # Get foot velocities
        left_foot_vel = torch.norm(self.rigid_state[:, self.feet_indices[0], 7:10], dim=1)
        right_foot_vel = torch.norm(self.rigid_state[:, self.feet_indices[1], 7:10], dim=1)
        
        # Penalize high foot velocities during transition (indicates disturbance)
        foot_disturbance = (left_foot_vel + right_foot_vel) / 2.0
        disturbance_penalty = torch.clamp(foot_disturbance - 0.5, 0.0, 2.0)  # Penalize > 0.5 m/s
        
        # Penalize simultaneous high velocities (both feet moving fast = poor coordination)
        simultaneous_movement = torch.minimum(left_foot_vel, right_foot_vel)
        coordination_penalty = torch.clamp(simultaneous_movement - 0.3, 0.0, 1.0)
        
        total_penalty = disturbance_penalty + coordination_penalty
        penalty[transitioning_mask] = total_penalty[transitioning_mask]
        
        return penalty

    def _reward_ankle_roll_stability(self):
        """V2.9: 踝关节稳定性奖励 - 添加函数调用验证"""
        # V2.9: 验证函数被调用
        if hasattr(self, '_ankle_debug_counter'):
            self._ankle_debug_counter += 1
        else:
            self._ankle_debug_counter = 1
            print(f"[V2.9 VERIFY] _reward_ankle_roll_stability function called! Counter: {self._ankle_debug_counter}")
        
        # Get ankle roll joint positions and velocities
        left_ankle_roll_pos = self.dof_pos[:, 5]   # left_ankle_roll_joint
        right_ankle_roll_pos = self.dof_pos[:, 11] # right_ankle_roll_joint
        left_ankle_roll_vel = self.dof_vel[:, 5]
        right_ankle_roll_vel = self.dof_vel[:, 11]
        
        # Reward for keeping ankle roll positions close to neutral (0)
        position_stability = torch.exp(-10.0 * (torch.abs(left_ankle_roll_pos) + torch.abs(right_ankle_roll_pos)))
        
        # Reward for low ankle roll velocities (less oscillation)
        velocity_stability = torch.exp(-5.0 * (torch.abs(left_ankle_roll_vel) + torch.abs(right_ankle_roll_vel)))
        
        # Combined stability reward
        stability_reward = (position_stability + velocity_stability) / 2.0
        
        # V2.8: 始终给予基础奖励，过渡状态时给额外奖励
        transitioning_mask = self.gait_state == 1
        reward = stability_reward * 0.2  # 基础奖励
        reward[transitioning_mask] = stability_reward[transitioning_mask] * 1.0  # 过渡状态全额奖励
        
        return reward

    def _reward_speed_limit(self):
        """
        V2.11: 独立的速度限制奖励，防止过快移动
        对超过安全速度上限的情况施加额外惩罚
        """
        max_safe_speed = 0.7  # m/s 安全速度上限
        current_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        
        # 超过安全速度时施加惩罚
        speed_penalty = torch.where(
            current_speed > max_safe_speed,
            -2.0 * (current_speed - max_safe_speed),  # 超速惩罚，与超速量成正比
            0.0
        )
        
        return speed_penalty

    def _reward_soft_landing(self):
        """
        V2.12: 软着陆奖励函数，鼓励平缓着地，惩罚冲击性着地
        基于足部接触力变化率来评估着地质量，实现轻盈步态
        """
        # 获取当前足部接触力
        current_contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        
        # 初始化上一步接触力（如果还没有记录）
        if not hasattr(self, 'last_contact_forces'):
            self.last_contact_forces = torch.zeros_like(current_contact_forces)
        
        # 计算接触力变化率
        contact_force_change = current_contact_forces - self.last_contact_forces
        
        # 对每只脚计算软着陆奖励
        soft_landing_reward = torch.zeros(self.num_envs, device=self.device)
        
        for foot_idx in range(len(self.feet_indices)):
            force_change = contact_force_change[:, foot_idx]
            current_force = current_contact_forces[:, foot_idx]
            
            # 检测着地瞬间（接触力从低到高的变化）
            landing_mask = (current_force > 10.0) & (force_change > 0.0)
            
            # 计算着地质量奖励
            if torch.any(landing_mask):
                # 平缓着地：力变化小 -> 正奖励
                gentle_landing = torch.exp(-force_change * 0.01)
                
                # 冲击着地：力变化大 -> 负奖励
                impact_penalty = torch.where(
                    force_change > 50.0,  # 接触力变化超过50N视为冲击
                    -0.5 * (force_change - 50.0) * 0.01,
                    0.0
                )
                
                # 综合奖励
                foot_reward = gentle_landing + impact_penalty
                soft_landing_reward += foot_reward * landing_mask.float()
        
        # 更新上一步接触力
        self.last_contact_forces = current_contact_forces.clone()
        
        return soft_landing_reward

    def _reward_vertical_speed_control(self):
        """
        V2.12: 垂直速度控制奖励，鼓励轻柔的抬脚和落脚动作
        控制足部垂直速度，避免过快下落导致冲击
        """
        # 获取足部垂直速度
        feet_vertical_vel = self.rigid_state[:, self.feet_indices, 9]  # z方向速度
        
        # 获取足部高度
        feet_height = self.rigid_state[:, self.feet_indices, 2]
        ground_height = 0.0  # 假设地面高度为0
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        for foot_idx in range(len(self.feet_indices)):
            foot_height = feet_height[:, foot_idx]
            foot_vel_z = feet_vertical_vel[:, foot_idx]
            
            # 检测接近地面的足部（高度小于0.1m）
            near_ground = foot_height < 0.1
            
            # 检测下降的足部（垂直速度为负）
            falling = foot_vel_z < 0.0
            
            # 对接近地面且下降的足部施加速度控制
            control_mask = near_ground & falling
            
            if torch.any(control_mask):
                # 理想的着地速度：接近地面时应该减速
                # 高度越低，期望速度越小
                ideal_speed = -0.5 * foot_height[control_mask]  # 负值表示向下
                actual_speed = foot_vel_z[control_mask]
                
                # 如果实际速度过快（更负），给予惩罚
                speed_error = torch.abs(actual_speed - ideal_speed)
                speed_reward = torch.exp(-speed_error * 10.0)  # 速度越接近理想值，奖励越高
                
                # 对速度过快的情况额外惩罚
                too_fast_penalty = torch.where(
                    actual_speed < -0.8,  # 下降速度超过0.8m/s
                    -0.5 * torch.abs(actual_speed + 0.8),
                    0.0
                )
                
                foot_reward = speed_reward + too_fast_penalty
                reward[control_mask] += foot_reward
        
        return reward