# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X1DHStandCfgV19(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot - v1.9 Disturbance Resilience & Fast Recovery
    Based on v1.8 enhanced stability, focused on comprehensive disturbance handling and rapid recovery
    """
    class env(LeggedRobotCfg.env):
        # Enhanced observation space for v1.9: +3 dims (push_flag + 2 contact states)
        frame_stack = 66      #all histroy obs num
        short_frame_stack = 5   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        num_single_obs = 50  # v1.9: Increased from 47 to include push flag + contact states
        num_observations = int(frame_stack * num_single_obs)  # 3300
        single_num_privileged_obs = 73
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24 #episode length in seconds
        use_ref_actions = False
        num_commands = 5 # sin_pos cos_pos vx vy vz

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/x1.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'

        name = "x1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # v1.9: Enhanced terrain for comprehensive disturbance training
        mesh_type = 'trimesh'
        curriculum = True  # v1.9: Enable curriculum for gradual difficulty increase
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        
        # v1.9: Enhanced terrain distribution for disturbance training
        terrain_dict = {"flat": 0.2,  # Reduced flat terrain 
                        "rough flat": 0.15,
                        "slope up": 0.15,
                        "slope down": 0.15, 
                        "rough slope up": 0.1,  # v1.9: Added challenging slopes
                        "rough slope down": 0.1, 
                        "stairs up": 0.05,  # v1.9: Added stairs
                        "stairs down": 0.05,
                        "discrete": 0.15,  # v1.9: Increased obstacles
                        "wave": 0.1,}  # v1.9: Added wave terrain
        terrain_proportions = list(terrain_dict.values())

        # v1.9: More challenging terrain parameters
        rough_flat_range = [0.005, 0.02]  # Increased roughness
        slope_range = [0, 0.2]   # Increased max slope to ~11 degrees
        rough_slope_range = [0.005, 0.03]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.05, 0.15]  # 5-15cm steps
        discrete_height_range = [0.0, 0.1]  # Up to 10cm obstacles/holes
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 2.0    # v1.9: Increased noise for robustness

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.03  # v1.9: Slightly increased sensor noise
            dof_vel = 2.0   # v1.9: Increased velocity noise
            ang_vel = 0.3   # v1.9: Increased IMU noise
            lin_vel = 0.15  
            quat = 0.15
            gravity = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': 0.4,
            'left_hip_roll_joint': 0.05,
            'left_hip_yaw_joint': -0.31,
            'left_knee_pitch_joint': 0.49,
            'left_ankle_pitch_joint': -0.21,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.05,
            'right_hip_yaw_joint': 0.31,
            'right_knee_pitch_joint': 0.49,
            'right_ankle_pitch_joint': -0.21, 
            'right_ankle_roll_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40,'hip_yaw_joint': 35,
                     'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
        damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0,'hip_yaw_joint': 4, 
                   'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        # v1.9: Enhanced randomization for comprehensive disturbance training
        randomize_friction = True
        friction_range = [0.1, 1.5]  # v1.9: Wider friction range (very slick to very sticky)
        restitution_range = [0.0, 0.5]  # v1.9: More elastic collisions

        # v1.9: Enhanced push configuration for disturbance resilience training
        push_robots = True
        push_interval_s = 3  # v1.9: More frequent pushes (every 3 seconds)
        update_step = 1500 * 24 # v1.9: Faster curriculum progression
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]  # v1.9: Extended duration range
        max_push_vel_xy = 0.3  # v1.9: Stronger pushes
        max_push_ang_vel = 0.3  # v1.9: Stronger rotational disturbances

        randomize_base_mass = True
        added_mass_range = [-5, 5] # v1.9: Wider mass variation

        randomize_com = True
        com_displacement_range = [[-0.08, 0.08],  # v1.9: Larger CoM displacement
                                  [-0.08, 0.08],
                                  [-0.08, 0.08]]

        randomize_gains = True
        stiffness_multiplier_range = [0.7, 1.3]  # v1.9: Wider gain variation
        damping_multiplier_range = [0.7, 1.3]

        randomize_torque = True
        torque_multiplier_range = [0.7, 1.3]  # v1.9: Wider torque variation

        randomize_link_mass = True
        added_link_mass_range = [0.8, 1.2]  # v1.9: More link mass variation

        randomize_motor_offset = True
        motor_offset_range = [-0.05, 0.05] # v1.9: Larger motor offset range
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.005, 1.5]  # v1.9: Wider friction range
        joint_1_friction_range = [0.005, 1.5]
        joint_2_friction_range = [0.005, 1.5]
        joint_3_friction_range = [0.005, 1.5]
        joint_4_friction_range = [0.3, 1.5]
        joint_5_friction_range = [0.3, 1.5]
        joint_6_friction_range = [0.005, 1.5]
        joint_7_friction_range = [0.005, 1.5]
        joint_8_friction_range = [0.005, 1.5]
        joint_9_friction_range = [0.3, 1.5]
        joint_10_friction_range = [0.3, 1.5]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.2, 2.0]  # v1.9: Wider damping range
        joint_1_damping_range = [0.2, 2.0]
        joint_2_damping_range = [0.2, 2.0]
        joint_3_damping_range = [0.2, 2.0]
        joint_4_damping_range = [0.5, 2.0]
        joint_5_damping_range = [0.5, 2.0]
        joint_6_damping_range = [0.2, 2.0]
        joint_7_damping_range = [0.2, 2.0]
        joint_8_damping_range = [0.2, 2.0]
        joint_9_damping_range = [0.5, 2.0]
        joint_10_damping_range = [0.5, 2.0]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.1]     # v1.9: Wider armature range
        joint_1_armature_range = [0.0001, 0.1]
        joint_2_armature_range = [0.0001, 0.1]
        joint_3_armature_range = [0.0001, 0.1]
        joint_4_armature_range = [0.0001, 0.1]
        joint_5_armature_range = [0.0001, 0.1]
        joint_6_armature_range = [0.0001, 0.1]
        joint_7_armature_range = [0.0001, 0.1]
        joint_8_armature_range = [0.0001, 0.1]
        joint_9_armature_range = [0.0001, 0.1]
        joint_10_armature_range = [0.0001, 0.1]

        # v1.9: Enhanced lag simulation for sensor/actuator delays
        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 50]  # v1.9: Extended lag range
        
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 50]  # v1.9: Extended DOF lag
        
        add_dof_pos_vel_lag = True  # v1.9: Enable position/velocity lag
        randomize_dof_pos_lag_timesteps = True
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [5, 30]
        randomize_dof_vel_lag_timesteps = True
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [5, 30]
        
        add_imu_lag = True  # v1.9: Enable IMU lag
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 15]  # v1.9: Extended IMU lag
        
        randomize_coulomb_friction = True
        joint_coulomb_range = [0.05, 1.2]  # v1.9: Wider Coulomb friction
        joint_viscous_range = [0.02, 0.15]  # v1.9: Wider viscous friction
        
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0  # v1.9: Higher curriculum limit for challenge
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 20.  # v1.9: More frequent command changes for adaptability
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,4],  # v1.9: More rotation training
                           "stand": [2,3],
                           "walk_omnidirectional": [4,8]}  # v1.9: Extended omni walking

        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.6, 1.5] # v1.9: Extended velocity range
            lin_vel_y = [-0.6, 0.6]   # v1.9: Extended lateral velocity
            ang_vel_yaw = [-0.8, 0.8]    # v1.9: Extended angular velocity
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61
        foot_min_dist = 0.2
        foot_max_dist = 1.0

        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos
        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
        target_feet_height = 0.03 
        target_feet_height_max = 0.06
        feet_to_ankle_distance = 0.041
        cycle_time = 0.7
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(-error*sigma)
        tracking_sigma = 5 
        max_contact_force = 700  # forces above this value are penalized
        
        # v1.8: Enhanced stability parameters (inherited)
        max_foot_clearance = 0.08  # Maximum allowed foot clearance height (8cm)
        action_smoothness_window = 3  # Number of timesteps to consider for action smoothness
        torso_orientation_target = [0.0, 0.0, 0.0]  # Target roll, pitch, yaw for torso
        lateral_velocity_threshold = 0.1  # Threshold for lateral velocity penalty
        
        # v1.9: New fast recovery and disturbance resilience parameters
        balance_stability_threshold = 0.3  # Maximum tilt angle (rad) for "balanced" state
        recovery_time_window = 50  # Timesteps to track for recovery performance
        disturbance_detection_threshold = 0.1  # Threshold for detecting external disturbances
        velocity_recovery_threshold = 0.2  # Velocity error threshold for recovery tracking
        
        class scales:
            ref_joint_pos = 2.2
            feet_clearance = 1.
            feet_contact_number = 2.0
            # gait
            feet_air_time = 1.2
            foot_slip = -0.15  # v1.9: Increased foot slip penalty
            feet_distance = 0.2
            knee_distance = 0.2
            # contact 
            feet_contact_forces = -0.01
            # vel tracking - v1.9: Enhanced tracking for faster recovery
            tracking_lin_vel = 2.5  # v1.9: Increased from 1.8 for faster velocity recovery
            tracking_ang_vel = 1.8  # v1.9: Increased from 1.1 for faster angular recovery
            vel_mismatch_exp = 0.8  # v1.9: Increased penalty for velocity mismatches
            low_speed = 0.3  # v1.9: Increased low speed penalty
            track_vel_hard = 0.8  # v1.9: Increased hard velocity tracking
            # base pos
            default_joint_pos = 1.0
            orientation = 2.5  # v1.9: Further increased from 2.0 for stability under disturbances
            feet_rotation = 0.3
            base_height = 0.3  # v1.9: Increased base height importance
            base_acc = 0.3  # v1.9: Increased base acceleration penalty
            # energy and smoothness (inherited from v1.8)
            action_smoothness = -0.01
            torques = -2e-7
            dof_vel = -5e-7
            dof_acc = -1e-6
            collision = -1.
            stand_still = 2.5
            
            # v1.8: Enhanced stability components (inherited)
            torso_angular_velocity = -0.8  # v1.9: Increased from -0.5
            lateral_movement = -1.2  # v1.9: Increased from -0.8
            excessive_foot_clearance = -1.5  # v1.9: Increased from -1.0
            action_rate = -0.008  # v1.9: Increased from -0.005
            joint_acceleration = -0.002  # v1.9: Increased from -0.001
            
            # v1.9: NEW - Disturbance Resilience & Fast Recovery Rewards
            balance_recovery = 2.0  # Reward for maintaining/quickly returning to balanced posture
            disturbance_response = 1.5  # Reward for appropriate response to disturbances
            velocity_recovery = 1.2  # Reward for quickly recovering target velocity after disruption
            postural_stability = 1.0  # Reward for maintaining stable posture during disturbances
            contact_stability = 0.8  # Reward for maintaining appropriate foot contact during recovery
            angular_momentum_control = -1.0  # Penalty for excessive angular momentum during disturbances
            
            # limits
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class X1DHStandCfgPPOV19(LeggedRobotCfgPPO):
    seed = 7  # v1.9: Different seed for new training
    runner_class_name = 'DHOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        #for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   #long history output dim
        in_channels = X1DHStandCfgV19.env.frame_stack

        # v1.4: LSTM configuration (enhanced for v1.9)
        use_lstm = True
        lstm_hidden_size = 512
        sequence_length = 20  # v1.9: Increased from 16 for better temporal modeling
        num_layers = 2  # v1.9: Added second LSTM layer for complex disturbance patterns

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 3  # v1.9: Increased from 2 for better convergence
        gamma = 0.995  # v1.9: Slightly increased discount factor
        lam = 0.9
        num_mini_batches = 4
        if X1DHStandCfgV19.terrain.measure_heights:
            lin_vel_idx = (X1DHStandCfgV19.env.single_num_privileged_obs + X1DHStandCfgV19.terrain.num_height) * (X1DHStandCfgV19.env.c_frame_stack - 1) + X1DHStandCfgV19.env.single_linvel_index
        else:
            lin_vel_idx = X1DHStandCfgV19.env.single_num_privileged_obs * (X1DHStandCfgV19.env.c_frame_stack - 1) + X1DHStandCfgV19.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 25000  # v1.9: Increased iterations for comprehensive training

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'x1_dh_stand_v1_9'  # v1.9: New experiment name
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt