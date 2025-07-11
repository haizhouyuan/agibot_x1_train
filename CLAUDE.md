# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the AgiBot X1 humanoid robot reinforcement learning training codebase. It uses Isaac Gym for simulation and PPO algorithm for training legged locomotion policies. The codebase is structured as a Python package with environments, algorithms, and utilities.

## Installation Commands

```bash
# Create Python 3.8 environment
conda create -n myenv python=3.8
conda activate myenv

# Install PyTorch with CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install NumPy
conda install numpy=1.23

# Install Isaac Gym (requires manual download)
# Download from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

# Install this package
pip install -e .
```

## Core Development Commands

```bash
# Train a model
python scripts/train.py --task=x1_dh_stand --run_name=<run_name> --headless

# Test/play trained model
python scripts/play.py --task=x1_dh_stand --load_run=<date_time><run_name>

# Export policy to JIT format
python scripts/export_policy_dh.py --task=x1_dh_stand --load_run=<date_time><run_name>

# Export policy to ONNX format
python scripts/export_onnx_dh.py --task=x1_dh_stand --load_run=<date_time>

# Sim2sim validation with MuJoCo
python scripts/sim2sim.py --task=x1_dh_stand --load_model /path/to/exported_policies/

# Fine-tuning from checkpoint
python scripts/finetune.py --task=x1_dh_stand --load_run=<date_time><run_name>
```

## Architecture Overview

### Environment Structure
- **Base Classes**: `LeggedRobot` and `LeggedRobotCfg` in `humanoid/envs/base/` provide the foundation
- **X1 Environment**: `X1DHStandEnv` in `humanoid/envs/x1/` implements the specific humanoid standing task
- **Config Versions**: Multiple config versions exist (v1.6, v1.7, finetune) with different hyperparameters and stability improvements

### Algorithm Components
- **PPO Implementation**: Custom PPO in `humanoid/algo/ppo/` with actor-critic architecture
- **Policy Networks**: `actor_critic_dh.py` contains the neural network architectures
- **Training Runner**: `dh_on_policy_runner.py` manages the training loop

### Key Configuration System
- Environment configs inherit from `LeggedRobotCfg` and define observation/action spaces, rewards, and physics parameters
- PPO configs define training hyperparameters
- Task registration system in `humanoid/utils/task_registry.py` manages environment-algorithm pairings

### Data Flow
1. **Observations**: Frame-stacked proprioceptive data (joint angles, velocities, IMU)
2. **Actions**: 12-DoF joint position targets
3. **Rewards**: Multi-component reward combining stability, tracking, and efficiency terms
4. **Privileged Information**: Additional state info for asymmetric training (terrain, contact forces)

### Version Management
- v1.7 includes stability improvements with residual angular momentum penalties
- Manual registration for v1.7 task in `train.py` ensures backward compatibility
- Config files maintain different hyperparameter sets for different training phases

## Working with Models

Trained models are saved in `logs/<experiment_name>/exported_data/<timestamp>/` with:
- `.pt` files for PyTorch checkpoints
- `.jit` files for deployment-ready TorchScript models
- `.onnx` files for cross-platform inference

## Version-Specific Training

### Current Versions Available
- **v1.6**: Base version (`x1_dh_stand_config_v1.6.py`)
- **v1.7**: Stability improvements (`x1_dh_stand_config_v1_7.py`)
- **v1.8**: Enhanced stability (`x1_dh_stand_config_v1_8.py`)
- **v1.9**: Disturbance resilience & fast recovery (`x1_dh_stand_config_v1_9.py`)
- **Finetune**: Specialized tuning config (`x1_dh_stand_finetune_config.py`)

### V1.9 Disturbance Resilience Training (Latest)
```bash
# Train v1.9 with disturbance resilience features
python scripts/train.py --task=x1_dh_stand_v1.9 --run_name=disturbance_resilience --headless

# Progressive training approach (recommended)
python scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_stage1 --headless --max_iterations=5000
python scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_stage2 --headless --load_run=<timestamp>v19_stage1 --max_iterations=20000
```

## Testing and Validation

### Manual Testing
```bash
# Interactive testing with visual feedback
python scripts/play.py --task=x1_dh_stand_v1.9 --load_run=<date_time><run_name>

# Extended duration stress testing
python scripts/play.py --task=x1_dh_stand_v1.9 --load_run=<model> --duration=600
```

### Sim2Sim Validation
```bash
# Cross-simulator validation with MuJoCo
python scripts/sim2sim.py --task=x1_dh_stand --load_model /path/to/exported_policies/
```

### Performance Monitoring
Key metrics to track during training:
- **Recovery Performance**: Balance recovery time, velocity recovery, push survival rate
- **Reward Components**: balance_recovery, disturbance_response, velocity_recovery rewards
- **Stability Indicators**: Episode length, curriculum progress, reward convergence

## Debugging and Development

### Common Debug Commands
```bash
# Check observation dimensions (v1.9: 50 single obs vs v1.8: 47)
# Verify push flag and contact state integration
# Monitor LSTM processing of 50×66 observation history

# Training divergence troubleshooting:
# - Check reward balance if training becomes unstable
# - Reduce new reward scales if total rewards become negative
# - Verify observation dimensions match config expectations
```

### Configuration Debugging
- **Poor Recovery**: Increase balance_recovery and disturbance_response reward scales
- **Excessive Conservatism**: Reduce penalty scales (torso_angular_velocity, lateral_movement)
- **Training Instability**: Verify curriculum progression and reward component balance

## Common Tasks

- **Adding New Robots**: Create config and env files in `humanoid/envs/<robot_name>/`, inherit from base classes, register in `__init__.py`
- **Modifying Rewards**: Edit reward functions in the environment's `_reward_*` methods
- **Training Hyperparameters**: Adjust in the PPO config classes
- **Observation Space**: Modify `num_observations` and observation computation in env classes
- **Version Switching**: Change task name in training commands to switch between config versions
- **Performance Benchmarking**: Use testing guides in `V1_8_TESTING_GUIDE.md` and `V1_9_TESTING_GUIDE.md`