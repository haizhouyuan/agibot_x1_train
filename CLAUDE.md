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
python humanoid/scripts/train.py --task=x1_dh_stand --run_name=<run_name> --headless

# Train with specific version (v1.8)
python humanoid/scripts/train.py --task=x1_dh_stand --version=1.8 --run_name=<run_name> --headless

# Test/play trained model
python humanoid/scripts/play.py --task=x1_dh_stand --load_run=<date_time><run_name>

# Export policy to JIT format
python humanoid/scripts/export_policy_dh.py --task=x1_dh_stand --load_run=<date_time><run_name>

# Export policy to JIT format (v1.8)
python humanoid/scripts/export_policy_dh.py --task=x1_dh_stand_v1.8 --load_run=<date_time><run_name>

# Export policy to ONNX format
python humanoid/scripts/export_onnx_dh.py --task=x1_dh_stand --load_run=<date_time>

# Export policy to ONNX format (v1.8)
python humanoid/scripts/export_onnx_dh.py --task=x1_dh_stand_v1.8 --load_run=<date_time>

# Sim2sim validation with MuJoCo
python humanoid/scripts/sim2sim.py --task=x1_dh_stand --load_model /path/to/exported_policies/

# Fine-tuning from checkpoint
python humanoid/scripts/finetune.py --task=x1_dh_stand --load_run=<date_time><run_name>
```

## Architecture Overview

### Environment Structure
- **Base Classes**: `LeggedRobot` and `LeggedRobotCfg` in `humanoid/envs/base/` provide the foundation
- **X1 Environment**: `X1DHStandEnv` in `humanoid/envs/x1/` implements the specific humanoid standing task
- **Config Versions**: Multiple config versions exist (v1.6, v1.7, v1.8, finetune) with different hyperparameters and stability improvements

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
- v1.8 includes enhanced stability improvements and disturbance resilience features
- Manual registration for v1.7 and v1.8 tasks in `train.py` ensures backward compatibility
- Config files maintain different hyperparameter sets for different training phases

## Working with Models

Trained models are saved in `logs/<experiment_name>/exported_data/<timestamp>/` with:
- `.pt` files for PyTorch checkpoints
- `.jit` files for deployment-ready TorchScript models (in `exported_policies/`)
- `.onnx` files for cross-platform inference (in `exported_onnx/`)

### Latest Training Results (v1.8)
- **Training Run**: `2025-07-10_08-13-42v1_8_production`
- **Iterations**: 20,000 training iterations completed
- **JIT Export**: Available at `logs/x1_dh_stand_v1_8/exported_policies/2025-07-10_16-14-03/policy_dh.jit`
- **ONNX Export**: Available at `logs/x1_dh_stand_v1_8/exported_onnx/2025-07-10_16-14-15/x1_policy.onnx`
- **Training Data Archive**: Complete training data archived at `/autodl-tmp/agibot_x1_train_v1_8_complete.tar.gz`

## Common Tasks

- **Adding New Robots**: Create config and env files in `humanoid/envs/<robot_name>/`, inherit from base classes, register in `__init__.py`
- **Modifying Rewards**: Edit reward functions in the environment's `_reward_*` methods
- **Training Hyperparameters**: Adjust in the PPO config classes
- **Observation Space**: Modify `num_observations` and observation computation in env classes