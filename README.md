# AgiBot X1 Train - v1.7 Clean Version

## Overview
This is a cleaned version of the AgiBot X1 training codebase, focusing on the v1.7 stability improvements. This repository contains only the essential code and configuration files needed for training, with all large binary files removed to keep the repository size under 100MB.

## Features
- PPO-based reinforcement learning for humanoid robot locomotion
- Isaac Gym physics simulation environment
- Modular reward function design for stability improvements
- Support for distributed training across multiple environments

## Quick Start

### Installation
1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Train the model:
   ```bash
   python humanoid/scripts/train.py --task=x1_dh_stand --run_name=my_training --headless
   ```

3. Test the trained model:
   ```bash
   python humanoid/scripts/play.py --task=x1_dh_stand --load_run=<run_name>
   ```

## Repository Structure
```
├── humanoid/           # Main training code
│   ├── algo/          # PPO algorithm implementation
│   ├── envs/          # Environment configurations
│   ├── scripts/       # Training and testing scripts
│   └── utils/         # Utility functions
├── resources/         # Robot URDF and configuration files
└── README.md          # This file
```

## Key Improvements in v1.7
- Enhanced stability through improved reward function design
- Better torso angular velocity control
- Reduced joint acceleration penalties
- Improved action smoothness

## Requirements
- Python 3.8+
- PyTorch 1.13.1
- Isaac Gym Preview 4
- CUDA-capable GPU

## License
BSD-3-Clause License

## Note
This is a cleaned version with training logs, large media files, and binary assets removed to maintain a lightweight repository suitable for code sharing and collaboration.