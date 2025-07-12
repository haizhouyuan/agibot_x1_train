# AgiBot X1 v2.0 Training Enhancement Documentation

## Overview

AgiBot X1 v2.0 represents a significant advancement in humanoid robot training, featuring enhanced anti-disturbance capabilities, energy efficiency optimization, and advanced neural network architectures. This version builds upon the solid foundation of v1.7 with systematic improvements across five key phases.

## New Features Summary

### Phase 1: Enhanced Stability & Rapid Recovery
- **Enhanced Reward Weights**: Increased orientation reward weight (1.0 → 2.0) for better posture control
- **Tilt Penalty System**: New penalty function for excessive body tilt (>5°)
- **Rapid Recovery Rewards**: Exponential rewards for quick balance recovery (<1 second)
- **Improved Base Acceleration Control**: Enhanced base acceleration rewards (0.2 → 0.5)

### Phase 2: Advanced Disturbance Training
- **Stronger Push Forces**: Increased disturbance strength (0.2 → 0.4 m/s linear, 0.2 → 0.3 rad/s angular)
- **More Frequent Disturbances**: Reduced push intervals (4s → 3s) 
- **Extended Duration**: Added longer disturbance durations (up to 0.3s)
- **Complex Terrain Types**: Enabled waves and stairs terrain (10% each)
- **Enhanced Friction Range**: Expanded friction randomization (0.1-1.5 vs 0.2-1.3)

### Phase 3: Energy Efficiency Optimization
- **Enhanced Torque Penalties**: Increased torque penalties (-1e-8 → -5e-8)
- **Joint Velocity Control**: Enhanced dof_vel penalties (-2e-8 → -1e-7)
- **Action Smoothness**: Improved action smoothness penalties (-0.002 → -0.005)
- **Power Consumption Monitoring**: New direct power penalty based on τ×ω calculation
- **Conservative Action Policy**: Reduced entropy coefficient for more efficient exploration

### Phase 4: Transformer Architecture (Future Enhancement)
- **Multi-Head Attention**: Replace LSTM with Transformer for better sequential processing
- **Positional Encoding**: Temporal position awareness for sequences
- **Improved Memory**: Better long-term dependency modeling
- **Separate Actor-Critic Transformers**: Independent processing for value and policy functions

### Phase 5: Adversarial Training (Future Enhancement)
- **Intelligent Disturbances**: AI-controlled disturbance generation
- **Min-Max Training**: Adversarial optimization between robot and disturbance policies
- **Enhanced Robustness**: Training against worst-case scenarios

## Configuration Files

### Primary Configuration: `x1_dh_stand_config_v2_0.py`

Key configuration classes and their purposes:

```python
class X1DHStandCfgV2(LeggedRobotCfg):
    """Main environment configuration with v2.0 enhancements"""
    
class X1DHStandCfgPPOV2(LeggedRobotCfgPPO):
    """PPO algorithm configuration optimized for v2.0"""
```

## New Reward Functions

### 1. Tilt Penalty (`_reward_tilt_penalty`)
- **Purpose**: Penalizes excessive body tilt to promote rapid balance recovery
- **Threshold**: 5 degrees (0.0873 radians)
- **Implementation**: Quadratic penalty for tilt exceeding threshold
- **Weight**: -2.0

### 2. Rapid Recovery (`_reward_rapid_recovery`)
- **Purpose**: Rewards quick stabilization after disturbances
- **Mechanism**: Tracks recovery time and provides exponential rewards
- **Target**: Recovery within 1 second
- **Implementation**: Detects disturbances via acceleration changes, monitors stability

### 3. Power Consumption (`_reward_power_consumption`)
- **Purpose**: Promotes energy-efficient gaits
- **Calculation**: P = |τ × ω| for each joint, summed across all joints
- **Weight**: -0.001
- **Benefit**: Reduces motor stress and improves battery life

## Training Instructions

### Basic Usage

```bash
# Train with v2.0 configuration
python humanoid/scripts/train.py --task=x1_dh_stand --version=2.0

# Resume training from checkpoint
python humanoid/scripts/train.py --task=x1_dh_stand --version=2.0 --resume

# Train with custom experiment name
python humanoid/scripts/train.py --task=x1_dh_stand --version=2.0 --run_name=my_experiment
```

### Progressive Training Approach

For best results, follow this training progression:

1. **Phase 1 Training** (5,000-10,000 iterations)
   - Focus on basic stability improvements
   - Use standard disturbance parameters
   - Monitor tilt recovery metrics

2. **Phase 2 Training** (20,000-30,000 iterations)
   - Enable enhanced disturbances
   - Add complex terrain types
   - Verify robustness improvements

3. **Phase 3 Training** (5,000-10,000 iterations)
   - Apply energy efficiency penalties
   - Monitor power consumption reduction
   - Ensure stability is maintained

4. **Phase 4 Training** (Optional, 20,000+ iterations)
   - Switch to Transformer architecture
   - Longer training for convergence
   - Enhanced sequential decision making

## Performance Metrics

### Key Performance Indicators

1. **Recovery Time**: Target <1 second from disturbance
2. **Tilt Angle**: Maximum sustained tilt <5 degrees
3. **Energy Efficiency**: 20% reduction in power consumption
4. **Robustness**: Zero-fall rate under standard disturbances
5. **Terrain Adaptability**: Stable performance on waves and stairs

### Monitoring During Training

```python
# Key metrics to monitor in training logs
- episode_rewards/tilt_penalty: Should decrease (less penalty)
- episode_rewards/rapid_recovery: Should increase (faster recovery)
- episode_rewards/power_consumption: Should decrease (less power)
- episode_rewards/orientation: Should increase (better posture)
- episode_length: Should remain high (no early terminations)
```

## Architecture Details

### Transformer Network (Phase 4)

```python
# Configuration for Transformer architecture
use_transformer = True
transformer_hidden_size = 128
transformer_num_heads = 4
transformer_num_layers = 2
transformer_sequence_length = 16
```

**Benefits**:
- Better temporal reasoning
- Improved response to sequential disturbances
- Enhanced adaptability to complex scenarios

### Adversarial Training (Phase 5)

```python
# Adversarial training components (future implementation)
class AdversaryActor:
    """Generates intelligent disturbances"""
    
class AdversarialTrainer:
    """Manages dual-policy training loop"""
```

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rate if loss oscillates
   - Check reward scaling if total rewards become negative
   - Ensure proper initialization from v1.7 checkpoint

2. **Poor Recovery Performance**
   - Verify tilt_threshold is appropriate (5°)
   - Check disturbance detection sensitivity (0.5 m/s²)
   - Monitor rapid_recovery reward activation

3. **High Energy Consumption**
   - Increase power_consumption penalty weight
   - Verify torque and velocity penalties are active
   - Check action smoothness constraints

### Performance Optimization

1. **Memory Usage**: Transformer requires more GPU memory
   - Reduce batch size if needed
   - Consider gradient checkpointing for large models

2. **Training Speed**: Complex rewards may slow training
   - Profile reward computation time
   - Optimize tensor operations where possible

## File Structure

```
humanoid/
├── envs/x1/
│   ├── x1_dh_stand_config_v2_0.py     # Main v2.0 configuration
│   └── x1_dh_stand_env.py             # Environment with new rewards
├── algo/ppo/
│   ├── actor_critic_dh.py             # Original architecture
│   └── actor_critic_transformer.py    # New Transformer architecture
└── scripts/
    └── train.py                       # Updated training script
```

## Changelog

### v2.0.0
- Added Phase 1: Enhanced stability and rapid recovery
- Added Phase 2: Advanced disturbance training
- Added Phase 3: Energy efficiency optimization
- Added Phase 4: Transformer architecture framework
- Added Phase 5: Adversarial training framework
- Enhanced configuration system for multi-phase training
- Improved documentation and usage instructions

## Future Enhancements

### v2.1 Planned Features
- Automatic curriculum learning for disturbance strength
- Real-time adaptation to different terrain types
- Multi-objective optimization for stability vs. efficiency
- Integration with real robot hardware validation

### Research Directions
- Hierarchical control strategies
- Domain transfer learning
- Online adaptation mechanisms
- Safety-constrained reinforcement learning

## Support and Contribution

For issues, feature requests, or contributions, please refer to the project repository. The v2.0 enhancements maintain backward compatibility while providing significant improvements in robot performance and training efficiency.

## References

1. Original AgiBot X1 project documentation
2. Humanoid-Gym framework specifications
3. Transformer neural network architectures for robotics
4. Adversarial training methods in reinforcement learning
5. Energy-efficient gait optimization techniques