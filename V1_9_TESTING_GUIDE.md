# AgiBot X1 v1.9 Disturbance Resilience & Fast Recovery Testing Guide

## Overview
This guide provides comprehensive testing procedures for validating the v1.9 disturbance resilience and fast recovery improvements. v1.9 builds upon v1.8's stability enhancements with advanced disturbance handling, comprehensive domain randomization, and rapid recovery capabilities.

## Key v1.9 Features

### New Disturbance Resilience Components
1. **Enhanced Observation Space**: Push indicators and foot contact state feedback
2. **Fast Recovery Rewards**: Balance recovery, disturbance response, velocity recovery
3. **Multi-Modal Disturbance Training**: Terrain variations, enhanced randomization, sensor noise
4. **Curriculum Learning**: Progressive difficulty increase for robust training
5. **Advanced LSTM Architecture**: Improved temporal modeling for disturbance patterns

### Core Improvements Over v1.8
- **50-dimensional observations** (vs 47): Added push flag + contact states
- **6 new reward functions** for disturbance resilience and recovery
- **Enhanced domain randomization**: Wider parameter ranges, extended delays
- **Comprehensive terrain training**: Slopes, stairs, obstacles, wave patterns
- **Curriculum progression**: Graduated disturbance intensity

## Training Commands

### Basic v1.9 Training
```bash
# Train v1.9 from scratch
python humanoid/scripts/train.py --task=x1_dh_stand_v1.9 --run_name=disturbance_resilience --headless

# Train with specific configuration
python humanoid/scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_baseline --headless --max_iterations=25000

# Resume training from checkpoint
python humanoid/scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_resume --resume --load_run=<timestamp>disturbance_resilience
```

### Progressive Training (Recommended)
```bash
# Stage 1: Base stability training (reduced disturbances)
# Modify config to temporarily reduce push frequency and terrain difficulty
python humanoid/scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_stage1 --headless --max_iterations=5000

# Stage 2: Full disturbance training
# Use default v1.9 config with full disturbance parameters
python humanoid/scripts/train.py --task=x1_dh_stand_v1.9 --run_name=v19_stage2 --headless --load_run=<timestamp>v19_stage1 --max_iterations=20000
```

### Testing/Evaluation
```bash
# Test trained v1.9 model
python humanoid/scripts/play.py --task=x1_dh_stand_v1.9 --load_run=<timestamp>disturbance_resilience

# Export v1.9 model for deployment
python humanoid/scripts/export_policy_dh.py --task=x1_dh_stand_v1.9 --load_run=<timestamp>disturbance_resilience

# Export to ONNX format
python humanoid/scripts/export_onnx_dh.py --task=x1_dh_stand_v1.9 --load_run=<timestamp>disturbance_resilience
```

## Validation Metrics

### Primary Disturbance Resilience Metrics

#### 1. Recovery Time Performance
Monitor these key recovery metrics during training:

- **Balance Recovery Time**: Time to return to upright posture after push (target: <2 seconds)
- **Velocity Recovery Time**: Time to resume commanded velocity (target: <3 seconds)  
- **Stability Margin**: Maximum tilt angle during recovery (target: <15°)
- **Push Survival Rate**: Percentage of pushes survived without falling (target: >95%)

#### 2. Enhanced Reward Components
Track these new v1.9 reward terms:

```python
# New v1.9 reward scales and targets:
# - balance_recovery: 2.0 (upright maintenance reward)
# - disturbance_response: 1.5 (appropriate reaction to pushes)
# - velocity_recovery: 1.2 (quick return to commanded motion)
# - postural_stability: 1.0 (posture maintenance during disturbances)
# - contact_stability: 0.8 (proper foot contact during recovery)
# - angular_momentum_control: -1.0 (controlled angular momentum)
```

#### 3. Observation Space Validation
Verify enhanced observation functionality:

- **Push Flag**: Correctly detects disturbance timing (dim 48 in obs)
- **Contact States**: Accurate foot contact detection (dims 49-50 in obs)
- **Temporal Correlation**: LSTM properly processes 50×66 observation history

### Advanced Testing Scenarios

#### Scenario 1: Multi-Directional Push Resistance
```bash
# Test with various push magnitudes and directions
# Observe recovery time and stability maintenance
# Expected: <2s recovery, <15° max tilt, maintained balance
```

**Validation Steps:**
1. Apply pushes from front, back, left, right, diagonal directions
2. Measure time to return to ±5° of vertical
3. Check velocity command tracking resumption
4. Verify no falls or excessive drift

#### Scenario 2: Terrain Adaptability
Test on various terrain types enabled in v1.9:

- **Slopes**: Up to 11° inclines and declines
- **Stairs**: 5-15cm step heights  
- **Discrete Obstacles**: 10cm height variations
- **Wave Terrain**: Dynamic uneven surfaces
- **Rough Terrain**: Surface irregularities

**Success Criteria:**
- Maintains stable gait across all terrain types
- Adapts foot placement for obstacles
- No excessive foot clearance (>8cm penalty active)
- Smooth transitions between terrain types

#### Scenario 3: Sensor/Actuator Robustness
Validate performance under realistic hardware conditions:

- **IMU Lag**: 1-15 timestep delays
- **Joint Sensor Lag**: 0-50 timestep delays  
- **Actuator Delays**: 5-30 timestep control lag
- **Sensor Noise**: 2x increased noise levels
- **Parameter Variations**: Mass, friction, gain variations

#### Scenario 4: Long-Duration Stress Testing
Extended operation validation:

```bash
# 10-minute continuous operation test
python humanoid/scripts/play.py --task=x1_dh_stand_v1.9 --load_run=<model> --duration=600

# Multi-command stress test with frequent velocity changes
# Monitor drift, stability degradation, recovery consistency
```

## Performance Benchmarks

### Quantitative Targets

#### Recovery Performance
- **Push Recovery**: 95% success rate for 0.3 m/s pushes
- **Recovery Time**: Median <2 seconds to stable posture
- **Velocity Tracking**: Resume within 0.2 m/s of command in <3 seconds
- **Angular Stability**: Peak angular velocity <3 rad/s during recovery

#### Gait Quality  
- **Lateral Drift**: <0.1 m per 10m forward travel
- **Energy Efficiency**: <30% increase in torque variance vs v1.8
- **Foot Clearance**: 90% of steps <8cm clearance
- **Symmetry**: <10% difference in left/right step parameters

#### Robustness Metrics
- **Terrain Success**: >90% stable operation on challenging terrain
- **Noise Tolerance**: Stable under 2x sensor noise
- **Parameter Robustness**: Stable with ±30% parameter variations
- **Long-Duration**: <5% performance degradation over 10 minutes

### Comparative Validation

#### v1.9 vs v1.8 Expected Improvements
- **50% improvement in push recovery time**
- **70% reduction in post-disturbance velocity error**
- **40% better terrain adaptability**
- **60% improved robustness to sensor/actuator variations**
- **30% faster curriculum learning convergence**

## Training Monitoring

### Key Metrics to Track

#### Reward Components
Monitor these reward trends during training:

```python
# High-priority rewards (should increase):
rewards/balance_recovery        # Target: >0.8 average
rewards/disturbance_response    # Target: >0.7 average  
rewards/velocity_recovery       # Target: >0.6 average
rewards/postural_stability      # Target: >0.5 average

# Penalty components (should decrease):
rewards/torso_angular_velocity  # Target: <0.1 magnitude
rewards/lateral_movement        # Target: <0.05 magnitude
rewards/excessive_foot_clearance # Target: <0.02 magnitude
```

#### Training Stability Indicators
- **Episode Length**: Should increase as stability improves
- **Curriculum Progress**: Terrain difficulty and push intensity advancement
- **Reward Convergence**: Stable reward accumulation after ~15k iterations
- **Policy Variance**: Consistent performance across environments

### Debugging Common Issues

#### Training Divergence
```bash
# Check reward balance if training unstable
# Reduce new reward scales if total rewards become negative
# Verify observation dimensions match config (50 vs 47)
```

#### Poor Recovery Performance
```bash
# Increase balance_recovery and disturbance_response reward scales
# Verify push_step_counter tracking in compute_observations
# Check LSTM sequence length and hidden size adequacy
```

#### Excessive Conservatism
```bash
# If robot becomes overly passive:
# Reduce penalty scales (torso_angular_velocity, lateral_movement)
# Increase velocity tracking rewards
# Verify curriculum progression is advancing properly
```

## Deployment Validation

### Real Robot Transfer
Before deploying to hardware:

1. **Sim2Sim Validation**: Test exported policy in MuJoCo
2. **Parameter Sensitivity**: Verify robustness to real robot variations
3. **Safety Limits**: Confirm torque and velocity limits are appropriate
4. **Emergency Stops**: Test failure detection and safe shutdown

### Hardware Testing Protocol
1. **Static Balance**: Verify standing stability before locomotion
2. **Gentle Pushes**: Start with very light manual pushes
3. **Progressive Testing**: Gradually increase disturbance intensity
4. **Recovery Timing**: Measure actual recovery performance
5. **Long-Term Operation**: Monitor performance degradation

## Success Criteria

### Training Success Indicators
- ✅ Training converges without instability (<5% episode failures)
- ✅ All reward components show expected trends
- ✅ Policy achieves target recovery times in simulation
- ✅ Robust performance across curriculum stages
- ✅ Observation space enhancement functions correctly

### Deployment Readiness
- ✅ Passes all quantitative benchmarks listed above
- ✅ Survives comprehensive stress testing scenarios  
- ✅ Demonstrates clear improvement over v1.8 baseline
- ✅ Maintains safety margins for real robot deployment
- ✅ Consistent performance across multiple trained models

## Next Steps

### If v1.9 Testing Succeeds
1. Document comprehensive performance analysis
2. Create detailed comparison study with v1.8
3. Prepare deployment package for real robot testing
4. Begin development of specialized task variants (stairs, outdoor, etc.)

### If Issues Arise
1. Analyze specific failure modes and root causes
2. Adjust reward weights or training hyperparameters
3. Consider ablation studies on individual v1.9 components
4. Iterate on observation space or network architecture as needed

### Future Development (v2.0)
- Advanced predictive recovery algorithms
- Multi-robot coordination under disturbances
- Adaptive curriculum based on real-time performance
- Integration with higher-level navigation and planning

---

**Note**: This testing guide assumes v1.9 implementation is complete and properly integrated. All training should be performed with adequate computational resources (GPU recommended) and sufficient iteration counts for convergence (~25k iterations minimum).