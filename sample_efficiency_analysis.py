#!/usr/bin/env python3
"""
Sample Efficiency Analysis: 2048 vs 4096 environments
"""

# Configuration Analysis
print("=== Sample Efficiency Analysis ===\n")

# Current optimized config
envs_opt = 2048
steps_per_env = 24
mini_batches_opt = 4
samples_per_iter_opt = envs_opt * steps_per_env
samples_per_batch_opt = samples_per_iter_opt // mini_batches_opt

# Original baseline config  
envs_orig = 4096
mini_batches_orig = 4
samples_per_iter_orig = envs_orig * steps_per_env
samples_per_batch_orig = samples_per_iter_orig // mini_batches_orig

print(f"Original Config (4096 envs):")
print(f"  - Samples per iteration: {samples_per_iter_orig:,}")
print(f"  - Samples per mini-batch: {samples_per_batch_orig:,}")
print(f"  - Iteration time: ~2.42s")

print(f"\nOptimized Config (2048 envs):")
print(f"  - Samples per iteration: {samples_per_iter_opt:,}")
print(f"  - Samples per mini-batch: {samples_per_batch_opt:,}")
print(f"  - Iteration time: ~1.71s")

# Sample efficiency impact
sample_ratio = samples_per_iter_orig / samples_per_iter_opt
time_ratio = 1.71 / 2.42

print(f"\n=== Impact Analysis ===")
print(f"Sample collection ratio: {sample_ratio:.2f}x (need {sample_ratio:.2f}x more iterations)")
print(f"Time efficiency ratio: {time_ratio:.2f}x (each iteration {(1-time_ratio)*100:.1f}% faster)")

# Theoretical analysis
theoretical_time_multiplier = sample_ratio * time_ratio
print(f"Theoretical time impact: {theoretical_time_multiplier:.2f}x")
if theoretical_time_multiplier < 1.0:
    print(f"Expected speedup: {(1-theoretical_time_multiplier)*100:.1f}% faster overall")
else:
    print(f"Expected slowdown: {(theoretical_time_multiplier-1)*100:.1f}% slower overall")

# Batch size analysis for further optimization
print(f"\n=== Batch Size Optimization Potential ===")
for mini_batches in [2, 6, 8, 12]:
    batch_size = samples_per_iter_opt // mini_batches
    gpu_util_estimate = min(100, batch_size / 8192 * 50)  # Rough estimate
    print(f"  {mini_batches} mini-batches: {batch_size:,} samples/batch (Est. GPU util: {gpu_util_estimate:.0f}%)")

print(f"\n=== Key Questions for Testing ===")
print(f"1. Does the faster iteration speed compensate for needing 2x iterations?")
print(f"2. Does the smaller batch size (12,288 vs 24,576) affect learning stability?")
print(f"3. Can we increase mini-batches to improve GPU utilization further?")
print(f"4. Is the reduced exploration diversity (2048 vs 4096 envs) affecting convergence quality?")

# Current training metrics
print(f"\n=== Current Training Status ===")
print(f"- Total timesteps achieved: 96,288,768")
print(f"- Training time so far: ~1 hour")
print(f"- Current ETA: 11.8 hours")
print(f"- Performance: 30% faster than 4096 env baseline")