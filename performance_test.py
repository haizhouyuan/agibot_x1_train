#!/usr/bin/env python3
"""
Performance testing script for finding optimal configurations
"""
import os
import time
import subprocess
import sys

def test_environment_count(num_envs, duration_seconds=180):
    """Test specific environment count for a duration"""
    print(f"Testing {num_envs} environments...")
    
    # Modify config file
    config_file = "humanoid/envs/x1/x1_dh_stand_config_v1_9.py"
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Backup original
    with open(config_file + '.backup', 'w') as f:
        f.write(content)
    
    # Replace num_envs
    import re
    new_content = re.sub(r'num_envs = \d+.*', f'num_envs = {num_envs}  # Test: {num_envs} environments', content)
    
    with open(config_file, 'w') as f:
        f.write(new_content)
    
    try:
        # Start training process
        env = os.environ.copy()
        env['PYTHONPATH'] = '/root/autodl-tmp/agibot_x1_train-1.9-disturbance-resilience:' + env.get('PYTHONPATH', '')
        
        cmd = [
            'python', 'humanoid/scripts/train.py',
            '--task=x1_dh_stand',
            '--version=1.9', 
            f'--run_name=test_{num_envs}envs',
            '--headless'
        ]
        
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for training to start and measure performance
        time.sleep(duration_seconds)
        proc.terminate()
        
        stdout, stderr = proc.communicate(timeout=10)
        
        # Parse performance metrics from output
        lines = stdout.split('\n')
        iteration_times = []
        for line in lines:
            if 'Iteration time:' in line:
                try:
                    time_str = line.split('Iteration time:')[1].split('s')[0].strip()
                    iteration_times.append(float(time_str))
                except:
                    pass
        
        avg_time = sum(iteration_times) / len(iteration_times) if iteration_times else None
        
        return {
            'num_envs': num_envs,
            'avg_iteration_time': avg_time,
            'num_samples': len(iteration_times),
            'last_iterations': iteration_times[-5:] if iteration_times else []
        }
        
    except Exception as e:
        print(f"Error testing {num_envs} environments: {e}")
        return None
    
    finally:
        # Restore original config
        with open(config_file + '.backup', 'r') as f:
            original_content = f.read()
        with open(config_file, 'w') as f:
            f.write(original_content)
        os.remove(config_file + '.backup')

def run_performance_tests():
    """Run comprehensive performance tests"""
    env_counts = [2048, 3072, 4096, 5120, 6144]
    results = []
    
    for num_envs in env_counts:
        result = test_environment_count(num_envs, duration_seconds=120)
        if result:
            results.append(result)
            print(f"Results for {num_envs} envs: {result['avg_iteration_time']:.2f}s avg iteration time")
        
        # Brief pause between tests
        time.sleep(5)
    
    # Print summary
    print("\n=== Performance Test Summary ===")
    print("Environments | Avg Iteration Time | Efficiency")
    print("-" * 45)
    
    for result in results:
        if result['avg_iteration_time']:
            efficiency = 4096 / result['num_envs'] * result['avg_iteration_time']  # Normalized to 4096 baseline
            print(f"{result['num_envs']:11d} | {result['avg_iteration_time']:17.2f}s | {efficiency:.2f}")
    
    return results

if __name__ == "__main__":
    results = run_performance_tests()