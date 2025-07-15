#!/usr/bin/env python3
"""
深度分析V2.10与V2.11训练差异
通过捕获tmux会话输出进行实时分析
"""

import subprocess
import re
import json
from datetime import datetime

def capture_tmux_output(session_name, lines=50):
    """捕获tmux会话的输出"""
    try:
        result = subprocess.run(
            ['tmux', 'capture-pane', '-S', f'-{lines}', '-p', '-t', session_name],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error capturing tmux session {session_name}: {e}")
        return None

def parse_training_metrics(output):
    """解析训练输出中的指标"""
    metrics = {}
    
    # 查找迭代信息
    iter_match = re.search(r'Learning iteration (\d+)/(\d+)', output)
    if iter_match:
        metrics['current_iteration'] = int(iter_match.group(1))
        metrics['total_iterations'] = int(iter_match.group(2))
    
    # 查找总奖励
    reward_match = re.search(r'Mean reward: ([\d.]+)', output)
    if reward_match:
        metrics['mean_reward'] = float(reward_match.group(1))
    
    # 查找Episode长度
    episode_match = re.search(r'Mean episode length: ([\d.]+)', output)
    if episode_match:
        metrics['mean_episode_length'] = float(episode_match.group(1))
    
    # 查找总时间步
    timestep_match = re.search(r'Total timesteps: (\d+)', output)
    if timestep_match:
        metrics['total_timesteps'] = int(timestep_match.group(1))
    
    # 查找训练时间
    time_match = re.search(r'Total time: ([\d.]+)s', output)
    if time_match:
        metrics['total_time'] = float(time_match.group(1))
    
    # 查找ETA
    eta_match = re.search(r'ETA: ([\d.]+)s', output)
    if eta_match:
        metrics['eta'] = float(eta_match.group(1))
    
    # 解析所有奖励指标
    reward_patterns = [
        r'Mean episode (rew_[\w_]+): ([-\d.]+)',
        r'Mean episode ([\w_]+): ([-\d.]+)'
    ]
    
    rewards = {}
    for pattern in reward_patterns:
        matches = re.findall(pattern, output)
        for match in matches:
            reward_name = match[0]
            reward_value = float(match[1])
            if reward_name.startswith('rew_'):
                rewards[reward_name] = reward_value
    
    metrics['rewards'] = rewards
    
    return metrics

def analyze_speed_control_effectiveness(v210_metrics, v211_metrics):
    """分析速度控制的有效性"""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'v210_metrics': v210_metrics,
        'v211_metrics': v211_metrics,
        'comparison': {}
    }
    
    print("🔍 V2.10 vs V2.11 训练对比分析")
    print("=" * 60)
    
    # 基础指标对比
    if 'mean_reward' in v210_metrics and 'mean_reward' in v211_metrics:
        v210_reward = v210_metrics['mean_reward']
        v211_reward = v211_metrics['mean_reward']
        reward_diff = ((v211_reward - v210_reward) / v210_reward) * 100
        print(f"📊 总奖励对比:")
        print(f"  V2.10: {v210_reward:.2f}")
        print(f"  V2.11: {v211_reward:.2f}")
        print(f"  差异: {reward_diff:+.1f}% {'✅' if reward_diff > -10 else '⚠️'}")
        analysis['comparison']['reward_change'] = reward_diff
    
    # Episode长度对比
    if 'mean_episode_length' in v210_metrics and 'mean_episode_length' in v211_metrics:
        v210_length = v210_metrics['mean_episode_length']
        v211_length = v211_metrics['mean_episode_length']
        length_diff = ((v211_length - v210_length) / v210_length) * 100
        print(f"\n📊 Episode长度对比:")
        print(f"  V2.10: {v210_length:.1f} 步")
        print(f"  V2.11: {v211_length:.1f} 步")
        print(f"  差异: {length_diff:+.1f}% {'✅' if length_diff > -20 else '⚠️'}")
        analysis['comparison']['episode_length_change'] = length_diff
    
    # 速度相关奖励分析
    speed_rewards = [
        'rew_tracking_lin_vel',
        'rew_tracking_ang_vel', 
        'rew_low_speed',
        'rew_track_vel_hard',
        'rew_speed_limit'  # V2.11新增
    ]
    
    print(f"\n🎯 速度相关奖励对比:")
    for reward in speed_rewards:
        v210_val = v210_metrics.get('rewards', {}).get(reward, 0)
        v211_val = v211_metrics.get('rewards', {}).get(reward, 0)
        
        if v210_val != 0 or v211_val != 0:
            if v210_val != 0:
                change = ((v211_val - v210_val) / abs(v210_val)) * 100
                print(f"  {reward}:")
                print(f"    V2.10: {v210_val:.6f}")
                print(f"    V2.11: {v211_val:.6f}")
                print(f"    变化: {change:+.1f}%")
            else:
                print(f"  {reward}: V2.11新增 = {v211_val:.6f}")
            
            analysis['comparison'][reward] = {
                'v210': v210_val,
                'v211': v211_val,
                'change': change if v210_val != 0 else 'new'
            }
    
    # 稳定性相关奖励
    stability_rewards = [
        'rew_orientation',
        'rew_base_height',
        'rew_feet_contact_forces',
        'rew_ankle_roll_stability'
    ]
    
    print(f"\n🎯 稳定性相关奖励对比:")
    for reward in stability_rewards:
        v210_val = v210_metrics.get('rewards', {}).get(reward, 0)
        v211_val = v211_metrics.get('rewards', {}).get(reward, 0)
        
        if v210_val != 0 or v211_val != 0:
            if v210_val != 0:
                change = ((v211_val - v210_val) / abs(v210_val)) * 100
                print(f"  {reward}:")
                print(f"    V2.10: {v210_val:.6f}")
                print(f"    V2.11: {v211_val:.6f}")
                print(f"    变化: {change:+.1f}%")
            else:
                print(f"  {reward}: V2.11新增 = {v211_val:.6f}")
    
    # 训练效率对比
    if 'total_timesteps' in v210_metrics and 'total_timesteps' in v211_metrics:
        v210_steps = v210_metrics['total_timesteps']
        v211_steps = v211_metrics['total_timesteps']
        print(f"\n📊 训练进度对比:")
        print(f"  V2.10: {v210_steps:,} 步")
        print(f"  V2.11: {v211_steps:,} 步")
        print(f"  V2.11相对进度: {(v211_steps/v210_steps)*100:.1f}%")
    
    # 保存分析结果
    with open('/home/claude_user/agibot_x1/agibot_x1_train/training_comparison_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📁 分析结果已保存到: training_comparison_analysis.json")
    return analysis

def main():
    print("🚀 开始分析V2.10和V2.11训练差异...")
    
    # 捕获两个训练的输出
    v210_output = capture_tmux_output('v29_final_fix', 100)
    v211_output = capture_tmux_output('v211_training', 100)
    
    if not v210_output:
        print("❌ 无法捕获V2.10训练输出")
        return
    
    if not v211_output:
        print("❌ 无法捕获V2.11训练输出")
        return
    
    # 解析指标
    v210_metrics = parse_training_metrics(v210_output)
    v211_metrics = parse_training_metrics(v211_output)
    
    print(f"✅ V2.10指标解析完成: {len(v210_metrics.get('rewards', {}))} 个奖励")
    print(f"✅ V2.11指标解析完成: {len(v211_metrics.get('rewards', {}))} 个奖励")
    
    # 进行对比分析
    analysis = analyze_speed_control_effectiveness(v210_metrics, v211_metrics)
    
    # 输出关键结论
    print("\n" + "=" * 60)
    print("🎯 关键结论:")
    
    if 'reward_change' in analysis['comparison']:
        reward_change = analysis['comparison']['reward_change']
        if reward_change > -10:
            print(f"✅ 速度控制实现良好，总奖励变化: {reward_change:+.1f}%")
        else:
            print(f"⚠️ 速度控制导致性能下降: {reward_change:+.1f}%")
    
    if 'rew_tracking_lin_vel' in analysis['comparison']:
        lin_vel_change = analysis['comparison']['rew_tracking_lin_vel']
        if isinstance(lin_vel_change, dict) and lin_vel_change.get('change', 0) < 0:
            print(f"✅ 线性速度追踪降低，速度控制生效")
        else:
            print(f"⚠️ 线性速度追踪未如预期降低")
    
    if 'rew_speed_limit' in analysis['comparison']:
        print(f"✅ 新增速度限制奖励功能正常")
    
    print("\n🔄 建议:")
    print("1. 继续监控V2.11训练，等待更多收敛数据")
    print("2. 如果V2.11性能稳定，可考虑部署到实际机器人")
    print("3. 对比更长时间的训练数据以确认速度控制有效性")

if __name__ == "__main__":
    main()