#!/usr/bin/env python3
"""
简化的奖励注册调试 - 避免Isaac Gym初始化
"""

import re

def analyze_config_file():
    """分析配置文件中的奖励"""
    config_file = '/home/claude_user/agibot_x1/agibot_x1_train/humanoid/envs/x1/x1_dh_stand_config.py'
    
    print("=== 分析配置文件 ===")
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 查找scales类中的奖励配置
    scales_pattern = r'class scales\([^)]*\):(.*?)(?=class|\Z)'
    scales_match = re.search(scales_pattern, content, re.DOTALL)
    
    if not scales_match:
        print("❌ 未找到scales类")
        return []
    
    scales_content = scales_match.group(1)
    
    # 提取奖励配置
    reward_pattern = r'(\w+)\s*=\s*([-+]?\d*\.?\d+)'
    rewards = re.findall(reward_pattern, scales_content)
    
    print(f"📊 配置文件中的奖励数量: {len(rewards)}")
    
    ankle_rewards = [(name, float(value)) for name, value in rewards if 'ankle' in name or 'transition' in name or 'smooth' in name]
    print(f"🦶 踝关节相关奖励: {ankle_rewards}")
    
    return rewards

def analyze_env_file():
    """分析环境文件中的奖励函数"""
    env_file = '/home/claude_user/agibot_x1/agibot_x1_train/humanoid/envs/x1/x1_dh_stand_env.py'
    
    print("\n=== 分析环境文件 ===")
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    # 查找所有_reward_开头的方法
    reward_pattern = r'def (_reward_\w+)\s*\('
    reward_methods = re.findall(reward_pattern, content)
    
    print(f"📊 环境中的奖励方法数量: {len(reward_methods)}")
    
    ankle_methods = [method for method in reward_methods if 'ankle' in method or 'transition' in method or 'smooth' in method]
    print(f"🦶 踝关节相关方法: {ankle_methods}")
    
    return reward_methods

def cross_check():
    """交叉检查配置和实现"""
    print("\n=== 交叉检查 ===")
    
    # 获取配置和方法
    rewards = analyze_config_file()
    methods = analyze_env_file()
    
    # 转换为集合便于比较
    reward_names = set([name for name, value in rewards])
    method_names = set([method.replace('_reward_', '') for method in methods])
    
    print(f"\n🔍 详细对比:")
    
    # 检查每个配置的奖励
    target_rewards = ['ankle_roll_stability', 'smooth_transition', 'foot_coordination_during_transition']
    
    for reward_name in target_rewards:
        print(f"\n   {reward_name}:")
        
        # 检查配置
        config_found = False
        for name, value in rewards:
            if name == reward_name:
                print(f"     配置: ✅ {name} = {value}")
                config_found = True
                break
        if not config_found:
            print(f"     配置: ❌ 未找到")
        
        # 检查方法
        if reward_name in method_names:
            print(f"     方法: ✅ _reward_{reward_name}")
        else:
            print(f"     方法: ❌ 缺失 _reward_{reward_name}")
    
    # 总结
    print(f"\n📋 总结:")
    print(f"   配置奖励数量: {len(rewards)}")
    print(f"   实现方法数量: {len(methods)}")
    print(f"   配置但未实现: {reward_names - method_names}")
    print(f"   实现但未配置: {method_names - reward_names}")

if __name__ == "__main__":
    cross_check()