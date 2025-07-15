#!/usr/bin/env python3
"""
深度调试奖励注册机制
"""

import sys
import os
import inspect
import importlib.util
sys.path.append('/home/claude_user/agibot_x1/agibot_x1_train')

# 直接分析源代码文件，避免Isaac Gym初始化
env_file = '/home/claude_user/agibot_x1/agibot_x1_train/humanoid/envs/x1/x1_dh_stand_env.py'
config_file = '/home/claude_user/agibot_x1/agibot_x1_train/humanoid/envs/x1/x1_dh_stand_config.py'

def debug_reward_registration():
    print("=== 深度奖励注册调试 ===")
    
    # 创建配置
    cfg = X1DHStandCfg()
    
    # 检查配置的奖励scales
    print("\n1. 配置检查:")
    scales_dict = class_to_dict(cfg.rewards.scales)
    print(f"   配置中的奖励数量: {len(scales_dict)}")
    
    ankle_rewards = {k: v for k, v in scales_dict.items() if 'ankle' in k or 'transition' in k or 'smooth' in k}
    print(f"   踝关节相关奖励: {ankle_rewards}")
    
    # 检查环境类的方法
    print("\n2. 环境类方法检查:")
    ankle_methods = [method for method in dir(X1DHStandEnv) if method.startswith('_reward_') and ('ankle' in method or 'transition' in method or 'smooth' in method)]
    print(f"   踝关节相关方法: {ankle_methods}")
    
    # 模拟奖励注册过程
    print("\n3. 模拟奖励注册过程:")
    
    # 模拟_prepare_reward_function的逻辑
    reward_scales = scales_dict.copy()
    print(f"   初始奖励数量: {len(reward_scales)}")
    
    # 移除零scale的奖励
    for key in list(reward_scales.keys()):
        scale = reward_scales[key]
        if scale == 0:
            print(f"   移除零权重奖励: {key}")
            reward_scales.pop(key)
    
    print(f"   非零奖励数量: {len(reward_scales)}")
    
    # 检查奖励函数获取
    print("\n4. 奖励函数获取测试:")
    missing_functions = []
    existing_functions = []
    
    for name, scale in reward_scales.items():
        if name == "termination":
            continue
        method_name = '_reward_' + name
        
        if hasattr(X1DHStandEnv, method_name):
            existing_functions.append((name, scale, method_name))
            print(f"   ✅ {name} (scale={scale}) -> {method_name}")
        else:
            missing_functions.append((name, scale, method_name))
            print(f"   ❌ {name} (scale={scale}) -> 缺失 {method_name}")
    
    print(f"\n5. 总结:")
    print(f"   可用奖励函数: {len(existing_functions)}")
    print(f"   缺失奖励函数: {len(missing_functions)}")
    
    if missing_functions:
        print(f"\n   缺失的奖励函数:")
        for name, scale, method_name in missing_functions:
            print(f"     - {method_name} (for {name}, scale={scale})")
    
    # 特别检查踝关节奖励
    print(f"\n6. 踝关节奖励专项检查:")
    target_rewards = ['ankle_roll_stability', 'smooth_transition', 'foot_coordination_during_transition']
    
    for reward in target_rewards:
        print(f"\n   {reward}:")
        
        # 检查配置
        if reward in scales_dict:
            scale = scales_dict[reward]
            print(f"     配置: ✅ scale={scale}")
        else:
            print(f"     配置: ❌ 未找到")
            continue
            
        # 检查方法
        method_name = f'_reward_{reward}'
        if hasattr(X1DHStandEnv, method_name):
            print(f"     方法: ✅ {method_name}")
        else:
            print(f"     方法: ❌ 缺失 {method_name}")
            continue
            
        # 检查是否会被注册
        if scale != 0:
            print(f"     注册: ✅ 非零权重")
        else:
            print(f"     注册: ❌ 零权重")

if __name__ == "__main__":
    debug_reward_registration()