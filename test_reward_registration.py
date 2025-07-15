#!/usr/bin/env python3
"""
测试奖励函数注册机制
"""

import sys
import os
sys.path.append('/home/claude_user/agibot_x1/agibot_x1_train')

from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg

def test_reward_registration():
    print("=== 奖励函数注册测试 ===")
    
    # 创建配置
    cfg = X1DHStandCfg()
    
    # 检查奖励配置结构
    print("奖励配置结构检查:")
    print(f"  cfg.rewards: {hasattr(cfg, 'rewards')}")
    print(f"  cfg.rewards.scales: {hasattr(cfg.rewards, 'scales')}")
    
    if hasattr(cfg.rewards, 'scales'):
        # 尝试多种方式获取属性
        try:
            print("方法1: 直接检查属性")
            print(f"  ankle_roll_stability: {getattr(cfg.rewards.scales, 'ankle_roll_stability', 'NOT_FOUND')}")
            print(f"  smooth_transition: {getattr(cfg.rewards.scales, 'smooth_transition', 'NOT_FOUND')}")
            print(f"  foot_coordination_during_transition: {getattr(cfg.rewards.scales, 'foot_coordination_during_transition', 'NOT_FOUND')}")
            
            print("\n方法2: dir() 列出所有属性")
            all_attrs = [attr for attr in dir(cfg.rewards.scales) if not attr.startswith('_')]
            print(f"  所有属性 ({len(all_attrs)}): {all_attrs[:10]}...")
            
            ankle_attrs = [attr for attr in all_attrs if 'ankle' in attr]
            print(f"  踝关节相关属性: {ankle_attrs}")
            
        except Exception as e:
            print(f"检查属性时出错: {e}")
    else:
        print("没有找到scales配置")
    
    # 创建环境实例（不初始化GPU部分）
    try:
        # 检查类中是否有对应的奖励函数
        env_class = X1DHStandEnv
        
        for name in ankle_rewards:
            method_name = f"_reward_{name}"
            if hasattr(env_class, method_name):
                print(f"✓ 找到奖励函数: {method_name}")
            else:
                print(f"✗ 缺失奖励函数: {method_name}")
                
        # 检查所有以_reward_开头的方法
        all_reward_methods = [method for method in dir(env_class) if method.startswith('_reward_')]
        print(f"\n所有奖励方法 ({len(all_reward_methods)}):")
        for method in sorted(all_reward_methods):
            print(f"  {method}")
            
        # 检查配置vs方法的匹配
        print(f"\n配置vs方法匹配检查:")
        if hasattr(cfg.rewards, 'scales'):
            scales_dict = cfg.rewards.scales.__dict__ if hasattr(cfg.rewards.scales, '__dict__') else vars(cfg.rewards.scales)
            for name, scale in scales_dict.items():
                method_name = f"_reward_{name}"
                if hasattr(env_class, method_name):
                    print(f"✓ {name} (scale={scale}) -> {method_name}")
                else:
                    print(f"✗ {name} (scale={scale}) -> 缺失 {method_name}")
        else:
            print("无法检查匹配：没有scales配置")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reward_registration()