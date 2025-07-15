#!/usr/bin/env python3
"""
分析TensorBoard数据中的踝关节奖励
"""

import os
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_ankle_rewards(log_dir):
    print(f"=== 分析TensorBoard日志: {log_dir} ===")
    
    # 查找事件文件
    event_files = list(Path(log_dir).glob("events.out.tfevents*"))
    if not event_files:
        print("❌ 未找到TensorBoard事件文件")
        return
    
    event_file = str(event_files[0])
    print(f"📁 分析文件: {event_file}")
    
    # 加载TensorBoard数据
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # 获取所有标量标签
        scalar_tags = ea.Tags()['scalars']
        print(f"\n📊 总共找到 {len(scalar_tags)} 个标量指标")
        
        # 查找踝关节相关标签
        ankle_tags = [tag for tag in scalar_tags if 'ankle' in tag.lower()]
        transition_tags = [tag for tag in scalar_tags if 'transition' in tag.lower()]
        smooth_tags = [tag for tag in scalar_tags if 'smooth' in tag.lower()]
        coordination_tags = [tag for tag in scalar_tags if 'coordination' in tag.lower()]
        
        print(f"\n🔍 踝关节相关指标 ({len(ankle_tags)}):")
        for tag in ankle_tags:
            print(f"  ✅ {tag}")
            
        print(f"\n🔍 过渡相关指标 ({len(transition_tags)}):")
        for tag in transition_tags:
            print(f"  ✅ {tag}")
            
        print(f"\n🔍 平滑相关指标 ({len(smooth_tags)}):")
        for tag in smooth_tags:
            print(f"  ✅ {tag}")
            
        print(f"\n🔍 协调相关指标 ({len(coordination_tags)}):")
        for tag in coordination_tags:
            print(f"  ✅ {tag}")
        
        # 查看所有标签的模式
        print(f"\n📋 所有标量标签 (前20个):")
        for i, tag in enumerate(sorted(scalar_tags)[:20]):
            print(f"  {i+1:2d}. {tag}")
        
        if len(scalar_tags) > 20:
            print(f"  ... 还有 {len(scalar_tags)-20} 个标签")
        
        # 分析所有奖励标签 (多种可能的格式)
        reward_patterns = ['rew_', 'reward', 'Episode/rew_', 'Train/Episode/rew_']
        all_reward_tags = []
        for pattern in reward_patterns:
            pattern_tags = [tag for tag in scalar_tags if pattern in tag]
            all_reward_tags.extend(pattern_tags)
        
        # 去重
        all_reward_tags = list(set(all_reward_tags))
        
        print(f"\n🎯 所有奖励相关指标 ({len(all_reward_tags)}):")
        for tag in sorted(all_reward_tags):
            # 获取最新数据点
            try:
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    latest_value = scalar_events[-1].value
                    step = scalar_events[-1].step
                    print(f"  📈 {tag}: {latest_value:.6f} (step {step})")
                else:
                    print(f"  📈 {tag}: 无数据")
            except Exception as e:
                print(f"  📈 {tag}: 读取错误 - {e}")
        
        # 特别关注的踝关节奖励
        target_rewards = [
            'Train/Episode/rew_ankle_roll_stability',
            'Train/Episode/rew_smooth_transition', 
            'Train/Episode/rew_foot_coordination_during_transition',
            'episode/rew_ankle_roll_stability',
            'episode/rew_smooth_transition',
            'episode/rew_foot_coordination_during_transition'
        ]
        
        print(f"\n🎯 重点关注的踝关节奖励:")
        found_any = False
        for reward in target_rewards:
            if reward in scalar_tags:
                found_any = True
                try:
                    scalar_events = ea.Scalars(reward)
                    if scalar_events:
                        latest_value = scalar_events[-1].value
                        step = scalar_events[-1].step
                        total_points = len(scalar_events)
                        print(f"  ✅ {reward}: {latest_value:.6f} (step {step}, {total_points} 数据点)")
                    else:
                        print(f"  ⚠️ {reward}: 标签存在但无数据")
                except Exception as e:
                    print(f"  ❌ {reward}: 读取错误 - {e}")
            else:
                print(f"  ❌ {reward}: 未找到")
        
        if not found_any:
            print("  ⚠️ 未找到任何目标踝关节奖励")
            
        return ankle_tags, transition_tags, smooth_tags, coordination_tags, reward_tags
        
    except Exception as e:
        print(f"❌ 分析TensorBoard数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 分析最新的几个训练日志
    log_base = "/autodl-fs/data/agibot_x1_train/logs/x1_dh_stand/exported_data"
    
    # 获取最新的训练目录
    log_dirs = sorted([d for d in Path(log_base).iterdir() if d.is_dir()], 
                     key=lambda x: x.name, reverse=True)
    
    print("🔍 分析最新的TensorBoard日志...")
    
    # 分析最新的几个训练
    for i, log_dir in enumerate(log_dirs[:3]):
        print(f"\n{'='*60}")
        print(f"分析训练 {i+1}: {log_dir.name}")
        result = analyze_ankle_rewards(str(log_dir))
        if result:
            ankle_tags, transition_tags, smooth_tags, coordination_tags, reward_tags = result
            if ankle_tags or transition_tags or smooth_tags or coordination_tags:
                print(f"✅ 在 {log_dir.name} 中找到踝关节相关数据!")
                break
    else:
        print("\n⚠️ 在最近的训练中未找到踝关节奖励数据")

if __name__ == "__main__":
    main()