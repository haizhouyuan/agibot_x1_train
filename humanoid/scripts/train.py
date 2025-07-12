# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.


from humanoid.envs import *
from humanoid.utils import get_args, task_registry

# Manual registration for v1.7 task
def ensure_v17_task_registered():
    try:
        from humanoid.envs.x1.x1_dh_stand_config_v1_7 import X1DHStandCfg, X1DHStandCfgPPO
        from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
        if 'x1_dh_stand_v1.7' not in task_registry.task_classes:
            task_registry.register('x1_dh_stand_v1.7', X1DHStandEnv, X1DHStandCfg(), X1DHStandCfgPPO())
            print("Successfully registered x1_dh_stand_v1.7 task")
        else:
            print("x1_dh_stand_v1.7 task already registered")
    except ImportError as e:
        print(f"Failed to register v1.7 task: {e}")

# Manual registration for v1.8 task
def ensure_v18_task_registered():
    try:
        from humanoid.envs.x1.x1_dh_stand_config_v1_8 import X1DHStandCfgV18, X1DHStandCfgPPOV18
        from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
        if 'x1_dh_stand_v1.8' not in task_registry.task_classes:
            task_registry.register('x1_dh_stand_v1.8', X1DHStandEnv, X1DHStandCfgV18(), X1DHStandCfgPPOV18())
            print("Successfully registered x1_dh_stand_v1.8 task")
        else:
            print("x1_dh_stand_v1.8 task already registered")
    except ImportError as e:
        print(f"Failed to register v1.8 task: {e}")

# Manual registration for v2.0 task
def ensure_v20_task_registered():
    try:
        from humanoid.envs.x1.x1_dh_stand_config_v2_0 import X1DHStandCfgV2, X1DHStandCfgPPOV2
        from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
        if 'x1_dh_stand_v2.0' not in task_registry.task_classes:
            task_registry.register('x1_dh_stand_v2.0', X1DHStandEnv, X1DHStandCfgV2(), X1DHStandCfgPPOV2())
            print("Successfully registered x1_dh_stand_v2.0 task")
        else:
            print("x1_dh_stand_v2.0 task already registered")
    except ImportError as e:
        print(f"Failed to register v2.0 task: {e}")

def train(args):
    # Ensure v1.7, v1.8, and v2.0 tasks are registered
    ensure_v17_task_registered()
    ensure_v18_task_registered()
    ensure_v20_task_registered()
    
    # v1.7: Allow dynamic task/config selection based on version
    task_name = args.task
    if hasattr(args, 'version') and args.version and args.version != "1.6": # Assuming 1.6 is the default/base
        task_name = f"{task_name}_v{args.version}"
        print(f"Running with custom config version: {args.version}. Task name: {task_name}")

    env, env_cfg = task_registry.make_env(name=task_name, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=task_name, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    train(args)
