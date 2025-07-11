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

import os
from humanoid.envs import *
from humanoid.utils import get_args, task_registry

def finetune(args):
    # --- Start of Gemini modifications for fine-tuning ---

    # 1. Force the task to 'x1_dh_stand' as we are fine-tuning this specific task.
    args.task = "x1_dh_stand"

    # 2. Point to the specific model checkpoint to load.
    #    This assumes the 'train.py' script and the log directory structure are set up
    #    to handle a direct path for the checkpoint.
    #    We will set 'resume' to True, and specify the checkpoint number.
    args.resume = True
    args.load_run = "v1.6_training_logs"  # The directory containing the model
    args.checkpoint = 6900  # The model iteration to load

    # 3. Specify a new experiment name to avoid overwriting original logs.
    args.experiment_name = "x1_dh_stand_finetune"
    
    print("--- Fine-tuning session initiated by Gemini ---")
    print(f"Task: {args.task}")
    print(f"Loading model from run: {args.load_run}, checkpoint: {args.checkpoint}")
    print(f"New experiment name: {args.experiment_name}")
    print("Using custom fine-tuning configuration: x1_dh_stand_finetune_config.py")
    print("-------------------------------------------------")

    # --- End of Gemini modifications ---

    # The task registry will now use our new 'x1_dh_stand_finetune_config.py'
    # if we modify the registry to point to it for the 'x1_dh_stand' task.
    # A safer way is to modify how the environment is created.
    # Let's adjust the task registry on the fly for this script.

    from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
    from humanoid.envs.x1.x1_dh_stand_finetune_config import X1DHStandCfg, X1DHStandCfgPPO

    task_registry.register("x1_dh_stand_finetune", X1DHStandEnv, X1DHStandCfg, X1DHStandCfgPPO)
    args.task = "x1_dh_stand_finetune"


    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Make sure the runner knows the absolute path to the checkpoint
    if args.resume:
        # Construct the full path to the model
        # Assumes the script is run from the 'agibot_x1_train' directory.
        log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        run_path = os.path.join(log_root, args.load_run)
        if args.checkpoint == -1:
            # Logic to find the last checkpoint if needed
            pass 
        else:
            model_path = os.path.join(run_path, f"model_{args.checkpoint}.pt")
            if os.path.exists(model_path):
                 ppo_runner.load(model_path)
                 print(f"Successfully loaded model from: {model_path}")
            else:
                 print(f"Error: Model path not found: {model_path}")
                 return

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    finetune(args)
