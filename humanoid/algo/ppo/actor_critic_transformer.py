# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import torch
import torch.nn as nn
import math
from torch.distributions import Normal

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer sequences.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ActorCriticTransformer(nn.Module):
    """
    v2.0 Phase 4: Transformer-based Actor-Critic network for improved sequential decision making.
    Replaces LSTM with multi-head attention mechanism for better long-term memory.
    """
    def __init__(self, 
                 num_short_obs,
                 num_proprio_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[768, 256, 128],
                 state_estimator_hidden_dims=[256, 128, 64],
                 in_channels=66,
                 kernel_size=[6, 4],
                 filter_size=[32, 16],
                 stride_size=[3, 2],
                 lh_output_dim=64,
                 init_noise_std=1.0,
                 # Transformer specific parameters
                 transformer_hidden_size=128,
                 transformer_num_heads=4,
                 transformer_num_layers=2,
                 transformer_sequence_length=16,
                 activation=nn.ELU(),
                 **kwargs):
        
        if kwargs:
            print("ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticTransformer, self).__init__()

        # Transformer configuration
        self.transformer_hidden_size = transformer_hidden_size
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.sequence_length = transformer_sequence_length
        
        # Input dimensions
        mlp_input_dim_a = num_short_obs + lh_output_dim + 3
        mlp_input_dim_c = num_critic_obs
        
        # Transformer components for actor
        self.obs_embed_actor = nn.Linear(mlp_input_dim_a, transformer_hidden_size)
        self.pos_encoding_actor = PositionalEncoding(transformer_hidden_size, max_len=transformer_sequence_length)
        
        encoder_layer_actor = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_hidden_size * 2,
            batch_first=False  # [seq_len, batch, features]
        )
        self.transformer_actor = nn.TransformerEncoder(encoder_layer_actor, num_layers=transformer_num_layers)
        
        # Transformer components for critic
        self.obs_embed_critic = nn.Linear(mlp_input_dim_c, transformer_hidden_size)
        self.pos_encoding_critic = PositionalEncoding(transformer_hidden_size, max_len=transformer_sequence_length)
        
        encoder_layer_critic = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_hidden_size * 2,
            batch_first=False
        )
        self.transformer_critic = nn.TransformerEncoder(encoder_layer_critic, num_layers=transformer_num_layers)
        
        # Actor network (after transformer)
        actor_layers = []
        actor_layers.append(nn.Linear(transformer_hidden_size, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic network (after transformer)
        critic_layers = []
        critic_layers.append(nn.Linear(transformer_hidden_size, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Transformer Actor: {self.transformer_actor}")
        print(f"Actor MLP: {self.actor}")
        print(f"Transformer Critic: {self.transformer_critic}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        
        # Long history CNN (same as original)
        long_history_layers = []
        self.in_channels = in_channels
        cnn_output_dim = num_proprio_obs
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                               kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)
        print(f"long_history CNN: {self.long_history}")
        
        # State estimator MLP (same as original)
        self.num_short_obs = num_short_obs
        state_estimator_input_dim = num_short_obs
        state_estimator_output_dim = 3
        state_estimator_layers = []
        state_estimator_layers.append(nn.Linear(state_estimator_input_dim, state_estimator_hidden_dims[0]))
        state_estimator_layers.append(activation)
        for l in range(len(state_estimator_hidden_dims)):
            if l == len(state_estimator_hidden_dims) - 1:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], state_estimator_output_dim))
            else:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], state_estimator_hidden_dims[l + 1]))
                state_estimator_layers.append(activation)
        self.state_estimator = nn.Sequential(*state_estimator_layers)
        print(f"state_estimator MLP: {self.state_estimator}")
        
        self.num_proprio_obs = num_proprio_obs
        
        # Sequence memory for transformer
        self.actor_sequence_buffer = None
        self.critic_sequence_buffer = None

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        """Reset sequence buffers for environments that are done."""
        if dones is not None and (self.actor_sequence_buffer is not None):
            # Reset sequence buffers for done environments
            batch_size = dones.shape[0]
            if self.actor_sequence_buffer.shape[1] != batch_size:
                # Initialize buffers if they don't match batch size
                self._init_sequence_buffers(batch_size, dones.device)
            else:
                # Reset only done environments
                self.actor_sequence_buffer[:, dones] = 0
                self.critic_sequence_buffer[:, dones] = 0

    def _init_sequence_buffers(self, batch_size, device):
        """Initialize sequence buffers for transformer memory."""
        actor_input_dim = self.transformer_hidden_size
        critic_input_dim = self.transformer_hidden_size
        
        self.actor_sequence_buffer = torch.zeros(
            self.sequence_length, batch_size, actor_input_dim, 
            device=device, dtype=torch.float32
        )
        self.critic_sequence_buffer = torch.zeros(
            self.sequence_length, batch_size, critic_input_dim,
            device=device, dtype=torch.float32
        )

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations_actor):
        """
        Update action distribution using transformer-processed observations.
        """
        # Process observations through transformer
        if len(observations_actor.shape) == 2:
            # Single timestep, expand for sequence
            batch_size = observations_actor.shape[0]
            device = observations_actor.device
            
            # Initialize sequence buffers if needed
            if self.actor_sequence_buffer is None or self.actor_sequence_buffer.shape[1] != batch_size:
                self._init_sequence_buffers(batch_size, device)
            
            # Embed current observation
            embedded_obs = self.obs_embed_actor(observations_actor)  # [batch, hidden_size]
            
            # Shift sequence buffer and add new observation
            self.actor_sequence_buffer[:-1] = self.actor_sequence_buffer[1:].clone()
            self.actor_sequence_buffer[-1] = embedded_obs
            
            # Apply positional encoding
            sequence_with_pos = self.pos_encoding_actor(self.actor_sequence_buffer)
            
            # Process through transformer
            transformer_output = self.transformer_actor(sequence_with_pos)  # [seq_len, batch, hidden_size]
            
            # Use the last timestep output for action prediction
            final_features = transformer_output[-1]  # [batch, hidden_size]
            
        else:
            # Sequence input [batch, seq_len, features] or [seq_len, batch, features]
            if len(observations_actor.shape) == 3:
                if observations_actor.shape[0] != self.sequence_length:
                    # Assume batch-first format, transpose to seq-first
                    observations_actor = observations_actor.transpose(0, 1)
                
                # Embed sequence
                embedded_seq = self.obs_embed_actor(observations_actor)  # [seq_len, batch, hidden_size]
                sequence_with_pos = self.pos_encoding_actor(embedded_seq)
                transformer_output = self.transformer_actor(sequence_with_pos)
                final_features = transformer_output[-1]  # Use last timestep
            else:
                raise ValueError(f"Unexpected observation shape: {observations_actor.shape}")
        
        # Generate actions from final features
        mean = self.actor(final_features)
        self.distribution = Normal(mean, mean*0. + self.std)
        return self.distribution

    def act(self, observations_actor, **kwargs):
        """Generate actions using transformer-processed observations."""
        self.update_distribution(observations_actor)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probabilities for given actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations_actor):
        """Inference mode action generation (deterministic)."""
        self.update_distribution(observations_actor)
        return self.distribution.mean

    def evaluate(self, observations_critic, observations_actor, actions, **kwargs):
        """
        Evaluate actions and observations to get value, action log probs, and entropy.
        """
        # Process critic observations through transformer
        batch_size = observations_critic.shape[0]
        device = observations_critic.device
        
        # Initialize critic sequence buffer if needed
        if self.critic_sequence_buffer is None or self.critic_sequence_buffer.shape[1] != batch_size:
            if self.critic_sequence_buffer is None:
                self._init_sequence_buffers(batch_size, device)
        
        # Process critic observations
        if len(observations_critic.shape) == 2:
            # Embed current observation
            embedded_obs_critic = self.obs_embed_critic(observations_critic)
            
            # Shift sequence buffer and add new observation
            self.critic_sequence_buffer[:-1] = self.critic_sequence_buffer[1:].clone()
            self.critic_sequence_buffer[-1] = embedded_obs_critic
            
            # Apply positional encoding and process through transformer
            sequence_with_pos_critic = self.pos_encoding_critic(self.critic_sequence_buffer)
            transformer_output_critic = self.transformer_critic(sequence_with_pos_critic)
            final_features_critic = transformer_output_critic[-1]
        else:
            # Handle sequence input for critic
            if observations_critic.shape[0] != self.sequence_length:
                observations_critic = observations_critic.transpose(0, 1)
            embedded_seq_critic = self.obs_embed_critic(observations_critic)
            sequence_with_pos_critic = self.pos_encoding_critic(embedded_seq_critic)
            transformer_output_critic = self.transformer_critic(sequence_with_pos_critic)
            final_features_critic = transformer_output_critic[-1]
        
        # Get value from critic
        value = self.critic(final_features_critic)
        
        # Update action distribution and get log probs
        self.update_distribution(observations_actor)
        actions_log_prob = self.get_actions_log_prob(actions)
        entropy = self.entropy
        
        return value, actions_log_prob, entropy