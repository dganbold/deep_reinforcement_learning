import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from network import *
from agent.OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------- #
# DDPG
# -------------------------------------------------------------------- #
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, param, seed):
        """Initialize an Agent object.

        Params
        ======
            param: hyperparameter
            seed (int): random seed
        """
        self.name = 'DDPG'
        self.seed = random.seed(seed)

        # Learning params
        self.gamma = param['gamma']

        # Actor Network
        self.actor_local = Actor(param['actor_input_size'], param['actor_output_size'], param['actor_hidden_layers'], seed).to(device)
        self.actor_target = Actor(param['actor_input_size'], param['actor_output_size'], param['actor_hidden_layers'], seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=param['actor_learning_rate'], weight_decay=param['actor_weight_decay'])

        # Critic Network
        self.critic_local = Critic(param['critic_input_size'], param['critic_output_size'], param['critic_hidden_layers'], seed).to(device)
        self.critic_target = Critic(param['critic_input_size'], param['critic_output_size'], param['critic_hidden_layers'], seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=param['critic_learning_rate'], weight_decay=param['critic_weight_decay'])

        # initialize targets same as original networks
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Initialize update parameters
        self.actor_thau = param['actor_thau']
        self.critic_thau = param['critic_thau']

        # Noise process
        self.noise = OUNoise(param['actor_output_size'], seed)

        # Print model summary
        #print(self.Q_network)

    def act(self, state, noise_amplitude=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            add_noise (bool):
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise_amplitude*self.noise.sample()
        return np.clip(action, -1, 1)

    def target_act(self, state, noise_amplitude=0.0):
        state = state.to(device)
        action = self.target_actor(state) + noise_amplitude*self.noise.sample()
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.critic_soft_update()
        self.actor_soft_update()

    def actor_soft_update(self):
        self.soft_update(self.actor_local, self.actor_target, self.actor_thau)

    def critic_soft_update(self):
        self.soft_update(self.critic_local, self.critic_target, self.critic_thau)

    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
    def soft_update(self, source, target, tau):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(self, source, target):
        """
        θ_target = θ_local
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def export_network(self,filename):
        torch.save(self.actor_local.state_dict(), '%s_actor.pth'% (filename))
        torch.save(self.critic_local.state_dict(), '%s_critic.pth'% (filename))

    def import_network(self,filename):
        self.actor_local.load_state_dict(torch.load('%s_actor.pth'% (filename)))
        self.critic_local.load_state_dict(torch.load('%s_critic.pth'% (filename)))

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
