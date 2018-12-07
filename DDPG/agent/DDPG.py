import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from network import *
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------- #
# Neural Q-Learning
# -------------------------------------------------------------------- #
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, param, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.name = 'DDPG'
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Learning params
        self.gamma = param['gamma']

        # Actor Network
        self.actor_local = Actor(state_size, action_size, param['actor_hidden_layers'], seed).to(device)
        self.actor_target = Actor(state_size, action_size, param['actor_hidden_layers'], seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=param['actor_learning_rate'])

        # Critic Network
        self.critic_local = Critic(state_size, action_size, param['critic_hidden_layers'], seed).to(device)
        self.critic_target = Critic(state_size, action_size, param['critic_hidden_layers'], seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=param['actor_learning_rate'], weight_decay=param['weight_decay'])

        # Initialize update parameters
        self.t_updates = 0
        self.fix_target_updates = param['fix_target_updates']
        self.thau = param['thau']

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Print model summary
        #print(self.Q_network)

    def act(self, state, add_noise=True):
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
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

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
        # Update target network
        if (self.t_updates % self.fix_target_updates) == 0:
            self.update_target(self.critic_local, self.critic_target, self.thau)
            self.update_target(self.actor_local, self.actor_target, self.thau)

        self.t_updates += 1

    def update_target(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def export_network(self,filename):
        torch.save(self.actor_local.state_dict(), '%s_actor.pth'% (filename))
        torch.save(self.critic_local.state_dict(), '%s_critic.pth'% (filename))

    def import_network(self,filename):
        self.actor_local.load_state_dict(torch.load('%s_actor.pth'% (filename)))
        self.critic_local.load_state_dict(torch.load('%s_critic.pth'% (filename)))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
