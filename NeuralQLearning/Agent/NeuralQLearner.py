import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------- #
# Q-Network
# -------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers: list of integers, the sizes of the hidden layers
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Forward through each layer in `hidden_layers`, with ReLU activation
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        # Returns the action values
        return self.output(x)

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
        self.name = 'NeuralQLearner'
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Learning params
        self.gamma = param['gamma']

        # Q-Network (Fully connected)
        self.Q_network = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(device)
        self.Q_network_target = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(device)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=param['learning_rate'])

        # Initialize update parameters
        self.t_updates = 0
        self.fix_target_updates = param['fix_target_updates']
        self.thau = param['thau']

    def greedy(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q_network.eval()
        with torch.no_grad():
            action_values = self.Q_network(state)
        self.Q_network.train()
        # Greedy action selection
        return np.argmax(action_values.cpu().data.numpy())


    def eGreedy(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            return self.greedy(state)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.Q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.Q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if (self.t_updates % self.fix_target_updates) == 0:
            self.update_target(self.Q_network, self.Q_network_target, self.thau)
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
        torch.save(self.Q_network.state_dict(), '%s.pth'% (filename))

    def import_network(self,filename):
        self.Q_network.load_state_dict(torch.load('%s.pth'% (filename)))

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
