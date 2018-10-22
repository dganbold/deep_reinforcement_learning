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
# Double Q-Learning
# -------------------------------------------------------------------- #
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, name, state_size, action_size, param, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Learning params
        self.gamma = param['gamma']

        # Q-Network (Fully connected)
        self.Q_network = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(device)
        self.Q_network_target = QNetwork(state_size, action_size, param['hidden_layers'], seed).to(device)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=param['learning_rate'])

        # Replay memory
        self.memory = ReplayBuffer(action_size, param['replay_size'], param['batch_size'], seed)
        # If enough samples are available in memory then start sampling
        self.replay_start = param['replay_initial']
        # Initialize time step (for updating every update_interval steps)
        self.t_step = 0
        self.update_interval = param['update_interval']
        self.thau = param['thau']

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_interval
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_start:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = state_buffer.step(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q_network.eval()
        with torch.no_grad():
            action_values = self.Q_network(state)
        self.Q_network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get best action for next states from Q function
        _,next_state_actions = self.Q_network(next_states).detach().max(1)

        # Get predicted Q values for next states from target Q function
        Q_targets_next = self.Q_network_target(next_states).gather(1, next_state_actions.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.Q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.Q_network, self.Q_network_target, self.thau)

    def soft_update(self, local_model, target_model, tau):
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

# -------------------------------------------------------------------- #
# Experience replay
# -------------------------------------------------------------------- #
class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
