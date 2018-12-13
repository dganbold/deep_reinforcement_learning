import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# -------------------------------------------------------------------- #
# Fully Connected Q-Network
# -------------------------------------------------------------------- #
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, output_size, hidden_layers, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each state
            output_size (int): Dimension of each action
            hidden_layers (int): Number of nodes and hidden layers
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, input):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, output_size, hidden_layers, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each state
            output_size (int): Dimension of each action
            hidden_layers (int): Number of nodes and hidden layers
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        sa = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
