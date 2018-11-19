import torch
import torch.nn.functional as F
import torch.nn as nn

# -------------------------------------------------------------------- #
# Fully Connected Q-Network
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
# EOF
# -------------------------------------------------------------------- #
