import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# -------------------------------------------------------------------- #
# CNN Policy-Network
# -------------------------------------------------------------------- #
class Policy(nn.Module):
    """Policy Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers
            seed (int): Random seed
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], action_size)

        #kernel_size = 4
        #stride = 4
        #self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride)
        #self.size = 1*(state_size - kernel_size + stride)/stride
        #self.fc2 = nn.Linear(self.size,hidden_layers[0])
        #self.fc3 = nn.Linear(hidden_layers[1], action_size)
        #self.fc2 = nn.Linear(self.size, action_size)
        #self.sigmoid = nn.Sigmoid()
        #self.reset_parameters()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    #def forward(self, state):
    #    """Build a policy network that maps states -> actions."""
    #    x = F.relu(self.conv(state))
    #    x = x.view(-1,self.size)
    #    return self.sigmoid(self.fc2(x))

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
