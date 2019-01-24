import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.distributions import Categorical
from network import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------- #
# REINFORCE Agent
# -------------------------------------------------------------------- #
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, param, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            param: hyperparameter
            seed (int): random seed
        """
        self.name = 'REINFORCE'
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Learning params
        self.gamma = param['gamma']

        # Policy Network
        self.policy = Policy(state_size, action_size, param['hidden_layers'], seed).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

        # Print model summary
        #print(self.policy)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    '''
    def surrogate(policy, old_probs, states, actions, rewards, discount = 0.995, beta=0.01):
        """Returns loss.

        """
        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        
        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        R = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
    
        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    
        # include a regularization term
        # this steers new_policy towards 0.5
        # which prevents policy to become exactly 0 or 1
        # this helps with exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+(1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    
        with torch.no_grad():
            L = R*torch.div(new_probs, old_probs+1.e-10)
        #
        return torch.mean(L + beta*entropy)
    '''

    def learn(self, experiences):
        """Update policy using given batch of experience tuples.
        where:
            policy(state) -> action

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (prob, s, a, r, ) tuples
        """
        saved_log_probs, rewards = experiences
        
        discounts = [self.gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)]) 

        # --------------------------- update policy --------------------------- #
        # Calculate surrogate loss
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        # --------------------------------------------------------------------- #


    def export_network(self,filename):
        torch.save(self.policy.state_dict(), '%s_policy.pth'% (filename))

    def import_network(self,filename):
        self.policy.load_state_dict(torch.load('%s_policy.pth'% (filename)))

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
