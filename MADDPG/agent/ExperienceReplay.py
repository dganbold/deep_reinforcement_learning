import numpy as np
import random
from collections import namedtuple, deque

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------- #
# Experience replay
# -------------------------------------------------------------------- #
class ReplayBuffer():
    """Fixed-size buffer to store transitions tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def push(self, transitions):
        """Add a new transitions to memory."""
        self.memory.append(transitions)


    def recall(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states  = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None])
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        dones = np.vstack([e[4] for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
