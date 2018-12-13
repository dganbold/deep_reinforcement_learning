import numpy as np
import random
from collections import namedtuple, deque

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

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

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
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def push(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def recall(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([[e.states] for e in experiences if e is not None])
        actions = np.vstack([[e.actions] for e in experiences if e is not None])
        rewards = np.vstack([[e.rewards] for e in experiences if e is not None])
        next_states = np.vstack([[e.next_states] for e in experiences if e is not None])
        dones = np.vstack([[e.dones] for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
