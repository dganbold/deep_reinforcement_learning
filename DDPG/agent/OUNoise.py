import numpy as np
import numpy.random as nr

# -------------------------------------------------------------------- #
# Ornstein-Uhlenbeck noise for exploration
# https://github.com/floodsung/DDPG/blob/master/ou_noise.py
# -------------------------------------------------------------------- #
class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
