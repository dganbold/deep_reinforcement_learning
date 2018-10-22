import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import OpenAIGym_Box2d

from DoubleQLearner import Agent
from collections import deque

# Initialize environment object
params = OpenAIGym_Box2d.HYPERPARAMS['LunarLander']
env_name = params['env_name']
env = gym.make(env_name)
env.seed(0)

# Get environment parameter
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Initialize Double-DQN agent
agent = Agent(name='ddqn', state_size=state_size, action_size=action_size, param=params, seed=0)
# Load the pre-trained network
agent.Q_network.load_state_dict(torch.load('models/%s_%s.pth'% (agent.name, env_name)))

# Define parameters for test
episodes = 2                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment and Capture the current state
    state = env.reset()

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    while not done:
        # Action selection by Epsilon-Greedy policy
        action = agent.act(state)
        env.render()
        # Take action and get rewards and new state
        next_state, reward, done, _ = env.step(action)

        # State transition
        state = next_state

        # Update total score
        score += reward


    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, score))

""" End of the Test """

# Close environment
env.close()
