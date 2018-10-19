import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas

from unityagents import UnityEnvironment
from DoubleQLearner import Agent
from collections import deque

# Initialize environment object
import os
env_name = 'banana_collector'
env = UnityEnvironment(file_name=os.environ['HOME']+"/ML/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Get environment parameter
number_of_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])
print('Number of agents  : ', number_of_agents)
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Initialize Double-DQN agent
agent = Agent(name='ddqn', state_size=state_size, action_size=action_size, seed=0)
# Load the pre-trained network
agent.Q_network.load_state_dict(torch.load('models/%s_%s.pth'% (agent.name, env_name)))

# Define parameters for test
episodes = 2                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # Capture the current state
    state = env_info.vector_observations[0]

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    while not done:
        # Action selection by Epsilon-Greedy policy
        action = agent.act(state)

        # Take action and get rewards and new state
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]              # if next is terminal state

        # State transition
        state = next_state

        # Update total score
        score += reward

    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, score))

""" End of the Test """

# Close environment
env.close()
