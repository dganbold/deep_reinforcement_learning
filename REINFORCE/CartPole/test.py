import sys
sys.path.append('../')
from utils.misc import *
# Config
from config.OpenAIGym_ClassicControl import *
# Environment
import gym
# Agent
from agent.REINFORCE import Agent

# Initialize environment object
params = HYPERPARAMS['CartPole']
env_name = params['env_name']
env = gym.make(env_name)
env.seed(params['random_seed'])

# Get environment parameter
action_size = 2
state_size = env.observation_space.shape[0]

print('Action space:', env.action_space)
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)
print('  - low:', env.observation_space.low)
print('  - high:', env.observation_space.high)

# Initialize agent
agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=params['random_seed'])

# Filename string
filename_format = "{:s}_{:s}_{:.1E}_{:d}_{:.1E}_{:d}"
# Filename string

filename = filename_format.format(  params['env_name'],agent.name,      \
                                    params['learning_rate'],            \
                                    params['hidden_layers'][0],         \
                                    params['weight_decay'],params['batch_size'])

# Load the pre-trained network
agent.import_network('./models/{:s}'.format(filename))

# Define parameters for test
episodes = 5                                       # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment
    state = env.reset()

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    while not done:
        # Action selection
        action, log_prob = agent.act(state)
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
