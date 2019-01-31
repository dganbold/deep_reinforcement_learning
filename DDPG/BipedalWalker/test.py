import sys
sys.path.append('../')
from utils.misc import *
# Config
from config.OpenAIGym_Box2d import *
# Environment
import gym
# Agent
from agent.DDPG import Agent
from agent.ExperienceReplay import ReplayBuffer

# Initialize environment object
params = HYPERPARAMS['BipedalWalker']
env_name = params['env_name']
env = gym.make(env_name)
env.seed(params['random_seed'])

# Get environment parameter
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
print('Number of actions : ', action_size)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)
print('Dimension of state space : ', state_size)

# Initialize agent
agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=params['random_seed'])

# Filename string
filename_format = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}"
filename = filename_format.format(  params['env_name'],agent.name,      \
                                    params['actor_learning_rate'],      \
                                    params['critic_learning_rate'],     \
                                    params['actor_hidden_layers'][0],   \
                                    params['actor_thau'],params['batch_size'])

# Load the pre-trained network
agent.import_network('./models/{:s}'.format(filename))

print('Hyperparameter values:')
pprint.pprint(params)

# Define parameters for test
episodes = 1                                       # maximum number of test episodes

#from gym.wrappers import Monitor
#env = Monitor(env, './video')
#state = env.reset()
#env.render()
#input('Press enter to start:')

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
        action = agent.act(state,noise_amplitude=0.0)
        env.render()

        # Take action and get rewards and new state
        next_state, reward, done, _ = env.step(2*action)

        # State transition
        state = next_state

        # Update total score
        score += reward

    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, score))

""" End of the Test """

# Close environment
env.close()
