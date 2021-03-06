import sys
sys.path.append('../')
# Config
from utils.misc import *
from config.UnityML_Agent import *
# Environment
from unityagents import UnityEnvironment
# Agent
from agent.DDPG import Agent
from agent.ExperienceReplay import ReplayBuffer

# Initialize environment object
params = HYPERPARAMS['Reacher']
env = UnityEnvironment(file_name='{:s}_Linux/{:s}.x86_64'.format(params['env_name'],params['env_name']))

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

# Initialize agent
agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0)

# Filename string
filename_format = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}"
filename = filename_format.format(  params['env_name'],agent.name,      \
                                    params['actor_learning_rate'],      \
                                    params['critic_learning_rate'],     \
                                    params['actor_hidden_layers'][0],   \
                                    params['thau'],params['batch_size'])

# Load the pre-trained network
agent.import_network('./models/{:s}'.format(filename))

# Define parameters for test
episodes = 2                                       # maximum number of test episodes

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
        # Action selection
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
