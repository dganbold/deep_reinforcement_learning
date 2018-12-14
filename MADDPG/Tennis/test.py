import sys
sys.path.append('../')
from utils.misc import *
#from utilities import transpose_list
# Config
from config.UnityML_Agent import *
# Environment
from unityagents import UnityEnvironment
# Agent
from agent.MADDPG import MultiAgent
from agent.ExperienceReplay import ReplayBuffer

# Initialize environment object
params = HYPERPARAMS['Tennis']
env = UnityEnvironment(file_name='{:s}_Linux/{:s}.x86'.format(params['env_name'],params['env_name']),no_graphics=False)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Get environment parameter
number_of_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]
print('Number of agents  : ', number_of_agents)
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Initialize agent
agents = MultiAgent(number_of_agents=number_of_agents, state_size=state_size, action_size=action_size, param=params, seed=params['random_seed'])

# Filename string
filename_format = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}"
filename = filename_format.format(  params['env_name'],'MADDPG',        \
                                    params['actor_learning_rate'],      \
                                    params['critic_learning_rate'],     \
                                    params['actor_hidden_layers'][0],   \
                                    params['actor_thau'],params['batch_size'])

# Load the pre-trained network
agents.import_network('./models/{:s}'.format(filename))

# Define parameters for test
episodes = 10                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    # Capture the current state
    states = env_info.vector_observations
    dones = env_info.local_done
    # Reset score collector
    scores = np.zeros(number_of_agents)
    # One episode loop
    while not np.any(dones):
        # Get actions from all agents
        actions = agents.act(states)

        # Take action and get rewards and new state
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)

        # State transition
        states = next_states

    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, np.max(scores)))

""" End of the Test """

# Close environment
env.close()
