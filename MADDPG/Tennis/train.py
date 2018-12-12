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
env = UnityEnvironment(file_name='{:s}_Linux/{:s}.x86'.format(params['env_name'],params['env_name']))

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

# Initialize replay buffer
memory = ReplayBuffer(params['replay_size'], params['batch_size'], seed=params['random_seed'])
update_interval = params['update_interval']
replay_start = params['replay_initial']

# Define parameters for training
episodes = params['train_episodes']         # maximum number of training episodes
stop_scores = params['stop_scores']
scores_window_size = params['scores_window_size']

# Define parameters for exploration
noise_amplitude = params['noise_amplitude_start']
noise_amplitude_final = params['noise_amplitude_final']
noise_amplitude_decay = params['noise_amplitude_decay']

print('Hyperparameter values:')
pprint.pprint(params)

#log_path = os.getcwd()+"/log"
#model_dir= os.getcwd()+"/model_dir"
#os.makedirs(model_dir, exist_ok=True)
#logger = SummaryWriter(log_dir=log_path)

""" Training loop  """
scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
for i_episode in range(1, episodes+1):
    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Capture the current state
    states = env_info.vector_observations
    dones = env_info.local_done
    # Reset score collector
    scores = np.zeros(number_of_agents)
    # One episode loop
    step = 0
    while not np.any(dones):
        # Get actions from all agents
        actions = agents.act(states, noise_amplitude=noise_amplitude)
        noise_amplitude *= noise_amplitude_decay

        # Take action and get rewards and new state
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step

        # Store experience
        transitions = (states, actions, rewards, next_states, dones)
        memory.push(transitions)

        # Update the Critics and Actors of all the agents
        step += 1
        if (step % update_interval) == 0 and len(memory) > replay_start:
            # Recall experiences (miniBatch)
            experiences = memory.recall()

            # Train agent
            agents.learn(experiences)

        # State transition
        states = next_states

    # Push to score list
    #scores_window.append(score)
    #scores.append([score, np.mean(scores_window), np.std(scores_window)])
    print('Score (max over agents) from episode {}: {}'.format(i_episode, np.max(scores)))

    # Print episode summary
    #print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
    #if i_episode % 100 == 0:
    #    print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
    #if np.mean(scores_window)>=13.0:
    #    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
    #    agent.export_network('models/%s_%s'% (agent.name,env_name))
    #    break

    # Update exploration
    #epsilon = max(epsilon_floor, epsilon*epsilon_decay)
""" End of the Training """

# Export scores to csv file
#df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
#df.to_csv('scores/%s_%s_batch_%d_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],i_episode), sep=',',index=False)

# Plot the scores
#fig = plt.figure(num=None,figsize=(10, 5))
#ax = fig.add_subplot(111)
#episode = np.arange(len(scores))
#plt.plot(episode,df['average_scores'])
#plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
#plt.title(env_name)
#ax.legend([agent.name + ' [ Average scores ]'])
#plt.ylabel('Score')
#plt.xlabel('Episode')
#plt.show()
#fig.savefig('scores/%s_%s_batch_%d_trained_%d_episodes.png'% (agent.name,env_name,params['batch_size'],i_episode))   # save the figure to file

# Close environment
env.close()
