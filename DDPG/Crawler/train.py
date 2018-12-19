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
params = HYPERPARAMS['Crawler']
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
agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=params['random_seed'])

# Initialize replay buffer
memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=params['random_seed'])

# Define parameters for exploration
noise_amplitude = 1.0 #params['noise_amplitude_start']
noise_amplitude_final = 0.1 #params['noise_amplitude_final']
noise_amplitude_decay = 0.999 #params['noise_amplitude_decay']

print('Hyperparameter values:')
pprint.pprint(params)

""" Training loop  """
filename_format = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}"
scores_history = []                                          # list containing scores from each episode
scores_window = deque(maxlen=params['scores_window_size'])   # last (window_size) scores
for i_episode in range(1, params['train_episodes']+1):
    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    agent.reset()
    # Capture the current state
    states = env_info.vector_observations
    dones = env_info.local_done
    # Reset score collector
    scores = np.zeros(number_of_agents)                     # initialize the score (for each agent)
    # One episode loop
    step = 0
    while not np.any(dones):
        # Action selection
        actions = agent.act(states)

        # Take action and get rewards and new state
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done                  # if next is terminal state

        # Store experience
        #memory.push(state, action, reward, next_state, done)

        # Update critic and actor policy
        #step += 1
        #if (step % params['update_interval']) == 0 and len(memory) > params['batch_size']:
        #    # Recall experiences (miniBatch)
        #    experiences = memory.recall()
        #    # Train agent
        #    agent.learn(experiences)

        # State transition
        states = next_states

        # Update total score
        scores += rewards

    # Push to score list
    scores_window.append(np.max(scores))
    scores_history.append([scores, np.max(scores), np.mean(scores_window,axis=0), np.std(scores_window,axis=0)])

    # Print episode summary
    print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}, Steps:{}'.format(i_episode, np.max(scores), np.mean(scores_window), noise_amplitude, step), end="")
    if i_episode % 100 == 0:
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}, Steps:{}'.format(i_episode, np.max(scores), np.mean(scores_window), noise_amplitude, step))
    if np.mean(scores_window) >= params['stop_scores']:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        break

    # Update exploration
    noise_amplitude = max(noise_amplitude_final, noise_amplitude*noise_amplitude_decay)
""" End of the Training """

# Filename string
filename = filename_format.format(  params['env_name'],agent.name,      \
                                    params['actor_learning_rate'],      \
                                    params['critic_learning_rate'],     \
                                    params['actor_hidden_layers'][0],   \
                                    params['thau'],params['batch_size'])
# Export trained agent's parameters
agent.export_network('./models/{:s}'.format(filename))

# Export scores to csv file
df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
df.to_csv('./scores/{:s}.csv'.format(filename), sep=',',index=False)

# Plot the scores
fig = plt.figure(num=None,figsize=(10, 5))
ax = fig.add_subplot(111)
episode = np.arange(len(scores))
plt.plot(episode,df['average_scores'])
plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
plt.title(params['env_name'])
ax.legend([agent.name + ' [ Average scores ]'])
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
fig.savefig('scores/{:s}.png'.format(filename))   # save the figure to file

# Close environment
env.close()
