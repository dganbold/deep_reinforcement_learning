import sys
sys.path.append('../')
from utils.misc import *
# Config
from config.UnityML_Agent import *
# Environment
from unityagents import UnityEnvironment
# Agent
#from agent.DoubleQLearner import Agent
from agent.NeuralQLearner import Agent
from agent.ExperienceReplay import ReplayBuffer

# Initialize environment object
params = HYPERPARAMS['Banana']
env_name = params['env_name']
env = UnityEnvironment(file_name = "Banana_Linux/Banana.x86_64")

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

# Initialize replay buffer
memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0)
update_interval = params['update_interval']
replay_start = params['replay_initial']

# Define parameters for training
episodes = params['train_episodes']         # maximum number of training episodes
stop_scores = params['stop_scores']
scores_window_size = params['scores_window_size']

# Define parameters for e-Greedy policy
epsilon = params['epsilon_start']           # starting value of epsilon
epsilon_floor = params['epsilon_final']     # minimum value of epsilon
epsilon_decay = params['epsilon_decay']     # factor for decreasing epsilon

""" Training loop  """
scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
for i_episode in range(1, episodes+1):
    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Capture the current state
    state = env_info.vector_observations[0]

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    step = 0
    while not done:
        # Action selection by Epsilon-Greedy policy
        action = agent.eGreedy(state, epsilon)

        # Take action and get rewards and new state
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]                  # if next is terminal state

        # Store experience
        memory.push(state, action, reward, next_state, done)

        # Update Q-Learning
        step += 1
        if (step % update_interval) == 0 and len(memory) > replay_start:
            # Recall experiences (miniBatch)
            experiences = memory.recall()
            # Train agent
            agent.learn(experiences)

        # State transition
        state = next_state

        # Update total score
        score += reward

    # Push to score list
    scores_window.append(score)
    scores.append([score, np.mean(scores_window), np.std(scores_window)])

    # Print episode summary
    print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
    if i_episode % 100 == 0:
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
    if np.mean(scores_window)>=13.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        agent.export_network('models/%s_%s'% (agent.name,env_name))
        break

    # Update exploration
    epsilon = max(epsilon_floor, epsilon*epsilon_decay)
""" End of the Training """

# Export scores to csv file
df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
df.to_csv('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.csv'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode), sep=',',index=False)

# Plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
episode = np.arange(len(scores))
plt.plot(episode,df['average_scores'])
plt.fill_between(episode,df['average_scores'].add(df['std']),df['average_scores'].sub(df['std']),alpha=0.3)
plt.title(env_name)
ax.legend([agent.name + ' [ Average scores ]'])
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
fig.savefig('scores/%s_%s_batch_%d_lr_%.E_trained_%d_episodes.png'% (agent.name,env_name,params['batch_size'],params['learning_rate'],i_episode))   # save the figure to file

# Close environment
env.close()
