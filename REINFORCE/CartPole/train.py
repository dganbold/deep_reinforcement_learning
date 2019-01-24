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

print('Hyperparameter values:')
pprint.pprint(params)

""" Training loop  """
filename_format = "{:s}_{:s}_{:.1E}_{:d}_{:.1E}_{:d}"
# Filename string

filename = filename_format.format(  params['env_name'],agent.name,      \
                                    params['learning_rate'],            \
                                    params['hidden_layers'][0],         \
                                    params['weight_decay'],params['batch_size'])
max_step = 1000
#max_score = -np.Inf
scores = []                                                  # list containing scores from each episode
scores_window = deque(maxlen=params['scores_window_size'])   # last (window_size) scores
for i_episode in range(1, params['train_episodes']+1):
    # Reset the environment
    state = env.reset()

    # Reset score collector
    score = 0

    # Collecting trajectory
    saved_log_probs = []
    rewards = []
    for t in range(max_step):
        action, log_prob = agent.act(state)
        saved_log_probs.append(log_prob)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)

        # Update total score
        score += reward

        if done:
            break 
    
    # Push to score list
    scores_window.append(score)
    scores.append([score, np.mean(scores_window), np.std(scores_window)])
    
    # 
    # Train agent
    experiences = (saved_log_probs, rewards)
    agent.learn(experiences)

    # Print episode summary
    print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}'.format(i_episode, score, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}'.format(i_episode, score, np.mean(scores_window)))
    if np.mean(scores_window) >= params['stop_scores']:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-params['scores_window_size'], np.mean(scores_window)))
        # Export trained agent's parameters
        agent.export_network('./models/{:s}'.format(filename))
        break

""" End of the Training """
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
