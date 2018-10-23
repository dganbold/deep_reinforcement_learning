import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import OpenAIGym_ClassicControl

from DoubleQLearner import Agent
from collections import deque

# Initialize environment object
params = OpenAIGym_ClassicControl.HYPERPARAMS['MountainCar']
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

# Define parameters for training
episodes = params['train_episodes']         # maximum number of training episodes
stop_scores = params['stop_scores']
scores_window_size = params['scores_window_size']

# Define parameters for e-Greedy policy
epsilon = params['epsilon_start']           # starting value of epsilon
epsilon_floor = params['epsilon_final']     # minimum value of epsilon
epsilon_decay = params['epsilon_decay']     # factor for decreasing epsilon

""" Training loop  """
scores = []                         # list containing scores from each episode
scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
for i_episode in range(1, episodes+1):
    # Reset the environment and Capture the current state
    state = env.reset()

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    while not done:
        # Action selection by Epsilon-Greedy policy
        action = agent.act(state, epsilon)

        # Take action and get rewards and new state
        next_state, reward, done, _ = env.step(action)

        # Store experience
        agent.step(state, action, reward, next_state, done)

        # State transition
        state = next_state

        # Update total score
        score += reward

    # Push to score list
    scores_window.append(score)
    scores.append([score, np.mean(scores_window)])

    # Print episode summary
    print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon), end="")
    if i_episode % 100 == 0:
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), epsilon))
    if np.mean(scores_window)>=stop_scores:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-scores_window_size, np.mean(scores_window)))
        torch.save(agent.Q_network.state_dict(), 'models/%s_%s.pth'% (agent.name, env_name))
        break

    # Update exploration
    epsilon = max(epsilon_floor, epsilon*epsilon_decay)
""" End of the Training """

# Export scores to csv file
df = pandas.DataFrame(scores,columns=["scores","average_scores"])
#df = pandas.DataFrame(data={"score": scores[0],"average_score": scores[1]})
df.to_csv('scores/%s_%s_trained_%d_episodes.csv'% (agent.name, env_name, i_episode), sep=',',index=False)

# Plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
ax.legend(['Raw scores','Average scores'])
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Close environment
env.close()
