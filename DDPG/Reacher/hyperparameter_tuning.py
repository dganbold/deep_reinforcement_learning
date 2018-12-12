#!/usr/bin/env python
# coding: utf-8

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
# Hyperparameter optimizer
import optuna

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

# Save results to csv file
log_filename = 'hyperparameter_optimization'
hyperscores = []

def train_agent(actor_learning_rate, critic_learning_rate, fc_units, thau, batch_size):
    # Set tunable parameters
    params['actor_hidden_layers']  = [int(fc_units), int(fc_units/2)]
    params['critic_hidden_layers'] = [int(fc_units), int(fc_units/2)]
    params['actor_learning_rate']  = actor_learning_rate
    params['critic_learning_rate'] = critic_learning_rate
    params['thau'] = thau
    params['batch_size'] = int(batch_size)

    # Create agent instance
    print("Created agent with following hyperparameter values:")
    pprint.pprint(params)

    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0)

    # Initialize replay buffer
    memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=0)
    update_interval = params['update_interval']
    replay_start = params['replay_initial']

    """ Training loop  """
    scores = []                                                   # list containing scores from each episode
    scores_window = deque(maxlen=params['scores_window_size'])    # last (window_size) scores
    filemeta = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}_solved{:d}"
    for i_episode in range(1, params['train_episodes']+1):
        # Reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()

        # Capture the current state
        state = env_info.vector_observations[0]

        # Reset score collector
        score = 0
        # One episode loop
        step = 0
        done = False
        while not done:
            # Action selection
            action = agent.act(state)

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
                # Rechyperparameter_optimizationall experiences (miniBatch)
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
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}'.format(i_episode, score, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}'.format(i_episode, score, np.mean(scores_window)))
        if np.mean(scores_window) >= params['stop_scores']:
            print('\nEnvironment solved in {:d} episodes!\tAverageimport time Score: {:.2f}'.format(i_episode-params['scores_window_size'], np.mean(scores_window)))
            break

    """ End of the Training """
    print('\n')

    # Filename string
    filename = filemeta.format( params['env_name'],agent.name,      \
                                params['actor_learning_rate'],      \
                                params['critic_learning_rate'],     \
                                fc_units,params['thau'],            \
                                params['batch_size'], i_episode-100)
    agent.export_network('./models/{:s}'.format(filename))
    # Export scores to csv file
    df = pandas.DataFrame(scores,columns=['scores','average_scores','std'])
    df.to_csv('./scores/{:s}.csv'.format(filename), sep=',',index=False)

    hyperscores.append([params['actor_learning_rate'], params['critic_learning_rate'], fc_units, params['thau'], params['batch_size'], np.mean(scores_window), i_episode-params['scores_window_size']])
    log_df = pandas.DataFrame(hyperscores,columns=['actor_learning_rate', 'critic_learning_rate', 'fc_units', 'thau', 'batch_size', 'i_episode'])
    log_df.to_csv('scores/{:s}.csv'.format(log_filename))

    return (params['stop_scores']-np.mean(scores_window))

def objective(trial):
    # Optuna objective function
    actor_learning_rate = trial.suggest_categorical('actor_learning_rate', [1e-4, 5e-4, 1e-3])
    critic_learning_rate = trial.suggest_categorical('critic_learning_rate', [1e-4, 5e-4, 1e-3])
    fc_units = trial.suggest_categorical('fc_units', [64, 128, 256])
    thau = 1e-3     #trial.suggest_categorical('thau', [1e-3, 2e-3])
    batch_size = trial.suggest_categorical('batch_size', [128, 256])

    return train_agent(actor_learning_rate, critic_learning_rate, fc_units, thau, batch_size)

# Create a new Optuna study object.
study = optuna.create_study()
# Invoke optimization of the objective function.
study.optimize(objective , n_trials=300, n_jobs=1)
# Print and Save result to .csv file
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
# Close the environment
env.close()
