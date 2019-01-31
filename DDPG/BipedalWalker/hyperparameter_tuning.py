#!/usr/bin/env python
# coding: utf-8

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
# Hyperparameter optimizer
import optuna
# Initialize environment object
params = HYPERPARAMS['BipedalWalker']
env_name = params['env_name']
env = gym.make(env_name)
env.seed(params['random_seed'])

# Get environment parameter
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Save results to csv file
log_filename = 'hyperparameter_optimization'
optuna_log = []

def train_agent(trail_id):
    # Create agent instance
    print("Created agent with following hyperparameter values:")
    pprint.pprint(params)

    # Initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=params['random_seed'])

    # Initialize replay buffer
    memory = ReplayBuffer(action_size, params['replay_size'], params['batch_size'], seed=params['random_seed'])

    # Define parameters for exploration
    noise_amplitude = params['noise_amplitude_start']
    noise_amplitude_final = params['noise_amplitude_final']
    noise_amplitude_decay = params['noise_amplitude_decay']

    """ Training loop  """
    max_step = 500
    max_score = -np.Inf
    filename_format = "{:d}"
    scores_history = []                                          # list containing scores from each episode
    scores_window = deque(maxlen=params['scores_window_size'])   # last (window_size) scores
    for i_episode in range(1, params['train_episodes']+1):
        # Reset the environment
        state = env.reset()
        agent.reset()
        # Reset score collector
        score = 0
        # One episode loop
        step = 0
        done = False
        while not np.any(done):
            # Get actions from all agent
            action = agent.act(state, noise_amplitude=noise_amplitude)

            # Take action and get rewards and new state
            next_state, reward, done, _ = env.step(action)

            # Store experience
            memory.push(state, action, reward, next_state, done)

            # Update the Critics and Actors of all the agents
            step += 1
            if (step % params['update_interval']) == 0 and len(memory) > params['replay_initial']:
                # Recall experiences (miniBatch)
                experiences = memory.recall()
                # Train agent
                agent.learn(experiences)

            # State transition
            state = next_state

            # Update total score
            score += reward

            if max_step < step:
                break

        # Push to score list
        scores_window.append(score)
        scores_history.append([score, np.mean(scores_window), np.std(scores_window)])

        # Print episode summary
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), noise_amplitude), end="")
        if i_episode % 100 == 0:
            print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, score, np.mean(scores_window), noise_amplitude))
        if np.mean(scores_window) >= params['stop_scores']:
            max_score = np.mean(scores_window)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
        elif max_score < np.mean(scores_window):
            max_score = np.mean(scores_window)

        # Update exploration
        noise_amplitude = max(noise_amplitude_final, noise_amplitude*noise_amplitude_decay)
    """ End of the Training """
    print('\n')

    # Filename string
    filename = "{:05d}".format(trail_id)
    # Export trained agent's parameters
    #agents.export_network('./models/{:s}'.format(filename))
    # Export scores to csv file
    df = pandas.DataFrame(scores_history,columns=['scores','average_scores','std'])
    df.to_csv('./scores/optuna_logs/{:s}.csv'.format(filename), sep=',',index=False)
    #
    param_metas = [key for key in params.keys()]
    param_metas.extend(['scores', 'trained_episodes', 'filename'])
    param_values = [value for value in params.values()]
    param_values.extend([np.mean(scores_window), i_episode, filename])
    #
    optuna_log.append(param_values)
    optuna_df = pandas.DataFrame(optuna_log,columns=param_metas)
    optuna_df.to_csv('scores/{:s}.csv'.format(log_filename))
    #
    return (params['stop_scores'] - max_score)

def objective(trial):
    # Set tunable parameters
    fc_units = trial.suggest_categorical('fc_units', [64, 128, 256, 512])
    params['actor_hidden_layers']  = [int(fc_units), int(fc_units/2)]
    params['critic_hidden_layers'] = [int(fc_units), int(fc_units/2)]

    actor_learning_rate = trial.suggest_categorical('actor_learning_rate', [1e-4, 5e-4, 1e-3])
    critic_learning_rate = trial.suggest_categorical('critic_learning_rate', [1e-4, 5e-4, 1e-3])
    params['actor_learning_rate']  = actor_learning_rate
    params['critic_learning_rate'] = critic_learning_rate

    thau = trial.suggest_categorical('thau', [5e-2, 1e-3, 5e-3])
    params['actor_thau'] = thau
    params['critic_thau'] = thau

    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    params['batch_size'] = batch_size

    update_interval = 1 #trial.suggest_int('update_interval', 1, 2)
    params['update_interval'] = update_interval

    noise_amplitude_start = 1.0 #trial.suggest_discrete_uniform('noise_amplitude_start', 0.1, 5.0, 1.0)
    params['noise_amplitude_start'] = noise_amplitude_start

    noise_amplitude_decay = 1#0.9999 #trial.suggest_loguniform('noise_amplitude_decay', 0.99, 0.99999)
    params['noise_amplitude_decay'] = noise_amplitude_decay

    #replay_size = int(1000000) #"trial.suggest_categorical('batch_size', [100000, 1000000])
    #params['replay_size'] = replay_size

    # Optuna objective function
    return train_agent(trial.trial_id)

# Create a new Optuna study object.
study = optuna.create_study()
# Invoke optimization of the objective function.
study.optimize(objective , n_trials=500, n_jobs=1)
# Print and Save result to .csv file
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
# Close the environment
env.close()
