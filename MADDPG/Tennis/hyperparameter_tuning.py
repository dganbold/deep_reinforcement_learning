#!/usr/bin/env python
# coding: utf-8

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
# Hyperparameter optimizer
import optuna

# Initialize environment object
params = HYPERPARAMS['Tennis']
env = UnityEnvironment(file_name='{:s}_Linux/{:s}.x86'.format(params['env_name'],params['env_name']),no_graphics=True)

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

# Save results to csv file
log_filename = 'hyperparameter_optimization'
hyperscores = []

def train_agent():
    # Create agent instance
    print("Created agent with following hyperparameter values:")
    pprint.pprint(params)

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

    """ Training loop  """
    filename_format = "{:s}_{:s}_{:.1E}_{:.1E}_{:d}_{:.1E}_{:d}"
    scores_history = []                                # list containing scores from each episode
    scores_window = deque(maxlen=scores_window_size)   # last (window_size) scores
    for i_episode in range(1, episodes+1):
        # Reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        agents.reset()
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

            # Take action and get rewards and new state
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)

            # Store experience
            memory.push(states, actions, rewards, next_states, dones)

            # Update the Critics and Actors of all the agents
            step += 1
            if (step % update_interval) == 0 and len(memory) > replay_start:
                for agent_id in range(number_of_agents):
                    # Recall experiences (miniBatch)
                    experiences = memory.recall()
                    # Train agent
                    agents.learn(experiences,agent_id)

            # State transition
            states = next_states

        # Push to score list
        scores_window.append(np.max(scores))
        scores_history.append([np.max(scores), np.mean(scores_window,axis=0), np.std(scores_window,axis=0)])

        # Print episode summary
        print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, np.max(scores), np.mean(scores_window), noise_amplitude), end="")
        if i_episode % 100 == 0:
            print('\r#TRAIN Episode:{}, Score:{:.2f}, Average Score:{:.2f}, Exploration:{:1.4f}'.format(i_episode, np.max(scores), np.mean(scores_window), noise_amplitude))
        if np.mean(scores_window) >= params['stop_scores']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break

        # Update exploration
        noise_amplitude = max(noise_amplitude_final, noise_amplitude*noise_amplitude_decay)
    """ End of the Training """
    print('\n')

    # Filename string
    filename = filename_format.format(  params['env_name'],'MADDPG',        \
                                        params['actor_learning_rate'],      \
                                        params['critic_learning_rate'],     \
                                        params['actor_hidden_layers'][0],   \
                                        params['actor_thau'],params['batch_size'],i_episode-100)

    # Export trained agent's parameters
    agents.export_network('./models/{:s}'.format(filename))
    # Export scores to csv file
    df = pandas.DataFrame(scores_history,columns=['scores','average_scores','std'])
    df.to_csv('./scores/{:s}.csv'.format(filename), sep=',',index=False)

    hyperscores.append([[value for param, value in params.items()], np.mean(scores_window), i_episode])
    log_df = pandas.DataFrame(hyperscores,columns=[[param for param, value in params.items()], 'scores', 'trained_episodes'])

    log_df.to_csv('scores/{:s}.csv'.format(log_filename))

    return (params['stop_scores']-np.mean(scores_window))

def objective(trial):
    # Set tunable parameters
    fc_units = trial.suggest_categorical('fc_units', [64, 128, 256, 512])
    params['actor_hidden_layers']  = [int(fc_units), int(fc_units/2)]
    params['critic_hidden_layers'] = [int(fc_units), int(fc_units/2)]

    actor_learning_rate = trial.suggest_categorical('actor_learning_rate', [1e-4, 5e-4, 1e-3, 2e-3])
    critic_learning_rate = trial.suggest_categorical('critic_learning_rate', [1e-4, 5e-4, 1e-3, 2e-3])
    params['actor_learning_rate']  = actor_learning_rate
    params['critic_learning_rate'] = critic_learning_rate

    thau = trial.suggest_discrete_uniform('thau', 1e-3, 1e-1, 1e-3)
    params['actor_thau'] = thau
    params['critic_thau'] = thau

    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    params['batch_size'] = batch_size

    update_interval = trial.suggest_int('update_interval', 1, 4)
    params['update_interval'] = update_interval

    noise_amplitude_start = trial.suggest_discrete_uniform('noise_amplitude_start', 0.1, 5.0, 1.0)
    params['noise_amplitude_start'] = noise_amplitude_start

    noise_amplitude_decay = trial.suggest_loguniform('noise_amplitude_decay', 0.99, 0.99999)
    params['noise_amplitude_decay'] = noise_amplitude_decay

    #replay_size = int(1000000) #"trial.suggest_categorical('batch_size', [100000, 1000000])
    #params['replay_size'] = replay_size

    # Optuna objective function
    return train_agent()

# Create a new Optuna study object.
study = optuna.create_study()
# Invoke optimization of the objective function.
study.optimize(objective , n_trials=2000, n_jobs=1)
# Print and Save result to .csv file
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
# Close the environment
env.close()
