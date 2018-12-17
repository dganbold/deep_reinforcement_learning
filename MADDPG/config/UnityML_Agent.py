# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'Tennis': {
        # Global parameters
        'random_seed':          0,                  # Random seed
        'update_interval':      2,                  # Freeze the Agent for the update_interval

        # Exploration parameters
        'noise_amplitude_start': 5.0,               # starting value of noise_amplitude
        'noise_amplitude_final': 0.1,               # minimum value of noise_amplitude
        'noise_amplitude_decay': 0.994,             # factor for decreasing noise_amplitude

        # Environment parameters
        'env_name':             "Tennis",
        'stop_scores':          0.5,
        'scores_window_size':   100,
        'train_episodes':       1000,

        # Replay buffer parameters
        'replay_size':          1000000,            # replay buffer size
        'replay_initial':       1000,               # replay buffer initialize
        'batch_size':           256,                # minibatch size

        # Actor parameters
        'actor_state_size':     0,
        'actor_action_size':    0,
        'actor_hidden_layers':  [256, 128],         # hidden units and layers of Actor-network
        'actor_learning_rate':  1e-3,               # actor learning rate
        'actor_weight_decay':   0.0000,             # L2 weight decay
        'actor_thau':           5e-2,               # for soft update of target parameters

        # Critic parameters
        'critic_state_size':    0,
        'critic_action_size':   0,
        'critic_hidden_layers': [256, 128],         # hidden units and layers of Critic-network
        'critic_learning_rate': 1e-3,               # actor learning rate
        'critic_weight_decay':  0.0000,             # L2 weight decay
        'critic_thau':          5e-2,               # for soft update of target parameters

        'gamma':                0.99,               # discount factor
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
