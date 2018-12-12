# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'Tennis': {
        # Global parameters
        'random_seed':          0,                  # Random seed
        'update_interval':      1,                  # Freeze the Agent for the update_interval

        # Exploration parameters
        'noise_amplitude_start': 2,                 # starting value of noise_amplitude
        'noise_amplitude_final': 0.05,              # minimum value of noise_amplitude
        'noise_amplitude_decay': 0.993,             # factor for decreasing noise_amplitude

        # Environment parameters
        'env_name':             "Tennis",
        'stop_scores':          30.0,
        'scores_window_size':   100,
        'train_episodes':       400,

        # Replay buffer parameters
        'replay_size':          100000,             # replay buffer size
        'replay_initial':       1000,               # replay buffer initialize
        'batch_size':           128,                # minibatch size

        # Actor parameters
        'actor_input_size':     2,
        'actor_output_size':    2,
        'actor_hidden_layers':  [256, 128],         # hidden units and layers of Actor-network
        'actor_learning_rate':  1e-4,               # actor learning rate
        'actor_weight_decay':   0.0001,             # L2 weight decay
        'actor_thau':           1e-3,               # for soft update of target parameters

        # Critic parameters
        'critic_input_size':    2,
        'critic_output_size':   1,
        'critic_hidden_layers': [256, 128],         # hidden units and layers of Critic-network
        'critic_learning_rate': 1e-4,               # actor learning rate
        'critic_weight_decay':  0.0001,             # L2 weight decay
        'critic_thau':          1e-3,               # for soft update of target parameters

        'gamma':                0.99,               # discount factor
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
