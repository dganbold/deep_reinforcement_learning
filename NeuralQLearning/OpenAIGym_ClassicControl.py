HYPERPARAMS = {
    'Acrobot': {
        'env_name':             "Acrobot-v1",
        'stop_scores':          200.0,
        'scores_window_size':   100,
        'train_episodes':       1800,

        'replay_size':          100000,             # replay buffer size
        'replay_initial':       10000,              # replay buffer initialize
        'update_interval':      4,                  # network updating every update_interval steps

        'hidden_layers':        [64, 64],          # hidden units and layers of Q-network

        'epsilon_start':        1.0,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.995,              # factor for decreasing epsilon

        'learning_rate':        5e-4,               # learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           64                  # minibatch size
    },
    'CartPole': {
        'env_name':             "CartPole-v0",
        'stop_scores':          195.0,
        'scores_window_size':   100,
        'train_episodes':       500,

        'replay_size':          10000,              # replay buffer size
        'replay_initial':       100,                # replay buffer initialize
        'update_interval':      4,                  # network updating every update_interval steps

        'hidden_layers':        [64, 64],           # hidden units and layers of Q-network

        'epsilon_start':        0.5,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.997,              # factor for decreasing epsilon

        'learning_rate':        5e-4,               # learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           64                  # minibatch size
    },
    'MountainCar': {
        'env_name':             "MountainCar-v0",
        'stop_scores':          -110.0,
        'scores_window_size':   100,
        'train_episodes':       1800,

        'replay_size':          10000,              # replay buffer size
        'replay_initial':       1000,               # replay buffer initialize
        'update_interval':      4,                  # network updating every update_interval steps

        'hidden_layers':        [64, 64],           # hidden units and layers of Q-network

        'epsilon_start':        0.5,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.995,              # factor for decreasing epsilon

        'learning_rate':        1e-3,               # learning rate
        'gamma':                0.98,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           64                  # minibatch size
    },
}
