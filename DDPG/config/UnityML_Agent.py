# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'Reacher': {
        'env_name':             "Reacher",
        'stop_scores':          30.0,
        'scores_window_size':   100,
        'train_episodes':       1000,

        'replay_size':          1000000,            # replay buffer size
        'replay_initial':       256,                # replay buffer initialize
        'update_interval':      4,
        'fix_target_updates':   1,                  # fix the target Q for the fix_target_updates

        'actor_hidden_layers':  [256, 256],         # hidden units and layers of Actor-network
        'critic_hidden_layers': [256, 256, 128],    # hidden units and layers of Critic-network

        'epsilon_start':        1.0,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.993,              # factor for decreasing epsilon

        'actor_learning_rate':  1e-4,               # actor learning rate
        'critic_learning_rate': 3e-4,               # critic learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           128,                # minibatch size
        'weight_decay':         0.0001              # L2 weight decay
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
