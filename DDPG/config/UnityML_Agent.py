# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'Reacher': {
        'env_name':             "Reacher",
        'stop_scores':          30.0,
        'scores_window_size':   100,
        'train_episodes':       5,

        'replay_size':          1000000,            # replay buffer size
        'replay_initial':       1000,               # replay buffer initialize
        'update_interval':      1,
        'fix_target_updates':   1,                  # fix the target Q for the fix_target_updates

        'actor_hidden_layers':  [64, 64],           # hidden units and layers of Actor-network
        'critic_hidden_layers': [64, 64, 32],       # hidden units and layers of Critic-network

        'epsilon_start':        1.0,                # starting value of epsilon
        'epsilon_final':        0.05,               # minimum value of epsilon
        'epsilon_decay':        0.999,              # factor for decreasing epsilon

        'actor_learning_rate':  1e-4,               # actor learning rate
        'critic_learning_rate': 1e-4,               # critic learning rate
        'gamma':                0.99,               # discount factor
        'thau':                 1e-3,               # for soft update of target parameters
        'batch_size':           128,                # minibatch size
        'weight_decay':         0.0001              # L2 weight decay
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
