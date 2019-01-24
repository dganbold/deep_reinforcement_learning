# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'Pong': {
        'env_name':             "PongDeterministic-v4",
        'stop_scores':          90,
        'scores_window_size':   100,
        'train_episodes':       500,

        'hidden_layers':        [128, 64],          # hidden units and layers of Policy-network

        'learning_rate':        1e-4,               # actor learning rate
        'gamma':                0.99,               # discount factor
        'betta':                0.01,               #
        'batch_size':           64,                 # minibatch size
        'weight_decay':         0.0000,             # L2 weight decay
        'random_seed':          0
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
