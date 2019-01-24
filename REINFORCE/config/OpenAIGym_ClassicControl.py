# -------------------------------------------------------------------- #
# Hyperparameters
# -------------------------------------------------------------------- #
HYPERPARAMS = {
    'CartPole': {
        'env_name':             "CartPole-v0",
        'stop_scores':          195.0,
        'scores_window_size':   100,
        'train_episodes':       1000,

        'hidden_layers':        [16, 16],           # hidden units and layers of Actor-network
        
        'learning_rate':        1e-2,               # actor learning rate
        'gamma':                1.00,               # discount factor
        'batch_size':           64,                 # minibatch size
        'weight_decay':         0.0000,             # L2 weight decay
        'random_seed':          0
    },
}
# -------------------------------------------------------------------- #
# EOF
# -------------------------------------------------------------------- #
