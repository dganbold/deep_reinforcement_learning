<p align="center">
    <img src="assets/Pytorch_logo.png" height="60px">
</p>

## PyTorch implementation of Deep Reinforcement Learning Algorithms

This repository contains :
  1. Value-Based Methods : (Neural Q-Learning, DQN, Double-DQN, Memory improved DQN)
  2. Policy-Based Methods: (DDPG)

## Important notes
- The code works for PyTorch.
- The agents interact with OpenAI gym and Unity environments.

## Features
* Support CUDA.(Faster than CPU implementation)
* Support discrete and continous state space.    
* Support discrete and continous action space.

## Resources
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [Deep Reinforcement Learning UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [Udacity Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/)  

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with _Coming soon!_
- `Cartpole-v0` with [REINFORCE](https://github.com/dganbold/deep_reinforcement_learning/tree/master/REINFORCE/CCartPole) | solved in 691 episodes.
- `MountainCarContinuous-v0` with [DDPG](https://github.com/dganbold/deep_reinforcement_learning/tree/master/DDPG)
- `MountainCar-v0` with _Coming soon!_
- `Pendulum-v0` with [DDPG](https://github.com/dganbold/deep_reinforcement_learning/tree/master/DDPG)

### Box2d
- `BipedalWalker-v2` with _Coming soon!_
- `CarRacing-v0` with _Coming soon!_
- `LunarLander-v2` with [NeuralQLearner](https://github.com/dganbold/deep_reinforcement_learning/tree/master/NeuralQLearning) | solved in 314 episodes. Average Score: 200.5
### Toy Text
- `FrozenLake-v0` with _Coming soon!_
- `Blackjack-v0` with _Coming soon!_
- `CliffWalking-v0` with _Coming soon!_

## Unity-ML-Agents Benchmarks
- `BananaCollector` with [NeuralQLearner](https://github.com/dganbold/deep_reinforcement_learning/tree/master/NeuralQLearning) | solved in 345 episodes. Average Score: 13.02
- `Reacher` with [DDPG](https://github.com/dganbold/deep_reinforcement_learning/tree/master/DDPG/Reacher) | solved in 147 episodes. Average Score: 30
- `Tennis` with [MADDPG](https://github.com/dganbold/deep_reinforcement_learning/tree/master/MADDPG/Tennis) | solved in 427 episodes. Average Score: 0.5

## Linux dependencies
To set up your python environment to run the code in this repository, follow the instructions below.

1. Install [conda](https://conda.io/docs/user-guide/install/linux.html) and create a new environment with Python 3.6.
```bash
conda create --name drlenv python=3.6
```

2. To activate this environment
```bash
source activate drlenv
```

3. To install and use [OpenAI gym](https://github.com/openai/gym).
  - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
  - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

4. `[Optional]` To install and use [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

5. Clone the repository
```bash
git clone https://github.com/dganbold/deep_reinforcement_learning
```

6. Install the dependencies
```bash
pip install .
```

# References
* Udacity Deep-Reinforcement-Learning [[Github]](https://github.com/udacity/deep-reinforcement-learning)
