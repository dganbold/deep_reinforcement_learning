## PyTorch implementation of Deep Reinforcement Learning Algorithms

This repository contains : 
  1. Value-Based Methods : (Neural Q-Learning, DQN, Double-DQN, Memory improved DQN)
  2. Policy-Based Methods:

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
- `Cartpole-v0` with _Coming soon!_
- `Cartpole-v0` with _Coming soon!_
- `MountainCarContinuous-v0` with _Coming soon!_
- `MountainCar-v0` with _Coming soon!_
- `Pendulum-v0` with _Coming soon!_

### Box2d
- `BipedalWalker-v2` with _Coming soon!_
- `CarRacing-v0` with _Coming soon!_
- `LunarLander-v2` with _Coming soon!_
### Toy Text
- `FrozenLake-v0` with _Coming soon!_
- `Blackjack-v0` with _Coming soon!_
- `CliffWalking-v0` with _Coming soon!_

## Unity-ML-Agents Benchmarks
- `BananaCollector` with [Neural Q-Learning](https://github.com/dganbold/deep_reinforcement_learning/tree/master/NeuralQLearning/BananaCollector) | solved in 326 episodes. Average Score: 13.02


## Linux dependencies
To set up your python environment to run the code in this repository, follow the instructions below.

1. Create a new environment with Python 3.6.
```bash
conda create --name drlenv python=3.6
```

2. Activate 
```bash
conda activate drlenv
```

3. To setup RL environment
- `[Optional]` To install and use [OpenAI gym](https://github.com/openai/gym).
  - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
  - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

- `[Optional]` To install and use [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

4. Clone the repository 
```bash
git clone https://github.com/dganbold/deep_reinforcement_learning
```

5. Install the dependencies
```bash
pip install .
```

# References
  [Udacity Deep-Reinforcement-Learning](https://github.com/udacity/deep-reinforcement-learning)
