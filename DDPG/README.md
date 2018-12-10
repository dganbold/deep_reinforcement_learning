# Deep Deterministic Policy Gradient(DDPG)

## Description
In this project, implemented Deep Deterministic Policy Gradient(DDPG) algorithm based on following papers with [PyTorch](https://www.pytorch.org/) and applied to continuous control environment, where the goal is agent is to maintain its position at the target location for as many time steps as possible.

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)

## Background
Policy-based methods are well-suited for continuous action spaces but it has several drawbacks suck as evaluating policy is generally inefficient and high variance. The Actor-Critic methods reduce variance with respect to pure policy search methods. It uses function approximation to learn a policy(Actor) and a value function(Critic).

<p align="center">
    <img src="../assets/actor_critic.png" height="220px">
</p>

## DDPG algorithm
The [DDPG](https://arxiv.org/abs/1509.02971) is off-policy Actor-Critic approach which combination of Policy learning method and Deep Q-Network(DQN). It maintains a parameterized actor function which specifies the current policy by deterministically mapping states to a specific action. The critic is learned using the Bellman equation as in Q-learning which evaluates the policy.

<p align="center">
    <img src="../assets/ddpg.png" height="200px">
</p>

## Result

## Dependencies
- [Conda](https://conda.io/docs/user-guide/install/index.html)
- Python 3.6
- [PyTorch 0.4.0](http://pytorch.org/)
- [NumPy >= 1.11.0](http://www.numpy.org/)
- [OpenAI Gym](https://github.com/openai/gym)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://www.scipy.org/)

If you want to run the code in this repository, check this [instructions](https://github.com/dganbold/deep_reinforcement_learning).

## Supported environments

### OpenAI Gym

#### Classic Control
- `Pendulum-v0` with _Coming soon!_

#### Box2d
- `BipedalWalker-v2` with _Coming soon!_

### Unity
- [`Reacher`](https://github.com/dganbold/deep_reinforcement_learning/tree/master/DDPG/Reacher) with DDPG | solved in X episodes

## Usage

- Execute the following command to train the agent:

```
$ cd [Environment]
$ python train.py
```

- Execute the following command to test the pre-trained agent:

```
$ python test.py
```

## Future work
- Implement a D4PG(Distributed Distributional Deterministic Policy Gradients) [[arxiv]](https://arxiv.org/abs/1804.08617)
