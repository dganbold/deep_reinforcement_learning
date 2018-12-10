[//]: # (Image References)

# Project 2: Continuous Control

<p align="center">
    <img src="https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif" height="250px">
</p>

## Description
In this project, implemented DDPG(Deep Deterministic Policy Gradient) algorithm based on following papers with [PyTorch](https://www.pytorch.org/) and applied to continuous control environment, where the goal is agent is to maintain its position at the target location for as many time steps as possible.

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)

## Background
Policy-based methods are well-suited for continuous action spaces but it has several drawbacks suck as evaluating policy is generally inefficient and high variance. The Actor-Critic methods reduce variance with respect to pure policy search methods. It uses function approximation to learn a policy(Actor) and a value function(Critic).

<p align="center">
    <img src="../../assets/actor_critic.png" height="220px">
</p>

## DDPG algorithm
The [DDPG](https://arxiv.org/abs/1509.02971) is off-policy Actor-Critic approach which combination of Policy learning method and Deep Q-Network(DQN). It maintains a parameterized actor function which specifies the current policy by deterministically mapping states to a specific action. The critic is learned using the Bellman equation as in Q-learning which evaluates the policy.

<p align="center">
    <img src="../../assets/ddpg.png" height="200px">
</p>

Some other interesting aspects of the DDPG are shown below.

<p align="center">
    <img src="../../assets/ddpg_algorithm.png" height="480px">
</p>

## Implementation
The baseline code from [Deep Reinforcement Learning nanodegree course's GitHub](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) which intended for solving OpenAI gym's BipedalWalker-v2 problem.

In this project, Agent is modified to interact with Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment and hyperparameters are tuned.

## Hyperparameter tuning
Bayesian Optimization based software framework [Optuna](https://optuna.org/) is used it as hyperparameter tuning.

## Result


## Future work
- Distributed Distributional Deterministic Policy Gradients [[arxiv]](https://arxiv.org/abs/1804.08617)
