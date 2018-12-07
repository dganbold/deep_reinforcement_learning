[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

<p align="center">
    <img src="../../assets/banana_collector_test.gif" height="300px">
</p>

## Description
In this project, implemented DDPG(Deep Deterministic Policy Gradient) algorithm based on following papers with [PyTorch](https://www.pytorch.org/) and applied to continuous control environment, where the goal is agent is to maintain its position at the target location for as many time steps as possible.

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)

## Background
Policy-based methods are well-suited for continuous action spaces but it has several drawbacks suck as evaluating policy is generally inefficient and high variance. The Actor-Critic methods reduce variance with respect to pure policy search methods. It uses function approximation to learn a policy(Actor) and a value function(Critic).

<p align="center">
    <img src="../../assets/actor_critic.png" height="300px">
</p>

The DDPG is combination of Actor-Critic method and Deep Q-Network(DQN) that maintains a parameterized actor function which specifies the current policy by deterministically mapping states to a specific action. The critic is learned using the Bellman equation as in Q-learning. 

<p align="center">
    <img src="../../assets/q_target.png" height="100px">
</p>

## Result

## Future work
