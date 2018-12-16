# Project 3: Collaboration and Competition

<p align="center">
    <img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif" height="250px">
</p>

## Description
In this project, implemented Deep Deterministic Policy Gradient (DDPG) algorithm based on following papers with [PyTorch](https://www.pytorch.org/) and applied to continuous control environment, where the goal is agent is to maintain its position at the target location for as many time steps as possible.

- Continuous control with deep reinforcement learning [[arxiv]](https://arxiv.org/abs/1509.02971)
- Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments [[arxiv]](https://arxiv.org/abs/1706.02275)
- Multi-Agent Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1807.09427)

## Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Background
The formal model of single-agent RL is the Markov decision process (MDP).
The agent selects actions and the environment responds by giving a reward and new state. A canonical view of the this interaction between agent and environment is shown below.

<p align="center">
    <img src="../../../assets/agent_environment_interaction.png" height="170px">
</p>
<p align="center">
    <em>The agent-environment interaction in reinforcement learning. (Source: Sutton and Barto, 2017)</em>
</p>

The multi-agent extention of MDPs called partially observable Markov games
[[Markov games as a framework for multi-agent reinforcement learning]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjYwpTgnaLfAhXNc94KHUVGA5YQFjAAegQIBhAC&url=https%3A%2F%2Fwww2.cs.duke.edu%2Fcourses%2Fspring07%2Fcps296.3%2Flittman94markov.pdf&usg=AOvVaw3Z8842P_QFvL9BePhnSKUY) by Littman, Michael L. ICML, 1994.<br />
A Markov game for N agents defined by a set of states describing the possible configurations of all agents, a set of action and a set of observations for each agent. In the multi-agent case, the state transitions and rewards are the result of the joint action of all the agents.

<p align="center">
    <img src="../../../assets/markov_game.png" height="300px">
</p>
<p align="center">
    <em>The multi-agent environment interaction.</em>
</p>

## Methods
### Single-Agent Actor Critic
The [DDPG](https://arxiv.org/abs/1509.02971) is off-policy Actor-Critic approach which combination of Policy learning method and Deep Q-Network(DQN). It maintains a parameterized actor function which specifies the current policy by deterministically mapping states to a specific action. The critic is learned using the Bellman equation as in Q-learning which evaluates the policy.

<p align="center">
    <img src="../../../assets/ddpg.png" height="200px">
</p>
<p align="center">
    <em>Overview of single-agent DDPG.</em>
</p>

### Multi-Agent Actor Critic

<p align="center">
    <img src="../../../assets/maddpg.png" height="500px">
</p>
<p align="center">
    <em>Overview of multi-agent DDPG.</em>
</p>

## Implementation
The baseline code from DDPG Implementation [[Github]](https://github.com/dganbold/deep_reinforcement_learning/tree/master/DDPG) which intended for solving Unity's Reacher problem.

In this project, single-agent DDPG algorithm is extended to multi-agent DDPG for [Unity's Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and hyperparameters are tuned.

## Hyperparameter tuning
Bayesian Optimization based software framework [Optuna](https://optuna.org/) is used it as hyperparameter tuning.

## Result


## Future work
- To apply MADDPG agent to solve [Unity's Soccer Twos environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos),
[[Youtube link]](https://www.youtube.com/watch?v=Hg3nmYD3DjQ&feature=youtu.be).
