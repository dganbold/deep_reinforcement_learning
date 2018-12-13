# Reacher

<p align="center">
    <img src="https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif" height="250px">
</p>

## Description
In this [environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher), a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.

## Environment setup (for Linux)

#### Step 1: Clone the Repository
If you haven't already, please follow the [instructions](https://github.com/dganbold/deep_reinforcement_learning) to set up your Python environment.

#### Step 2: Download the Unity Environment
Download pre-built environment for Linux from one of the [links](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip).<br />
Then, place the file in the DDPG/Reacher/ folder in the cloned Repository, and decompress the file.<br />

```
$ cp Reacher_Linux.zip DDPG/Reacher/
$ cd DDPG/Reacher/
$ unzip Reacher_Linux.zip
```

Next, change the file_name parameter in train.py and test.py to match the binary file name of the Unity environment that you downloaded.
- Linux (x86): "Reacher_Linux/Reacher.x86"
- Linux (x86_64): "Reacher_Linux/Reacher.x86_64"

## State space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

## Action space
In this environment, a double-jointed arm can move to target locations. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Reward
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

## Usage

- Execute the following command to train the agent:

```
$ python train.py
```

- Execute the following command to test the pre-trained agent:

```
$ python test.py
```

# Result
This environment is solved in 147 episodes by [DDPG](https://github.com/dganbold/deep_reinforcement_learning/blob/master/DDPG/agent/DDPG.py) with [hyperparameters](https://github.com/dganbold/deep_reinforcement_learning/blob/master/DDPG/config/UnityML_Agent.py). [[score history]](../scores/Reacher_DDPG_1.0E-04_1.0E-04_256_1.0E-03_128.csv).<br />
See the detailed project report from [Report.md](./docs/Report.md) file.
<p align="center">
    <img src="./docs/best_score_history.png" height="260">
</p>
