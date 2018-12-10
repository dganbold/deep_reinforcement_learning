# Tennis

<p align="center">
    <img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif" height="250px">
</p>

[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

## Environment setup (for Linux)
In this environment, two agents control rackets to bounce a ball over a net.
The task is episodic, environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

#### Step 1: Clone the Repository
If you haven't already, please follow the [instructions](https://github.com/dganbold/deep_reinforcement_learning) to set up your Python environment.

#### Step 2: Download the Unity Environment
Download pre-built environment for Linux from one of the [links](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip).<br />
Then, place the file in the DDPG/Tennis/ folder in the cloned Repository, and decompress the file.<br />

```
$ cp Tennis_Linux.zip DDPG/Tennis/
$ cd DDPG/Tennis/
$ unzip Tennis_Linux.zip
```

Next, change the file_name parameter in train.py and test.py to match the binary file name of the Unity environment that you downloaded.
- Linux (x86): "Tennis_Linux/Tennis.x86"
- Linux (x86_64): "Tennis_Linux/Tennis.x86_64"

## State space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.

## Action space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Reward
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

## Usage

- Execute the following command to train the agent:

```
$ python train.py
```

- Execute the following command to test the pre-trained agent:

```
$ python test.py
```

## Result
"ToDo"
