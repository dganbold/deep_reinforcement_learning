# Tennis

## Description
In this [environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos), the goal is to train a team of agents to play soccer.
Set-up: Environment where four agents compete in a 2 vs 2 toy soccer game.
* Goal:
  * Striker: Get the ball into the opponent's goal.
  * Goalie: Prevent the ball from entering its own goal.
* Agents:
The environment contains four agents, with two linked to one Brain (strikers) and two linked to another (goalies).

## Environment setup (for Linux)

#### Step 1: Clone the Repository
If you haven't already, please follow the [instructions](https://github.com/dganbold/deep_reinforcement_learning) to set up your Python environment.

#### Step 2: Download the Unity Environment
Download pre-built environment for Linux from one of the [links](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip).<br />
Then, place the file in the MADDPG/Soccer/ folder in the cloned Repository, and decompress the file.<br />

```
$ cp Soccer_Linux.zip MADDPG/Soccer/
$ cd MADDPG/Soccer/
$ unzip Soccer_Linux.zip
```

Next, make sure the file_name parameter in train.py and test.py to match the binary file name of the Unity environment that you downloaded.
- Linux (x86): "Soccer_Linux/Soccer.x86"
- Linux (x86_64): "Soccer_Linux/Soccer.x86_64"

## State space
Vector Observation space: 112 corresponding to local 14 ray casts, each detecting 7 possible object types, along with the object's distance. Perception is in 180 degree view from front of agent.

## Action space
Vector Action space: (Discrete) One Branch
* Striker: 6 actions corresponding to forward, backward, sideways movement, as well as rotation.
* Goalie: 4 actions corresponding to forward, backward, sideways movement.

## Reward
Agent Reward Function (dependent):
* Striker: If an agent receives a reward of +1 when ball enters opponent's goal. If an agent enters own team's goal then receives a reward of -0.1 and -0.001 existential penalty.
* Goalie: If an agent receives a reward of -1 when ball enters team's goal. If ball enters opponents goal then receives a reward of +0.1 and +0.001 existential bonus.

If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

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
ToDo
