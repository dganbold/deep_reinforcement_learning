from config import *
# Environment
from unityagents import UnityEnvironment
# Agent
#from Agent.DoubleQLearner import Agent
from Agent.NeuralQLearner import Agent

# Initialize environment object
params = HYPERPARAMS['LunarLander']
env_name = params['env_name']
env = gym.make(env_name)
env.seed(0)

# Get environment parameter
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
print('Number of actions : ', action_size)
print('Dimension of state space : ', state_size)

# Initialize Double-DQN agent
agent = Agent(state_size=state_size, action_size=action_size, param=params, seed=0)
# Load the pre-trained network
agent.import_network('models/%s_%s'% (agent.name,env_name))

# Define parameters for test
episodes = 2                        # maximum number of test episodes

""" Test loop  """
for i_episode in range(1, episodes+1):
    # Reset the environment and Capture the current state
    state = env.reset()

    # Reset score collector
    score = 0
    done = False
    # One episode loop
    while not done:
        # Action selection by Epsilon-Greedy policy
        action = agent.greedy(state)
        env.render()
        # Take action and get rewards and new state
        next_state, reward, done, _ = env.step(action)

        # State transition
        state = next_state

        # Update total score
        score += reward


    # Print episode summary
    print('\r#TEST Episode:{}, Score:{:.2f}'.format(i_episode, score))

""" End of the Test """

# Close environment
env.close()
