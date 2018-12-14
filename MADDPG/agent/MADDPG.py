import numpy as np
from agent.DDPG import Agent
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class MultiAgent():
    def __init__(self, number_of_agents, state_size, action_size, param, seed=0):
        super(MultiAgent, self).__init__()

        # Parameter settings
        param['actor_state_size'] = state_size
        param['actor_action_size'] = action_size

        # Critic input = all_states + all_actions
        param['critic_state_size'] = state_size*number_of_agents
        param['critic_action_size'] = action_size*number_of_agents

        # Create Agent instance
        self.number_of_agents = number_of_agents
        self.agents = [ Agent(0,param, seed), Agent(1,param, seed) ]

    def act(self, states, noise_amplitude=0.0):
        """ Get actions from all agents in the MADDPG object"""
        actions = [agent.act(state, noise_amplitude) for agent, state in zip(self.agents, states)]
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[array]): tuple of (s, a, r, s', done) tuples
            agent_id (int):
        """
        states, actions, rewards, next_states, dones = experiences
        #
        states  = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        #
        states = [s for s in states.t()]
        actions = [a for a in actions.t()]
        next_states = [s for s in next_states.t()]
        next_actions = [agent.actor_target(next_state) for agent, next_state in zip(self.agents, next_states)]
        predicted_actions = [agent.actor_local(state) for agent, state in zip(self.agents, states)]
        #
        agent = self.agents[agent_id]
        #
        states = torch.cat(states, dim=1).to(device)
        actions = torch.cat(actions, dim=1).to(device)
        next_states = torch.cat(next_states, dim=1).to(device)
        #
        experiences = (states, actions, rewards, next_states, dones)

        agent.learn(experiences,next_actions,predicted_actions)

    def export_network(self,filename):
        for id, agent in enumerate(self.agents):
            agent.export_network('{:s}_{:d}'.format(filename,id))

    def import_network(self,filename):
        for id, agent in enumerate(self.agents):
            agent.import_network('{:s}_{:d}'.format(filename,id))
