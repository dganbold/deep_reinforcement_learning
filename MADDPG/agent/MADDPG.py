import numpy as np
from agent.DDPG import Agent
import torch
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class MultiAgent():
    def __init__(self, number_of_agents, state_size, action_size, param, seed=0):
        super(MultiAgent, self).__init__()

        # Parameter settings
        param['actor_input_size'] = state_size
        param['actor_output_size'] = action_size

        # Critic input = all_states + all_actions
        param['critic_input_size'] = (state_size + action_size)*number_of_agents

        # Create Agent instance
        self.number_of_agents = number_of_agents
        self.agents = [ Agent(0,param, seed), Agent(1,param, seed) ]

    def act(self, states, noise_amplitude=0.0):
        """ Get actions from all agents in the MADDPG object"""
        actions = [agent.act(state, noise_amplitude) for agent, state in zip(self.agents, states)]
        return actions

    #def target_act(self, states, noise_amplitude=0.0):
    #    """ Get target network actions from all the agents in the MADDPG object """
    #    target_actions = [agent.target_act(state, noise_amplitude) for agent, state in zip(self.agents, states)]
    #    return target_actions

    #def target_act(self, states, noise_amplitude=0.0):
    #    """ Get target network actions from all the agents in the MADDPG object """
    #    target_actions = [agent.target_act(state, noise_amplitude) for agent, state in zip(self.agents, states)]
    #    return target_actions

    #def learn(self, experiences, agent_number, logger):
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
        #states = [s for s in states.t()]
        #actions = [a for a in actions.t()]
        #next_states = [s for s in next_states.t()]
        states_list = []
        actions_list = []
        next_states_list = []
        for id, agent in enumerate(self.agents):
            agent_id = torch.tensor([id]).to(device)
            next_state = states[:,id,:]
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

            states = [s for s in states.t()]
            actions = [a for a in actions.t()]
            next_states = [s for s in next_states.t()]
        #
        agent = self.agents[agent_id]
        next_actions = [agent.actor_target(next_state) for agent, next_state in zip(self.agents, next_states)]
        predicted_actions = [agent.actor_local(state) for agent, state in zip(self.agents, states)]
        #
        states = torch.cat(states, dim=1).to(device)
        actions = torch.cat(actions, dim=1).to(device)
        next_states = torch.cat(next_states, dim=1).to(device)
        #
        experiences = (states, actions, rewards, next_states, dones)
        agent.learn(experiences,next_actions,predicted_actions)

        """
        actions_next = torch.cat(actions_next_list, dim=1).to(device)
        # Concatenate next_states and actions_next
        target_critic_input = torch.cat((next_states,actions_next), dim=1).to(device)
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = agent.critic_target(target_critic_input)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[agent_number].view(-1, 1) + agent.gamma * Q_targets_next * (1 - dones[agent_number].view(-1, 1))
        # Compute critic loss
        actions_list = [a for a in actions]
        actions = torch.cat(actions_list, dim=1)
        states_list = [s for s in states]
        states = torch.cat(states_list, dim=1)
        # Concatenate states and actions
        local_critic_input = torch.cat((states,actions), dim=1).to(device)
        Q_expected = agent.critic_local(local_critic_input)
        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(Q_expected, Q_targets.detach())
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Make input to agent
        # detach the other agents to save computation saves some time for computing derivative
        predicted_actions_list = [ self.maddpg_agent[n].actor_local(state) if n == agent_number \
                else self.maddpg_agent[n].actor_local(state).detach() for n, state in enumerate(states_list) ]
        # Concatenate states and actions
        # combine all the actions and observations for input to critic
        # many of the states are redundant, and states[1] contains all useful information already
        predicted_actions = torch.cat(predicted_actions_list, dim=1).to(device)
        critic_input = torch.cat((states, predicted_actions), dim=1).to(device)
        # Compute actor loss
        actor_loss = -agent.critic_local(critic_input).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        # Update actor network using policy gradient
        agent.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        agent.critic_soft_update()
        agent.actor_soft_update()
        """
        # ---------------------------- loss logging ---------------------------- #
        #al = actor_loss.cpu().detach().item()
        #cl = critic_loss.cpu().detach().item()
        #logger.add_scalars('agent%i/losses' % agent_number,{'critic loss': cl,'actor_loss': al},self.iter)
