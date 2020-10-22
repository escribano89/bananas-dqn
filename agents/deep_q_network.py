# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:47:01 2020

@author: Javier Escribano
"""
from models.neural_network import NeuralNetwork
from utils.replay_buffer import ReplayBuffer
from collections import deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 10000     # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.05              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
EPSILON_START = 1       # epsilon start value
EPSILON_MIN = 0.001     # epsilon min value
EPSILON_DECAY = 0.99    # epsilon decayment value
N_EPISODES = 2000       # number of episodes
MAX_TIMESTEPS = 1000    # max number of timesteps per episode
GOAL = 14               # goal to consider the problem solved

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = NeuralNetwork(state_size, action_size).to(device)
        self.qnetwork_target = NeuralNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def train(self, env, brain_name):
        scores = []                       
        scores_window = deque(maxlen=100) 
        eps = EPSILON_START                   
        for i_episode in range(1, N_EPISODES+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(MAX_TIMESTEPS):
                action = self.act(state, eps)
                env_info = env.step(action.astype(np.int32))[brain_name] 
                next_state = env_info.vector_observations[0]   
                reward = env_info.rewards[0]               
                done = env_info.local_done[0]      
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
                
            scores_window.append(score)       
            scores.append(score)              
            eps = max(EPSILON_MIN, EPSILON_DECAY*eps)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
            if np.mean(scores_window)>= GOAL:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores
    
    def test(self, env, brain_name,filename):
        weights = torch.load(filename)
        self.qnetwork_local.load_state_dict(weights)
        self.qnetwork_target.load_state_dict(weights)
            
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]

        for t in range(MAX_TIMESTEPS):
            action = self.act(state, 0)
            env_info = env.step(action.astype(np.int32))[brain_name] 
            next_state = env_info.vector_observations[0]         
            done = env_info.local_done[0]      
            state = next_state
 
            if done:
                break 
                
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network using soft update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


