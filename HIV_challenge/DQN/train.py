from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim import lr_scheduler as lr_scheduler

from Challenge.src.ReplayBuffer import ReplayBuffer
from Challenge.src.DeepQN import BaseQNetwork, DeepQNetwork, DDeepQNetwork, DDDeepQNetwork, DuelingNetwork, DeepDuelingNetwork, DDeepDuelingNetwork

from evaluate import evaluate_HIV, evaluate_HIV_population

class ProjectAgent:
    def __init__(self, config, DUELING=True):
        """
        Initialize the agent here
        Inputs : 
            config : a dictionary containing the parameters of the agent
            model : the model of the agent
            DUELING : a boolean indicating if the agent is a dueling agent
         
        Output : 
            None
        """
        self.save_path = 'project_agent.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        
        self.target_update_freq = config['target_update_freq']  # Parameter for target network update frequency
        self.update_counter = 0  # Counter for target network updates

        self.epsilon_max = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay_period = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_decay_period

        self.state_size = 6
        self.action_size = 4
        self.hidden_dimension = config['hidden_dimension']

        self.lr = 0.001
        self.criterion = torch.nn.HuberLoss()

        if DUELING:
            print("Using Dueling Network Architecture")
            self.q_network = DeepDuelingNetwork(state_size=self.state_size, 
                                              action_size=self.action_size, 
                                              hidden_dimension=self.hidden_dimension).to(self.device)
            self.target_network = DeepDuelingNetwork(state_size=self.state_size, 
                                                   action_size=self.action_size, 
                                                   hidden_dimension=self.hidden_dimension).to(self.device)
        else:
            print("Using Basic QNetwork Architecture")
            self.q_network = DeepQNetwork(state_size=self.state_size, 
                                        action_size=self.action_size, 
                                        hidden_dimension=self.hidden_dimension).to(self.device)
            self.target_network = DeepQNetwork(state_size=self.state_size, 
                                             action_size=self.action_size, 
                                             hidden_dimension=self.hidden_dimension).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.epsilon_delay, gamma=0.5)


    def take_greedy_action(self, observation):
        """
        Inputs : 
            network : the network to use
            observation : the current state
        
        Output :
            action : the action to take
        """
        obs_tensor = torch.Tensor(observation).unsqueeze(0).to(self.device)
        q_tensor = self.q_network(obs_tensor)
        return q_tensor.argmax(dim=1).item()


    def act(self, observation, use_random=False):
        """
        Inputs : 
            observation : the current observation
            use_random : a boolean indicating if the agent should act randomly
        
        Output :
            action : the action to take
        """
        if use_random and random.random() < self.epsilon: 
            return np.random.randint(0, self.action_size) # Random action
        else : 
            return self.take_greedy_action(observation) # Greedy action
        

    def gradient_step(self):
        """Perform a gradient step using Double DQN"""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        s, a, r, s_, d = batch

        # Double DQN: Use online network to select actions and target network to evaluate them
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(s_).argmax(1, keepdim=True)
            # Evaluate actions using target network
            next_q_values = self.target_network(s_)
            max_next_q = next_q_values.gather(1, next_actions).squeeze(1)
            target_q_values = r + self.gamma * max_next_q * (1 - d)

        # Current Q-values
        current_q_values = self.q_network(s)
        current_q = current_q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)

        # Compute loss and update
        loss = self.criterion(current_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")


    def load(self):
        path = self.save_path
        self.q_network.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Model loaded from {path}")
        

def train():
    env = TimeLimit(
        HIVPatient(domain_randomization=True), max_episode_steps=200
    )
    
    config = {
        'gamma': 0.99,
        'batch_size': 1024,
        'buffer_size': 70000,
        'hidden_dimension': 64,
        'epsilon_decay_period': 880,
        'epsilon_delay_decay': 100,
        'target_update_freq': 40  # Update target network every 100 steps
    }
    
    agent = ProjectAgent(config, DUELING=True)
    n_episodes = 1200
    n_step = 200

    episode_returns = []
    epsilons = []
    agent.epsilon = agent.epsilon_max
    BEST_SCORE = 0.0
    current_episode = 0

    for episode in tqdm(range(n_episodes)):
        epsilons.append(agent.epsilon)
        state, _ = env.reset()
        episode_cum_reward = 0

        for step in range(n_step):
            action = agent.act(state, use_random=True)
            next_state, reward, done, trunc, _ = env.step(action)
            agent.memory.append(state, action, reward, next_state, done)
            agent.gradient_step()

            state = next_state
            episode_cum_reward += reward

            if done:
                break

        current_episode += 1
        if current_episode > agent.epsilon_delay:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_step)
        episode_returns.append(episode_cum_reward)

        # Validation and model saving logic
        if episode_cum_reward > 3.5e10:
            validation_score = evaluate_HIV(agent=agent, nb_episode=3)
            validation_score_dr = evaluate_HIV_population(agent=agent, nb_episode=5)
            if (validation_score + validation_score_dr)/2 > BEST_SCORE:
                BEST_SCORE = (validation_score + validation_score_dr)/2
                print(f"New best model with validation score: {BEST_SCORE:.2e}")
                agent.save("best_" + agent.save_path)
            print(f"Validation scores - Standard: {validation_score:.2e}, DR: {validation_score_dr:.2e}")
            if validation_score > 4e10 and validation_score_dr > 2.5e10:
                agent.save("best_top_score_" + agent.save_path)
                break

    agent.save(agent.save_path)