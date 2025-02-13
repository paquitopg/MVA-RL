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
        """Initialize the Dueling Double DQN agent"""
        self.save_path = 'project_agent.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        
        self.target_update_freq = config['target_update_freq']
        self.update_counter = 0

        # Modified epsilon parameters for slower decay
        self.epsilon_max = 1.0 
        self.epsilon_min = 0.05  # Increased minimum exploration
        self.epsilon_decay_period = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_decay_period

        self.state_size = 6
        self.action_size = 4
        self.hidden_dimension = config['hidden_dimension']

        self.lr = config['learning_rate']  # Made learning rate configurable
        self.criterion = torch.nn.HuberLoss()

        # Initialize reward scaling factors
        self.reward_scale = config['reward_scale']

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
        """Select greedy action"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()

    def act(self, observation, use_random=False):
        """Select action using epsilon-greedy policy"""
        if use_random and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.take_greedy_action(observation)

    def gradient_step(self):
        """Perform a gradient step using Double DQN"""
        if len(self.memory) < self.batch_size:
            return

        # Get batch
        batch = self.memory.sample(self.batch_size)
        s, a, r, s_, d = batch

        # Scale rewards
        r = r * self.reward_scale

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
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

def train():
    env = TimeLimit(
        HIVPatient(domain_randomization=True), max_episode_steps=200
    )
    
    config = {
        'gamma': 0.99,
        'batch_size': 128,  # Reduced batch size
        'buffer_size': 100000,  # Increased buffer size
        'hidden_dimension': 64,
        'epsilon_decay_period': 1000,  # Slower epsilon decay
        'epsilon_delay_decay': 100,  # Longer delay before decay
        'target_update_freq': 500,  # Less frequent target updates
        'learning_rate': 0.0003,  # Reduced learning rate
        'reward_scale': 1e-7  # Scale factor for rewards
    }
    
    agent = ProjectAgent(config, DUELING=True)
    n_episodes = 500  # More episodes
    n_step = 150

    episode_returns = []
    epsilons = []
    losses = []
    agent.epsilon = agent.epsilon_max
    BEST_SCORE = 0.0
    current_episode = 0

    for episode in tqdm(range(n_episodes)):
        epsilons.append(agent.epsilon)
        state, _ = env.reset()
        episode_cum_reward = 0
        episode_losses = []

        for step in range(n_step):
            action = agent.act(state, use_random=True)
            next_state, reward, done, trunc, _ = env.step(action)
            agent.memory.append(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                loss = agent.gradient_step()
                episode_losses.append(loss)

            state = next_state
            episode_cum_reward += reward

            if done:
                break

        current_episode += 1
        if current_episode > agent.epsilon_delay:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_step)
        
        episode_returns.append(episode_cum_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Print training information every 10 episodes
        if episode % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            print(f"\nEpisode {episode}")
            print(f"Average Return: {avg_return:.2e}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epsilon: {agent.epsilon:.3f}")

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
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_returns)
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    
    plt.subplot(1, 3, 2)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    train()
