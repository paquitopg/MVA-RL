from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch.nn as nn
import torch
import numpy as np
import random
from copy import deepcopy
# from tqdm import tqdm

# Import the necessary classes from the Challenge/src folder
from ReplayBuffer import ReplayBuffer
from DeepQN import DDeepDuelingNetwork
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProjectAgent:

    config = {'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 100000,
          'initial_memory_size': 800,
          'epsilon_min': 0.01,
          'epsilon_max': 1.0,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 400,
          'batch_size': 1024,
          'episode_max_length': 200,
          'gradient_steps': 2,
          'hidden_dim': 256,
          'update_target_freq': 600,
          'update_target_tau': 0.001,
          'update_target_strategy': 'ema',
          'criterion': nn.HuberLoss()}

    def greedy_action(self,network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def __init__(self, config = config):
        self.device = device
        self.model = DDeepDuelingNetwork(hdim=config['hidden_dim']).to(self.device)
        self.state_size = 6
        self.nb_actions = 4
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.initial_memory_size = config['initial_memory_size'] if 'initial_memory_size' in config.keys() else 1000
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] 
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        self.lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.episode_count = 0
        self.max_ep = config['episode_max_length'] if 'episode_max_length' in config.keys() else 300

    def gradient_step(self):
        if len(self.memory) >= self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def init_replay_buffer(self, env):
        state, _ = env.reset()
        for _ in range(self.initial_memory_size):
            action = env.action_space.sample()
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if done or trunc:
                state, _ = env.reset()
            else:
                state = next_state
        self.episode_count += 1


    def train(self, env, max_episode=None):
        BEST_SCORE_GEN = 5e9
        BEST_REWARD = 1e10
        BEST_SCORE = 1e10
        if self.episode_count == 0:
            self.init_replay_buffer(env)
        if max_episode is None:
            max_episode = self.max_ep
        
        self.training = True
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        # Create progress bar for episodes
        # episode_pbar = tqdm(total=max_episode, desc='Episodes', position=0)
        # Create progress bar for steps
        # step_pbar = tqdm(desc='Steps', position=1, leave=True)
        
        # Initialize reward tracking
        reward_window = []
        reward_tracking_interval = 50  # Print every 50 episodes
        
        metrics_dict = {
            'epsilon': epsilon,
            'reward': 0.0,
            'buffer_size': len(self.memory)
        }
        
        try:
            while episode < max_episode:
                # Update epsilon
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
                
                # Select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.greedy_action(self.model, state)
                
                # Step
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward
                
                # Train
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()
                
                # Update target network if needed
                if self.update_target_strategy == 'replace':
                    if step % self.update_target_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.update_target_strategy == 'ema':
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    tau = self.update_target_tau
                    for key in model_state_dict:
                        target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                    self.target_model.load_state_dict(target_state_dict)
                
                # Update progress bars and metrics
                step += 1
                #step_pbar.update(1)
                metrics_dict.update({
                    'epsilon': round(epsilon, 3),
                    'reward': round(episode_cum_reward * 0.0000001, 4),
                    'buffer_size': len(self.memory)
                })
                #step_pbar.set_postfix(metrics_dict)

                if done or trunc:
                    episode += 1
                    #episode_pbar.update(1)
                    episode_return.append(episode_cum_reward)
                    
                    # Track rewards for reporting
                    reward_window.append(episode_cum_reward * 0.0000001)  # Scale reward
                    
                    # Print reward statistics every reward_tracking_interval episodes
                    if episode % reward_tracking_interval == 0:
                        avg_reward = np.mean(reward_window)
                        std_reward = np.std(reward_window)
                        min_reward = np.min(reward_window)
                        max_reward = np.max(reward_window)
                        print(f"\nEpisode {episode}/{max_episode}")
                        print(f"Last {reward_tracking_interval} episodes statistics:")
                        print(f"  Average reward: {avg_reward:.4f}")
                        print(f"  Std reward: {std_reward:.4f}")
                        print(f"  Min reward: {min_reward:.4f}")
                        print(f"  Max reward: {max_reward:.4f}")
                        reward_window = []  # Reset window

                    if episode_cum_reward > 1e10 and episode_cum_reward > BEST_REWARD:
                        print('Estimating validation score')
                        validation_score = evaluate_HIV(agent=self, nb_episode=3)
                        validation_score_dr = evaluate_HIV_population(agent=self, nb_episode=10)

                        if validation_score > BEST_SCORE:
                            print(f"New best model with particular validation score: {validation_score:.2e}")
                            BEST_SCORE = validation_score
                            self.save(f"src/bestModel_part{episode}.pth")

                        if validation_score_dr > BEST_SCORE_GEN:
                            BEST_REWARD = episode_cum_reward
                            BEST_SCORE_GEN =  validation_score_dr
                            print(f"New best model with population validation score: {BEST_SCORE_GEN:.2e}")
                            self.save(f"src/best_genModel{episode}.pth")

                    # Reset for next episode
                    state, _ = env.reset()
                    episode_cum_reward = 0
                else:
                    state = next_state
        
        finally:
            # Clean up progress bars
            # episode_pbar.close()
            # step_pbar.close()
            pass
        
        return episode_return

    def act(self, observation, use_random=False):
        self.training = False
        return self.greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        pass

    def load(self):
        self.model.load_state_dict(torch.load("src/best_genModel_256.pth", weights_only=True, map_location=self.device))
        self.target_model = deepcopy(self.model).to(self.device)
        pass

if __name__ == "__main__":
    agent = ProjectAgent()
    print("Training the agent")
    agent.train(env)
    agent.save("src/fully_trained_agent.pth")
    print("Agent trained and saved")
    pass