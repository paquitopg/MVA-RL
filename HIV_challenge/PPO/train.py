import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import matplotlib.pyplot as plt

from PPONetworks import PolNetwork, ValNetwork


class ProjectAgent:
    def __init__(self, state_dim=6, action_dim=4, lr=1e-3, gamma=0.99, clip_eps=0.2, 
                 entropy_coeff=0.01, max_grad_norm=0.5, gae_lambda=0.95, 
                 total_timesteps=1000000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Normalization statistics
        self.mean_state = np.array([7.81713173e+05, 5.44882828e+02, 6.46098854e+02, 
                                  2.74172803e+01, 4.17874874e+03, 3.12276867e+04])
        self.std_state = np.array([2.04371166e+05, 3.55292671e+03, 3.17949758e+02, 
                                 1.12591133e+01, 1.77395840e+04, 1.77111172e+04])
        self.mean_rwd = 15.73507607215155
        self.std_rwd = 2.457368983645282

        # Networks
        self.policy = PolNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value = ValNetwork(self.state_dim).to(self.device)

        # Optimizers and schedulers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr = lr)
        
        # Cosine learning rate decay over total timesteps
        self.policy_scheduler = CosineAnnealingLR(self.policy_optimizer, T_max= total_timesteps)
        self.value_scheduler = CosineAnnealingLR(self.value_optimizer, T_max= total_timesteps)

        # PPO parameters
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

    def scale(self, x):
        return np.where(x > 1, np.log(x), x)

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        rewards = [(self.scale(r) - self.mean_rwd) / self.std_rwd for r in rewards]
        
        # Convert to tensor and move to device
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Calculate deltas
        next_values = torch.cat([values[1:], torch.zeros(1).to(self.device)])
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        # Calculate advantages using GAE
        advantages = []
        gae = 0
        for delta, done in zip(reversed(deltas), reversed(dones)):
            if done:
                gae = 0
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            advantages.insert(0, gae)

        # Convert to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values
        
        return advantages, returns

    def update(self, states, actions, log_probs_old, values, rewards, dones):
        # Prepare inputs
        states = torch.tensor(
            ((states - self.mean_state) / self.std_state), 
            dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)

        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Policy update
        for _ in range(10):
            # Forward pass
            logits = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate policy loss
            ratios = torch.exp(log_probs - log_probs_old)
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratios * advantages_normalized
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_normalized
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

            # Gradient step with clipping
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

        # Value function update
        for _ in range(10):
            value_pred = self.value(states).squeeze()
            value_loss = nn.HuberLoss()(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

        # Step the schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def get_action(self, state, use_random=False, use_best=False):
        with torch.no_grad():
            state = torch.tensor(
                ((state - self.mean_state) / self.std_state), 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            logits = self.policy(state)
            dist = torch.distributions.Categorical(logits=logits)
            
            if use_random:
                action = torch.randint(0, self.action_dim, (1,))
            elif use_best:
                action = torch.argmax(logits)
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).item()
            value = self.value(state).squeeze().item()
            
            return action.item(), log_prob, value

    def act(self, observation, use_random=False):
        return self.get_action(observation, use_random=False, use_best=True)[0]
    
    def save(self, path = "src/models/model_"):
        torch.save(self.policy.state_dict(), path + "network.pth")
        torch.save(self.value.state_dict(), path + "value.pth")
        torch.save(self.policy_optimizer.state_dict(), path + "policy_optimizer.pth")
        torch.save(self.value_optimizer.state_dict(), path + "value_optimizer.pth")
        torch.save(self.policy_scheduler.state_dict(), path + "policy_scheduler.pth")
        torch.save(self.value_scheduler.state_dict(), path + "value_scheduler.pth")

    def load(self):
        path = "src/models/model_"
        if torch.cuda.is_available():
            self.policy.load_state_dict(torch.load(path + "network.pth"))
            self.value.load_state_dict(torch.load(path + "value.pth"))
            self.policy_optimizer.load_state_dict(torch.load(path + "policy_optimizer.pth"))
            self.value_optimizer.load_state_dict(torch.load(path + "value_optimizer.pth"))
            self.policy_scheduler.load_state_dict(torch.load(path + "policy_scheduler.pth"))
            self.value_scheduler.load_state_dict(torch.load(path + "value_scheduler.pth"))

        else:
            self.policy.load_state_dict(torch.load(path + "network.pth", map_location=torch.device('cpu')))
            self.value.load_state_dict(torch.load(path + "value.pth", map_location=torch.device('cpu')))
            self.policy_optimizer.load_state_dict(torch.load(path + "policy_optimizer.pth", map_location=torch.device('cpu')))
            self.value_optimizer.load_state_dict(torch.load(path + "value_optimizer.pth", map_location=torch.device('cpu')))
            self.policy_scheduler.load_state_dict(torch.load(path + "policy_scheduler.pth", map_location=torch.device('cpu')))
            self.value_scheduler.load_state_dict(torch.load(path + "value_scheduler.pth", map_location=torch.device('cpu')))

    def train(self, env, num_episodes=300, max_steps=200, rolling_window=15):
        # Initialize trackers
        episode_returns = []
        policy_losses = []
        value_losses = []
        entropies = []
        rolling_returns = deque(maxlen=rolling_window)
        BEST_SCORE_Individual = 1e8
        BEST_SCORE_GEN = 5e7

        # Training loop
        with tqdm(total=num_episodes, desc="Training") as pbar:
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                
                # Storage for episode data
                states = []
                actions = []
                log_probs = []
                values = []
                rewards = []
                dones = []
                
                # Episode loop
                for step in range(max_steps):
                    # Get action from policy
                    action, log_prob, value = self.get_action(state)
                   
                    # Take step in environment
                    next_state, reward, done, trunc, _ = env.step(action)
                    
                    # Store transition
                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)
                    dones.append(done)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done or trunc:
                        break
                
                # Update policy using collected data
                policy_loss, value_loss, entropy = self.update(
                    np.array(states),
                    np.array(actions),
                    np.array(log_probs),
                    np.array(values),
                    np.array(rewards),
                    np.array(dones)
                )
                
                # Store metrics
                episode_returns.append(episode_reward)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                rolling_returns.append(episode_reward)
                
                # Update progress bar
                rolling_reward = np.mean(rolling_returns)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.2e}',
                    'rolling_reward': f'{rolling_reward:.2e}',
                    'policy_loss': f'{policy_loss:.4f}'
                })
                pbar.update(1)
                
                # Optional: Save best model
                if episode_reward > BEST_SCORE_Individual:
                    print('New best model')
                    BEST_SCORE_Individual = episode_reward
                    self.save()

                    # Validation and model saving logic
                    if episode_reward > 1e10 :
                        print('Estimating validation score')
                        validation_score = evaluate_HIV(agent = self, nb_episode=3)
                        validation_score_dr = evaluate_HIV_population(agent=self, nb_episode=5)

                        if (validation_score + validation_score_dr)/2 > BEST_SCORE_GEN:
                            BEST_SCORE_GEN = (validation_score + validation_score_dr)/2
                            print(f"New best model with validation score: {BEST_SCORE_GEN:.2e}")
                            self.save()
                    
                        if validation_score > 4e10 and validation_score_dr > 2.5e10:
                            print("Found ultimate model")
                            self.save()
                            break
                    
                # Print detailed stats every 50 episodes
                if episode % 50 == 0:
                    print(f"\nEpisode {episode}")
                    print(f"Episode Reward: {episode_reward:.2e}")
                    print(f"Rolling Reward: {rolling_reward:.2e}")
                    print(f"Policy Loss: {policy_loss:.4f}")
                    print(f"Value Loss: {value_loss:.4f}")
                    print(f"Entropy: {entropy:.4f}")
                    print(f"Policy LR: {self.policy_scheduler.get_last_lr()[0]:.2e}")
                    print(f"Value LR: {self.value_scheduler.get_last_lr()[0]:.2e}")
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode returns
        ax1.plot(episode_returns)
        ax1.plot(np.convolve(episode_returns, 
                            np.ones(rolling_window)/rolling_window, 
                            mode='valid'), 
                label=f'{rolling_window}-episode average')
        ax1.set_title('Episode Returns')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.legend()
        ax1.grid(True)
        
        # Plot policy loss
        ax2.plot(policy_losses)
        ax2.set_title('Policy Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Plot value loss
        ax3.plot(value_losses)
        ax3.set_title('Value Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Plot entropy
        ax4.plot(entropies)
        ax4.set_title('Policy Entropy')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Entropy')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('ppo_training_curves.png')
        plt.close()
        
        # Also plot reward on a log scale
        plt.figure(figsize=(10, 5))
        plt.plot(np.log10(np.array(episode_returns)))
        plt.plot(np.log10(np.convolve(episode_returns, 
                                    np.ones(rolling_window)/rolling_window, 
                                    mode='valid')),
                label=f'{rolling_window}-episode average')
        plt.title('Episode Returns (Log Scale)')
        plt.xlabel('Episode')
        plt.ylabel('Log10(Return)')
        plt.legend()
        plt.grid(True)
        plt.savefig('ppo_training_curves_log.png')
        plt.close()
        print("Overall best model saved performances : ", BEST_SCORE_Individual, BEST_SCORE_GEN)
        return episode_returns, policy_losses, value_losses, entropies
            
if __name__ == "__main__":
    env = TimeLimit(
        HIVPatient(domain_randomization=True), 
        max_episode_steps=200
    )
    agent = ProjectAgent()
    agent.train(env, num_episodes=200, max_steps=200)