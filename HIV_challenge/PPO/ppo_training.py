import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from Challenge.src.train import ProjectAgent
from evaluate import evaluate_HIV, evaluate_HIV_population

def train_ppo(env, agent, num_episodes=1000, max_steps=200, rolling_window=100):
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
                action, log_prob, value = agent.get_action(state)
                
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
            policy_loss, value_loss, entropy = agent.update(
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
                agent.save('best_ppo_model.pth')

                # Validation and model saving logic
                if episode_reward > 1e10 :
                    print('Estimating validation score')
                    validation_score = evaluate_HIV(agent=agent, nb_episode=3)
                    validation_score_dr = evaluate_HIV_population(agent=agent, nb_episode=5)

                    if (validation_score + validation_score_dr)/2 > BEST_SCORE_GEN:
                        BEST_SCORE_GEN = (validation_score + validation_score_dr)/2
                        print(f"New best model with validation score: {BEST_SCORE_GEN:.2e}")
                        agent.save("best_genModel.pth")
                
                    if validation_score > 4e10 and validation_score_dr > 2.5e10:
                        agent.save("ultimate_model.pth")
                        break
                
            # Print detailed stats every 50 episodes
            if episode % 100 == 0:
                print(f"\nEpisode {episode}")
                print(f"Episode Reward: {episode_reward:.2e}")
                print(f"Rolling Reward: {rolling_reward:.2e}")
                print(f"Policy Loss: {policy_loss:.4f}")
                print(f"Value Loss: {value_loss:.4f}")
                print(f"Entropy: {entropy:.4f}")
                print(f"Policy LR: {agent.policy_scheduler.get_last_lr()[0]:.2e}")
                print(f"Value LR: {agent.value_scheduler.get_last_lr()[0]:.2e}")
    
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

    return episode_returns, policy_losses, value_losses, entropies

if __name__ == "__main__":
    env = TimeLimit(
        HIVPatient(domain_randomization=True),
        max_episode_steps=200
    )
    agent = ProjectAgent(
        state_dim=6,
        action_dim=4,
        lr=1e-3,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        total_timesteps=1000000
    )
    returns, policy_losses, value_losses, entropies = train_ppo(
        env=env,
        agent=agent,
        num_episodes=150,
        max_steps=200,
        rolling_window=10
    )
