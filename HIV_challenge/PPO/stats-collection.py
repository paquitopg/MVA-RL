import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm

def collect_statistics(num_episodes=600, max_steps=200):
    """
    Collect state and reward statistics from random trajectories
    """
    env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=max_steps)
    
    # Lists to store all states and rewards
    all_states = []
    all_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        all_states.append(state)
        
        for step in range(max_steps):
            # Take random action
            action = np.random.randint(0, 4)
            next_state, reward, done, _, _ = env.step(action)
            
            all_states.append(next_state)
            all_rewards.append(reward)
            
            if done:
                break
            state = next_state
    
    # Convert to numpy arrays
    all_states = np.array(all_states)
    all_rewards = np.array(all_rewards)
    
    # Calculate statistics
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    reward_mean = np.mean(all_rewards)
    reward_std = np.std(all_rewards)
    
    print("\nState Statistics:")
    print(f"Mean: {state_mean}")
    print(f"Std: {state_std}")
    print("\nReward Statistics:")
    print(f"Mean: {reward_mean}")
    print(f"Std: {reward_std}")
    
    return state_mean, state_std, reward_mean, reward_std

if __name__ == "__main__":
    state_mean, state_std, reward_mean, reward_std = collect_statistics()
