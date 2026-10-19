"""Train and evaluate a Q-learning agent on deterministic FrozenLake-v1."""
import os
import numpy as np
import gymnasium as gym
from agent import QLearningAgent
from utils import save_training_plot, evaluate_policy

SEED = 42
EPISODES = 20_000

def train():
    os.makedirs("outputs", exist_ok=True)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env.action_space.seed(SEED)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.8, gamma=0.95, epsilon=1.0,
        epsilon_min=0.02, epsilon_decay=0.0005, seed=SEED
    )
    rewards, lengths = [], []
    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        done, total_reward, steps = False, 0.0, 0
        while not done and steps < 100:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
        agent.decay_epsilon()
        rewards.append(total_reward)
        lengths.append(steps)
    np.save("outputs/q_table.npy", agent.q_table)
    np.save("outputs/episode_rewards.npy", np.asarray(rewards))
    save_training_plot(rewards, "outputs/training_curve.png")
    success_rate, avg_steps = evaluate_policy(env, agent.q_table, episodes=500, seed=50_000)
    print(f"Evaluation success rate over 500 episodes: {success_rate:.3f}")
    print(f"Average episode length: {avg_steps:.2f}")
    print("Saved: outputs/q_table.npy, outputs/episode_rewards.npy, outputs/training_curve.png")
    env.close()
    return agent.q_table, rewards, success_rate, avg_steps

if __name__ == "__main__":
    train()
