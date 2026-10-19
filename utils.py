"""Plotting and evaluation helpers."""
import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.array([values.mean()]) if len(values) else np.array([])
    return np.convolve(values, np.ones(window)/window, mode="valid")

def save_training_plot(rewards, output_path, window=100):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    ma = moving_average(rewards, window)
    if len(ma):
        x = np.arange(len(ma)) + (window-1 if len(rewards) >= window else 0)
        plt.plot(x, ma, linewidth=2, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (1 = reached goal)")
    plt.title("FrozenLake Q-learning training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def evaluate_policy(env, q_table, episodes=500, seed=1000):
    successes, lengths = 0, []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done, steps = False, 0
        while not done and steps < 100:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        successes += int(reward > 0)
        lengths.append(steps)
    return successes / episodes, float(np.mean(lengths))
