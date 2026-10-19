# Custom Reinforcement Learning Agent — FrozenLake-v1

A beginner-friendly, from-scratch tabular Q-learning implementation using Farama Gymnasium.

## Environment choice
`FrozenLake-v1`, 4×4 map, `is_slippery=False` (deterministic). The agent starts at S, must reach G, and must avoid holes H. Reward is 1 for reaching G and 0 otherwise. The deterministic setting is deliberate so the learned policy can be inspected and reliably demonstrated.

## Repository files
- `main.py` — creates the Gymnasium environment, trains and evaluates the agent.
- `agent.py` — Q-learning table, epsilon-greedy action selection and update rule.
- `utils.py` — moving average, plot saving and policy evaluation.
- `notebook.ipynb` — step-by-step notebook.
- `requirements.txt` — Python dependencies.
- `outputs/q_table.npy` — Q-table produced with the included deterministic training setup.
- `outputs/training_curve.png` — illustrative training curve from the bundled reference run; rerun notebook/main.py to generate a fresh curve.
- `outputs/agent_demo.gif` — short grid animation of the learned route.

## Run in Google Colab (recommended)
1. Upload/unzip this repository in Colab, or upload the `.ipynb` notebook.
2. Run `!pip install -r requirements.txt`.
3. Run all notebook cells from top to bottom.
4. The notebook saves the Q-table, reward plot and GIF under `outputs/`.
5. Download the repository folder and push it to GitHub.

## Run locally
```bash
python -m pip install -r requirements.txt
python main.py
```


