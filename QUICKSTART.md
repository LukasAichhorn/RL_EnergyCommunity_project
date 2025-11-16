# Quick Start Guide - Stable-Baselines3 Version

## What Changed

The project has been rebuilt to use **Stable-Baselines3 (SB3)** instead of Ray RLlib. This is simpler, more stable, and easier to use.

## Installation

```bash
pip install -r requirements.txt
```

This installs:
- `stable-baselines3` - The RL library
- `gymnasium` - The environment interface
- Other dependencies (pandas, numpy, matplotlib, torch)

## Quick Test

### Step 1: Test the Environment

```bash
python battery_env.py
```

Should show a few test steps with random actions.

### Step 2: Test with Simple Policy

```bash
python example_usage.py
```

Shows how the environment works with a basic rule-based policy.

### Step 3: Train an Agent (Quick Test)

Start with a short training run:

```bash
python train_agent.py \
    --algorithm PPO \
    --episode-length 24 \
    --total-timesteps 10000 \
    --checkpoint-freq 5000
```

This will:
- Train for 10,000 timesteps (~2-5 minutes)
- Save checkpoints every 5,000 steps
- Create models in `./models/` directory

### Step 4: Evaluate the Agent

```bash
python evaluate_agent.py \
    --model ./models/best_model/best_model \
    --algorithm PPO \
    --num-episodes 3
```

This will:
- Load the best model
- Run 3 evaluation episodes
- Create plots in `./evaluation_results/`

## Full Training

For a proper training run:

```bash
python train_agent.py \
    --algorithm PPO \
    --episode-length 96 \
    --total-timesteps 200000 \
    --checkpoint-freq 20000 \
    --eval-freq 10000
```

**Parameters:**
- `--total-timesteps 200000`: Train for 200k steps (30-60 minutes)
- `--checkpoint-freq 20000`: Save checkpoint every 20k steps
- `--eval-freq 10000`: Evaluate every 10k steps
- `--episode-length 96`: 1 day episodes (96 steps × 15 min)

## Monitoring Training

You can monitor training with TensorBoard:

```bash
tensorboard --logdir ./models/tensorboard
```

Then open http://localhost:6006 in your browser.

## What You'll See

### During Training:
```
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
----------------------------------
| rollout/           |          |
|    ep_len_mean     | 24       |
|    ep_rew_mean     | -45.2    |  ← Should increase over time
| time/              |          |
|    fps             | 1200     |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 2048     |
----------------------------------
```

### Model Files:
- `./models/checkpoints/model_10000_steps.zip` - Checkpoints
- `./models/best_model/best_model.zip` - Best model
- `./models/final_model_PPO.zip` - Final model

## Tips

1. **Start small**: Use `--total-timesteps 10000` first to test
2. **Watch rewards**: `ep_rew_mean` should increase (less negative)
3. **Use TensorBoard**: Monitor training progress visually
4. **Try both algorithms**: PPO is more stable, SAC can be more efficient
5. **Adjust episode length**: Shorter episodes (24) train faster, longer (96) are more realistic

## Troubleshooting

- **Import errors**: Make sure you installed requirements: `pip install -r requirements.txt`
- **CUDA errors**: SB3 will use CPU by default, which is fine for this project
- **Memory issues**: Reduce `--total-timesteps` or `--episode-length`

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Test environment: `python battery_env.py`
3. Quick train: `python train_agent.py --total-timesteps 10000`
4. Evaluate: `python evaluate_agent.py --model ./models/best_model/best_model --algorithm PPO`
