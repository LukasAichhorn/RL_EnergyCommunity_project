# RL Battery Control Gym Environment

A Gymnasium-compatible reinforcement learning environment for training agents to control a community battery system. The agent learns to minimize grid energy usage by optimizing charge/discharge decisions based on community consumption and production patterns.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** This project uses Stable-Baselines3 (SB3) instead of Ray RLlib for simpler, more stable training.

## Project Structure

- `data_processor.py`: Data loading and preprocessing utilities
- `battery_env.py`: Gymnasium environment for battery control
- `train_agent.py`: Training script using Ray RLlib
- `evaluate_agent.py`: Evaluation and visualization script
- `requirements.txt`: Python dependencies

## Quick Start

### 1. Test Data Processing

```bash
python data_processor.py
```

### 2. Test Environment

```bash
python battery_env.py
```

### 3. Train an Agent

Train a PPO agent (default):

```bash
python train_agent.py \
    --algorithm PPO \
    --episode-length 96 \
    --total-timesteps 100000 \
    --checkpoint-freq 10000 \
    --eval-freq 5000
```

Train a SAC agent:

```bash
python train_agent.py \
    --algorithm SAC \
    --episode-length 96 \
    --total-timesteps 100000 \
    --checkpoint-freq 10000 \
    --eval-freq 5000
```

### 4. Evaluate Trained Agent

```bash
python evaluate_agent.py \
    --model ./models/best_model/best_model \
    --algorithm PPO \
    --num-episodes 5 \
    --output-dir ./evaluation_results
```

Or use a checkpoint:

```bash
python evaluate_agent.py \
    --model ./models/checkpoints/model_100000_steps \
    --algorithm PPO \
    --num-episodes 5
```

## Configuration

### Environment Parameters

- `battery_capacity_kwh`: Battery capacity in kWh (default: 100.0)
- `max_charge_rate_kw`: Maximum charging rate in kW (default: 50.0)
- `max_discharge_rate_kw`: Maximum discharging rate in kW (default: 50.0)
- `efficiency`: Battery charge/discharge efficiency 0-1 (default: 0.95)
- `episode_length`: Number of steps per episode (default: 96 = 1 day)
- `normalize_state`: Whether to normalize state features (default: True)

### Training Parameters

- `--algorithm`: RL algorithm to use (PPO or SAC)
- `--total-timesteps`: Total number of timesteps to train (default: 100000)
- `--checkpoint-freq`: Frequency of checkpoint saving in timesteps (default: 10000)
- `--eval-freq`: Frequency of evaluation in timesteps (default: 5000)

## Environment Details

### State Space

The observation space consists of:
- `total_consumption`: Total community consumption (normalized)
- `total_production`: Total community production (normalized)
- `surplus_production`: Surplus production available (normalized)
- `battery_soc`: Battery state of charge (0-1)

### Action Space

Continuous action space:
- Range: `[-max_discharge_rate_kw, max_charge_rate_kw]`
- Negative values = discharge battery
- Positive values = charge battery

### Reward Function

The reward is designed to minimize grid energy usage:
- `reward = -grid_usage`
- `grid_usage = total_consumption - own_coverage`
- `own_coverage = min(total_consumption, total_production + battery_discharge - battery_charge)`

Higher rewards indicate lower grid dependency.

## Data Format

The environment expects a CSV file with the following columns:
- `metering_timestamp`: Timestamp (15-minute intervals)
- `total_consumption`: Energy consumption in kWh
- `total_production`: Energy production in kWh
- `surplus_production`: Surplus production in kWh
- `own_coverage`: Energy covered by own production
- `community_share`: Energy shared within community

Data is automatically aggregated by timestamp across all meterpoints.

## Example Usage

```python
from battery_env import BatteryControlEnv

# Create environment
env = BatteryControlEnv(
    data_path="metering_data_last_year.csv",
    battery_capacity_kwh=100.0,
    max_charge_rate_kw=50.0,
    episode_length=96
)

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(96):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

## Notes

- Each timestep represents 15 minutes of real time
- The environment automatically handles battery constraints (SOC limits, charge/discharge rates)
- State normalization improves training stability
- Models are saved in `./models/` by default
  - `./models/checkpoints/` - Regular checkpoints during training
  - `./models/best_model/` - Best model based on evaluation
  - `./models/final_model_<algorithm>` - Final model after training
  - `./models/tensorboard/` - TensorBoard logs for monitoring
- Evaluation results and plots are saved in `./evaluation_results/` by default

