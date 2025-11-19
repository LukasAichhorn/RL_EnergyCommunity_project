# Training on Modal GPU

This guide explains how to train your battery RL agent on Modal's cloud GPUs.

## Setup (One-time)

1. **Install Modal:**
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal:**
   ```bash
   modal setup
   ```
   
   This will open a browser window to create a Modal account (if you don't have one) and authenticate.

## Training on Modal

### Basic Training

Train with default parameters (100,000 timesteps on A10G GPU):

```bash
modal run train_agent_modal.py
```

### Custom Training

Train with custom parameters:

```bash
# Train for 200,000 timesteps with PPO
modal run train_agent_modal.py --total-timesteps 200000

# Train with SAC algorithm
modal run train_agent_modal.py --algorithm SAC --total-timesteps 150000

# Custom battery parameters
modal run train_agent_modal.py \
    --total-timesteps 300000 \
    --battery-capacity 150.0 \
    --max-charge-rate 75.0 \
    --checkpoint-freq 15000
```

### Available Parameters

All the same parameters from `train_agent.py` are supported:

- `--algorithm` - RL algorithm (PPO or SAC), default: PPO
- `--total-timesteps` - Training timesteps, default: 100000
- `--battery-capacity` - Battery capacity in kWh, default: 100.0
- `--max-charge-rate` - Max charge rate in kW, default: 50.0
- `--max-discharge-rate` - Max discharge rate in kW, default: 50.0
- `--efficiency` - Battery efficiency, default: 0.95
- `--episode-length` - Episode length in steps, default: 96 (1 day)
- `--checkpoint-freq` - Checkpoint frequency, default: 10000
- `--eval-freq` - Evaluation frequency, default: 5000
- `--seed` - Random seed, default: 42
- `--train-split` - Train/test split ratio, default: 0.8
- `--continue-battery-state` - Continue battery SOC across episodes, default: True
- `--local-output-dir` - Local directory for downloaded models, default: ./models_modal

## What Happens

1. **Upload**: Your local `metering_data_last_year.csv` is uploaded to Modal
2. **Train**: Training runs on Modal's A10G GPU (much faster than local)
3. **Download**: Trained models are automatically downloaded to `./models_modal/`

## Training Progress

During training, you'll see:
- Real-time progress bar
- Episode rewards
- Evaluation results
- TensorBoard logs (saved in downloaded models)

## After Training

Once training completes, evaluate your model:

```bash
python evaluate_agent.py --model ./models_modal/best_model/best_model.zip
```

View TensorBoard logs:

```bash
tensorboard --logdir ./models_modal/tensorboard
```

## GPU Specifications

- **GPU**: NVIDIA A10G
- **Memory**: 24GB VRAM
- **Cost**: ~$1-2 per hour (Modal pricing)
- **Timeout**: 2 hours max per training run

## Troubleshooting

### "Module modal not found"
```bash
pip install modal
modal setup
```

### "Data file not found"
Make sure `metering_data_last_year.csv` exists in your current directory.

### Training timeout
For very long training runs (>2 hours), split into multiple sessions:
```bash
# First session
modal run train_agent_modal.py --total-timesteps 100000

# Continue from checkpoint (future feature)
modal run train_agent_modal.py --total-timesteps 200000 --load-model ./models_modal/checkpoints/model_100000_steps.zip
```

## Comparison: Local vs Modal

| Aspect | Local (CPU) | Modal (A10G GPU) |
|--------|-------------|------------------|
| Speed | ~2-3 hours for 100k steps | ~15-20 minutes for 100k steps |
| Cost | Electricity | $1-2/hour |
| Setup | None | One-time auth |
| Scalability | Limited | Infinite |

## Benefits of Modal Training

1. **10x faster** - GPU acceleration
2. **No local resources** - Your laptop stays free
3. **Reproducible** - Same environment every time
4. **Scalable** - Train multiple models in parallel
5. **Automatic** - Upload, train, download all handled for you



