"""
Training script for battery control RL agent using Stable-Baselines3.
"""

import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from battery_env import BatteryControlEnv
from data_simulator import generate_synthetic_data, save_synthetic_data


def make_env(env_config, rank=0, seed=0):
    """Create and wrap environment for vectorization."""
    def _init():
        env = BatteryControlEnv(**env_config)
        env.seed(seed + rank)
        # Wrap with Monitor for statistics
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for battery control")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm to use (PPO or SAC)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="metering_data_last_year.csv",
        help="Path to metering data CSV file"
    )
    parser.add_argument(
        "--battery-capacity",
        type=float,
        default=100.0,
        help="Battery capacity in kWh"
    )
    parser.add_argument(
        "--max-charge-rate",
        type=float,
        default=50.0,
        help="Maximum charge rate in kW"
    )
    parser.add_argument(
        "--max-discharge-rate",
        type=float,
        default=50.0,
        help="Maximum discharge rate in kW"
    )
    parser.add_argument(
        "--efficiency",
        type=float,
        default=0.95,
        help="Battery efficiency (0-1)"
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=96,
        help="Episode length in steps (96 = 1 day)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Checkpoint frequency (timesteps)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (0-1). Rest is for testing."
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data for training (test will still use real data)"
    )
    parser.add_argument(
        "--synthetic-timesteps",
        type=int,
        default=20000,
        help="Number of timesteps to generate for synthetic training data"
    )
    parser.add_argument(
        "--continue-battery-state",
        action="store_true",
        help="Continue battery SOC across episodes (no reset between episodes). Default: True"
    )
    parser.add_argument(
        "--no-continue-battery-state",
        dest="continue_battery_state",
        action="store_false",
        help="Reset battery SOC each episode (disable continuous battery state)"
    )
    parser.set_defaults(continue_battery_state=True)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data if requested
    train_data_path = args.data_path
    if args.use_synthetic:
        synthetic_path = os.path.join(args.output_dir, "synthetic_train_data.csv")
        print(f"Generating synthetic training data ({args.synthetic_timesteps} timesteps)...")
        save_synthetic_data(
            output_path=synthetic_path,
            num_timesteps=args.synthetic_timesteps,
            data_path=args.data_path,  # Use real data stats
            seed=args.seed
        )
        train_data_path = synthetic_path
        print(f"Using synthetic data for training: {synthetic_path}")
    
    # Environment configuration for training
    train_env_config = {
        "data_path": train_data_path,
        "battery_capacity_kwh": args.battery_capacity,
        "max_charge_rate_kw": args.max_charge_rate,
        "max_discharge_rate_kw": args.max_discharge_rate,
        "efficiency": args.efficiency,
        "episode_length": args.episode_length,
        "normalize_state": True,
        "train_split": args.train_split,
        "mode": "train",  # Use training data split
        "continue_battery_state": args.continue_battery_state,  # Continue SOC across episodes
        "seed": args.seed,
    }
    
    # Environment configuration for evaluation (always use real data)
    eval_env_config = {
        "data_path": args.data_path,  # Always use real data for testing
        "battery_capacity_kwh": args.battery_capacity,
        "max_charge_rate_kw": args.max_charge_rate,
        "max_discharge_rate_kw": args.max_discharge_rate,
        "efficiency": args.efficiency,
        "episode_length": args.episode_length,
        "normalize_state": True,
        "train_split": args.train_split,
        "mode": "test",  # Use test data split (real data)
        "continue_battery_state": args.continue_battery_state,  # Continue SOC across episodes
        "seed": args.seed + 1000,
    }
    
    # Create vectorized environment for training
    env = DummyVecEnv([make_env(train_env_config, seed=args.seed)])
    
    # Create evaluation environment (uses test data)
    eval_env = DummyVecEnv([make_env(eval_env_config, seed=args.seed + 1000)])
    
    # TensorBoard logging (optional)
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = os.path.join(args.output_dir, "tensorboard")
        print(f"TensorBoard logging enabled: {tensorboard_log}")
    except ImportError:
        print("TensorBoard not installed. Training will continue without TensorBoard logging.")
        print("Install with: pip install tensorboard")
    
    # Algorithm configuration
    # CHANGED: Increased entropy coefficient from 0.01 to 0.05 to encourage more exploration
    # This helps the agent explore larger actions and find better policies
    # The higher entropy means the policy will be more stochastic during training,
    # allowing it to try different action magnitudes (not just 0.1-0.2 kW)
    if args.algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.999,  # INCREASED from 0.99 to 0.999: Values future rewards more
                # Higher gamma means the agent cares more about long-term consequences
                # This helps with temporal credit assignment - charging now helps reduce grid usage later
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.3,  # HIGH: Maximum exploration to encourage larger actions
                vf_coef=0.5,
                verbose=1,
                tensorboard_log=tensorboard_log,
                seed=args.seed,
                policy_kwargs=dict(
                    # Initialize policy with VERY large standard deviation
                    # Default log_std_init is usually -0.5 to -1.0 (std ≈ 0.6-0.37)
                    # Using 1.5 means std=exp(1.5)≈4.48, encouraging very large initial actions
                    # This forces the policy to explore the full action space from the start
                    log_std_init=1.5,
                    # Larger network for better capacity
                    net_arch=dict(pi=[256, 256], vf=[256, 256])
                ),
            )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="model",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.output_dir, "best_model"),
        log_path=os.path.join(args.output_dir, "logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    
    # Training
    print(f"Starting training with {args.algorithm}...")
    print(f"Training data: {'SYNTHETIC' if args.use_synthetic else 'REAL'}")
    print(f"Testing data: REAL (unseen during training)")
    print(f"Training for {args.total_timesteps} timesteps")
    print(f"Battery state: {'CONTINUOUS across episodes' if args.continue_battery_state else 'RESETS each episode'}")
    print(f"Models will be saved to: {args.output_dir}")
    if not args.use_synthetic:
        print(f"Train/test split: {args.train_split:.0%} train, {1-args.train_split:.0%} test")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"final_model_{args.algorithm}")
    model.save(final_model_path)
    print(f"\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model', 'best_model')}")


if __name__ == "__main__":
    main()
