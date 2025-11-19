"""
Modal-based training script for battery control RL agent.
Runs training on Modal's A10G GPU and downloads trained models locally.

Usage:
    modal run train_agent_modal.py --total-timesteps 200000
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("battery-rl-training")

# Create container image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    )
)

# Create Modal Volume for data and models
data_volume = modal.Volume.from_name("battery-data", create_if_missing=True)
models_volume = modal.Volume.from_name("battery-models", create_if_missing=True)


@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/data": data_volume,
        "/models": models_volume,
    },
    timeout=7200,  # 2 hours
)
def train_agent(
    algorithm: str = "PPO",
    battery_capacity: float = 100.0,
    max_charge_rate: float = 50.0,
    max_discharge_rate: float = 50.0,
    efficiency: float = 0.95,
    episode_length: int = 96,
    total_timesteps: int = 100000,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    seed: int = 42,
    train_split: float = 0.8,
    continue_battery_state: bool = True,
    battery_env_code: str = "",
    data_processor_code: str = "",
):
    """Train battery control RL agent on Modal GPU."""
    import os
    import sys
    import numpy as np
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Write the Python modules to files
    with open("/root/battery_env.py", "w") as f:
        f.write(battery_env_code)
    
    with open("/root/data_processor.py", "w") as f:
        f.write(data_processor_code)
    
    # Add current directory to Python path
    sys.path.insert(0, "/root")
    
    from battery_env import BatteryControlEnv
    
    print("=" * 80)
    print("TRAINING ON MODAL GPU (A10G)")
    print("=" * 80)
    
    # Learning rate decay callback
    class LearningRateDecayCallback(BaseCallback):
        """Callback to decay learning rate and entropy coefficient over time."""
        def __init__(self, initial_lr, final_lr, initial_ent_coef, final_ent_coef, total_timesteps, verbose=0):
            super().__init__(verbose)
            self.initial_lr = initial_lr
            self.final_lr = final_lr
            self.initial_ent_coef = initial_ent_coef
            self.final_ent_coef = final_ent_coef
            self.total_timesteps = total_timesteps
        
        def _on_step(self) -> bool:
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            current_lr = self.initial_lr * (1 - progress) + self.final_lr * progress
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = current_lr
            if hasattr(self.model, 'ent_coef'):
                current_ent_coef = self.initial_ent_coef * (1 - progress) + self.final_ent_coef * progress
                self.model.ent_coef = current_ent_coef
            return True
    
    def make_env(env_config, rank=0, seed=0):
        """Create and wrap environment for vectorization."""
        def _init():
            env = BatteryControlEnv(**env_config)
            env.seed(seed + rank)
            env = Monitor(env, filename=None, allow_early_resets=True)
            return env
        return _init
    
    # Data path in Modal volume
    data_path = "/data/metering_data_last_year.csv"
    output_dir = "/models/current_training"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    
    # Environment configuration for training
    train_env_config = {
        "data_path": data_path,
        "battery_capacity_kwh": battery_capacity,
        "max_charge_rate_kw": max_charge_rate,
        "max_discharge_rate_kw": max_discharge_rate,
        "efficiency": efficiency,
        "episode_length": episode_length,
        "normalize_state": True,
        "train_split": train_split,
        "mode": "train",
        "continue_battery_state": continue_battery_state,
        "seed": seed,
    }
    
    # Environment configuration for evaluation
    eval_env_config = {
        "data_path": data_path,
        "battery_capacity_kwh": battery_capacity,
        "max_charge_rate_kw": max_charge_rate,
        "max_discharge_rate_kw": max_discharge_rate,
        "efficiency": efficiency,
        "episode_length": episode_length,
        "normalize_state": True,
        "train_split": train_split,
        "mode": "test",
        "continue_battery_state": continue_battery_state,
        "seed": seed + 1000,
    }
    
    # Create vectorized environments
    env = DummyVecEnv([make_env(train_env_config, seed=seed)])
    eval_env = DummyVecEnv([make_env(eval_env_config, seed=seed + 1000)])
    
    # TensorBoard logging
    tensorboard_log = os.path.join(output_dir, "tensorboard")
    
    # Create model
    print(f"\nCreating {algorithm} model...")
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=seed,
            policy_kwargs=dict(
                log_std_init=1.0,
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
            seed=seed,
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="model",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    
    lr_decay_callback = LearningRateDecayCallback(
        initial_lr=3e-4,
        final_lr=1e-5,
        initial_ent_coef=0.1 if algorithm == "PPO" else 0.0,
        final_ent_coef=0.01 if algorithm == "PPO" else 0.0,
        total_timesteps=total_timesteps
    )
    
    # Training info
    print("\n" + "=" * 80)
    print(f"Starting training with {algorithm}...")
    print(f"Training data: REAL battery data")
    print(f"Training for {total_timesteps} timesteps")
    print(f"Battery state: {'CONTINUOUS' if continue_battery_state else 'RESETS'} across episodes")
    print(f"GPU: A10G")
    print(f"Models will be saved to: {output_dir}")
    print(f"Learning rate: 3e-4 → 1e-5 (decay over training)")
    if algorithm == "PPO":
        print(f"Entropy coefficient: 0.1 → 0.01 (decay over training)")
    print(f"Train/test split: {train_split:.0%} train, {1-train_split:.0%} test")
    print("=" * 80 + "\n")
    
    # Train (disable progress bar for Modal environment)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, lr_decay_callback],
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"final_model_{algorithm}")
    model.save(final_model_path)
    
    # Commit changes to Modal volumes
    data_volume.commit()
    models_volume.commit()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model', 'best_model.zip')}")
    print("=" * 80)
    
    return {
        "status": "success",
        "output_dir": output_dir,
        "total_timesteps": total_timesteps,
    }


@app.local_entrypoint()
def main(
    algorithm: str = "PPO",
    battery_capacity: float = 100.0,
    max_charge_rate: float = 50.0,
    max_discharge_rate: float = 50.0,
    efficiency: float = 0.95,
    episode_length: int = 96,
    total_timesteps: int = 100000,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    seed: int = 42,
    train_split: float = 0.8,
    continue_battery_state: bool = True,
    local_output_dir: str = "./models_modal",
):
    """
    Local entrypoint: uploads data, runs training on Modal, downloads models.
    
    Example usage:
        modal run train_agent_modal.py --total-timesteps 200000
        modal run train_agent_modal.py --algorithm SAC --total-timesteps 150000
    """
    import os
    import shutil
    from pathlib import Path
    
    print("\n" + "=" * 80)
    print("BATTERY RL TRAINING WITH MODAL")
    print("=" * 80)
    
    # Step 1: Upload data file and Python modules
    print("\n[1/3] Uploading data file and code to Modal...")
    local_data_path = Path("metering_data_last_year.csv")
    
    if not local_data_path.exists():
        raise FileNotFoundError(f"Data file not found: {local_data_path}")
    
    # Read Python module files
    battery_env_code = Path("battery_env.py").read_text()
    data_processor_code = Path("data_processor.py").read_text()
    
    # Upload data file to Modal volume (force=True to overwrite if exists)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_data_path), "metering_data_last_year.csv")
    
    print(f"   ✓ Uploaded {local_data_path} to Modal volume")
    print(f"   ✓ Prepared Python modules for Modal")
    
    # Step 2: Run training on Modal GPU
    print("\n[2/3] Starting training on Modal A10G GPU...")
    print(f"   Training for {total_timesteps} timesteps with {algorithm}...")
    
    result = train_agent.remote(
        algorithm=algorithm,
        battery_capacity=battery_capacity,
        max_charge_rate=max_charge_rate,
        max_discharge_rate=max_discharge_rate,
        efficiency=efficiency,
        episode_length=episode_length,
        total_timesteps=total_timesteps,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        seed=seed,
        train_split=train_split,
        continue_battery_state=continue_battery_state,
        battery_env_code=battery_env_code,
        data_processor_code=data_processor_code,
    )
    
    print(f"   ✓ Training completed: {result['status']}")
    
    # Step 3: Download trained models from Modal volume
    print("\n[3/3] Downloading trained models from Modal...")
    
    # Create local output directory
    local_output_path = Path(local_output_dir)
    local_output_path.mkdir(parents=True, exist_ok=True)
    
    # Download models from Modal volume using recursive listing
    remote_base_dir = "current_training"
    
    try:
        # List all files recursively from the models volume
        files_downloaded = 0
        for remote_file in models_volume.listdir(remote_base_dir, recursive=True):
            # remote_file.path is already the full path from volume root
            remote_path = remote_file.path
            # Remove the base dir prefix for local path
            relative_path = remote_file.path.replace(f"{remote_base_dir}/", "")
            local_path = local_output_path / relative_path
            
            # Create parent directories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            try:
                # read_file returns a generator, need to concatenate chunks
                chunks = []
                for chunk in models_volume.read_file(remote_path):
                    chunks.append(chunk)
                file_bytes = b''.join(chunks)
                local_path.write_bytes(file_bytes)
                print(f"   ✓ Downloaded {remote_file.path}")
                files_downloaded += 1
            except Exception as e:
                # Skip directories silently
                error_msg = str(e).lower()
                if "is a directory" not in error_msg and "not a file" not in error_msg and "not a regular file" not in error_msg:
                    print(f"   ⚠ Could not download {remote_file.path}: {e}")
        
        if files_downloaded == 0:
            print(f"   ⚠ No files downloaded - volume may be empty")
    except Exception as e:
        print(f"   ⚠ Error listing files: {e}")
        print(f"   Note: Models are saved in Modal volume 'battery-models' under 'current_training/'")
    
    print(f"\n   ✓ All models downloaded to: {local_output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Local models directory: {local_output_path}")
    print(f"Best model: {local_output_path / 'best_model' / 'best_model.zip'}")
    print(f"Final model: {local_output_path / f'final_model_{algorithm}.zip'}")
    print("\nTo evaluate the model:")
    print(f"  python evaluate_agent.py --model {local_output_path / 'best_model' / 'best_model.zip'}")
    print("=" * 80 + "\n")

