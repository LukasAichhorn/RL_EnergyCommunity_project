"""
Evaluation and visualization script for trained battery control agent.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from battery_env import BatteryControlEnv


def make_env(env_config, rank=0, seed=0):
    """Create and wrap environment for vectorization."""
    def _init():
        env = BatteryControlEnv(**env_config)
        env.seed(seed + rank)
        return env
    return _init


def evaluate_agent(
    model_path: str,
    algorithm: str,
    env_config: dict,
    num_episodes: int = 5,
    render: bool = False
):
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to the model file
        algorithm: Algorithm name ("PPO" or "SAC")
        env_config: Environment configuration
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        
    Returns:
        Dictionary with evaluation results
    """
    # Create environment
    env = DummyVecEnv([make_env(env_config, seed=42)])
    
    # Load trained agent
    if algorithm == "PPO":
        model = PPO.load(model_path)
    else:  # SAC
        model = SAC.load(model_path)
    
    # Evaluation metrics
    all_episode_rewards = []
    all_episode_lengths = []
    all_battery_socs = []
    all_grid_usage = []
    all_consumption = []
    all_production = []
    all_actions = []
    
    print(f"Evaluating agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_battery_socs = []
        episode_grid_usage = []
        episode_consumption = []
        episode_production = []
        episode_actions = []
        
        # Get initial battery SOC from unwrapped env
        unwrapped_env = env.envs[0]
        episode_battery_socs.append(unwrapped_env.battery_soc)
        
        done = False
        
        while not done:
            # Get action from agent
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action[0][0] if isinstance(action, np.ndarray) else action[0])
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Extract info from unwrapped environment
            unwrapped_env = env.envs[0]
            episode_battery_socs.append(unwrapped_env.battery_soc)
            
            # Get info from last step
            if isinstance(info, list) and len(info) > 0:
                step_info = info[0]
                episode_grid_usage.append(step_info.get('grid_usage', 0))
                episode_consumption.append(step_info.get('total_consumption', 0))
                episode_production.append(step_info.get('total_production', 0))
            
            if render:
                unwrapped_env.render()
        
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        all_battery_socs.append(episode_battery_socs)
        all_grid_usage.append(episode_grid_usage)
        all_consumption.append(episode_consumption)
        all_production.append(episode_production)
        all_actions.append(episode_actions)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(all_episode_rewards),
        'std_reward': np.std(all_episode_rewards),
        'mean_length': np.mean(all_episode_lengths),
        'mean_grid_usage': np.mean([np.mean(ep) for ep in all_grid_usage if len(ep) > 0]),
        'total_grid_usage': np.mean([np.sum(ep) for ep in all_grid_usage if len(ep) > 0]),
        'battery_socs': all_battery_socs,
        'grid_usage': all_grid_usage,
        'consumption': all_consumption,
        'production': all_production,
        'actions': all_actions,
    }
    
    return results


def visualize_results(results: dict, output_dir: str = "./evaluation_results"):
    """Visualize evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create comprehensive plot showing agent actions and battery state
    num_episodes = len(results['battery_socs'])
    fig, axes = plt.subplots(num_episodes, 2, figsize=(16, 4 * num_episodes))
    
    # Handle single episode case
    if num_episodes == 1:
        axes = axes.reshape(1, -1)
    
    for episode_idx in range(num_episodes):
        # Left plot: Battery SOC and Actions
        ax1 = axes[episode_idx, 0]
        
        # Plot battery SOC
        socs = results['battery_socs'][episode_idx]
        ax1_twin = ax1.twinx()
        
        ax1.plot(socs, 'b-', linewidth=2, label='Battery SOC', alpha=0.8)
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Battery SOC (0-1)', color='b', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Episode {episode_idx + 1}: Battery State & Actions', fontsize=12, fontweight='bold')
        
        # Plot actions on twin axis
        if episode_idx < len(results['actions']) and len(results['actions'][episode_idx]) > 0:
            actions = results['actions'][episode_idx]
            # Pad actions to match SOC length if needed
            if len(actions) < len(socs):
                actions = list(actions) + [0] * (len(socs) - len(actions))
            elif len(actions) > len(socs):
                actions = actions[:len(socs)]
            
            ax1_twin.plot(actions, 'r-', linewidth=1.5, label='Action (kW)', alpha=0.7)
            ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax1_twin.fill_between(range(len(actions)), 0, actions, 
                                  where=[a > 0 for a in actions], 
                                  color='green', alpha=0.2, label='Charging')
            ax1_twin.fill_between(range(len(actions)), 0, actions, 
                                  where=[a < 0 for a in actions], 
                                  color='red', alpha=0.2, label='Discharging')
            ax1_twin.set_ylabel('Action (kW)', color='r', fontsize=10)
            ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # Right plot: Grid usage, Consumption, Production
        ax2 = axes[episode_idx, 1]
        
        if episode_idx < len(results['grid_usage']) and len(results['grid_usage'][episode_idx]) > 0:
            grid_usage = results['grid_usage'][episode_idx]
            ax2.plot(grid_usage, 'orange', linewidth=2, label='Grid Usage', alpha=0.8)
        
        if episode_idx < len(results['consumption']) and len(results['consumption'][episode_idx]) > 0:
            consumption = results['consumption'][episode_idx]
            ax2.plot(consumption, 'purple', linewidth=1.5, label='Consumption', alpha=0.6, linestyle='--')
        
        if episode_idx < len(results['production']) and len(results['production'][episode_idx]) > 0:
            production = results['production'][episode_idx]
            ax2.plot(production, 'green', linewidth=1.5, label='Production', alpha=0.6, linestyle='--')
        
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel('Energy (kWh)', fontsize=10)
        ax2.set_title(f'Episode {episode_idx + 1}: Energy Flows', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "comprehensive_plots.png", dpi=150, bbox_inches='tight')
    print(f"Comprehensive plots saved to {output_path / 'comprehensive_plots.png'}")
    
    # Original 2x2 plot for quick overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Battery SOC
    ax = axes[0, 0]
    for i, socs in enumerate(results['battery_socs']):  # Plot all episodes
        ax.plot(socs, label=f'Episode {i+1}', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Battery SOC')
    ax.set_title('Battery State of Charge Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grid usage
    ax = axes[0, 1]
    for i, grid_usage in enumerate(results['grid_usage']):  # Plot all episodes
        if len(grid_usage) > 0:
            ax.plot(grid_usage, label=f'Episode {i+1}', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Grid Usage (kWh)')
    ax.set_title('Grid Energy Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Consumption vs Production
    ax = axes[1, 0]
    if results['consumption'] and len(results['consumption'][0]) > 0:
        ax.plot(results['consumption'][0], label='Consumption', alpha=0.7)
        ax.plot(results['production'][0], label='Production', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Consumption vs Production (Episode 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Actions (charge/discharge)
    ax = axes[1, 1]
    for i, actions in enumerate(results['actions']):  # Plot all episodes
        if len(actions) > 0:
            ax.plot(actions, label=f'Episode {i+1}', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No action')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action (kW)')
    ax.set_title('Battery Actions (Charge/Discharge)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "evaluation_plots.png", dpi=150, bbox_inches='tight')
    print(f"Overview plots saved to {output_path / 'evaluation_plots.png'}")
    
    # Create dedicated SOC plot for full evaluation period
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Concatenate all episodes for continuous SOC plot
    all_socs = []
    all_timesteps = []
    current_timestep = 0
    
    for episode_idx, socs in enumerate(results['battery_socs']):
        timesteps = range(current_timestep, current_timestep + len(socs))
        all_socs.extend(socs)
        all_timesteps.extend(timesteps)
        current_timestep += len(socs)
    
    ax.plot(all_timesteps, all_socs, 'b-', linewidth=2, label='Battery SOC', alpha=0.8)
    ax.fill_between(all_timesteps, 0, all_socs, alpha=0.3, color='blue')
    ax.set_xlabel('Time Step (15-minute intervals)', fontsize=12)
    ax.set_ylabel('Battery State of Charge (0-1)', fontsize=12)
    ax.set_title('Battery SOC Over Full Evaluation Period', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add episode boundaries
    current_timestep = 0
    for episode_idx in range(len(results['battery_socs'])):
        if episode_idx > 0:
            ax.axvline(x=current_timestep, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        current_timestep += len(results['battery_socs'][episode_idx])
    
    plt.tight_layout()
    plt.savefig(output_path / "battery_soc_full_period.png", dpi=150, bbox_inches='tight')
    print(f"Battery SOC plot saved to {output_path / 'battery_soc_full_period.png'}")
    plt.close()
    
    # Create grid usage comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top plot: Grid usage comparison (with vs without battery)
    ax1 = axes[0]
    
    # Calculate baseline grid usage (without battery - just consumption - production)
    all_grid_with_battery = []
    all_grid_without_battery = []
    all_timesteps_grid = []
    current_timestep = 0
    
    for episode_idx in range(len(results['grid_usage'])):
        grid_usage = results['grid_usage'][episode_idx]
        consumption = results['consumption'][episode_idx] if episode_idx < len(results['consumption']) else []
        production = results['production'][episode_idx] if episode_idx < len(results['production']) else []
        
        # Calculate baseline (without battery)
        baseline_grid = []
        for i in range(len(grid_usage)):
            if i < len(consumption) and i < len(production):
                # Without battery: grid usage = max(0, consumption - production)
                baseline = max(0, consumption[i] - production[i])
                baseline_grid.append(baseline)
        
        # Pad to match lengths
        min_len = min(len(grid_usage), len(baseline_grid))
        grid_usage = grid_usage[:min_len]
        baseline_grid = baseline_grid[:min_len]
        
        timesteps = range(current_timestep, current_timestep + len(grid_usage))
        all_grid_with_battery.extend(grid_usage)
        all_grid_without_battery.extend(baseline_grid)
        all_timesteps_grid.extend(timesteps)
        current_timestep += len(grid_usage)
    
    ax1.plot(all_timesteps_grid, all_grid_without_battery, 'r-', linewidth=2, 
             label='Grid Usage (Without Battery)', alpha=0.7)
    ax1.plot(all_timesteps_grid, all_grid_with_battery, 'g-', linewidth=2, 
             label='Grid Usage (With Agent)', alpha=0.8)
    ax1.fill_between(all_timesteps_grid, all_grid_with_battery, all_grid_without_battery, 
                     where=[w < wo for w, wo in zip(all_grid_with_battery, all_grid_without_battery)],
                     alpha=0.3, color='green', label='Energy Saved')
    ax1.set_xlabel('Time Step (15-minute intervals)', fontsize=12)
    ax1.set_ylabel('Grid Usage (kWh)', fontsize=12)
    ax1.set_title('Grid Usage Comparison: With vs Without Battery Agent', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Cumulative grid usage savings
    ax2 = axes[1]
    
    cumulative_without = np.cumsum(all_grid_without_battery)
    cumulative_with = np.cumsum(all_grid_with_battery)
    cumulative_savings = cumulative_without - cumulative_with
    
    ax2.plot(all_timesteps_grid, cumulative_without, 'r-', linewidth=2, 
             label='Cumulative Grid Usage (Without Battery)', alpha=0.7)
    ax2.plot(all_timesteps_grid, cumulative_with, 'g-', linewidth=2, 
             label='Cumulative Grid Usage (With Agent)', alpha=0.8)
    ax2.fill_between(all_timesteps_grid, cumulative_with, cumulative_without, 
                     alpha=0.3, color='green', label='Cumulative Energy Saved')
    ax2.set_xlabel('Time Step (15-minute intervals)', fontsize=12)
    ax2.set_ylabel('Cumulative Grid Usage (kWh)', fontsize=12)
    ax2.set_title('Cumulative Grid Usage Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "grid_usage_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Grid usage comparison saved to {output_path / 'grid_usage_comparison.png'}")
    plt.close()
    
    # Create comprehensive validation plot with all episodes in one continuous timeline
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    
    # Concatenate all episodes for continuous plots
    all_socs_continuous = []
    all_actions_continuous = []
    all_grid_usage_continuous = []
    all_consumption_continuous = []
    all_production_continuous = []
    all_timesteps_continuous = []
    episode_boundaries = [0]  # Track where episodes start
    
    current_timestep = 0
    for episode_idx in range(len(results['battery_socs'])):
        socs = results['battery_socs'][episode_idx]
        actions = results['actions'][episode_idx] if episode_idx < len(results['actions']) else []
        grid_usage = results['grid_usage'][episode_idx] if episode_idx < len(results['grid_usage']) else []
        consumption = results['consumption'][episode_idx] if episode_idx < len(results['consumption']) else []
        production = results['production'][episode_idx] if episode_idx < len(results['production']) else []
        
        # Pad actions to match SOC length (SOC has one extra point at start)
        if len(actions) < len(socs) - 1:
            actions = list(actions) + [0] * (len(socs) - 1 - len(actions))
        elif len(actions) > len(socs) - 1:
            actions = actions[:len(socs) - 1]
        
        # Pad other arrays to match SOC length
        min_len = min(len(socs), len(grid_usage) + 1, len(consumption) + 1, len(production) + 1)
        socs = socs[:min_len]
        if len(grid_usage) < min_len - 1:
            grid_usage = list(grid_usage) + [0] * (min_len - 1 - len(grid_usage))
        if len(consumption) < min_len - 1:
            consumption = list(consumption) + [0] * (min_len - 1 - len(consumption))
        if len(production) < min_len - 1:
            production = list(production) + [0] * (min_len - 1 - len(production))
        
        timesteps = range(current_timestep, current_timestep + len(socs))
        all_socs_continuous.extend(socs)
        all_actions_continuous.extend(actions + [0])  # Add one action for last SOC point
        all_grid_usage_continuous.extend([0] + grid_usage[:min_len-1])  # Add 0 at start
        all_consumption_continuous.extend([0] + consumption[:min_len-1])
        all_production_continuous.extend([0] + production[:min_len-1])
        all_timesteps_continuous.extend(timesteps)
        
        current_timestep += len(socs)
        episode_boundaries.append(current_timestep)
    
    # Plot 1: Battery SOC
    ax = axes[0]
    ax.plot(all_timesteps_continuous, all_socs_continuous, 'b-', linewidth=2, label='Battery SOC', alpha=0.8)
    ax.fill_between(all_timesteps_continuous, 0, all_socs_continuous, alpha=0.3, color='blue')
    # Mark episode boundaries
    for boundary in episode_boundaries[1:-1]:  # Skip first and last
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Episode boundary' if boundary == episode_boundaries[1] else '')
    ax.set_ylabel('Battery SOC (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('All Validation Episodes - Continuous Timeline', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticklabels([])  # Remove x-axis labels for this subplot
    
    # Plot 2: Actions
    ax = axes[1]
    ax.plot(all_timesteps_continuous, all_actions_continuous, 'purple', linewidth=1.5, label='Action (kW)', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.fill_between(all_timesteps_continuous, 0, all_actions_continuous, 
                     where=[a > 0 for a in all_actions_continuous], 
                     color='green', alpha=0.2, label='Charging')
    ax.fill_between(all_timesteps_continuous, 0, all_actions_continuous, 
                     where=[a < 0 for a in all_actions_continuous], 
                     color='red', alpha=0.2, label='Discharging')
    for boundary in episode_boundaries[1:-1]:
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Action (kW)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticklabels([])
    
    # Plot 3: Energy flows (Consumption, Production)
    ax = axes[2]
    ax.plot(all_timesteps_continuous, all_consumption_continuous, 'purple', linewidth=2, 
            label='Consumption', alpha=0.8, linestyle='-')
    ax.plot(all_timesteps_continuous, all_production_continuous, 'green', linewidth=2, 
            label='Production', alpha=0.8, linestyle='-')
    for boundary in episode_boundaries[1:-1]:
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticklabels([])
    
    # Plot 4: Grid Usage
    ax = axes[3]
    ax.plot(all_timesteps_continuous, all_grid_usage_continuous, 'orange', linewidth=2, 
            label='Grid Usage', alpha=0.8)
    ax.fill_between(all_timesteps_continuous, 0, all_grid_usage_continuous, 
                     alpha=0.3, color='orange')
    for boundary in episode_boundaries[1:-1]:
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Time Step (15-minute intervals)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grid Usage (kWh)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add episode number annotations at boundaries
    for i, boundary in enumerate(episode_boundaries[:-1]):
        if i < len(episode_boundaries) - 1:
            mid_point = (boundary + episode_boundaries[i+1]) / 2
            axes[0].text(mid_point, 0.95, f'Ep {i+1}', ha='center', va='top', 
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / "all_episodes_continuous.png", dpi=150, bbox_inches='tight')
    print(f"All episodes continuous plot saved to {output_path / 'all_episodes_continuous.png'}")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.2f}")
    if results['mean_grid_usage'] > 0:
        print(f"Mean Grid Usage per Step: {results['mean_grid_usage']:.2f} kWh")
        print(f"Total Grid Usage per Episode: {results['total_grid_usage']:.2f} kWh")
    
    # Calculate and print grid usage savings
    if len(all_grid_with_battery) > 0 and len(all_grid_without_battery) > 0:
        total_with_battery = sum(all_grid_with_battery)
        total_without_battery = sum(all_grid_without_battery)
        total_savings = total_without_battery - total_with_battery
        savings_percentage = (total_savings / total_without_battery * 100) if total_without_battery > 0 else 0
        
        print(f"\nGrid Usage Comparison:")
        print(f"  Total Grid Usage (Without Battery): {total_without_battery:.2f} kWh")
        print(f"  Total Grid Usage (With Agent): {total_with_battery:.2f} kWh")
        print(f"  Total Energy Saved: {total_savings:.2f} kWh")
        print(f"  Reduction: {savings_percentage:.2f}%")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained battery control agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="Algorithm used for training"
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
        help="Episode length in steps"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes during evaluation"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Fraction of data used for training (for normalization stats). Use 1.0 to evaluate on full dataset."
    )
    parser.add_argument(
        "--continue-battery-state",
        action="store_true",
        help="Continue battery SOC across episodes (no reset between episodes)"
    )
    
    args = parser.parse_args()
    
    # Environment configuration (must match training config)
    # If train_split=1.0, use "train" mode to evaluate on full dataset
    # Otherwise use "test" mode for unseen data
    eval_mode = "train" if args.train_split >= 1.0 else "test"
    
    env_config = {
        "data_path": args.data_path,
        "battery_capacity_kwh": args.battery_capacity,
        "max_charge_rate_kw": args.max_charge_rate,
        "max_discharge_rate_kw": args.max_discharge_rate,
        "efficiency": args.efficiency,
        "episode_length": args.episode_length,
        "normalize_state": True,
        "train_split": args.train_split,
        "mode": eval_mode,  # "train" for full dataset, "test" for test split
        "continue_battery_state": args.continue_battery_state,  # Continue SOC across episodes
    }
    
    # Evaluate agent
    results = evaluate_agent(
        model_path=args.model,
        algorithm=args.algorithm,
        env_config=env_config,
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # Visualize results
    visualize_results(results, args.output_dir)


if __name__ == "__main__":
    main()
