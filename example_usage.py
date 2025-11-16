"""
Example script showing basic usage of the battery control environment.
This demonstrates how to interact with the environment without training.
"""

from battery_env import BatteryControlEnv
import numpy as np


def simple_control_policy(obs, env):
    """
    Simple heuristic policy: charge when production > consumption, discharge otherwise.
    This is just for demonstration - the RL agent will learn a better policy.
    """
    consumption = obs[0]
    production = obs[1]
    battery_soc = obs[3]
    
    # Simple rule: if production > consumption, charge; otherwise discharge
    if production > consumption and battery_soc < 0.9:
        # Charge at moderate rate
        action = env.max_charge_rate_kw * 0.5
    elif consumption > production and battery_soc > 0.1:
        # Discharge to cover consumption
        action = -min(env.max_discharge_rate_kw, (consumption - production) * 2)
    else:
        # Do nothing
        action = 0.0
    
    return np.array([action], dtype=np.float32)


def main():
    # Create environment
    print("Creating battery control environment...")
    env = BatteryControlEnv(
        data_path="metering_data_last_year.csv",
        battery_capacity_kwh=100.0,
        max_charge_rate_kw=50.0,
        max_discharge_rate_kw=50.0,
        efficiency=0.95,
        episode_length=96,  # 1 day
        normalize_state=True,
        seed=42
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # Run a few episodes with simple policy
    num_episodes = 3
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_grid_usage = []
        episode_battery_soc = [info['battery_soc']]
        
        print(f"Episode {episode + 1}")
        print(f"  Initial battery SOC: {info['battery_soc']:.2%}")
        
        step = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from simple policy
            action = simple_control_policy(obs, env)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_grid_usage.append(info['grid_usage'])
            episode_battery_soc.append(info['battery_soc'])
            step += 1
            
            # Print progress every 24 steps (6 hours)
            if step % 24 == 0:
                print(f"  Step {step}: SOC={info['battery_soc']:.2%}, "
                      f"Grid usage={info['grid_usage']:.2f} kWh, "
                      f"Reward={reward:.2f}")
        
        print(f"  Episode complete:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Average grid usage: {np.mean(episode_grid_usage):.2f} kWh/step")
        print(f"    Total grid usage: {np.sum(episode_grid_usage):.2f} kWh")
        print(f"    Final battery SOC: {episode_battery_soc[-1]:.2%}")
        print()


if __name__ == "__main__":
    main()

