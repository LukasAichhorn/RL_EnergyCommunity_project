"""
Gymnasium environment for community battery control using RL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from data_processor import load_and_preprocess_data, get_data_statistics


class BatteryControlEnv(gym.Env):
    """
    Gymnasium environment for controlling a community battery.
    
    The agent learns to charge/discharge the battery to minimize grid energy usage.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: str = "metering_data_last_year.csv",
        battery_capacity_kwh: float = 100.0,
        max_charge_rate_kw: float = 50.0,
        max_discharge_rate_kw: float = 50.0,
        efficiency: float = 0.95,
        initial_soc: float = 0.5,
        episode_length: int = 96,  # 1 day = 96 steps (15 min intervals)
        normalize_state: bool = True,
        train_split: float = 0.8,  # Fraction of data to use for training (0-1)
        mode: str = "train",  # "train" or "test" - which split to use
        continue_battery_state: bool = False,  # If True, battery SOC continues across episodes
        seed: Optional[int] = None,
    ):
        """
        Initialize the battery control environment.
        
        Args:
            data_path: Path to the metering data CSV file
            battery_capacity_kwh: Battery capacity in kWh
            max_charge_rate_kw: Maximum charging rate in kW
            max_discharge_rate_kw: Maximum discharging rate in kW
            efficiency: Battery charge/discharge efficiency (0-1)
            initial_soc: Initial state of charge (0-1)
            episode_length: Maximum number of steps per episode
            normalize_state: Whether to normalize state features
            train_split: Fraction of data for training (0-1). Rest is for testing.
            mode: "train" or "test" - which data split to use
            continue_battery_state: If True, battery SOC continues across episodes (useful for evaluation)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Load data - can be real or synthetic
        is_synthetic = "synthetic" in data_path.lower()
        
        if mode == "train" and is_synthetic:
            # Load synthetic training data (always normalized by participants)
            all_data = load_and_preprocess_data(data_path, normalize_by_participants=True)
            self.raw_data = all_data.copy()
            self.stats = get_data_statistics(self.raw_data)
            print(f"Using SYNTHETIC TRAIN data: {len(self.raw_data)} timesteps ({len(self.raw_data)/96:.1f} days)")
        else:
            # Load real data (always normalized by participants)
            all_data = load_and_preprocess_data(data_path, normalize_by_participants=True)
            
            # Handle 100% data case (train_split = 1.0)
            if train_split >= 1.0:
                # Use all data for training
                if mode == "train":
                    self.raw_data = all_data.copy().reset_index(drop=True)
                    self.stats = get_data_statistics(self.raw_data)
                    print(f"Using 100% REAL data for training: {len(self.raw_data)} timesteps ({len(self.raw_data)/96:.1f} days)")
                elif mode == "test":
                    # If train_split=1.0, there's no test data, so use all data
                    self.raw_data = all_data.copy().reset_index(drop=True)
                    self.stats = get_data_statistics(self.raw_data)
                    print(f"Using 100% REAL data for testing: {len(self.raw_data)} timesteps ({len(self.raw_data)/96:.1f} days)")
                else:
                    raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
            else:
                # Split data into train/test based on time (chronological split)
                split_idx = int(len(all_data) * train_split)
                
                if mode == "train":
                    self.raw_data = all_data.iloc[:split_idx].copy().reset_index(drop=True)
                    print(f"Using REAL TRAIN data: {len(self.raw_data)} timesteps ({len(self.raw_data)/96:.1f} days)")
                elif mode == "test":
                    self.raw_data = all_data.iloc[split_idx:].copy().reset_index(drop=True)
                    print(f"Using REAL TEST data: {len(self.raw_data)} timesteps ({len(self.raw_data)/96:.1f} days)")
                else:
                    raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
                
                # Use statistics from training data for normalization (important!)
                if mode == "test":
                    # For test mode, we should use training statistics for normalization
                    # This simulates real-world where we normalize based on training data
                    train_data = all_data.iloc[:split_idx]
                    self.stats = get_data_statistics(train_data)
                else:
                    self.stats = get_data_statistics(self.raw_data)
        
        self.normalize_state = normalize_state
        self.mode = mode
        
        # Battery parameters
        self.battery_capacity_kwh = battery_capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        
        # Episode parameters
        self.episode_length = episode_length
        self.timestep_duration_hours = 0.25  # 15 minutes
        self.continue_battery_state = continue_battery_state
        
        # State space: [total_consumption, total_production, surplus_production, battery_soc]
        # Optionally add time features: [hour_of_day, day_of_week]
        if normalize_state:
            # Normalized features (0-1 range)
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Raw features (unbounded, but we'll use reasonable bounds)
            max_consumption = self.stats['total_consumption']['max'] * 1.5
            max_production = self.stats['total_production']['max'] * 1.5
            max_surplus = self.stats['surplus_production']['max'] * 1.5
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([max_consumption, max_production, max_surplus, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        # Action space: continuous charge/discharge rate in kW
        # Negative = discharge, Positive = charge
        self.action_space = spaces.Box(
            low=np.array([-max_discharge_rate_kw], dtype=np.float32),
            high=np.array([max_charge_rate_kw], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.battery_soc = initial_soc
        self.current_data_idx = 0
        self.episode_data = None
        
        self.seed(seed)
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_data_idx >= len(self.episode_data):
            # Use last available data point
            row = self.episode_data.iloc[-1]
        else:
            row = self.episode_data.iloc[self.current_data_idx]
        
        total_consumption = row['total_consumption']
        total_production = row['total_production']
        surplus_production = row['surplus_production']
        
        # Normalize if requested
        if self.normalize_state:
            total_consumption = self._normalize_value(
                total_consumption, 'total_consumption'
            )
            total_production = self._normalize_value(
                total_production, 'total_production'
            )
            surplus_production = self._normalize_value(
                surplus_production, 'surplus_production'
            )
        
        obs = np.array([
            total_consumption,
            total_production,
            surplus_production,
            self.battery_soc
        ], dtype=np.float32)
        
        return obs
    
    def _normalize_value(self, value: float, feature_name: str) -> float:
        """Normalize a single value using stored statistics."""
        stats = self.stats[feature_name]
        min_val = stats['min']
        max_val = stats['max']
        if max_val - min_val > 0:
            return (value - min_val) / (max_val - min_val)
        return 0.0
    
    def _update_battery(
        self, 
        action: float, 
        total_production: float, 
        total_consumption: float
    ) -> Tuple[float, float]:
        """
        Update battery state based on action.
        If action is impossible, SOC does not update (returns 0 energy).
        Agent learns from reward penalties.
        
        Args:
            action: Charge/discharge rate in kW (positive=charge, negative=discharge)
            total_production: Current production in kWh
            total_consumption: Current consumption in kWh
            
        Returns:
            Tuple of (actual_charge_kwh, actual_discharge_kwh)
        """
        # Calculate excess production (surplus available for charging)
        excess_production_kwh = max(0, total_production - total_consumption)
        # Check actual difference with small epsilon for floating point precision
        surplus_exists = (total_production - total_consumption) > 1e-6

        # Check for invalid actions - if invalid, don't update SOC
        # Invalid action 1: Trying to charge when no surplus (no grid charging)
        if action > 0 and not surplus_exists:
            # Impossible action - return 0 energy, SOC doesn't change
            return 0.0, 0.0
        
        # Invalid action 2: Trying to discharge when battery is empty
        if action < 0 and self.battery_soc <= 0.0:
            # Impossible action - return 0 energy, SOC doesn't change
            return 0.0, 0.0
        
        
        # Action is valid - calculate energy change and update SOC
        if action > 0:  # Charging
            energy_charged_kwh = action * self.timestep_duration_hours * self.efficiency
            energy_discharged_kwh = 0.0
        elif action < 0:  # Discharging
            energy_charged_kwh = 0.0
            energy_discharged_kwh = abs(action) * self.timestep_duration_hours / self.efficiency
        else:  # No action
            energy_charged_kwh = 0.0
            energy_discharged_kwh = 0.0
        
        # Update SOC only if action was valid
        soc_change = (energy_charged_kwh - energy_discharged_kwh) / self.battery_capacity_kwh
        self.battery_soc = np.clip(self.battery_soc + soc_change, 0.0, 1.0)
        
        return energy_charged_kwh, energy_discharged_kwh
    
    def _calculate_reward(
        self,
        total_consumption: float,
        total_production: float,
        energy_charged_kwh: float,
        energy_discharged_kwh: float,
        action_value_kw: float,  # Add this parameter
        soc_before: float        # Add this parameter
    ) -> float:
        """
        Calculate reward based on grid usage and penalize invalid actions.
        
        Reward formula:
        reward = -grid_usage - invalid_action_penalty
        
        Where:
        - grid_usage: Energy drawn from the grid
        - invalid_action_penalty: Penalty for trying impossible actions
        """
        # Calculate available energy: production + battery discharge
        available_energy = total_production + energy_discharged_kwh
        
        # Grid usage = consumption not covered by available energy
        grid_usage = max(0, total_consumption - available_energy)
        
        # Calculate excess production (surplus available for charging)
        excess_production = max(0, total_production - total_consumption)
        # Check actual difference with small epsilon for floating point precision
        surplus_exists = (total_production - total_consumption) > 1e-6
        
        # Penalize invalid actions
        invalid_action_penalty = 0.0
        
        # Penalty 1: Trying to charge when no surplus (grid charging not allowed)
        if action_value_kw > 0 and not surplus_exists:
            # Agent tried to charge but no surplus available
            invalid_action_penalty = -10.0 * abs(action_value_kw) * self.timestep_duration_hours
        
        # Penalty 2: Trying to discharge when battery is empty
        elif action_value_kw < 0 and soc_before <= 0.001:
            # Agent tried to discharge but battery was empty
            invalid_action_penalty = -10.0 * abs(action_value_kw) * self.timestep_duration_hours
        
        # Penalty 3: Trying to discharge when surplus exists (surplus must be used first)
        elif action_value_kw < 0 and surplus_exists:
            # Agent tried to discharge when surplus was available
            invalid_action_penalty = -10.0 * abs(action_value_kw) * self.timestep_duration_hours
        
        # In _calculate_reward(), after calculating grid_usage:
        # Bonus for using stored energy to reduce grid usage
        if energy_discharged_kwh > 0 and total_consumption > total_production:
            # Agent discharged to reduce grid usage - give small bonus
            discharge_bonus = 0.1 * energy_discharged_kwh
        else:
            discharge_bonus = 0.0

        reward = -grid_usage + discharge_bonus + invalid_action_penalty
                
        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Select random starting point in data (with enough remaining steps)
        max_start_idx = max(0, len(self.raw_data) - self.episode_length)
        start_idx = self.np_random.integers(0, max_start_idx + 1)
        
        # Extract episode data
        end_idx = min(start_idx + self.episode_length, len(self.raw_data))
        self.episode_data = self.raw_data.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        
        # Reset internal state
        self.current_step = 0
        # Only reset battery SOC if not continuing state across episodes
        if not self.continue_battery_state:
            self.battery_soc = self.initial_soc
        # If continuing, battery_soc keeps its current value
        self.current_data_idx = 0
        
        observation = self._get_observation()
        info = {
            'battery_soc': self.battery_soc,
            'data_idx': self.current_data_idx
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Charge/discharge rate in kW
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Extract action value
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Get current data
        if self.current_data_idx >= len(self.episode_data):
            # Episode should have ended, but handle gracefully
            row = self.episode_data.iloc[-1]
        else:
            row = self.episode_data.iloc[self.current_data_idx]
        
        total_consumption = row['total_consumption']
        total_production = row['total_production']
        
        # Update battery (simple structure - just update SOC)
        # Store SOC before battery update (needed for penalty calculations)
        soc_before = self.battery_soc

        # Update battery (simple structure - just update SOC)
        energy_charged_kwh, energy_discharged_kwh = self._update_battery(
            action_value, total_production, total_consumption
        )

        # Calculate reward (pass action and SOC before update for penalty calculation)
        reward = self._calculate_reward(
            total_consumption,
            total_production,
            energy_charged_kwh,
            energy_discharged_kwh,
            action_value,  # Pass original action to detect invalid attempts
            soc_before     # Pass SOC before update to check if battery was empty
        )
                
        # Update step counter
        self.current_step += 1
        self.current_data_idx += 1
        
        # Check termination conditions
        terminated = False
        truncated = (
            self.current_step >= self.episode_length or
            self.current_data_idx >= len(self.episode_data)
        )
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate grid usage for info (simplified - no grid charging)
        available_energy = total_production + energy_discharged_kwh
        own_coverage = min(total_consumption, max(0, available_energy))
        total_grid_usage = total_consumption - own_coverage
        
        # Info dictionary
        info = {
            'battery_soc': self.battery_soc,
            'energy_charged_kwh': energy_charged_kwh,
            'energy_discharged_kwh': energy_discharged_kwh,
            'total_consumption': total_consumption,
            'total_production': total_production,
            'grid_usage': total_grid_usage,
            'own_coverage': own_coverage,
            'data_idx': self.current_data_idx
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional, for visualization)."""
        if self.current_data_idx < len(self.episode_data):
            row = self.episode_data.iloc[self.current_data_idx]
            print(f"Step: {self.current_step}")
            print(f"  Consumption: {row['total_consumption']:.2f} kWh")
            print(f"  Production: {row['total_production']:.2f} kWh")
            print(f"  Battery SOC: {self.battery_soc:.2%}")


if __name__ == "__main__":
    # Test environment
    env = BatteryControlEnv(episode_length=10, seed=42)
    
    print("Testing environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action[0]:.2f} kW")
        print(f"  Reward: {reward:.2f}")
        print(f"  Battery SOC: {info['battery_soc']:.2%}")
        print(f"  Grid Usage: {info['grid_usage']:.2f} kWh")
        
        if terminated or truncated:
            print("Episode ended!")
            break

