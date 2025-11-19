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
        Update battery state based on action with physical constraints.
        
        Physical constraints enforced:
        - Cannot discharge more than available energy in battery
        - Cannot charge beyond battery capacity
        - Cannot charge if no surplus available (no grid charging)
        
        Args:
            action: Charge/discharge rate in kW (positive=charge, negative=discharge)
            total_production: Current production in kWh
            total_consumption: Current consumption in kWh
            
        Returns:
            Tuple of (actual_charge_kwh, actual_discharge_kwh)
        """
        surplus = total_production - total_consumption
        
        # Calculate desired energy change based on action
        if action > 0:  # Charging
            # Physical constraint: Cannot charge without surplus
            if surplus <= 0.1:
                return 0.0, 0.0
            
            # Calculate energy to charge (with efficiency)
            energy_charged_kwh = action * self.timestep_duration_hours * self.efficiency
            
            # Physical constraint: Cannot charge beyond battery capacity
            max_chargeable = (1.0 - self.battery_soc) * self.battery_capacity_kwh
            energy_charged_kwh = min(energy_charged_kwh, max_chargeable)
            
            # Physical constraint: Cannot charge more than available surplus
            energy_charged_kwh = min(energy_charged_kwh, surplus * self.efficiency)
            
            energy_discharged_kwh = 0.0
            
        elif action < 0:  # Discharging
            # Physical constraint: Cannot discharge if battery is empty
            if self.battery_soc <= 0.01:
                return 0.0, 0.0
            
            # Calculate energy to discharge (with efficiency loss)
            energy_discharged_kwh = abs(action) * self.timestep_duration_hours / self.efficiency
            
            # Physical constraint: Cannot discharge more than what's in battery
            max_dischargeable = self.battery_soc * self.battery_capacity_kwh
            energy_discharged_kwh = min(energy_discharged_kwh, max_dischargeable)
            
            energy_charged_kwh = 0.0
            
        else:  # No action
            energy_charged_kwh = 0.0
            energy_discharged_kwh = 0.0
        
        # Update SOC
        soc_change = (energy_charged_kwh - energy_discharged_kwh) / self.battery_capacity_kwh
        self.battery_soc = np.clip(self.battery_soc + soc_change, 0.0, 1.0)
        
        return energy_charged_kwh, energy_discharged_kwh
    
    def _calculate_reward(
        self,
        total_consumption: float,
        total_production: float,
        energy_charged_kwh: float,
        energy_discharged_kwh: float,
        action_value_kw: float,
        soc_before: float
    ) -> float:
        """
        Reward function with NO double-counting:
        1. Charge when surplus exists
        2. Discharge when deficit exists to minimize grid usage
        3. Penalize impossible actions (even though they won't execute)
        """
        # Calculate surplus (excess production)
        surplus = total_production - total_consumption
        surplus_exists = surplus > 0.1  # Meaningful threshold
        
        # Calculate grid usage (what we actually draw from grid)
        available_energy = total_production + energy_discharged_kwh
        grid_usage = max(0, total_consumption - available_energy)
        
        # Primary objective: minimize grid usage
        reward = -grid_usage
        
        # Penalty for impossible actions (agent attempted but physical constraints prevented)
        # These help the agent learn to avoid impossible actions faster
        if action_value_kw < 0 and soc_before <= 0.01:  # Tried to discharge from empty battery
            reward -= 20.0 * abs(action_value_kw) * self.timestep_duration_hours
        elif action_value_kw > 0 and not surplus_exists:  # Tried to charge without surplus
            reward -= 20.0 * abs(action_value_kw) * self.timestep_duration_hours
        elif action_value_kw < 0 and surplus_exists:  # Tried to discharge when surplus exists
            reward -= 20.0 * abs(action_value_kw) * self.timestep_duration_hours
        
        # Reward/penalty for valid actions based on scenario
        elif surplus_exists:
            # Scenario 1: We have surplus - should charge
            if action_value_kw > 0:  # Charging (correct action)
                # Reward for storing surplus (future benefit)
                reward += 1.5 * energy_charged_kwh
            else:  # Doing nothing (wasting surplus)
                # Penalty for not storing available surplus
                reward -= 0.5 * surplus
        
        else:
            # Scenario 2: Deficit (no surplus) - should discharge to reduce grid usage
            deficit = total_consumption - total_production  # How much we need from battery/grid
            
            if action_value_kw < 0 and energy_discharged_kwh > 0:  # Discharging (correct action)
                # EXPLICIT BONUS for discharging to help reduce grid usage
                grid_without_discharge = max(0, total_consumption - total_production)
                grid_reduction = grid_without_discharge - grid_usage
                # Give positive reward for the reduction
                reward += 2.0 * grid_reduction  # Increased from 1.0 to 2.0
                
                # BONUS: Encourage stronger discharge when battery has more energy
                battery_availability_bonus = 1.0 * soc_before * energy_discharged_kwh  # Increased from 0.5 to 1.0
                reward += battery_availability_bonus
                
                # NEW: Penalty for INSUFFICIENT discharge when deficit is large
                # If we could discharge more (battery has energy) but didn't discharge enough
                max_possible_discharge_kwh = min(
                    soc_before * self.battery_capacity_kwh,  # What's available in battery
                    deficit  # What's actually needed
                )
                
                if max_possible_discharge_kwh > energy_discharged_kwh * 1.5:
                    # Agent is discharging less than 67% of what it could/should
                    discharge_shortfall = max_possible_discharge_kwh - energy_discharged_kwh
                    weak_discharge_penalty = 3.0 * discharge_shortfall * soc_before
                    reward -= weak_discharge_penalty
                
            # If doing nothing when deficit exists AND battery has energy, penalize HEAVILY
            elif soc_before > 0.1 and grid_usage > 0.1:
                # Battery has energy but not using it when grid usage exists
                unused_battery_penalty = 1.0 * soc_before * grid_usage  # Increased from 0.3 to 1.0
                reward -= unused_battery_penalty
        
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
        
        # Update battery state
        energy_charged_kwh, energy_discharged_kwh = self._update_battery(
            action_value, total_production, total_consumption
        )

        # Calculate reward (grid usage penalty only)
        reward = self._calculate_reward(
            total_consumption,
            total_production,
            energy_charged_kwh,
            energy_discharged_kwh,
            action_value,
            self.battery_soc
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

