"""
Data simulator for generating synthetic consumption and production patterns.
Uses statistics from real data to create realistic synthetic time series.
"""

import pandas as pd
import numpy as np
from typing import Optional
from data_processor import load_and_preprocess_data, get_data_statistics


def generate_synthetic_data(
    num_timesteps: int = 10000,
    data_path: Optional[str] = None,
    seed: Optional[int] = None,
    add_noise: bool = True,
    seasonal_patterns: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic consumption and production data.
    
    Args:
        num_timesteps: Number of timesteps to generate
        data_path: Path to real data file (used to extract statistics)
        seed: Random seed for reproducibility
        add_noise: Whether to add random noise
        seasonal_patterns: Whether to include daily/seasonal patterns
        
    Returns:
        DataFrame with synthetic data matching real data format
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Load real data statistics if available
    if data_path:
        real_data = load_and_preprocess_data(data_path)
        stats = get_data_statistics(real_data)
        
        # Use real data statistics
        cons_mean = stats['total_consumption']['mean']
        cons_std = stats['total_consumption']['std']
        cons_min = stats['total_consumption']['min']
        cons_max = stats['total_consumption']['max']
        
        prod_mean = stats['total_production']['mean']
        prod_std = stats['total_production']['std']
        prod_min = stats['total_production']['min']
        prod_max = stats['total_production']['max']
        
        surplus_mean = stats['surplus_production']['mean']
        surplus_std = stats['surplus_production']['std']
    else:
        # Default values if no real data provided
        cons_mean, cons_std = 20.0, 10.0
        cons_min, cons_max = 0.0, 80.0
        prod_mean, prod_std = 18.0, 15.0
        prod_min, prod_max = 0.0, 130.0
        surplus_mean, surplus_std = 8.0, 10.0
    
    # Generate timestamps (15-minute intervals)
    start_date = pd.Timestamp('2024-01-01 00:00:00+00:00')
    timestamps = pd.date_range(
        start=start_date,
        periods=num_timesteps,
        freq='15min'
    )
    
    # Generate base consumption with daily patterns
    hours = np.array([ts.hour for ts in timestamps])
    days_of_week = np.array([ts.dayofweek for ts in timestamps])
    
    if seasonal_patterns:
        # Daily pattern: higher consumption during day, lower at night
        daily_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * (hours - 6) / 24)
        daily_pattern = np.clip(daily_pattern, 0.5, 1.5)
        
        # Weekly pattern: lower consumption on weekends
        weekly_pattern = np.where(days_of_week < 5, 1.0, 0.85)
        
        # Seasonal pattern (simplified)
        day_of_year = np.array([ts.dayofyear for ts in timestamps])
        seasonal_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        base_consumption = cons_mean * daily_pattern * weekly_pattern * seasonal_pattern
    else:
        base_consumption = np.full(num_timesteps, cons_mean)
    
    # Add noise and ensure non-negative
    if add_noise:
        consumption = base_consumption + np.random.normal(0, cons_std, num_timesteps)
    else:
        consumption = base_consumption
    
    consumption = np.clip(consumption, cons_min, cons_max)
    
    # Generate production with daily patterns (solar-like)
    if seasonal_patterns:
        # Solar production: high during day, zero at night
        solar_pattern = np.where((hours >= 6) & (hours <= 18), 
                                 np.sin(np.pi * (hours - 6) / 12), 0.0)
        solar_pattern = np.clip(solar_pattern, 0.0, 1.0)
        
        # Seasonal variation (more production in summer)
        seasonal_prod = 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        seasonal_prod = np.clip(seasonal_prod, 0.3, 1.4)
        
        base_production = prod_mean * solar_pattern * seasonal_prod
    else:
        base_production = np.full(num_timesteps, prod_mean)
    
    # Add noise
    if add_noise:
        production = base_production + np.random.normal(0, prod_std, num_timesteps)
    else:
        production = base_production
    
    production = np.clip(production, prod_min, prod_max)
    
    # Calculate surplus production (production - consumption, but only positive)
    surplus = np.maximum(0, production - consumption)
    
    # Add some correlation: when production is high, consumption might be slightly lower
    # (simulating self-consumption)
    own_coverage = np.minimum(consumption, production)
    
    # Community share (simplified - assume some sharing)
    community_share = own_coverage * (0.8 + 0.2 * np.random.random(num_timesteps))
    
    # Create DataFrame
    synthetic_data = pd.DataFrame({
        'metering_timestamp': timestamps,
        'total_consumption': consumption,
        'total_production': production,
        'surplus_production': surplus,
        'own_coverage': own_coverage,
        'community_share': community_share
    })
    
    return synthetic_data


def save_synthetic_data(
    output_path: str,
    num_timesteps: int = 10000,
    data_path: Optional[str] = None,
    seed: Optional[int] = None
):
    """Generate and save synthetic data to CSV."""
    synthetic_data = generate_synthetic_data(
        num_timesteps=num_timesteps,
        data_path=data_path,
        seed=seed
    )
    synthetic_data.to_csv(output_path, index=False)
    print(f"Saved {len(synthetic_data)} synthetic timesteps to {output_path}")
    return synthetic_data


if __name__ == "__main__":
    # Test the simulator
    print("Generating synthetic data...")
    synthetic = generate_synthetic_data(
        num_timesteps=1000,
        data_path="metering_data_last_year.csv",
        seed=42
    )
    
    print(f"\nGenerated {len(synthetic)} timesteps")
    print(f"\nStatistics:")
    print(f"Consumption: mean={synthetic['total_consumption'].mean():.2f}, "
          f"std={synthetic['total_consumption'].std():.2f}")
    print(f"Production: mean={synthetic['total_production'].mean():.2f}, "
          f"std={synthetic['total_production'].std():.2f}")
    print(f"\nFirst few rows:")
    print(synthetic.head())

