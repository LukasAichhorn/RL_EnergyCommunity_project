"""
Data processing module for battery control RL environment.
Loads and preprocesses metering data for community battery control.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_and_preprocess_data(
    csv_path: str = "metering_data_last_year.csv",
    fill_na: bool = True,
    normalize_by_participants: bool = True,
    scale_to_reference_participants: Optional[float] = None
) -> pd.DataFrame:
    """
    Load and preprocess metering data.
    
    Args:
        csv_path: Path to the CSV file
        fill_na: Whether to fill NaN values with 0
        normalize_by_participants: If True, normalize values by number of participants
                                   to get per-participant averages (accounts for changing
                                   participant count over time)
        scale_to_reference_participants: If provided, scale per-participant values back
                                        to total values using this reference count
                                        (e.g., mean or median participant count)
        
    Returns:
        Preprocessed DataFrame with aggregated data by timestamp
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df['metering_timestamp'] = pd.to_datetime(df['metering_timestamp'])
    
    # Fill NaN values with 0 (for consumption-only or production-only meterpoints)
    if fill_na:
        df['total_consumption'] = df['total_consumption'].fillna(0)
        df['total_production'] = df['total_production'].fillna(0)
        df['surplus_production'] = df['surplus_production'].fillna(0)
        df['own_coverage'] = df['own_coverage'].fillna(0)
        df['community_share'] = df['community_share'].fillna(0)
    
    # Count number of unique participants per timestamp
    participant_count = df.groupby('metering_timestamp')['metering_point_hash'].nunique().reset_index()
    participant_count.columns = ['metering_timestamp', 'num_participants']
    
    # Aggregate by timestamp (sum across all meterpoints)
    aggregated = df.groupby('metering_timestamp').agg({
        'total_consumption': 'sum',
        'total_production': 'sum',
        'surplus_production': 'sum',
        'own_coverage': 'sum',
        'community_share': 'sum'
    }).reset_index()
    
    # Merge participant count
    aggregated = aggregated.merge(participant_count, on='metering_timestamp', how='left')
    
    # Normalize by number of participants (per-participant average)
    # This accounts for changing number of participants over time
    if normalize_by_participants:
        aggregated['total_consumption'] = aggregated['total_consumption'] / aggregated['num_participants']
        aggregated['total_production'] = aggregated['total_production'] / aggregated['num_participants']
        aggregated['surplus_production'] = aggregated['surplus_production'] / aggregated['num_participants']
        aggregated['own_coverage'] = aggregated['own_coverage'] / aggregated['num_participants']
        aggregated['community_share'] = aggregated['community_share'] / aggregated['num_participants']
        
        # Scale back to reference participant count (default: mean)
        # This makes values comparable to original scale but normalized for participant changes
        if scale_to_reference_participants is None:
            # Use mean participant count as default reference
            scale_factor = aggregated['num_participants'].mean()
        else:
            scale_factor = scale_to_reference_participants
        
        aggregated['total_consumption'] = aggregated['total_consumption'] * scale_factor
        aggregated['total_production'] = aggregated['total_production'] * scale_factor
        aggregated['surplus_production'] = aggregated['surplus_production'] * scale_factor
        aggregated['own_coverage'] = aggregated['own_coverage'] * scale_factor
        aggregated['community_share'] = aggregated['community_share'] * scale_factor
    
    # Sort by timestamp
    aggregated = aggregated.sort_values('metering_timestamp').reset_index(drop=True)
    
    return aggregated


def normalize_data(
    df: pd.DataFrame,
    columns: list = ['total_consumption', 'total_production', 'surplus_production']
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        df: DataFrame to normalize
        columns: List of column names to normalize
        
    Returns:
        Tuple of (normalized DataFrame, normalization parameters dict)
    """
    normalized_df = df.copy()
    norm_params = {}
    
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # Avoid division by zero
            if max_val - min_val > 0:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0.0
            norm_params[col] = {'min': min_val, 'max': max_val}
    
    return normalized_df, norm_params


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about the data for normalization and scaling.
    
    Args:
        df: DataFrame with aggregated data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_consumption': {
            'min': df['total_consumption'].min(),
            'max': df['total_consumption'].max(),
            'mean': df['total_consumption'].mean(),
            'std': df['total_consumption'].std()
        },
        'total_production': {
            'min': df['total_production'].min(),
            'max': df['total_production'].max(),
            'mean': df['total_production'].mean(),
            'std': df['total_production'].std()
        },
        'surplus_production': {
            'min': df['surplus_production'].min(),
            'max': df['surplus_production'].max(),
            'mean': df['surplus_production'].mean(),
            'std': df['surplus_production'].std()
        }
    }
    return stats


if __name__ == "__main__":
    # Test data loading
    df = load_and_preprocess_data()
    print(f"Loaded {len(df)} timesteps")
    print(f"Date range: {df['metering_timestamp'].min()} to {df['metering_timestamp'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nStatistics:")
    stats = get_data_statistics(df)
    for key, val in stats.items():
        print(f"{key}: min={val['min']:.2f}, max={val['max']:.2f}, mean={val['mean']:.2f}")

