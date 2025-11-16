"""
Data preparation utilities for VAE training.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import sys
import os

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import load_and_preprocess_data


def normalize_sequences(sequences: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize sequences to [0, 1] range.
    Returns normalized sequences and normalization parameters.
    """
    # Flatten for normalization
    flat = sequences.reshape(-1, sequences.shape[-1])
    
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    
    normalized = (sequences - mins) / ranges
    
    norm_params = {
        'mins': mins,
        'maxs': maxs,
        'ranges': ranges
    }
    
    return normalized, norm_params


def denormalize_sequences(normalized: np.ndarray, norm_params: dict) -> np.ndarray:
    """Denormalize sequences back to original scale."""
    return normalized * norm_params['ranges'] + norm_params['mins']


def create_sequences(
    data: pd.DataFrame,
    sequence_length: int = 96,  # 1 day
    stride: int = 1,
    features: list = None
) -> np.ndarray:
    """
    Create sequences from time series data using sliding window.
    
    Args:
        data: DataFrame with time series data
        sequence_length: length of each sequence
        stride: step size for sliding window
        features: list of feature columns to use
        
    Returns:
        sequences: (num_sequences, sequence_length, num_features)
    """
    if features is None:
        features = ['total_consumption', 'total_production', 'surplus_production']
    
    # Extract feature values
    values = data[features].values
    
    sequences = []
    for i in range(0, len(values) - sequence_length + 1, stride):
        seq = values[i:i + sequence_length]
        sequences.append(seq)
    
    return np.array(sequences)


def prepare_data(
    data_path: str,
    sequence_length: int = 96,
    stride: int = 1,
    train_split: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, dict, list]:
    """
    Load and prepare data for VAE training.
    
    Returns:
        train_sequences: training sequences (normalized)
        val_sequences: validation sequences (normalized)
        norm_params: normalization parameters
        feature_names: list of feature names
    """
    # Load data
    data = load_and_preprocess_data(data_path)
    
    # Select features
    features = ['total_consumption', 'total_production', 'surplus_production']
    
    # Create sequences
    sequences = create_sequences(data, sequence_length, stride, features)
    
    # Split train/val
    split_idx = int(len(sequences) * train_split)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # Normalize using training data statistics
    train_normalized, norm_params = normalize_sequences(train_sequences)
    
    # Normalize validation using training statistics
    val_normalized = (val_sequences - norm_params['mins']) / norm_params['ranges']
    
    return train_normalized, val_normalized, norm_params, features

