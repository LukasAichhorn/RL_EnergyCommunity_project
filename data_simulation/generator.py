"""
Generate synthetic data using trained VAE.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .vae_model import TimeSeriesVAE
from .data_preparation import denormalize_sequences


def generate_synthetic_data(
    model_path: str,
    num_sequences: int = 100,
    sequence_length: int = 96,
    device: str = None
) -> pd.DataFrame:
    """
    Generate synthetic time series data using trained VAE.
    
    Args:
        model_path: path to saved model checkpoint
        num_sequences: number of sequences to generate
        sequence_length: length of each sequence (default: 96 = 1 day)
        device: device to run on
        
    Returns:
        DataFrame with synthetic data
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    model = TimeSeriesVAE(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        latent_dim=checkpoint['latent_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    norm_params = checkpoint['norm_params']
    feature_names = checkpoint['feature_names']
    seq_len = checkpoint.get('sequence_length', sequence_length)
    
    # Generate sequences
    print(f"Generating {num_sequences} sequences of length {seq_len}...")
    with torch.no_grad():
        generated_normalized = model.generate(num_sequences, seq_len, device)
        generated_normalized = generated_normalized.cpu().numpy()
    
    # Denormalize
    generated = denormalize_sequences(generated_normalized, norm_params)
    
    # Ensure non-negative values
    generated = np.maximum(generated, 0)
    
    # Flatten sequences into single time series
    all_data = generated.reshape(-1, len(feature_names))
    
    # Create timestamps (15-minute intervals)
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [
        start_date + timedelta(minutes=15 * i)
        for i in range(len(all_data))
    ]
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=feature_names)
    df.insert(0, 'metering_timestamp', timestamps)
    
    # Add derived columns
    df['own_coverage'] = df[['total_consumption', 'total_production']].min(axis=1)
    df['community_share'] = df['own_coverage'] * 0.9  # Simplified
    
    return df

