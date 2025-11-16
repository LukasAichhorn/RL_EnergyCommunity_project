"""
Encoder-decoder based data simulation package.
"""

from .vae_model import TimeSeriesVAE
from .data_preparation import create_sequences, prepare_data, normalize_sequences, denormalize_sequences
from .trainer import train_vae
from .generator import generate_synthetic_data
from .visualizer import plot_comparison, plot_statistics_comparison

__all__ = [
    'TimeSeriesVAE',
    'create_sequences',
    'prepare_data',
    'normalize_sequences',
    'denormalize_sequences',
    'train_vae',
    'generate_synthetic_data',
    'plot_comparison',
    'plot_statistics_comparison'
]

