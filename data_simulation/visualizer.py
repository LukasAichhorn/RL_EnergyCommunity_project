"""
Visualization utilities for comparing original and simulated data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional


def plot_comparison(
    original_data: pd.DataFrame,
    simulated_data_list: List[pd.DataFrame],
    num_samples: int = 2000,
    output_path: Optional[str] = None
):
    """
    Plot comparison between original and simulated data.
    
    Args:
        original_data: original real data DataFrame
        simulated_data_list: list of simulated DataFrames (can be multiple)
        num_samples: number of timesteps to plot
        output_path: path to save plot (if None, shows plot)
    """
    # Sample data for plotting
    orig_sample = original_data.head(num_samples)
    
    # Create figure with subplots
    num_features = 3  # consumption, production, surplus
    fig, axes = plt.subplots(num_features, 1, figsize=(16, 12))
    
    features = ['total_consumption', 'total_production', 'surplus_production']
    colors = ['blue', 'green', 'orange']
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Plot original
        ax.plot(
            range(len(orig_sample)),
            orig_sample[feature].values,
            label='Original Data',
            color='black',
            linewidth=2,
            alpha=0.8
        )
        
        # Plot each simulated dataset
        for sim_idx, sim_data in enumerate(simulated_data_list):
            sim_sample = sim_data.head(num_samples)
            if len(sim_sample) < len(orig_sample):
                # Pad or truncate to match length
                sim_values = sim_sample[feature].values
                if len(sim_values) < len(orig_sample):
                    sim_values = np.pad(sim_values, (0, len(orig_sample) - len(sim_values)), 'constant')
                else:
                    sim_values = sim_values[:len(orig_sample)]
            else:
                sim_values = sim_sample[feature].values[:len(orig_sample)]
            
            ax.plot(
                range(len(orig_sample)),
                sim_values,
                label=f'Simulated {sim_idx + 1}',
                alpha=0.6,
                linewidth=1.5
            )
        
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{feature.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if idx == len(features) - 1:
            ax.set_xlabel('Time Step', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_statistics_comparison(
    original_data: pd.DataFrame,
    simulated_data_list: List[pd.DataFrame],
    output_path: Optional[str] = None
):
    """
    Compare statistics (mean, std, distribution) between original and simulated.
    """
    features = ['total_consumption', 'total_production', 'surplus_production']
    
    fig, axes = plt.subplots(len(features), 2, figsize=(16, 12))
    
    for idx, feature in enumerate(features):
        # Histogram
        ax = axes[idx, 0]
        ax.hist(original_data[feature], bins=50, alpha=0.7, label='Original', density=True, color='black')
        for sim_idx, sim_data in enumerate(simulated_data_list):
            ax.hist(sim_data[feature], bins=50, alpha=0.5, label=f'Sim {sim_idx+1}', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics comparison
        ax = axes[idx, 1]
        stats_data = {
            'Original': [
                original_data[feature].mean(),
                original_data[feature].std(),
                original_data[feature].min(),
                original_data[feature].max()
            ]
        }
        for sim_idx, sim_data in enumerate(simulated_data_list):
            stats_data[f'Sim {sim_idx+1}'] = [
                sim_data[feature].mean(),
                sim_data[feature].std(),
                sim_data[feature].min(),
                sim_data[feature].max()
            ]
        
        x = np.arange(len(['Mean', 'Std', 'Min', 'Max']))
        width = 0.8 / len(stats_data)
        
        for i, (label, values) in enumerate(stats_data.items()):
            offset = (i - len(stats_data)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label, alpha=0.7)
        
        ax.set_xlabel('Statistic')
        ax.set_ylabel('Value')
        ax.set_title(f'{feature} Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Std', 'Min', 'Max'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

