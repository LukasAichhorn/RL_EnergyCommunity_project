"""
Create histogram of consumption and production data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_processor import load_and_preprocess_data

# Load data
print("Loading data...")
data = load_and_preprocess_data("metering_data_last_year.csv", normalize_by_participants=True)

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nData statistics:")
print(data[['total_consumption', 'total_production']].describe())

# Create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Histogram for consumption
axes[0].hist(data['total_consumption'], bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Total Consumption (kWh)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Total Consumption', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(data['total_consumption'].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {data["total_consumption"].mean():.2f} kWh')
axes[0].axvline(data['total_consumption'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data["total_consumption"].median():.2f} kWh')
axes[0].legend()

# Histogram for production
axes[1].hist(data['total_production'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Total Production (kWh)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Total Production', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(data['total_production'].mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {data["total_production"].mean():.2f} kWh')
axes[1].axvline(data['total_production'].median(), color='lightgreen', linestyle='--', linewidth=2, label=f'Median: {data["total_production"].median():.2f} kWh')
axes[1].legend()

plt.tight_layout()
plt.savefig('data_histogram.png', dpi=300, bbox_inches='tight')
print("\nHistogram saved to: data_histogram.png")

# Create overlapping histogram for comparison
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.hist(data['total_consumption'], bins=50, color='red', alpha=0.5, label='Consumption', edgecolor='black')
ax.hist(data['total_production'], bins=50, color='green', alpha=0.5, label='Production', edgecolor='black')
ax.set_xlabel('Energy (kWh)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Consumption vs Production', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('data_histogram_overlay.png', dpi=300, bbox_inches='tight')
print("Overlay histogram saved to: data_histogram_overlay.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Consumption:")
print(f"  Mean:   {data['total_consumption'].mean():.2f} kWh")
print(f"  Median: {data['total_consumption'].median():.2f} kWh")
print(f"  Std:    {data['total_consumption'].std():.2f} kWh")
print(f"  Min:    {data['total_consumption'].min():.2f} kWh")
print(f"  Max:    {data['total_consumption'].max():.2f} kWh")

print(f"\nTotal Production:")
print(f"  Mean:   {data['total_production'].mean():.2f} kWh")
print(f"  Median: {data['total_production'].median():.2f} kWh")
print(f"  Std:    {data['total_production'].std():.2f} kWh")
print(f"  Min:    {data['total_production'].min():.2f} kWh")
print(f"  Max:    {data['total_production'].max():.2f} kWh")

print(f"\nSurplus Production:")
print(f"  Mean:   {data['surplus_production'].mean():.2f} kWh")
print(f"  Median: {data['surplus_production'].median():.2f} kWh")
print(f"  Max:    {data['surplus_production'].max():.2f} kWh")

# Calculate some insights
total_timesteps = len(data)
surplus_timesteps = (data['surplus_production'] > 0).sum()
deficit_timesteps = (data['surplus_production'] <= 0).sum()

print(f"\nTimestep Analysis:")
print(f"  Total timesteps: {total_timesteps}")
print(f"  Surplus periods: {surplus_timesteps} ({surplus_timesteps/total_timesteps*100:.1f}%)")
print(f"  Deficit periods: {deficit_timesteps} ({deficit_timesteps/total_timesteps*100:.1f}%)")

plt.show()


