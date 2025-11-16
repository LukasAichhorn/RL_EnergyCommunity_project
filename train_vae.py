"""
Main script to train VAE and generate synthetic data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_simulation.trainer import train_vae
from data_simulation.generator import generate_synthetic_data
from data_simulation.visualizer import plot_comparison, plot_statistics_comparison
from data_processor import load_and_preprocess_data
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train VAE and generate synthetic data")
    parser.add_argument("--data-path", type=str, default="metering_data_last_year.csv")
    parser.add_argument("--output-dir", type=str, default="./vae_models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-sequences", type=int, default=5, help="Number of synthetic datasets to generate")
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    parser.add_argument("--train-only", action="store_true", help="Only train, don't generate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Train VAE
    print("=" * 60)
    print("TRAINING VAE MODEL")
    print("=" * 60)
    model, norm_params, feature_names = train_vae(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        device=args.device
    )
    
    if args.train_only:
        return
    
    # Generate synthetic data
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)
    
    model_path = os.path.join(args.output_dir, 'best_model.pt')
    simulated_data_list = []
    
    for i in range(args.num_sequences):
        print(f"\nGenerating dataset {i+1}/{args.num_sequences}...")
        sim_data = generate_synthetic_data(
            model_path=model_path,
            num_sequences=100,  # Generate 100 sequences (days)
            sequence_length=args.sequence_length,
            device=args.device
        )
        simulated_data_list.append(sim_data)
        
        # Save to CSV
        output_path = os.path.join(args.output_dir, f'synthetic_data_{i+1}.csv')
        sim_data.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    # Load original data for comparison
    print("\nLoading original data for comparison...")
    original_data = load_and_preprocess_data(args.data_path)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_comparison(
        original_data=original_data,
        simulated_data_list=simulated_data_list,
        num_samples=2000,
        output_path=os.path.join(args.output_dir, 'comparison_plot.png')
    )
    
    plot_statistics_comparison(
        original_data=original_data,
        simulated_data_list=simulated_data_list,
        output_path=os.path.join(args.output_dir, 'statistics_comparison.png')
    )
    
    print("\nDone!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

