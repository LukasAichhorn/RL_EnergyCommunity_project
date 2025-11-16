"""
Training script for VAE model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

from .vae_model import TimeSeriesVAE
from .data_preparation import prepare_data


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss: reconstruction loss + KL divergence.
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(
    data_path: str,
    output_dir: str = "./vae_models",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sequence_length: int = 96,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    beta: float = 1.0,
    device: str = None
):
    """
    Train VAE model on time series data.
    
    Returns:
        model: trained VAE model
        norm_params: normalization parameters
        feature_names: list of feature names
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    train_sequences, val_sequences, norm_params, feature_names = prepare_data(
        data_path, sequence_length=sequence_length
    )
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Sequence shape: {train_sequences.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(train_sequences))
    val_dataset = TensorDataset(torch.FloatTensor(val_sequences))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = train_sequences.shape[-1]
    model = TimeSeriesVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Train - Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Val   - Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'norm_params': norm_params,
                'feature_names': feature_names,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'latent_dim': latent_dim,
                'sequence_length': sequence_length,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    return model, norm_params, feature_names

