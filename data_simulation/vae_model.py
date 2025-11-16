"""
Variational Autoencoder for time series generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesVAE(nn.Module):
    """
    Variational Autoencoder for time series generation.
    Encodes sequences into latent space, decodes to generate new sequences.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # consumption, production, surplus
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: LSTM → mean and log_var
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent → LSTM → output
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encode input sequence to latent space.
        Args:
            x: (batch, seq_len, features)
        Returns:
            mu, logvar: (batch, latent_dim)
        """
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]  # Last hidden state from last layer
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, sequence_length):
        """
        Decode latent vector to sequence.
        Args:
            z: (batch, latent_dim)
            sequence_length: length of sequence to generate
        Returns:
            output: (batch, sequence_length, features)
        """
        batch_size = z.size(0)
        num_layers = self.decoder.num_layers
        
        # Initialize hidden state for all layers
        h_0 = self.decoder_input(z).unsqueeze(0)  # (1, batch, hidden)
        # Expand to match number of layers
        h = h_0.repeat(num_layers, 1, 1)  # (num_layers, batch, hidden)
        c = torch.zeros_like(h)
        
        outputs = []
        input_t = torch.zeros(batch_size, 1, self.hidden_dim, device=z.device)
        
        for _ in range(sequence_length):
            out, (h, c) = self.decoder(input_t, (h, c))
            out_features = self.fc_output(out)  # (batch, 1, features)
            outputs.append(out_features)
            # Use output as next input (autoregressive)
            input_t = out
        
        return torch.cat(outputs, dim=1)
    
    def forward(self, x):
        """
        Forward pass: encode, sample, decode.
        Args:
            x: (batch, seq_len, features)
        Returns:
            recon: reconstructed sequence
            mu: latent mean
            logvar: latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar
    
    def generate(self, num_sequences, sequence_length, device='cpu'):
        """
        Generate synthetic sequences by sampling from latent space.
        Args:
            num_sequences: number of sequences to generate
            sequence_length: length of each sequence
            device: device to run on
        Returns:
            generated: (num_sequences, sequence_length, features)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_sequences, self.latent_dim, device=device)
            generated = self.decode(z, sequence_length)
        return generated

