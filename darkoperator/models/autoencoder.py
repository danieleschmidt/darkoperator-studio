"""Autoencoder models for anomaly detection and dimensionality reduction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import warnings
import logging
from ..utils.error_handling import model_validator, data_validator, DarkOperatorError


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional autoencoder for calorimeter data compression and anomaly detection."""
    
    def __init__(
        self, 
        input_channels: int = 1,
        latent_dim: int = 128,
        image_size: Tuple[int, int] = (50, 50)
    ):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 50 -> 25
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 25 -> 12
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 6 * 6),
            nn.ReLU(),
            nn.Unflatten(1, (128, 6, 6)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),  # -> 48x48
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(image_size),  # Ensure exact output size
        )
    
    @model_validator(check_parameters=True, check_device=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        # Input validation
        if x.dim() != 4 or x.size(-1) != self.image_size[1] or x.size(-2) != self.image_size[0]:
            raise DarkOperatorError(f"Expected input shape [N, C, {self.image_size[0]}, {self.image_size[1]}], got {x.shape}")
        
        if not torch.isfinite(x).all():
            raise DarkOperatorError("Input contains non-finite values")
        
        try:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            
            # Output validation
            if not torch.isfinite(decoded).all():
                raise DarkOperatorError("Output contains non-finite values")
                
            return decoded
            
        except Exception as e:
            logging.error(f"ConvolutionalAutoencoder forward pass failed: {e}")
            raise DarkOperatorError(f"Forward pass failed: {e}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        return self.decoder(z)
    
    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        # Input validation
        if x.shape != x_recon.shape:
            raise DarkOperatorError(f"Shape mismatch: original {x.shape} vs reconstruction {x_recon.shape}")
        
        if not torch.isfinite(x).all() or not torch.isfinite(x_recon).all():
            raise DarkOperatorError("Non-finite values in loss computation")
        
        loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Sanity check loss value
        if not torch.isfinite(loss).all() or loss < 0:
            raise DarkOperatorError(f"Invalid loss value: {loss}")
        
        return loss


class VariationalAutoencoder(nn.Module):
    """Variational autoencoder for probabilistic anomaly detection."""
    
    def __init__(
        self, 
        input_channels: int = 1,
        latent_dim: int = 128,
        image_size: Tuple[int, int] = (50, 50)
    ):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        h = self.decoder_fc(z)
        return self.decoder_conv(h)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss (reconstruction + KL divergence)."""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the learned distribution."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score based on reconstruction error and latent space density."""
        with torch.no_grad():
            output = self.forward(x)
            recon_error = F.mse_loss(output['x_recon'], x, reduction='none')
            recon_error = recon_error.view(x.size(0), -1).mean(dim=1)
            
            # Latent space density (negative log probability)
            mu, logvar = output['mu'], output['logvar']
            latent_density = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1, dim=1)
            
            # Combined anomaly score
            anomaly_scores = recon_error + 0.1 * latent_density
            return anomaly_scores


def test_autoencoders():
    """Test autoencoder implementations."""
    print("Testing autoencoder models...")
    
    # Test data
    batch_size, channels, height, width = 4, 1, 50, 50
    x = torch.randn(batch_size, channels, height, width)
    
    # Test ConvolutionalAutoencoder
    print("Testing ConvolutionalAutoencoder...")
    conv_ae = ConvolutionalAutoencoder(input_channels=channels)
    x_recon = conv_ae(x)
    assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} != {x.shape}"
    
    loss = conv_ae.reconstruction_loss(x, x_recon)
    print(f"ConvAE reconstruction loss: {loss.item():.4f}")
    
    # Test VariationalAutoencoder
    print("Testing VariationalAutoencoder...")
    vae = VariationalAutoencoder(input_channels=channels)
    vae_output = vae(x)
    
    assert vae_output['x_recon'].shape == x.shape
    assert vae_output['mu'].shape == (batch_size, 128)
    assert vae_output['logvar'].shape == (batch_size, 128)
    
    loss_dict = vae.loss_function(x, **{k: v for k, v in vae_output.items() if k != 'z'})
    print(f"VAE total loss: {loss_dict['loss'].item():.4f}")
    print(f"VAE recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"VAE KL loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Test anomaly scoring
    anomaly_scores = vae.anomaly_score(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Mean anomaly score: {anomaly_scores.mean().item():.4f}")
    
    print("✅ All autoencoder tests passed!")


if __name__ == "__main__":
    test_autoencoders()