"""
Variational Autoencoder (VAE) Model Implementation

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This module implements a Variational Autoencoder for image generation.
The VAE learns a probabilistic latent space representation of images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network that maps input images to latent space parameters (mean and log variance).
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    
    def __init__(self, input_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        """
        Initialize the encoder network.
        
        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
        """
        super(Encoder, self).__init__()
        
        modules = []
        in_channels = input_channels
        
        # Build encoder layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size after convolutions
        self.flatten_size = hidden_dims[-1] * 4 * 4  # Assuming 64x64 input -> 4x4 output
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that reconstructs images from latent space.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    
    def __init__(self, latent_dim=128, output_channels=3, hidden_dims=[256, 128, 64, 32]):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim: Dimension of the latent space
            output_channels: Number of output image channels (default: 3 for RGB)
            hidden_dims: List of hidden layer dimensions (reversed from encoder)
        """
        super(Decoder, self).__init__()
        
        # First fully connected layer
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        self.flatten_size = hidden_dims[0]
        
        modules = []
        
        # Build decoder layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Reconstructed image tensor [batch_size, channels, height, width]
        """
        x = self.fc(z)
        x = x.view(-1, self.flatten_size, 4, 4)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder with reparameterization trick.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    
    def __init__(self, input_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        """
        Initialize the VAE model.
        
        Args:
            input_channels: Number of input image channels
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder and decoder with reversed hidden dimensions
        self.encoder = Encoder(input_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_channels, list(reversed(hidden_dims)))
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input image tensor
            
        Returns:
            reconstructed: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent vector
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z
    
    def generate(self, num_samples=1, device='cpu'):
        """
        Generate new images by sampling from the latent space.
        
        Args:
            num_samples: Number of images to generate
            device: Device to run generation on
            
        Returns:
            Generated images
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated = self.decoder(z)
        return generated
    
    def encode(self, x):
        """
        Encode input images to latent space.
        
        Args:
            x: Input image tensor
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        return self.encoder(x)


def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """
    Calculate VAE loss (reconstruction loss + KL divergence).
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        reconstructed: Reconstructed images
        original: Original input images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
        
    Returns:
        Total loss, reconstruction loss, KL divergence loss
    """
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

