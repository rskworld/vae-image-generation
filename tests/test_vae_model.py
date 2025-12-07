"""
Unit tests for VAE model

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun
"""

import torch
import unittest
from vae_model import VAE, Encoder, Decoder, vae_loss


class TestVAE(unittest.TestCase):
    """Test cases for VAE model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.input_channels = 3
        self.latent_dim = 128
        self.image_size = 64
        self.device = 'cpu'
        
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = Encoder(input_channels=self.input_channels, latent_dim=self.latent_dim)
        x = torch.randn(self.batch_size, self.input_channels, self.image_size, self.image_size)
        mu, logvar = encoder(x)
        
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        decoder = Decoder(latent_dim=self.latent_dim, output_channels=self.input_channels)
        z = torch.randn(self.batch_size, self.latent_dim)
        output = decoder(z)
        
        self.assertEqual(output.shape, (self.batch_size, self.input_channels, self.image_size, self.image_size))
    
    def test_vae_forward(self):
        """Test VAE forward pass"""
        model = VAE(input_channels=self.input_channels, latent_dim=self.latent_dim)
        x = torch.randn(self.batch_size, self.input_channels, self.image_size, self.image_size)
        reconstructed, mu, logvar, z = model(x)
        
        self.assertEqual(reconstructed.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
    
    def test_vae_generate(self):
        """Test VAE image generation"""
        model = VAE(input_channels=self.input_channels, latent_dim=self.latent_dim)
        num_samples = 8
        generated = model.generate(num_samples=num_samples, device=self.device)
        
        self.assertEqual(generated.shape, (num_samples, self.input_channels, self.image_size, self.image_size))
    
    def test_vae_loss(self):
        """Test VAE loss calculation"""
        batch_size = 4
        reconstructed = torch.randn(batch_size, 3, 64, 64)
        original = torch.randn(batch_size, 3, 64, 64)
        mu = torch.randn(batch_size, 128)
        logvar = torch.randn(batch_size, 128)
        
        total_loss, recon_loss, kl_loss = vae_loss(reconstructed, original, mu, logvar, beta=1.0)
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(recon_loss, torch.Tensor)
        self.assertIsInstance(kl_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)


if __name__ == '__main__':
    unittest.main()

