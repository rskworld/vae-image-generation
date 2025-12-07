"""
Image Generation Script using Trained VAE Model

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script generates new images using a trained VAE model.
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse
import os
from vae_model import VAE


def generate_images(model_path, num_images=64, latent_dim=128, output_path='generated_images.png',
                    input_channels=3, device='cpu'):
    """
    Generate images from a trained VAE model.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model_path: Path to trained model weights
        num_images: Number of images to generate
        latent_dim: Latent dimension of the model
        output_path: Path to save generated images
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        device: Device to run generation on
    """
    device = torch.device(device)
    
    # Load model
    model = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f'Model loaded from {model_path}')
    print(f'Generating {num_images} images...')
    
    # Generate images
    with torch.no_grad():
        generated = model.generate(num_samples=num_images, device=device)
    
    # Save images
    save_image(generated, output_path, nrow=8, normalize=True)
    print(f'Generated images saved to {output_path}')


def interpolate_latent_space(model_path, latent_dim=128, num_steps=10, output_path='interpolation.png',
                            input_channels=3, device='cpu'):
    """
    Interpolate between two random points in latent space.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model_path: Path to trained model weights
        latent_dim: Latent dimension of the model
        num_steps: Number of interpolation steps
        output_path: Path to save interpolated images
        input_channels: Number of input channels
        device: Device to run generation on
    """
    device = torch.device(device)
    
    # Load model
    model = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    # Interpolate
    alphas = torch.linspace(0, 1, num_steps).to(device)
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = model.decoder(z_interp)
            interpolated_images.append(img)
    
    # Concatenate and save
    result = torch.cat(interpolated_images, dim=0)
    save_image(result, output_path, nrow=num_steps, normalize=True)
    print(f'Interpolation saved to {output_path}')


def main():
    """
    Main function for image generation.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Generate Images using Trained VAE')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--num-images', type=int, default=64, help='Number of images to generate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--input-channels', type=int, default=3, help='Input channels (1 or 3)')
    parser.add_argument('--output', type=str, default='generated_images.png', help='Output path')
    parser.add_argument('--interpolate', action='store_true', help='Generate interpolation instead')
    parser.add_argument('--interpolation-steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.interpolate:
        interpolate_latent_space(
            args.model, args.latent_dim, args.interpolation_steps,
            args.output, args.input_channels, args.device
        )
    else:
        generate_images(
            args.model, args.num_images, args.latent_dim,
            args.output, args.input_channels, args.device
        )


if __name__ == '__main__':
    main()

