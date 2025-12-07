"""
Advanced Visualization Script for VAE

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script provides advanced visualization capabilities for VAE models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from vae_model import VAE
from utils import visualize_latent_space


def plot_training_curves(log_file, save_path='training_curves.png'):
    """
    Plot training curves from log file.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    # This would read from a log file if implemented
    pass


def visualize_latent_manifold(model, latent_dim, device, save_path='latent_manifold.png', grid_size=10):
    """
    Visualize 2D latent space manifold.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: Trained VAE model
        latent_dim: Latent dimension
        device: Device to run on
        save_path: Path to save visualization
        grid_size: Size of the grid for visualization
    """
    model.eval()
    
    # Create 2D grid in latent space (using first 2 dimensions)
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Generate images for each point in grid
    images = []
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                # Create latent vector with first 2 dims from grid, rest zeros
                z = torch.zeros(1, latent_dim).to(device)
                z[0, 0] = xx[i, j]
                z[0, 1] = yy[i, j]
                
                img = model.decoder(z)
                images.append(img)
    
    # Create grid of images
    grid = torch.cat(images, dim=0)
    save_image(grid, save_path, nrow=grid_size, normalize=True)
    print(f'Latent manifold saved to {save_path}')


def compare_models(model_paths, labels, num_samples=64, device='cpu', save_path='model_comparison.png'):
    """
    Compare multiple trained models by generating samples.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model_paths: List of paths to model files
        labels: List of labels for each model
        num_samples: Number of samples to generate
        device: Device to run on
        save_path: Path to save comparison
    """
    device = torch.device(device)
    all_samples = []
    
    for model_path, label in zip(model_paths, labels):
        # Load model (assuming same architecture)
        model = VAE(input_channels=3, latent_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            samples = model.generate(num_samples=num_samples, device=device)
            all_samples.append(samples)
    
    # Concatenate all samples
    comparison = torch.cat(all_samples, dim=0)
    save_image(comparison, save_path, nrow=num_samples, normalize=True)
    print(f'Model comparison saved to {save_path}')


def create_interpolation_video(model_path, latent_dim=128, num_frames=60, device='cpu', output_dir='interpolation_frames'):
    """
    Create frames for latent space interpolation video.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model_path: Path to trained model
        latent_dim: Latent dimension
        num_frames: Number of frames to generate
        device: Device to run on
        output_dir: Directory to save frames
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    # Load model
    model = VAE(input_channels=3, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    # Interpolate
    alphas = torch.linspace(0, 1, num_frames).to(device)
    
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = model.decoder(z_interp)
            save_image(img, os.path.join(output_dir, f'frame_{i:04d}.png'), normalize=True)
    
    print(f'Interpolation frames saved to {output_dir}')
    print(f'Use ffmpeg to create video: ffmpeg -r 10 -i {output_dir}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p output.mp4')


def main():
    """
    Main visualization function.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Visualize VAE Model')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['manifold', 'compare', 'interpolation'],
                       default='manifold', help='Visualization mode')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='visualization.png', help='Output path')
    
    args = parser.parse_args()
    
    if args.mode == 'manifold' and args.model:
        visualize_latent_manifold(
            torch.load(args.model, map_location=args.device) if isinstance(args.model, str) else args.model,
            args.latent_dim,
            args.device,
            args.output
        )
    elif args.mode == 'interpolation' and args.model:
        create_interpolation_video(
            args.model,
            args.latent_dim,
            device=args.device
        )


if __name__ == '__main__':
    main()

