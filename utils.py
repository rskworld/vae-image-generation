"""
Utility Functions for VAE Image Generation

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This module contains utility functions for data loading, visualization, and model utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def denormalize(tensor, mean=(0.5,), std=(0.5,)):
    """
    Denormalize a tensor image.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def save_reconstructions(original, reconstructed, save_path, nrow=8):
    """
    Save side-by-side comparison of original and reconstructed images.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        save_path: Path to save the comparison
        nrow: Number of images per row
    """
    comparison = torch.cat([original, reconstructed], dim=0)
    grid = make_grid(comparison, nrow=nrow, normalize=True)
    
    # Convert to numpy and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_latent_space(model, dataloader, device, save_path='latent_space.png'):
    """
    Visualize the latent space using t-SNE or PCA (2D projection).
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for images
        device: Device to run on
        save_path: Path to save visualization
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn not installed. Skipping latent space visualization.")
        return
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Use PCA for dimensionality reduction if latent dim > 2
    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
    else:
        latents_2d = latents
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Latent space visualization saved to {save_path}')


def create_gif_from_images(image_dir, output_path, duration=0.5):
    """
    Create a GIF from a sequence of images.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save GIF
        duration: Duration between frames in seconds
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Cannot create GIF.")
        return
    
    images = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        images.append(Image.open(img_path))
    
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:],
                      duration=duration * 1000, loop=0)
        print(f'GIF saved to {output_path}')


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

