"""
Training Script for Variational Autoencoder (VAE)

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script trains a VAE model on image data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import argparse
from vae_model import VAE, vae_loss


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """
    Train the model for one epoch.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: VAE model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to run training on
        beta: Weight for KL divergence term
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, mu, logvar, z = model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item()/len(data):.4f}, '
                  f'Recon: {recon_loss.item()/len(data):.4f}, '
                  f'KL: {kl_loss.item()/len(data):.4f}')
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon_loss / len(dataloader.dataset)
    avg_kl = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def validate(model, dataloader, device, beta=1.0):
    """
    Validate the model.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: VAE model
        dataloader: DataLoader for validation data
        device: Device to run validation on
        beta: Weight for KL divergence term
        
    Returns:
        Average loss for validation set
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            reconstructed, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon_loss / len(dataloader.dataset)
    avg_kl = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def main():
    """
    Main training function.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Train VAE for Image Generation')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for KL divergence')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'celeba'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--save-model', type=str, default='vae_model.pth', help='Model save path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Device setup
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Data loading
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        input_channels = 1
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
        input_channels = 3
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model initialization
    model = VAE(input_channels=input_channels, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.beta)
        print(f'Train Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}')
        
        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, args.beta)
        print(f'Val Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.save_model))
            print(f'Model saved with validation loss: {val_loss:.4f}')
        
        # Generate sample images
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Generate from random latent vectors
                samples = model.generate(num_samples=64, device=device)
                save_image(samples, os.path.join(args.output_dir, 'samples', f'epoch_{epoch+1}.png'),
                          nrow=8, normalize=True)
                
                # Reconstruct some validation images
                val_data, _ = next(iter(val_loader))
                val_data = val_data[:8].to(device)
                recon_data, _, _, _ = model(val_data)
                comparison = torch.cat([val_data, recon_data], dim=0)
                save_image(comparison, os.path.join(args.output_dir, 'samples', f'recon_epoch_{epoch+1}.png'),
                          nrow=8, normalize=True)
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()

