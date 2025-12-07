"""
Model Evaluation Script

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script evaluates trained VAE models using various metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse
from vae_model import VAE, vae_loss
from utils import count_parameters


def calculate_reconstruction_error(model, dataloader, device):
    """
    Calculate reconstruction error (MSE) on test set.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Average reconstruction error
    """
    model.eval()
    total_error = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            reconstructed, _, _, _ = model(data)
            error = nn.functional.mse_loss(reconstructed, data, reduction='sum')
            total_error += error.item()
            total_samples += len(data)
    
    return total_error / total_samples


def calculate_latent_statistics(model, dataloader, device):
    """
    Calculate statistics of latent space distribution.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with latent space statistics
    """
    model.eval()
    all_mus = []
    all_logvars = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            all_mus.append(mu.cpu().numpy())
            all_logvars.append(logvar.cpu().numpy())
    
    all_mus = np.concatenate(all_mus, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)
    
    stats = {
        'mu_mean': np.mean(all_mus),
        'mu_std': np.std(all_mus),
        'logvar_mean': np.mean(all_logvars),
        'logvar_std': np.std(all_logvars),
        'latent_dim': all_mus.shape[1]
    }
    
    return stats


def evaluate_model(model_path, dataset='mnist', batch_size=128, device='cpu', input_channels=1):
    """
    Comprehensive model evaluation.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        model_path: Path to trained model
        dataset: Dataset name
        batch_size: Batch size for evaluation
        device: Device to run on
        input_channels: Number of input channels
    """
    device = torch.device(device)
    
    # Load model
    model = VAE(input_channels=input_channels, latent_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f'Model loaded from {model_path}')
    print(f'Total parameters: {count_parameters(model):,}')
    
    # Load test data
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate metrics
    print('\nEvaluating model...')
    recon_error = calculate_reconstruction_error(model, test_loader, device)
    latent_stats = calculate_latent_statistics(model, test_loader, device)
    
    # Print results
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'Reconstruction Error (MSE): {recon_error:.6f}')
    print(f'\nLatent Space Statistics:')
    print(f'  Mean (μ): {latent_stats["mu_mean"]:.4f} ± {latent_stats["mu_std"]:.4f}')
    print(f'  Log Variance: {latent_stats["logvar_mean"]:.4f} ± {latent_stats["logvar_std"]:.4f}')
    print(f'  Latent Dimension: {latent_stats["latent_dim"]}')
    print('='*50)


def main():
    """
    Main evaluation function.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Evaluate VAE Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset used for training')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--input-channels', type=int, default=1, help='Input channels (1 or 3)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model,
        args.dataset,
        args.batch_size,
        args.device,
        args.input_channels
    )


if __name__ == '__main__':
    main()

