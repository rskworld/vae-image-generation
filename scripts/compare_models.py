"""
Model Comparison Script

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script compares multiple trained VAE models.
"""

import torch
import argparse
from torchvision.utils import save_image
from vae_model import VAE
from evaluate import calculate_reconstruction_error
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def compare_models(model_paths, labels, dataset='mnist', device='cpu', num_samples=64):
    """
    Compare multiple models by generating samples and calculating metrics.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    device = torch.device(device)
    input_channels = 1 if dataset == 'mnist' else 3
    
    # Load test data
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    results = []
    
    for model_path, label in zip(model_paths, labels):
        print(f'\nEvaluating {label}...')
        
        # Load model
        model = VAE(input_channels=input_channels, latent_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Calculate reconstruction error
        recon_error = calculate_reconstruction_error(model, test_loader, device)
        
        # Generate samples
        with torch.no_grad():
            samples = model.generate(num_samples=num_samples, device=device)
        
        results.append({
            'label': label,
            'recon_error': recon_error,
            'samples': samples
        })
        
        print(f'  Reconstruction Error: {recon_error:.6f}')
    
    # Save comparison
    print('\nSaving comparison...')
    all_samples = []
    for result in results:
        all_samples.append(result['samples'])
    
    comparison = torch.cat(all_samples, dim=0)
    save_image(comparison, 'model_comparison.png', nrow=num_samples, normalize=True)
    
    # Print summary
    print('\n' + '='*50)
    print('COMPARISON SUMMARY')
    print('='*50)
    for result in results:
        print(f"{result['label']}: {result['recon_error']:.6f}")
    print('='*50)
    print('Comparison image saved to model_comparison.png')


def main():
    """
    Main comparison function.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Compare VAE Models')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model files')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                       help='Labels for each model')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'], help='Dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.labels):
        raise ValueError("Number of models must match number of labels")
    
    compare_models(args.models, args.labels, args.dataset, args.device)


if __name__ == '__main__':
    main()

