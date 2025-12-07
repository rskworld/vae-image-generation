"""
Advanced Training Script with Additional Features

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script provides advanced training features including:
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard logging
- Gradient clipping
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import argparse
import json
from datetime import datetime
from vae_model import VAE, vae_loss
from data_loader import create_dataloader, get_data_transforms


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore(self, model):
        """Restore best weights to model"""
        if self.best_weights:
            model.load_state_dict(self.best_weights)


def train_epoch_advanced(model, dataloader, optimizer, device, beta=1.0, 
                        use_amp=False, grad_clip=None, scaler=None):
    """
    Advanced training epoch with mixed precision and gradient clipping.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with torch.cuda.amp.autocast():
                reconstructed, mu, logvar, z = model(data)
                loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar, beta)
            
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, data, mu, logvar, beta)
            loss.backward()
            
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
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


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save training checkpoint.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load training checkpoint.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def main():
    """
    Main training function with advanced features.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Advanced VAE Training')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for KL divergence')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr-scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--grad-clip', type=float, help='Gradient clipping value')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--tensorboard', action='store_true', help='Use TensorBoard logging')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device setup
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(os.path.join(run_dir, 'logs'))
    
    # Data loading
    train_loader = create_dataloader(
        args.dataset, args.data_dir, args.batch_size,
        shuffle=True, augment=args.augment
    )
    val_loader = create_dataloader(
        args.dataset, args.data_dir, args.batch_size,
        shuffle=False, augment=False
    )
    
    # Determine input channels
    input_channels = 1 if args.dataset == 'mnist' else 3
    
    # Model initialization
    model = VAE(input_channels=input_channels, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Mixed precision scaler
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        print(f'Resumed from epoch {start_epoch}')
    
    print(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_recon, train_kl = train_epoch_advanced(
            model, train_loader, optimizer, device, args.beta,
            args.use_amp, args.grad_clip, scaler
        )
        print(f'Train Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}')
        
        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, args.beta)
        print(f'Val Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}')
        
        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Loss/Recon_Train', train_recon, epoch)
            writer.add_scalar('Loss/Recon_Val', val_recon, epoch)
            writer.add_scalar('Loss/KL_Train', train_kl, epoch)
            writer.add_scalar('Loss/KL_Val', val_kl, epoch)
            if scheduler:
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(run_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            print(f'Best model saved with validation loss: {val_loss:.4f}')
        
        # Generate samples
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                samples = model.generate(num_samples=64, device=device)
                save_image(samples, os.path.join(run_dir, 'samples', f'epoch_{epoch+1}.png'),
                          nrow=8, normalize=True)
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            if early_stopping.restore_best_weights:
                early_stopping.restore(model)
            break
    
    if writer:
        writer.close()
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Results saved to: {run_dir}')


if __name__ == '__main__':
    main()

