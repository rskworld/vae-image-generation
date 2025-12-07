"""
Configuration file for VAE Image Generation Project

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This file contains configuration parameters for training and inference.
"""

# Model Configuration
MODEL_CONFIG = {
    'input_channels': 3,  # 1 for grayscale, 3 for RGB
    'latent_dim': 128,
    'hidden_dims': [32, 64, 128, 256],  # Encoder/decoder hidden dimensions
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 1e-3,
    'beta': 1.0,  # KL divergence weight
    'dataset': 'mnist',  # Options: 'mnist', 'cifar10', 'celeba'
    'data_dir': './data',
    'output_dir': './outputs',
    'save_model': 'vae_model.pth',
    'device': 'cuda',  # 'cuda' or 'cpu'
}

# Generation Configuration
GENERATION_CONFIG = {
    'num_images': 64,
    'interpolation_steps': 10,
    'output_path': 'generated_images.png',
}

# Paths
PATHS = {
    'data_dir': './data',
    'output_dir': './outputs',
    'samples_dir': './outputs/samples',
    'model_path': './outputs/vae_model.pth',
}

# Data Transformations
TRANSFORMS = {
    'mnist': {
        'size': (64, 64),
        'mean': (0.5,),
        'std': (0.5,),
    },
    'cifar10': {
        'size': (64, 64),
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
    },
}

