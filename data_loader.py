"""
Custom Data Loader with Augmentation

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This module provides custom data loaders with augmentation support.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    
    def __init__(self, root_dir, transform=None, image_size=(64, 64)):
        """
        Initialize custom dataset.
        
        Args:
            root_dir: Root directory containing images
            transform: Optional transform to apply
            image_size: Target image size
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend([f for f in os.listdir(root_dir) if f.lower().endswith(ext.replace('*', ''))])
        
        if not self.image_files:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label


def get_data_transforms(dataset='mnist', augment=False, image_size=64):
    """
    Get data transformation pipeline.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        dataset: Dataset name ('mnist', 'cifar10', 'custom')
        augment: Whether to apply data augmentation
        image_size: Target image size
        
    Returns:
        Transform pipeline
    """
    if dataset == 'mnist':
        if augment:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    elif dataset == 'cifar10':
        if augment:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        # Custom dataset
        if augment:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    return transform


def create_dataloader(dataset='mnist', data_dir='./data', batch_size=128, 
                     shuffle=True, augment=False, image_size=64, num_workers=4):
    """
    Create DataLoader for specified dataset.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    
    Args:
        dataset: Dataset name
        data_dir: Data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        image_size: Image size
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    from torchvision import datasets
    
    transform = get_data_transforms(dataset, augment, image_size)
    
    if dataset == 'mnist':
        dataset_obj = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif dataset == 'cifar10':
        dataset_obj = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    elif dataset == 'custom':
        dataset_obj = CustomImageDataset(data_dir, transform=transform, image_size=(image_size, image_size))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

