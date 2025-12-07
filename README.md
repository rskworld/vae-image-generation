# Variational Autoencoder (VAE) for Image Generation

**Author:** Molla Samser  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277  
**Designer & Tester:** Rima Khatun

## Description

This project implements a Variational Autoencoder (VAE) for image generation. Unlike standard autoencoders, VAE learns a probabilistic latent space by encoding images to a distribution and sampling from it. The architecture includes encoder, decoder, reparameterization trick, and KL divergence loss for learning meaningful latent representations.

## Features

### Core Features
- ✅ Variational Autoencoder architecture
- ✅ Probabilistic latent space
- ✅ Reparameterization trick
- ✅ Image generation from latent space
- ✅ KL divergence regularization
- ✅ Support for multiple datasets (MNIST, CIFAR-10, Custom)
- ✅ Latent space interpolation
- ✅ Training and inference scripts
- ✅ Jupyter notebook for interactive use

### Advanced Features
- ✅ Learning rate scheduling
- ✅ Early stopping mechanism
- ✅ Model checkpointing and resume training
- ✅ TensorBoard logging
- ✅ Gradient clipping
- ✅ Mixed precision training (AMP)
- ✅ Data augmentation
- ✅ Custom dataset support
- ✅ Model evaluation metrics
- ✅ Advanced visualization tools
- ✅ Model export (ONNX, TorchScript)
- ✅ REST API for image generation
- ✅ Unit tests
- ✅ Comprehensive documentation

## Technologies

- Python
- PyTorch
- TensorFlow/Keras (optional)
- VAE (Variational Autoencoder)
- Latent Space
- Reparameterization Trick
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rskworld/vae-image-generation.git
cd vae-image-generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the VAE model using the training script:

```bash
python train.py --dataset mnist --epochs 50 --batch-size 128 --latent-dim 128
```

Options:
- `--dataset`: Dataset to use (mnist, cifar10)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--latent-dim`: Dimension of latent space
- `--beta`: Weight for KL divergence term (default: 1.0)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device to use (cuda/cpu)

### Image Generation

Generate new images from a trained model:

```bash
python generate.py --model outputs/vae_model.pth --num-images 64 --output generated.png
```

Options:
- `--model`: Path to trained model
- `--num-images`: Number of images to generate
- `--output`: Output path for generated images
- `--interpolate`: Generate interpolation between two latent vectors

### Advanced Training

Train with advanced features (learning rate scheduling, early stopping, TensorBoard):

```bash
python train_advanced.py --dataset mnist --epochs 50 --tensorboard --early-stopping 10 --lr-scheduler
```

### Model Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model outputs/vae_model.pth --dataset mnist
```

### Visualization

Create advanced visualizations:

```bash
python visualize.py --model outputs/vae_model.pth --mode manifold --output latent_manifold.png
```

### REST API

Start the REST API server:

```bash
python api.py --model outputs/vae_model.pth --port 5000
```

Then use the API endpoints:
- `GET /health` - Health check
- `POST /generate` - Generate images (JSON: `{"num_images": 10}`)
- `POST /interpolate` - Interpolate in latent space (JSON: `{"num_steps": 10}`)

### Jupyter Notebook

For interactive use, open the Jupyter notebook:

```bash
jupyter notebook VAE_Image_Generation.ipynb
```

## Project Structure

```
vae-image-generation/
├── vae_model.py              # VAE model implementation
├── train.py                  # Basic training script
├── train_advanced.py         # Advanced training with additional features
├── generate.py               # Image generation script
├── evaluate.py               # Model evaluation script
├── visualize.py              # Advanced visualization tools
├── data_loader.py            # Custom data loaders with augmentation
├── utils.py                  # Utility functions
├── api.py                    # REST API for image generation
├── config.py                 # Configuration file
├── setup.py                  # Package setup script
├── VAE_Image_Generation.ipynb  # Jupyter notebook
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # License file
├── index.html                # Demo web page
├── data/                     # Data directory
├── outputs/                  # Output directory
│   ├── samples/             # Generated samples
│   ├── models/              # Saved models
│   └── logs/                # Training logs
├── models/                   # Model storage
├── tests/                    # Unit tests
│   ├── __init__.py
│   └── test_vae_model.py
└── scripts/                  # Utility scripts
    └── export_model.py      # Model export utilities
```

## Model Architecture

The VAE consists of:

1. **Encoder**: Convolutional layers that map input images to latent space parameters (μ, σ)
2. **Reparameterization Trick**: Samples latent vectors z from N(μ, σ²)
3. **Decoder**: Transposed convolutional layers that reconstruct images from latent vectors

### Loss Function

The VAE loss combines:
- **Reconstruction Loss**: MSE between original and reconstructed images
- **KL Divergence**: Regularization term to ensure latent space follows standard normal distribution

```
Loss = Reconstruction Loss + β × KL Divergence
```

## Examples

### Training on MNIST

```bash
python train.py --dataset mnist --epochs 50 --batch-size 128 --latent-dim 128 --beta 1.0
```

### Generating Images

```bash
python generate.py --model outputs/vae_model.pth --num-images 64 --output samples.png
```

### Latent Space Interpolation

```bash
python generate.py --model outputs/vae_model.pth --interpolate --interpolation-steps 10 --output interpolation.png
```

## Results

After training, you can:
- Generate new images from random latent vectors
- Reconstruct input images
- Interpolate between points in latent space
- Visualize the learned latent space

## License

This project is provided for educational purposes.

## Contact

For questions, support, or contributions:
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in, support@rskworld.in
- **Phone:** +91 93305 39277

## Acknowledgments

- **Author:** Molla Samser
- **Designer & Tester:** Rima Khatun
- **Website:** https://rskworld.in

---

**Note:** This project is part of the RSK World collection of free programming resources and source code.

