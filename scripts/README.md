# Scripts Directory

**Author:** Molla Samser  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in, support@rskworld.in  
**Phone:** +91 93305 39277  
**Designer & Tester:** Rima Khatun

This directory contains utility scripts for the VAE Image Generation project.

## Available Scripts

### export_model.py

Export trained models to different formats (ONNX, TorchScript).

**Usage:**
```bash
python scripts/export_model.py --model outputs/vae_model.pth --format onnx --output model.onnx
python scripts/export_model.py --model outputs/vae_model.pth --format torchscript --output model.pt
```

### compare_models.py

Compare multiple trained models by generating samples and calculating metrics.

**Usage:**
```bash
python scripts/compare_models.py --models model1.pth model2.pth --labels "Model 1" "Model 2" --dataset mnist
```

## Adding New Scripts

When adding new scripts, please:
1. Include author information in file header
2. Add proper documentation
3. Use argparse for command-line arguments
4. Follow the project's code style

