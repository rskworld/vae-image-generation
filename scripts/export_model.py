"""
Model Export Script (ONNX, TorchScript)

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This script exports trained VAE models to various formats.
"""

import torch
import argparse
from vae_model import VAE


def export_to_onnx(model_path, output_path, input_channels=3, latent_dim=128, 
                   image_size=64, device='cpu'):
    """
    Export model to ONNX format.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    try:
        import onnx
    except ImportError:
        print("ONNX not installed. Install with: pip install onnx")
        return
    
    device = torch.device(device)
    model = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_channels, image_size, image_size).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['reconstructed', 'mu', 'logvar', 'z'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f'Model exported to ONNX: {output_path}')


def export_to_torchscript(model_path, output_path, input_channels=3, latent_dim=128,
                         device='cpu'):
    """
    Export model to TorchScript format.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    device = torch.device(device)
    model = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Trace the model
    dummy_input = torch.randn(1, input_channels, 64, 64).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    print(f'Model exported to TorchScript: {output_path}')


def main():
    """
    Main export function.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='Export VAE Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'], required=True,
                       help='Export format')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--input-channels', type=int, default=3, help='Input channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    if args.format == 'onnx':
        export_to_onnx(args.model, args.output, args.input_channels, args.latent_dim, device=args.device)
    elif args.format == 'torchscript':
        export_to_torchscript(args.model, args.output, args.input_channels, args.latent_dim, device=args.device)


if __name__ == '__main__':
    main()

