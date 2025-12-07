"""
REST API for VAE Image Generation

Author: Molla Samser
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Designer & Tester: Rima Khatun

This module provides a Flask REST API for generating images using trained VAE models.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import io
from PIL import Image
import base64
import argparse
from vae_model import VAE
import os

app = Flask(__name__)
CORS(app)

# Global model variable
model = None
device = None


def load_model(model_path, input_channels=3, latent_dim=128):
    """
    Load trained VAE model.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_channels=input_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Model loaded from {model_path}')


def image_to_base64(image_tensor):
    """
    Convert image tensor to base64 string.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    # Denormalize and convert to PIL
    img = image_tensor.squeeze(0).cpu()
    img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate images from random latent vectors.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() or {}
        num_images = data.get('num_images', 1)
        
        with torch.no_grad():
            generated = model.generate(num_samples=num_images, device=device)
        
        # Convert to base64
        images = []
        for i in range(num_images):
            img_str = image_to_base64(generated[i])
            images.append(img_str)
        
        return jsonify({
            'success': True,
            'num_images': num_images,
            'images': images
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/interpolate', methods=['POST'])
def interpolate():
    """
    Interpolate between two latent vectors.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() or {}
        num_steps = data.get('num_steps', 10)
        
        z1 = torch.randn(1, model.latent_dim).to(device)
        z2 = torch.randn(1, model.latent_dim).to(device)
        
        alphas = torch.linspace(0, 1, num_steps).to(device)
        images = []
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = model.decoder(z_interp)
                img_str = image_to_base64(img)
                images.append(img_str)
        
        return jsonify({
            'success': True,
            'num_steps': num_steps,
            'images': images
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """
    Main function to run the API server.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in
    Phone: +91 93305 39277
    Designer & Tester: Rima Khatun
    """
    parser = argparse.ArgumentParser(description='VAE Image Generation API')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input-channels', type=int, default=3, help='Input channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model, args.input_channels, args.latent_dim)
    
    # Run server
    print(f'Starting API server on {args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

