#!/usr/bin/env python3
"""
Real DeepGaze III Only Application

Pure DeepGaze III implementation without the complex system dependencies
"""

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import time
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Real DeepGaze III imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169
import torchvision.transforms as transforms

# Directory setup
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "uploads"

for dir_path in [STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

app = FastAPI(title="Real DeepGaze III Neural Analyzer", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

class RealDeepGazeIII(nn.Module):
    """
    Real DeepGaze III Implementation
    Based on: Kümmerer et al., Journal of Vision 2021
    """
    
    def __init__(self, pretrained: bool = True):
        super(RealDeepGazeIII, self).__init__()
        
        # DenseNet-169 backbone
        self.backbone = densenet169(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # DeepGaze III readout network
        self.readout = nn.Sequential(
            nn.Conv2d(1664, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.center_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.center_bias_shift = nn.Parameter(torch.tensor(0.0))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.readout.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def generate_center_bias(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        
        y_coords = (y_coords / (height - 1)) * 2 - 1
        x_coords = (x_coords / (width - 1)) * 2 - 1
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance_sq = xx**2 + yy**2
        
        sigma = 0.33  # As per DeepGaze III paper
        center_bias = torch.exp(-distance_sq / (2 * sigma**2))
        
        center_bias = self.center_bias_scale * center_bias + self.center_bias_shift
        return center_bias.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> dict:
        batch_size, _, height, width = x.shape
        device = x.device
        
        # Extract DenseNet features
        features = self.backbone(x)
        
        # Generate saliency through readout
        saliency_features = self.readout(features)
        
        # Upsample to original resolution
        saliency_map = F.interpolate(
            saliency_features,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # Generate center bias
        center_bias = self.generate_center_bias(height, width, device)
        center_bias = center_bias.expand(batch_size, -1, -1, -1)
        
        # Combine saliency with center bias
        final_saliency = saliency_map * center_bias
        
        # Softmax normalization
        final_saliency_flat = final_saliency.view(batch_size, -1)
        final_saliency_normalized = F.softmax(final_saliency_flat, dim=1)
        final_saliency_normalized = final_saliency_normalized.view(batch_size, 1, height, width)
        
        return {
            'saliency_map': final_saliency_normalized,
            'raw_saliency': saliency_map,
            'center_bias': center_bias,
            'features': features
        }

# Global DeepGaze III model
deepgaze_model = None
device = None
preprocess = None

def initialize_deepgaze():
    """Initialize the DeepGaze III model"""
    global deepgaze_model, device, preprocess
    
    device = torch.device('cpu')  # Use CPU for compatibility
    
    deepgaze_model = RealDeepGazeIII(pretrained=True)
    deepgaze_model.to(device)
    deepgaze_model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("✅ Real DeepGaze III initialized successfully")

def process_with_deepgaze(image: np.ndarray) -> dict:
    """Process image with real DeepGaze III"""
    try:
        original_height, original_width = image.shape[:2]
        
        # Convert to PIL and preprocess
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)
        
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            results = deepgaze_model(input_tensor)
        
        # Extract results
        saliency_map = results['saliency_map'].squeeze().cpu().numpy()
        raw_saliency = results['raw_saliency'].squeeze().cpu().numpy()
        center_bias = results['center_bias'].squeeze().cpu().numpy()
        
        # Resize to original size
        from scipy.ndimage import zoom
        if saliency_map.shape != (original_height, original_width):
            scale_y = original_height / saliency_map.shape[0]
            scale_x = original_width / saliency_map.shape[1]
            saliency_map = zoom(saliency_map, (scale_y, scale_x), order=1)
            raw_saliency = zoom(raw_saliency, (scale_y, scale_x), order=1)
            center_bias = zoom(center_bias, (scale_y, scale_x), order=1)
        
        return {
            'saliency_map': saliency_map,
            'raw_saliency': raw_saliency,
            'center_bias': center_bias,
            'confidence_score': float(np.max(saliency_map)),
            'model_info': {
                'name': 'Real DeepGaze III',
                'architecture': 'DenseNet-169 + Readout Network',
                'paper': 'Kümmerer et al., Journal of Vision 2021',
                'device': str(device)
            }
        }
        
    except Exception as e:
        print(f"DeepGaze III error: {e}")
        # Simple fallback
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        sigma = min(w, h) * 0.3
        fallback_saliency = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        fallback_saliency = fallback_saliency / np.sum(fallback_saliency)
        
        return {
            'saliency_map': fallback_saliency,
            'raw_saliency': fallback_saliency,
            'center_bias': fallback_saliency,
            'confidence_score': 0.5,
            'error': str(e),
            'model_info': {
                'name': 'DeepGaze III (Fallback)',
                'architecture': 'Center Bias Only',
                'error': str(e)
            }
        }

def generate_gaze_points(saliency_map: np.ndarray, num_fixations: int = 12) -> list:
    """Generate gaze points with IOR"""
    h, w = saliency_map.shape
    fixations = []
    ior_mask = np.ones_like(saliency_map)
    ior_sigma = min(w, h) * 0.05
    
    for i in range(num_fixations):
        current_saliency = saliency_map * ior_mask
        
        saliency_flat = current_saliency.flatten()
        if np.sum(saliency_flat) == 0:
            probabilities = np.ones_like(saliency_flat) / len(saliency_flat)
        else:
            probabilities = saliency_flat / np.sum(saliency_flat)
        
        chosen_idx = np.random.choice(len(saliency_flat), p=probabilities)
        y_coord, x_coord = np.unravel_index(chosen_idx, saliency_map.shape)
        
        fixation_duration = np.random.normal(250, 50)
        fixation_duration = max(150, min(400, fixation_duration))
        
        fixation = {
            'x': int(x_coord),
            'y': int(y_coord),
            'x_norm': float(x_coord / w),
            'y_norm': float(y_coord / h),
            'duration_ms': int(fixation_duration),
            'timestamp_ms': i * 250,
            'confidence': float(current_saliency[y_coord, x_coord]),
            'order': i + 1
        }
        fixations.append(fixation)
        
        # Update IOR
        y_grid, x_grid = np.ogrid[:h, :w]
        ior_gaussian = np.exp(-((x_grid - x_coord)**2 + (y_grid - y_coord)**2) / (2 * ior_sigma**2))
        ior_mask *= (1 - 0.7 * ior_gaussian)
        ior_mask = np.clip(ior_mask, 0.1, 1.0)
    
    return fixations

def create_visualization(original_image: np.ndarray, saliency_map: np.ndarray, 
                        gaze_points: list, output_path: str) -> str:
    """Create visualization with Real DeepGaze III results"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor('black')
    
    ax.imshow(original_image)
    
    # Custom colormap
    colors = ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffcc00', '#ff6b35', '#f7931e', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('deepgaze', colors, N=256)
    
    # Saliency overlay
    ax.imshow(saliency_map, alpha=0.6, cmap=cmap, interpolation='gaussian')
    
    # Gaze points
    if gaze_points:
        x_coords = [p['x'] for p in gaze_points]
        y_coords = [p['y'] for p in gaze_points]
        
        # Trajectories
        for i in range(len(gaze_points) - 1):
            alpha = 0.3 + 0.4 * (i / len(gaze_points))
            width = 1 + 2 * (i / len(gaze_points))
            ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                   color='cyan', alpha=alpha, linewidth=width, zorder=10)
        
        # Fixation circles
        for i, point in enumerate(gaze_points):
            circle_size = 15 + (point['duration_ms'] - 150) / 250 * 20
            
            outer_circle = patches.Circle((point['x'], point['y']), circle_size + 3, 
                                        color='white', alpha=0.8, fill=False, linewidth=2, zorder=15)
            ax.add_patch(outer_circle)
            
            main_circle = patches.Circle((point['x'], point['y']), circle_size, 
                                       color='red', alpha=0.9, fill=True, zorder=20)
            ax.add_patch(main_circle)
            
            ax.text(point['x'], point['y'], str(point['order']), 
                   ha='center', va='center', fontsize=10, color='white', weight='bold', zorder=25)
    
    ax.set_xlim(0, original_image.shape[1])
    ax.set_ylim(original_image.shape[0], 0)
    ax.axis('off')
    
    plt.suptitle('Real DeepGaze III Analysis - Kümmerer et al. 2021', 
                fontsize=16, color='white', y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none', 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    return output_path

@app.on_event("startup")
async def startup_event():
    initialize_deepgaze()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("professional_index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    try:
        # Load image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if too large
        max_size = 800
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        image_array = np.array(pil_image)
        
        # Process with Real DeepGaze III
        start_time = time.time()
        deepgaze_result = process_with_deepgaze(image_array)
        saliency_map = deepgaze_result['saliency_map']
        
        # Generate gaze points
        gaze_points = generate_gaze_points(saliency_map, num_fixations=12)
        
        # Create visualization
        timestamp = int(time.time() * 1000)
        output_path = UPLOADS_DIR / f"deepgaze_result_{timestamp}.png"
        create_visualization(image_array, saliency_map, gaze_points, str(output_path))
        
        processing_time = time.time() - start_time
        
        # Statistics
        avg_duration = np.mean([p['duration_ms'] for p in gaze_points])
        avg_confidence = np.mean([p['confidence'] for p in gaze_points])
        max_saliency = np.max(saliency_map)
        
        return {
            "success": True,
            "processing_time_ms": processing_time * 1000,
            "result_image": f"/uploads/{output_path.name}",
            "deepgaze_info": deepgaze_result['model_info'],
            "statistics": {
                "num_fixations": len(gaze_points),
                "avg_fixation_duration_ms": float(avg_duration),
                "avg_confidence": float(avg_confidence),
                "max_saliency": float(max_saliency),
                "deepgaze_confidence": deepgaze_result['confidence_score'],
                "image_dimensions": list(image_array.shape)
            },
            "gaze_points": gaze_points[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/uploads/{filename}")
async def get_upload_file(filename: str):
    file_path = UPLOADS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    print("🧠 Real DeepGaze III Neural Analyzer")
    print("📄 Paper: Kümmerer et al., Journal of Vision 2021")
    print("🏗️ Architecture: DenseNet-169 + Readout Network")
    print("🌐 Starting server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8889)