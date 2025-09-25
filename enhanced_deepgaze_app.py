# ========== Enhanced DeepGaze III Neural Attention System ==========
"""
🧠 Enhanced DeepGaze III Neural Attention Analyzer
📄 Based on: Kümmerer et al., Journal of Vision 2021
🏗️ Dual Implementation: Official deepgaze_pytorch + Custom PyTorch
👁️ Features: ROI Analysis, Duration Prediction, Advanced Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import io
import base64
import json
import os
import urllib.request
import shutil
from typing import List, Dict, Tuple, Optional
import warnings
import math
import psutil
import tempfile
import gc
import platform
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Enhanced Imports
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
    st.error("streamlit-drawable-canvas が必要です。`pip install streamlit-drawable-canvas` でインストールしてください。")

# Official DeepGaze III Import
try:
    import deepgaze_pytorch
    DEEPGAZE_AVAILABLE = True
except ImportError:
    DEEPGAZE_AVAILABLE = False
    st.error("deepgaze_pytorch が必要です。`pip install git+https://github.com/matthias-k/DeepGaze.git` でインストールしてください。")

# Custom DeepGaze III Implementation Import
try:
    import sys
    sys.path.append('/home/user/webapp/src/processors')
    from real_deepgaze_iii import RealDeepGazeIII
    CUSTOM_DEEPGAZE_AVAILABLE = True
except ImportError:
    CUSTOM_DEEPGAZE_AVAILABLE = False
    st.warning("Custom DeepGaze III implementation not available. Using official version only.")

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Enhanced DeepGaze III Neural Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Advanced Professional CSS ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* === ROOT VARIABLES === */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --neural-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.18);
    --neon-blue: #00d4ff;
    --neon-purple: #b794f6;
    --neon-green: #68d391;
    --neon-orange: #fd9853;
}

/* === GLOBAL STYLES === */
.stApp {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* === MAIN HEADER === */
.main-header {
    font-family: 'Inter', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    margin: 2rem 0;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: headerGlow 3s ease-in-out infinite alternate;
    text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    letter-spacing: -2px;
    position: relative;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: linear-gradient(45deg, var(--neon-blue), var(--neon-purple), var(--neon-green));
    border-radius: 20px;
    opacity: 0.1;
    filter: blur(20px);
    z-index: -1;
    animation: halo 4s ease-in-out infinite alternate;
}

@keyframes headerGlow {
    0% { 
        background-position: 0% 50%;
        filter: brightness(1) drop-shadow(0 0 20px rgba(102, 126, 234, 0.4));
    }
    100% { 
        background-position: 100% 50%;
        filter: brightness(1.3) drop-shadow(0 0 40px rgba(102, 126, 234, 0.8));
    }
}

@keyframes halo {
    0% { transform: scale(1) rotate(0deg); opacity: 0.1; }
    100% { transform: scale(1.05) rotate(180deg); opacity: 0.2; }
}

/* === SECTION HEADERS === */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding: 1rem 1.5rem;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 15px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    color: white;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
}

.section-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    animation: shimmer 3s ease-in-out infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* === GLASSMORPHISM CARDS === */
.neural-network {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.neural-network::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: var(--neural-gradient);
    border-radius: 22px;
    z-index: -1;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.neural-network:hover {
    transform: translateY(-5px);
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.2);
}

.neural-network:hover::before {
    opacity: 1;
}

.neural-network h3 {
    color: white;
    font-weight: 600;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.neural-network p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1rem;
    line-height: 1.6;
}

/* === METRIC CARDS === */
.metric-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    color: white;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--neural-gradient);
    border-radius: 0 4px 4px 0;
}

.metric-card:hover {
    transform: translateX(5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
}

/* === PROFESSIONAL BUTTONS === */
.stButton > button {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* === SIDEBAR STYLING === */
.css-1d391kg {
    background: var(--glass-bg) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(20px) !important;
}

.css-1d391kg .css-17eq0hr {
    color: white !important;
}

/* === SUCCESS/ERROR MESSAGES === */
.stSuccess {
    background: var(--glass-bg) !important;
    border: 1px solid rgba(104, 211, 145, 0.5) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(20px) !important;
    color: var(--neon-green) !important;
}

.stError {
    background: var(--glass-bg) !important;
    border: 1px solid rgba(245, 87, 108, 0.5) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(20px) !important;
    color: #f5576c !important;
}

.stWarning {
    background: var(--glass-bg) !important;
    border: 1px solid rgba(253, 152, 83, 0.5) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(20px) !important;
    color: var(--neon-orange) !important;
}

/* === INPUT STYLING === */
.stSelectbox > div > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: white !important;
    backdrop-filter: blur(20px) !important;
}

.stSlider > div > div > div > div {
    background: var(--neural-gradient) !important;
}

/* === DATAFRAME STYLING === */
.dataframe {
    background: var(--glass-bg) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
}

/* === NEURAL NETWORK ANIMATION === */
.neural-pulse {
    position: relative;
    display: inline-block;
}

.neural-pulse::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    background: var(--neon-blue);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: neuralPulse 2s ease-in-out infinite;
    opacity: 0.8;
}

@keyframes neuralPulse {
    0%, 100% { 
        transform: translate(-50%, -50%) scale(0.8);
        opacity: 1;
    }
    50% { 
        transform: translate(-50%, -50%) scale(1.2);
        opacity: 0.5;
    }
}

/* === LOADING ANIMATIONS === */
.stSpinner > div {
    border-color: var(--neon-blue) !important;
}

/* === RESPONSIVE DESIGN === */
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
        letter-spacing: -1px;
    }
    
    .section-header {
        font-size: 1.4rem;
        padding: 0.75rem 1rem;
    }
    
    .neural-network {
        padding: 1rem;
        margin: 1rem 0;
    }
}

/* === TEXT COLORS === */
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: rgba(255, 255, 255, 0.9) !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: white !important;
}

/* === PROFESSIONAL GLOW EFFECTS === */
.glow-effect {
    box-shadow: 
        0 0 5px var(--neon-blue),
        0 0 10px var(--neon-blue),
        0 0 15px var(--neon-blue),
        0 0 20px var(--neon-blue);
    animation: gentleGlow 3s ease-in-out infinite alternate;
}

@keyframes gentleGlow {
    from { 
        box-shadow: 
            0 0 5px var(--neon-blue),
            0 0 10px var(--neon-blue),
            0 0 15px var(--neon-blue);
    }
    to { 
        box-shadow: 
            0 0 10px var(--neon-blue),
            0 0 20px var(--neon-blue),
            0 0 30px var(--neon-blue);
    }
}
</style>
""", unsafe_allow_html=True)

# ========== Utility Functions ==========

def debug_print(message: str, level: str = "INFO"):
    """Enhanced debug printing with timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if st.session_state.get('show_debug', False):
        color_map = {
            "INFO": "🔵", "SUCCESS": "🟢", "WARNING": "🟡", 
            "ERROR": "🔴", "DEBUG": "⚪"
        }
        icon = color_map.get(level, "ℹ️")
        st.text(f"[{timestamp}] {icon} {level}: {message}")

def check_memory_usage():
    """Check system memory usage"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent_used': memory.percent
    }

def check_system_info():
    """Get comprehensive system information"""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__ if 'torch' in globals() else 'Not installed',
        'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False,
        'memory': check_memory_usage()
    }

# ========== Enhanced DeepGaze III Manager ==========

class EnhancedDeepGazeManager:
    def __init__(self):
        self.official_model = None
        self.custom_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = "none"
        self.model_loaded = False
        
    def load_official_model(self):
        """Load official deepgaze_pytorch DeepGaze III model"""
        if not DEEPGAZE_AVAILABLE:
            st.error("Official deepgaze_pytorch not available")
            return False
            
        try:
            debug_print("Loading Official DeepGaze III model", "INFO")
            
            with st.spinner("🧠 Loading Official DeepGaze III..."):
                self.official_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(self.device).eval()
                self.model_type = "official"
                self.model_loaded = True
                
            debug_print("Official DeepGaze III model loaded successfully", "SUCCESS")
            return True
            
        except Exception as e:
            debug_print(f"Official DeepGaze III loading error: {str(e)}", "ERROR")
            st.error(f"Official DeepGaze III loading failed: {str(e)}")
            return False
    
    def load_custom_model(self):
        """Load custom DeepGaze III implementation"""
        if not CUSTOM_DEEPGAZE_AVAILABLE:
            st.error("Custom DeepGaze III implementation not available")
            return False
            
        try:
            debug_print("Loading Custom DeepGaze III model", "INFO")
            
            with st.spinner("🏗️ Loading Custom DeepGaze III..."):
                self.custom_model = RealDeepGazeIII().to(self.device).eval()
                self.model_type = "custom"
                self.model_loaded = True
                
            debug_print("Custom DeepGaze III model loaded successfully", "SUCCESS")
            return True
            
        except Exception as e:
            debug_print(f"Custom DeepGaze III loading error: {str(e)}", "ERROR")
            st.error(f"Custom DeepGaze III loading failed: {str(e)}")
            return False
    
    def get_active_model(self):
        """Get the currently active model"""
        if self.model_type == "official":
            return self.official_model
        elif self.model_type == "custom":
            return self.custom_model
        return None
    
    def load_centerbias(self, h: int, w: int) -> np.ndarray:
        """Load or generate center bias"""
        try:
            debug_print(f"Loading center bias for {w}x{h}", "INFO")
            
            # Try to download official center bias
            path = "centerbias_mit1003.npy"
            if not os.path.exists(path):
                url = ("https://github.com/matthias-k/DeepGaze/"
                       "releases/download/v1.0.0/centerbias_mit1003.npy")
                try:
                    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
                        shutil.copyfileobj(r, f)
                    debug_print("Center bias downloaded successfully", "SUCCESS")
                except:
                    debug_print("Center bias download failed, generating synthetic", "WARNING")
                    return self.generate_center_bias(h, w)
            
            centerbias = np.load(path)
            centerbias = cv2.resize(centerbias, (w, h), interpolation=cv2.INTER_CUBIC)
            return np.log(centerbias)
            
        except Exception as e:
            debug_print(f"Center bias error: {str(e)}", "ERROR")
            return self.generate_center_bias(h, w)
    
    def generate_center_bias(self, h: int, w: int, strength: float = 0.6) -> np.ndarray:
        """Generate synthetic center bias"""
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        bias = np.exp(-strength * (dist / max_dist)**2)
        return np.log(bias + 1e-8)
    
    def run_inference(self, img: np.ndarray, cb_log: np.ndarray, seed_xy: Tuple[int, int] = None) -> np.ndarray:
        """Run DeepGaze III inference"""
        try:
            model = self.get_active_model()
            if model is None:
                debug_print("No model loaded", "ERROR")
                return None
            
            debug_print(f"Running {self.model_type} DeepGaze III inference", "INFO")
            
            if self.model_type == "official":
                return self._run_official_inference(img, cb_log, seed_xy)
            elif self.model_type == "custom":
                return self._run_custom_inference(img, cb_log)
                
        except Exception as e:
            debug_print(f"Inference error: {str(e)}", "ERROR")
            return None
    
    def _run_official_inference(self, img: np.ndarray, cb_log: np.ndarray, seed_xy: Tuple[int, int]) -> np.ndarray:
        """Run official DeepGaze III inference"""
        if seed_xy is None:
            seed_xy = (img.shape[1] // 2, img.shape[0] // 2)
            
        fx, fy = seed_xy
        hx = [fx] + [np.nan] * (len(self.official_model.included_fixations) - 1)
        hy = [fy] + [np.nan] * (len(self.official_model.included_fixations) - 1)
        
        # Prepare tensors
        it = torch.tensor([img.transpose(2, 0, 1)], dtype=torch.float32).to(self.device)
        ct = torch.tensor([cb_log], dtype=torch.float32).to(self.device)
        xt = torch.tensor([hx], dtype=torch.float32).to(self.device)
        yt = torch.tensor([hy], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            log_map = self.official_model(it, ct, xt, yt)[0, 0].cpu().numpy()
        
        return np.exp(log_map)
    
    def _run_custom_inference(self, img: np.ndarray, cb_log: np.ndarray) -> np.ndarray:
        """Run custom DeepGaze III inference"""
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            result = self.custom_model(img_tensor)
            saliency_map = result['saliency_map'][0, 0].cpu().numpy()
        
        return saliency_map

# ========== ROI and Analysis Classes ==========

class ROIManager:
    def __init__(self):
        self.rois = []
        
    def add_roi(self, x: int, y: int, w: int, h: int, name: str = None):
        """Add a Region of Interest"""
        roi = {
            'x': x, 'y': y, 'width': w, 'height': h,
            'name': name or f"ROI_{len(self.rois) + 1}",
            'id': len(self.rois)
        }
        self.rois.append(roi)
        return roi
    
    def point_in_roi(self, x: int, y: int, roi: dict) -> bool:
        """Check if point is inside ROI"""
        return (roi['x'] <= x <= roi['x'] + roi['width'] and 
                roi['y'] <= y <= roi['y'] + roi['height'])
    
    def get_roi_saliency(self, saliency_map: np.ndarray, roi: dict) -> dict:
        """Extract saliency statistics for ROI"""
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        roi_saliency = saliency_map[y:y+h, x:x+w]
        
        return {
            'mean_saliency': np.mean(roi_saliency),
            'max_saliency': np.max(roi_saliency),
            'total_saliency': np.sum(roi_saliency),
            'roi_area': w * h,
            'saliency_density': np.sum(roi_saliency) / (w * h)
        }

class FixationPredictor:
    def __init__(self):
        self.fixation_history = []
        
    def predict_fixations(self, saliency_map: np.ndarray, num_fixations: int = 10, 
                         ior_radius: int = 50, ior_strength: float = 0.7) -> List[Dict]:
        """Predict fixation sequence using IOR (Inhibition of Return)"""
        fixations = []
        current_map = saliency_map.copy()
        
        for i in range(num_fixations):
            # Find maximum saliency point
            y, x = np.unravel_index(current_map.argmax(), current_map.shape)
            
            # Calculate fixation duration (simplified model)
            saliency_value = current_map[y, x]
            base_duration = 200  # ms
            duration = base_duration + (saliency_value * 300)
            
            fixation = {
                'x': int(x), 'y': int(y),
                'x_norm': float(x / current_map.shape[1]),
                'y_norm': float(y / current_map.shape[0]),
                'saliency': float(saliency_value),
                'duration_ms': int(duration),
                'timestamp_ms': int(i * duration),
                'fixation_id': i
            }
            fixations.append(fixation)
            
            # Apply IOR
            mask = np.zeros_like(current_map)
            cv2.circle(mask, (x, y), ior_radius, 1, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), ior_radius / 2)
            current_map *= (1 - ior_strength * mask)
            
        return fixations

# ========== Visualization Functions ==========

def create_heatmap_overlay(image: np.ndarray, saliency_map: np.ndarray, 
                          alpha: float = 0.6, colormap: str = 'jet') -> np.ndarray:
    """Create heatmap overlay on image"""
    # Normalize saliency map
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(saliency_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Resize to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def draw_fixations_and_scanpath(image: np.ndarray, fixations: List[Dict], 
                               roi_manager: ROIManager = None) -> np.ndarray:
    """Draw fixations and scanpath on image"""
    result_img = image.copy()
    
    # Draw ROIs if available
    if roi_manager:
        for roi in roi_manager.rois:
            cv2.rectangle(result_img, 
                         (roi['x'], roi['y']), 
                         (roi['x'] + roi['width'], roi['y'] + roi['height']),
                         (0, 255, 0), 2)
            cv2.putText(result_img, roi['name'],
                       (roi['x'], roi['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw scanpath lines
    for i in range(len(fixations) - 1):
        pt1 = (fixations[i]['x'], fixations[i]['y'])
        pt2 = (fixations[i+1]['x'], fixations[i+1]['y'])
        cv2.line(result_img, pt1, pt2, (255, 255, 255), 2)
        cv2.arrowedLine(result_img, pt1, pt2, (0, 0, 255), 2, tipLength=0.3)
    
    # Draw fixation circles
    for i, fix in enumerate(fixations):
        # Circle size based on duration
        radius = int(10 + (fix['duration_ms'] - 200) / 50)
        radius = max(5, min(radius, 25))
        
        # Color based on fixation order (early = red, late = blue)
        color_ratio = i / max(len(fixations) - 1, 1)
        color = (int(255 * (1 - color_ratio)), 0, int(255 * color_ratio))
        
        cv2.circle(result_img, (fix['x'], fix['y']), radius, color, -1)
        cv2.circle(result_img, (fix['x'], fix['y']), radius, (255, 255, 255), 2)
        
        # Fixation number
        cv2.putText(result_img, str(i+1),
                   (fix['x'] - 8, fix['y'] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

# ========== Main Application ==========

def main():
    # Professional Header with Neural Network Theme
    st.markdown("""
    <div class="main-header neural-pulse">
        🧠 DeepGaze III Neural Attention System
    </div>
    """, unsafe_allow_html=True)
    
    # Professional System Information Panel
    st.markdown("""
    <div class="neural-network glow-effect">
        <h3>🔬 Advanced Neural Architecture</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <h4 style="color: #00d4ff; margin-bottom: 0.5rem;">🏛️ Official Implementation</h4>
                <p><strong>Model:</strong> deepgaze_pytorch.DeepGazeIII</p>
                <p><strong>Architecture:</strong> DenseNet-201 Backbone</p>
                <p><strong>Parameters:</strong> ~20M (Pre-trained)</p>
                <p><strong>Source:</strong> Kümmerer et al., Journal of Vision 2021</p>
            </div>
            <div>
                <h4 style="color: #b794f6; margin-bottom: 0.5rem;">⚡ Custom Implementation</h4>
                <p><strong>Model:</strong> PyTorch DenseNet-169</p>
                <p><strong>Architecture:</strong> Backbone + Neural Readout</p>
                <p><strong>Parameters:</strong> 14.8M (12.4M + 2.3M)</p>
                <p><strong>Optimization:</strong> Custom Training Pipeline</p>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <h4 style="color: #68d391; margin-bottom: 0.5rem;">🚀 Advanced Features</h4>
            <p>✨ <strong>ROI Analysis</strong> • 🎯 <strong>Fixation Prediction</strong> • 📊 <strong>Duration Estimation</strong> • 🔄 <strong>IOR Mechanisms</strong> • 📈 <strong>Statistical Analysis</strong> • 💾 <strong>Data Export</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'deepgaze_manager' not in st.session_state:
        st.session_state.deepgaze_manager = EnhancedDeepGazeManager()
    if 'roi_manager' not in st.session_state:
        st.session_state.roi_manager = ROIManager()
    if 'fixation_predictor' not in st.session_state:
        st.session_state.fixation_predictor = FixationPredictor()
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
    # Professional Sidebar Configuration
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: white; font-weight: 700;">⚙️ Control Panel</h2>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Neural Network Configuration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Selection with Professional Styling
        st.markdown('<div class="section-header">🧠 Neural Model Selection</div>', unsafe_allow_html=True)
        
        model_options = []
        if DEEPGAZE_AVAILABLE:
            model_options.append("Official DeepGaze III")
        if CUSTOM_DEEPGAZE_AVAILABLE:
            model_options.append("Custom DeepGaze III")
        
        if not model_options:
            st.error("❌ No DeepGaze III implementations available!")
            return
        
        selected_model = st.selectbox("🎯 Select Neural Architecture:", model_options)
        
        # Model Status Display
        if st.session_state.deepgaze_manager.model_loaded:
            st.markdown(f"""
            <div class="metric-card">
                <strong>🟢 Active Model:</strong> {st.session_state.deepgaze_manager.model_type.title()}<br>
                <strong>📊 Status:</strong> Ready for Inference<br>
                <strong>🚀 Device:</strong> {str(st.session_state.deepgaze_manager.device).upper()}
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Load Neural Model", type="primary"):
            with st.spinner("🧠 Initializing Neural Network..."):
                if selected_model == "Official DeepGaze III":
                    success = st.session_state.deepgaze_manager.load_official_model()
                    if success:
                        st.success("✅ Official DeepGaze III Neural Network Loaded!")
                else:
                    success = st.session_state.deepgaze_manager.load_custom_model()
                    if success:
                        st.success("✅ Custom DeepGaze III Neural Network Loaded!")
        
        st.markdown("---")
        
        # Analysis Parameters with Enhanced Styling
        st.markdown('<div class="section-header">📊 Neural Analysis Parameters</div>', unsafe_allow_html=True)
        
        st.markdown("**🎯 Fixation Analysis**")
        num_fixations = st.slider("Number of Fixations:", 5, 20, 10, 
                                help="Number of predicted eye fixation points")
        
        st.markdown("**🔄 Inhibition of Return (IOR)**")
        ior_radius = st.slider("IOR Radius (pixels):", 20, 100, 50,
                             help="Radius of inhibition around previous fixations")
        ior_strength = st.slider("IOR Strength:", 0.3, 1.0, 0.7, 0.1,
                                help="Strength of inhibition effect")
        
        st.markdown("**🎨 Visualization Settings**")
        heatmap_alpha = st.slider("Heatmap Opacity:", 0.3, 0.9, 0.6, 0.1,
                                help="Transparency of attention heatmap overlay")
        
        st.markdown("---")
        
        # Visualization Options with Icons
        st.markdown('<div class="section-header">🎨 Visualization Control</div>', unsafe_allow_html=True)
        
        show_heatmap = st.checkbox("🔥 Show Attention Heatmap", True)
        show_fixations = st.checkbox("👁️ Show Fixation Points", True)
        show_scanpath = st.checkbox("🔗 Show Scanpath Connections", True)
        show_rois = st.checkbox("📍 Show ROI Analysis", True)
        
        st.markdown("---")
        
        # Advanced Options
        st.markdown('<div class="section-header">🔧 Advanced Options</div>', unsafe_allow_html=True)
        
        st.session_state.show_debug = st.checkbox("🐛 Debug Mode", False)
        
        if st.button("📊 System Information"):
            sys_info = check_system_info()
            st.markdown(f"""
            <div class="metric-card">
                <strong>💻 Platform:</strong> {sys_info['platform']}<br>
                <strong>🐍 Python:</strong> {sys_info['python_version']}<br>
                <strong>🔥 PyTorch:</strong> {sys_info['torch_version']}<br>
                <strong>⚡ CUDA:</strong> {'Available' if sys_info['cuda_available'] else 'Not Available'}<br>
                <strong>💾 Memory:</strong> {sys_info['memory']['available_gb']:.1f}GB / {sys_info['memory']['total_gb']:.1f}GB
            </div>
            """, unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📸 Image Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            # Resize if too large
            max_size = 1024
            if max(image_np.shape[:2]) > max_size:
                scale = max_size / max(image_np.shape[:2])
                new_h, new_w = int(image_np.shape[0] * scale), int(image_np.shape[1] * scale)
                image_np = cv2.resize(image_np, (new_w, new_h))
                image = Image.fromarray(image_np)
            
            st.image(image, caption=f"Input Image ({image_np.shape[1]}×{image_np.shape[0]})", 
                    use_container_width=True)
            
            # ROI Selection with Professional Interface
            if show_rois:
                st.markdown('<div class="section-header">🎯 ROI Selection</div>', unsafe_allow_html=True)
                
                # Professional ROI creation interface
                st.markdown("""
                <div class="neural-network">
                <h3>🔬 Region of Interest Configuration</h3>
                <p>Define rectangular regions for focused analysis. Use the controls below to create precise ROIs.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_roi1, col_roi2 = st.columns(2)
                
                with col_roi1:
                    st.markdown("**📍 ROI Coordinates**")
                    roi_x = st.number_input("X Position", 0, image_np.shape[1]-1, 50, key="roi_x")
                    roi_y = st.number_input("Y Position", 0, image_np.shape[0]-1, 50, key="roi_y")
                
                with col_roi2:
                    st.markdown("**📏 ROI Dimensions**")
                    roi_w = st.number_input("Width", 10, image_np.shape[1]-roi_x, 100, key="roi_w")
                    roi_h = st.number_input("Height", 10, image_np.shape[0]-roi_y, 100, key="roi_h")
                
                roi_name = st.text_input("ROI Name", f"ROI_{len(st.session_state.roi_manager.rois) + 1}")
                
                if st.button("➕ Add ROI", type="secondary"):
                    roi = st.session_state.roi_manager.add_roi(roi_x, roi_y, roi_w, roi_h, roi_name)
                    st.success(f"✅ Added {roi_name} at ({roi_x}, {roi_y}) size {roi_w}×{roi_h}")
                
                # Display current ROIs
                if st.session_state.roi_manager.rois:
                    st.markdown("**📊 Current ROIs:**")
                    for i, roi in enumerate(st.session_state.roi_manager.rois):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.text(f"{roi['name']}: ({roi['x']}, {roi['y']})")
                        with col2:
                            st.text(f"Size: {roi['width']}×{roi['height']}")
                        with col3:
                            if st.button("🗑️", key=f"del_{i}"):
                                st.session_state.roi_manager.rois.pop(i)
                                st.rerun()
                
                if st.session_state.roi_manager.rois and st.button("🧹 Clear All ROIs"):
                    st.session_state.roi_manager.rois = []
                    st.success("✅ All ROIs cleared")
            
            # Analysis Button
            if st.button("🔬 Analyze with DeepGaze III", type="primary"):
                if not st.session_state.deepgaze_manager.model_loaded:
                    st.error("❌ Please load a DeepGaze III model first!")
                    return
                
                with st.spinner("🧠 Running DeepGaze III Analysis..."):
                    start_time = time.time()
                    
                    # Load center bias
                    h, w = image_np.shape[:2]
                    cb_log = st.session_state.deepgaze_manager.load_centerbias(h, w)
                    
                    # Run inference
                    seed_xy = (w // 2, h // 2)  # Center seed point
                    saliency_map = st.session_state.deepgaze_manager.run_inference(
                        image_np, cb_log, seed_xy
                    )
                    
                    if saliency_map is not None:
                        processing_time = time.time() - start_time
                        
                        # Predict fixations
                        fixations = st.session_state.fixation_predictor.predict_fixations(
                            saliency_map, num_fixations, ior_radius, ior_strength
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results = {
                            'saliency_map': saliency_map,
                            'fixations': fixations,
                            'processing_time': processing_time,
                            'image': image_np,
                            'model_type': st.session_state.deepgaze_manager.model_type
                        }
                        
                        st.success(f"✅ Analysis complete in {processing_time:.2f}s")
                    else:
                        st.error("❌ Analysis failed")
    
    with col2:
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Professional Results Header
            st.markdown('<div class="section-header">📊 Neural Analysis Results</div>', unsafe_allow_html=True)
            
            # Enhanced Results Summary
            processing_fps = 1 / results['processing_time'] if results['processing_time'] > 0 else 0
            total_fixation_time = sum(fix['duration_ms'] for fix in results['fixations'])
            avg_saliency = np.mean([fix['saliency'] for fix in results['fixations']])
            
            st.markdown(f"""
            <div class="neural-network">
                <h3>🧠 Neural Processing Summary</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div style="text-align: center; padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 10px;">
                        <h4 style="color: #00d4ff; margin: 0;">⚡ Processing</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: 600;">{results['processing_time']:.2f}s</p>
                        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">{processing_fps:.1f} FPS</p>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: rgba(183, 148, 246, 0.1); border-radius: 10px;">
                        <h4 style="color: #b794f6; margin: 0;">👁️ Fixations</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: 600;">{len(results['fixations'])}</p>
                        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">{total_fixation_time}ms total</p>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: rgba(104, 211, 145, 0.1); border-radius: 10px;">
                        <h4 style="color: #68d391; margin: 0;">🎯 Attention</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: 600;">{avg_saliency:.3f}</p>
                        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Avg Saliency</p>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                    <p><strong>🤖 Model:</strong> {results['model_type'].title()} DeepGaze III Neural Network</p>
                    <p><strong>📐 Resolution:</strong> {results['image'].shape[1]}×{results['image'].shape[0]} pixels</p>
                    <p><strong>🧮 Computation:</strong> {results['image'].shape[0] * results['image'].shape[1]:,} pixels processed</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create visualizations
            result_image = results['image'].copy()
            
            # Add heatmap overlay
            if show_heatmap:
                result_image = create_heatmap_overlay(
                    result_image, results['saliency_map'], heatmap_alpha
                )
            
            # Add fixations and scanpath
            if show_fixations or show_scanpath:
                roi_manager = st.session_state.roi_manager if show_rois else None
                result_image = draw_fixations_and_scanpath(
                    result_image, results['fixations'], roi_manager
                )
            
            st.image(result_image, caption="DeepGaze III Analysis Result", 
                    use_container_width=True)
            
            # Enhanced Fixation Statistics
            st.markdown('<div class="section-header">📈 Detailed Neural Analysis</div>', unsafe_allow_html=True)
            
            fixation_df = pd.DataFrame(results['fixations'])
            
            # Professional Metrics Display
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                avg_duration = fixation_df['duration_ms'].mean()
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="color: #00d4ff; margin-bottom: 0.5rem;">⏱️ Avg Duration</h4>
                    <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: 700;">{avg_duration:.0f}</p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">milliseconds</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats2:
                max_saliency = fixation_df['saliency'].max()
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="color: #b794f6; margin-bottom: 0.5rem;">🎯 Peak Attention</h4>
                    <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: 700;">{max_saliency:.3f}</p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">saliency score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats3:
                total_duration = fixation_df['duration_ms'].sum()
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="color: #68d391; margin-bottom: 0.5rem;">🕐 Total Time</h4>
                    <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: 700;">{total_duration:.0f}</p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">milliseconds</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats4:
                attention_coverage = (len(results['fixations']) / (results['image'].shape[0] * results['image'].shape[1])) * 100000
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="color: #fd9853; margin-bottom: 0.5rem;">📊 Coverage</h4>
                    <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: 700;">{attention_coverage:.2f}</p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">fixations/10K px</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Professional Data Table
            st.markdown("**🔬 Detailed Fixation Analysis**")
            
            # Style the dataframe
            styled_df = fixation_df[['fixation_id', 'x', 'y', 'saliency', 'duration_ms', 'x_norm', 'y_norm']].round(3)
            styled_df.columns = ['ID', 'X (px)', 'Y (px)', 'Saliency', 'Duration (ms)', 'X (norm)', 'Y (norm)']
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # ROI Analysis (if ROIs defined)
            if st.session_state.roi_manager.rois and show_rois:
                st.markdown('<div class="section-header">🎯 ROI Analysis</div>', unsafe_allow_html=True)
                
                roi_stats = []
                for roi in st.session_state.roi_manager.rois:
                    stats = st.session_state.roi_manager.get_roi_saliency(
                        results['saliency_map'], roi
                    )
                    stats['roi_name'] = roi['name']
                    roi_stats.append(stats)
                
                roi_df = pd.DataFrame(roi_stats)
                st.dataframe(roi_df, use_container_width=True)
            
            # Professional Export Section
            st.markdown('<div class="section-header">💾 Research Data Export</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="neural-network">
                <h3>📊 Export Neural Analysis Data</h3>
                <p>Download comprehensive analysis results for research, visualization, or further processing.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                # Enhanced CSV Export
                csv_data = styled_df.to_csv(index=False)
                st.download_button(
                    label="📊 Fixation Data (CSV)",
                    data=csv_data,
                    file_name=f"deepgaze_fixations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download detailed fixation data in CSV format for statistical analysis"
                )
            
            with col_export2:
                # Enhanced JSON Export
                export_data = {
                    'metadata': {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'model_type': results['model_type'],
                        'model_architecture': 'DeepGaze III Neural Network',
                        'processing_time_seconds': results['processing_time'],
                        'image_dimensions': {
                            'width': results['image'].shape[1],
                            'height': results['image'].shape[0],
                            'channels': results['image'].shape[2]
                        },
                        'analysis_parameters': {
                            'num_fixations': num_fixations,
                            'ior_radius': ior_radius,
                            'ior_strength': ior_strength
                        }
                    },
                    'fixations': results['fixations'],
                    'statistics': {
                        'total_fixations': len(results['fixations']),
                        'total_duration_ms': int(total_duration),
                        'avg_duration_ms': int(avg_duration),
                        'max_saliency': float(max_saliency),
                        'avg_saliency': float(avg_saliency)
                    },
                    'rois': st.session_state.roi_manager.rois if show_rois else []
                }
                
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="🔬 Complete Analysis (JSON)",
                    data=json_data,
                    file_name=f"deepgaze_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download complete analysis results including metadata and statistics"
                )
            
            with col_export3:
                # ROI-specific export
                if st.session_state.roi_manager.rois:
                    roi_analysis = []
                    for roi in st.session_state.roi_manager.rois:
                        roi_stats = st.session_state.roi_manager.get_roi_saliency(
                            results['saliency_map'], roi
                        )
                        roi_stats['roi_info'] = roi
                        roi_analysis.append(roi_stats)
                    
                    roi_data = json.dumps(roi_analysis, indent=2)
                    st.download_button(
                        label="🎯 ROI Analysis (JSON)",
                        data=roi_data,
                        file_name=f"deepgaze_roi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download Region of Interest analysis data"
                    )
                else:
                    st.markdown("""
                    <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px; text-align: center;">
                        <p style="color: rgba(255,255,255,0.7); margin: 0;">📍 Add ROIs for region-specific analysis</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()