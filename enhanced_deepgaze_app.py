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

# ========== Custom CSS ==========
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #1f77b4, #ff7f0e, #2ca02c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: pulse 2s ease-in-out infinite alternate;
}

@keyframes pulse {
    from { filter: brightness(1); }
    to { filter: brightness(1.2); }
}

.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #ff7f0e;
    padding-bottom: 0.5rem;
    background: linear-gradient(90deg, rgba(255,127,14,0.1) 0%, transparent 100%);
    padding-left: 1rem;
}

.neural-network {
    background: linear-gradient(135deg, rgba(31,119,180,0.1), rgba(255,127,14,0.1));
    border: 2px solid rgba(31,119,180,0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg, rgba(44,160,44,0.1), rgba(214,39,40,0.1));
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #2ca02c;
}

.warning-box {
    background: linear-gradient(135deg, rgba(255,193,7,0.1), rgba(255,87,34,0.1));
    border: 2px solid rgba(255,193,7,0.5);
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
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
    st.markdown('<h1 class="main-header">🧠 Enhanced DeepGaze III Neural Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="neural-network">
    <h3>🔬 Dual Implementation System</h3>
    <p><strong>Official:</strong> deepgaze_pytorch.DeepGaze III (Kümmerer et al.)</p>
    <p><strong>Custom:</strong> PyTorch DenseNet-169 + Neural Readout</p>
    <p><strong>Features:</strong> ROI Analysis • Duration Prediction • Advanced Visualization</p>
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
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("🧠 Model Selection")
        model_options = []
        if DEEPGAZE_AVAILABLE:
            model_options.append("Official DeepGaze III")
        if CUSTOM_DEEPGAZE_AVAILABLE:
            model_options.append("Custom DeepGaze III")
        
        if not model_options:
            st.error("No DeepGaze III implementations available!")
            return
        
        selected_model = st.selectbox("Select Model:", model_options)
        
        if st.button("🚀 Load Selected Model"):
            if selected_model == "Official DeepGaze III":
                success = st.session_state.deepgaze_manager.load_official_model()
                if success:
                    st.success("✅ Official DeepGaze III loaded!")
            else:
                success = st.session_state.deepgaze_manager.load_custom_model()
                if success:
                    st.success("✅ Custom DeepGaze III loaded!")
        
        # Analysis Parameters
        st.subheader("📊 Analysis Parameters")
        num_fixations = st.slider("Number of Fixations:", 5, 20, 10)
        ior_radius = st.slider("IOR Radius:", 20, 100, 50)
        ior_strength = st.slider("IOR Strength:", 0.3, 1.0, 0.7, 0.1)
        heatmap_alpha = st.slider("Heatmap Opacity:", 0.3, 0.9, 0.6, 0.1)
        
        # Visualization Options
        st.subheader("🎨 Visualization")
        show_heatmap = st.checkbox("Show Heatmap", True)
        show_fixations = st.checkbox("Show Fixations", True)
        show_scanpath = st.checkbox("Show Scanpath", True)
        show_rois = st.checkbox("Show ROIs", True)
        
        # Debug Options
        st.subheader("🔧 Debug")
        st.session_state.show_debug = st.checkbox("Show Debug Info", False)
        
        if st.button("📊 System Info"):
            sys_info = check_system_info()
            st.json(sys_info)
    
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
            
            # ROI Selection (if canvas available)
            if CANVAS_AVAILABLE and show_rois:
                st.markdown('<div class="section-header">🎯 ROI Selection</div>', unsafe_allow_html=True)
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="green",
                    background_image=image,
                    drawing_mode="rect",
                    key="roi_canvas",
                    height=min(400, image_np.shape[0]),
                    width=min(600, image_np.shape[1])
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    st.session_state.roi_manager.rois = []
                    
                    for i, obj in enumerate(objects):
                        if obj["type"] == "rect":
                            roi = st.session_state.roi_manager.add_roi(
                                int(obj["left"]), int(obj["top"]),
                                int(obj["width"]), int(obj["height"]),
                                f"ROI_{i+1}"
                            )
            
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
            
            st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
            
            # Display model info
            st.markdown(f"""
            <div class="metric-card">
            <strong>Model:</strong> {results['model_type'].title()} DeepGaze III<br>
            <strong>Processing Time:</strong> {results['processing_time']:.2f}s<br>
            <strong>Fixations Generated:</strong> {len(results['fixations'])}<br>
            <strong>Image Size:</strong> {results['image'].shape[1]}×{results['image'].shape[0]}
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
            
            # Fixation Statistics
            st.markdown('<div class="section-header">📈 Fixation Statistics</div>', unsafe_allow_html=True)
            
            fixation_df = pd.DataFrame(results['fixations'])
            
            # Summary statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Avg Duration", f"{fixation_df['duration_ms'].mean():.0f}ms")
            with col_b:
                st.metric("Max Saliency", f"{fixation_df['saliency'].max():.3f}")
            with col_c:
                st.metric("Total Duration", f"{fixation_df['duration_ms'].sum():.0f}ms")
            
            # Detailed fixation table
            st.dataframe(fixation_df[['fixation_id', 'x', 'y', 'saliency', 'duration_ms']], 
                        use_container_width=True)
            
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
            
            # Export Options
            st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # CSV Export
                csv_data = fixation_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Fixations CSV",
                    data=csv_data,
                    file_name=f"deepgaze_fixations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                # JSON Export
                export_data = {
                    'analysis_info': {
                        'model_type': results['model_type'],
                        'processing_time_ms': results['processing_time'] * 1000,
                        'timestamp': datetime.now().isoformat(),
                        'image_dimensions': list(results['image'].shape)
                    },
                    'fixations': results['fixations'],
                    'rois': st.session_state.roi_manager.rois if show_rois else []
                }
                
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📊 Download Full Data JSON",
                    data=json_data,
                    file_name=f"deepgaze_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()