# ========== インポート ==========
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
import matplotlib.font_manager as fm
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
import streamlit.elements.image as _st_img_mod

warnings.filterwarnings('ignore')

# streamlit-drawable-canvas のインポート
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
    st.error("streamlit-drawable-canvas が必要です。`pip install streamlit-drawable-canvas` でインストールしてください。")
    st.stop()

# DeepGaze III のインポート
try:
    import deepgaze_pytorch
    DEEPGAZE_AVAILABLE = True
except ImportError:
    DEEPGAZE_AVAILABLE = False
    st.error("deepgaze_pytorch が必要です。`pip install git+https://github.com/matthias-k/DeepGaze.git` でインストールしてください。")
    st.stop()

# YOLOv8のインポート
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("ultralytics が必要です。`pip install ultralytics` でインストールしてください。")
    st.stop()

# SAMのインポート
try:
    from segment_anything import sam_model_registry, SamPredictor
    import segment_anything
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    st.warning("SAMが利用できません。YOLOv8のみ使用可能です。")

# ========== ページ設定 ==========
st.set_page_config(
    page_title="ROI DeepGaze Analyzer Ultimate with Duration",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== カスタムCSS ==========
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #ff7f0e;
    padding-bottom: 0.5rem;
}
.subsection-header {
    font-size: 1.2rem;
    font-weight: bold;
    color: #2ca02c;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.debug-info {
    background: linear-gradient(135deg, #e8f4f8, #d0e9f0);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.error-info {
    background: linear-gradient(135deg, #ffe8e8, #ffcccb);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #ff4444;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.success-info {
    background: linear-gradient(135deg, #e8f5e8, #d4edda);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #22c55e;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.warning-info {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #ffc107;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.detection-info {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #6f42c1;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #e9ecef;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
}
.duration-info {
    background: linear-gradient(135deg, #f0e6ff, #e6d9ff);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #8b5cf6;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.no-roi-info {
    background: linear-gradient(135deg, #e6f3ff, #cce7ff);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #0080ff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ========== ヘルパー関数 ==========
def setup_japanese_font():
    """日本語フォントの設定"""
    system = platform.system()
    
    if system == 'Windows':
        font_name = 'Meiryo'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
    else:
        font_name = 'DejaVu Sans'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic']
    
    plt.rcParams['axes.unicode_minus'] = False
    return font_name

def debug_print(message: str, level: str = "INFO"):
    """デバッグ情報を出力"""
    if st.session_state.get('show_debug', False):
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": "🔵",
            "WARNING": "🟡", 
            "ERROR": "🔴",
            "SUCCESS": "🟢",
            "DEBUG": "🟣"
        }
        icon = color_map.get(level, "ℹ️")
        st.write(f"{icon} [{timestamp}] {level}: {message}")

def check_disk_space(path: str = ".") -> Dict[str, float]:
    """ディスク容量をチェック"""
    try:
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        used_bytes = total_bytes - free_bytes
        
        return {
            "free_gb": free_bytes / (1024**3),
            "total_gb": total_bytes / (1024**3),
            "used_gb": used_bytes / (1024**3),
            "free_percent": (free_bytes / total_bytes) * 100
        }
    except:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return {
                "free_gb": free / (1024**3),
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_percent": (free / total) * 100
            }
        except:
            return {"free_gb": 0, "total_gb": 0, "used_gb": 0, "free_percent": 0}

def check_memory_usage() -> Dict[str, float]:
    """メモリ使用量をチェック"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    except:
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0}

def cleanup_temp_files():
    """一時ファイルの清理"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith(('sam_', 'yolo', 'deepgaze')):
                try:
                    os.remove(os.path.join(temp_dir, filename))
                    debug_print(f"Cleaned up temp file: {filename}", "DEBUG")
                except:
                    pass
    except Exception as e:
        debug_print(f"Cleanup error: {str(e)}", "WARNING")

def _flex_image_to_url(*args, **kwargs) -> str:
    """Canvas互換性修正（エラーハンドリング強化）"""
    try:
        img = args[0] if args else kwargs.get("image")
        if img is None:
            debug_print("image_to_url: image argument missing", "ERROR")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        if isinstance(img, np.ndarray):
            debug_print(f"Converting numpy array to PIL: shape={img.shape}, dtype={img.dtype}", "DEBUG")
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = Image.fromarray(img[..., :3])
            elif img.ndim == 2:
                img = Image.fromarray(img, mode='L').convert('RGB')
            else:
                debug_print(f"Unsupported numpy array shape: {img.shape}", "ERROR")
                return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        if not isinstance(img, Image.Image):
            debug_print(f"Unsupported image type: {type(img)}", "ERROR")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        if img.size[0] * img.size[1] > 2048 * 2048:
            debug_print(f"Image too large: {img.size}, resizing", "WARNING")
            img.thumbnail((2048, 2048), Image.LANCZOS)
        
        if img.mode not in ['RGB', 'RGBA', 'L']:
            debug_print(f"Converting image mode from {img.mode} to RGB", "INFO")
            img = img.convert('RGB')
        
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=6)
        buf.seek(0)
        
        data_size = len(buf.getvalue())
        debug_print(f"Image data size: {data_size / 1024:.1f} KB", "DEBUG")
        
        if data_size > 10 * 1024 * 1024:
            debug_print("Image data too large, compressing", "WARNING")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
        
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        debug_print(f"Image encoded successfully: {len(encoded)} characters", "DEBUG")
        
        return f"data:image/png;base64,{encoded}"
        
    except Exception as e:
        debug_print(f"Canvas URL conversion error: {str(e)}", "ERROR")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def create_fixation_duration_table(scanpath_data: List[Dict]) -> pd.DataFrame:
    """注視点の滞留時間テーブルを作成"""
    try:
        table_data = []
        for fixation in scanpath_data:
            table_data.append({
                '番号': fixation['fixation_id'],
                'X座標': fixation['x'],
                'Y座標': fixation['y'],
                '滞留時間(ms)': f"{fixation['duration_ms']:.1f}",
                'ROI': fixation.get('roi_label', 'Background')
            })
        
        return pd.DataFrame(table_data)
        
    except Exception as e:
        debug_print(f"Fixation table creation error: {str(e)}", "ERROR")
        return pd.DataFrame()

# フォント設定を初期化
FONT_NAME = setup_japanese_font()

# Canvas互換性パッチ適用
_need_patch = (not hasattr(_st_img_mod, "image_to_url") or
               _st_img_mod.image_to_url is None or
               _st_img_mod.image_to_url.__code__.co_argcount != 1)
if _need_patch:
    _st_img_mod.image_to_url = _flex_image_to_url
    debug_print("Canvas compatibility patch applied", "SUCCESS")

# ========== クラス定義 ==========
class ROICoordinateManager:
    """ROI座標管理クラス - 回転ROI位置ずれ完全修正版"""
    
    def __init__(self):
        self.canvas_to_image_scale = 1.0
        self.image_to_canvas_scale = 1.0
        self.canvas_size = (0, 0)
        self.image_size = (0, 0)
        self.offset_x = 0
        self.offset_y = 0
        
    def set_coordinate_system(self, canvas_width: int, canvas_height: int, 
                            image_width: int, image_height: int):
        """座標系の設定"""
        self.canvas_size = (canvas_width, canvas_height)
        self.image_size = (image_width, image_height)
        
        self.canvas_to_image_scale = 1.0
        self.image_to_canvas_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        debug_print(f"Coordinate system set: canvas={self.canvas_size}, image={self.image_size}", "INFO")
        debug_print(f"Scale: canvas->image={self.canvas_to_image_scale:.3f}", "INFO")
    
    def canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """Canvas座標を画像座標に変換"""
        image_x = int(canvas_x * self.canvas_to_image_scale)
        image_y = int(canvas_y * self.canvas_to_image_scale)
        
        image_x = max(0, min(image_x, self.image_size[0] - 1))
        image_y = max(0, min(image_y, self.image_size[1] - 1))
        
        return image_x, image_y
    
    def canvas_rect_to_image_rect(self, canvas_rect: Dict) -> Dict:
        """Canvas矩形を画像矩形に変換（回転ROI対応強化版）"""
        try:
            canvas_x = float(canvas_rect.get('left', 0))
            canvas_y = float(canvas_rect.get('top', 0))
            canvas_width = float(canvas_rect.get('width', 0))
            canvas_height = float(canvas_rect.get('height', 0))
            angle = float(canvas_rect.get('angle', 0))
            
            debug_print(f"Canvas rect: ({canvas_x}, {canvas_y}, {canvas_width}, {canvas_height}), angle: {angle:.1f}°", "DEBUG")
            
            image_x = int(canvas_x * self.canvas_to_image_scale)
            image_y = int(canvas_y * self.canvas_to_image_scale)
            image_width = int(canvas_width * self.canvas_to_image_scale)
            image_height = int(canvas_height * self.canvas_to_image_scale)
            
            image_x = max(0, min(image_x, self.image_size[0] - 1))
            image_y = max(0, min(image_y, self.image_size[1] - 1))
            image_width = max(1, min(image_width, self.image_size[0] - image_x))
            image_height = max(1, min(image_height, self.image_size[1] - image_y))
            
            center_x = image_x + image_width / 2
            center_y = image_y + image_height / 2
            
            image_rect = {
                'x': image_x,
                'y': image_y,
                'width': image_width,
                'height': image_height,
                'center_x': center_x,
                'center_y': center_y,
                'angle': angle,
                'canvas_x': canvas_x,
                'canvas_y': canvas_y,
                'canvas_width': canvas_width,
                'canvas_height': canvas_height
            }
            
            debug_print(f"Image rect: ({image_rect['x']}, {image_rect['y']}, {image_rect['width']}, {image_rect['height']}), center: ({center_x:.1f}, {center_y:.1f})", "DEBUG")
            
            return image_rect
            
        except Exception as e:
            debug_print(f"Canvas rect conversion error: {str(e)}", "ERROR")
            return {
                'x': 0, 'y': 0, 'width': 10, 'height': 10,
                'center_x': 5, 'center_y': 5, 'angle': 0,
                'canvas_x': 0, 'canvas_y': 0,
                'canvas_width': 10, 'canvas_height': 10
            }

class AutoDetectionManager:
    def __init__(self):
        self.yolo_model = None
        self.sam_model = None
        self.sam_predictor = None
        
    def load_yolo_model(self, model_size: str = "yolov8n.pt"):
        """YOLOv8モデルの読み込み"""
        try:
            debug_print(f"Loading YOLOv8 model: {model_size}", "INFO")
            
            disk_info = check_disk_space()
            debug_print(f"Available disk space: {disk_info['free_gb']:.2f} GB", "INFO")
            
            if disk_info['free_gb'] < 1.0:
                debug_print("Insufficient disk space for model download", "ERROR")
                st.error("ディスク容量不足です。1GB以上の空き容量が必要です。")
                return None
            
            memory_info = check_memory_usage()
            debug_print(f"Available memory: {memory_info['available_gb']:.2f} GB", "INFO")
            
            with st.spinner(f"YOLOv8 ({model_size}) を読み込んでいます..."):
                cleanup_temp_files()
                gc.collect()
                
                model = YOLO(model_size)
                debug_print(f"YOLOv8 model loaded successfully", "SUCCESS")
                
                try:
                    model_info = {
                        "classes": len(model.names),
                        "class_names": list(model.names.values())[:10],
                        "model_size": model_size
                    }
                    debug_print(f"Model info: {model_info['classes']} classes", "INFO")
                except Exception as e:
                    debug_print(f"Could not get model info: {str(e)}", "WARNING")
                
                return model
                
        except Exception as e:
            debug_print(f"YOLOv8 loading error: {str(e)}", "ERROR")
            st.error(f"YOLOv8読み込みエラー: {str(e)}")
            return None
    
    def load_sam_model(self, model_type: str = "vit_b", checkpoint_path: str = None):
        """SAMモデルの読み込み"""
        if not SAM_AVAILABLE:
            debug_print("SAM not available", "WARNING")
            return None, None
        
        try:
            debug_print(f"Loading SAM model: {model_type}", "INFO")
            
            disk_info = check_disk_space()
            debug_print(f"Available disk space: {disk_info['free_gb']:.2f} GB", "INFO")
            
            if disk_info['free_gb'] < 2.0:
                debug_print("Insufficient disk space for SAM model", "ERROR")
                st.error("SAMモデルには2GB以上の空き容量が必要です。")
                return None, None
            
            memory_info = check_memory_usage()
            debug_print(f"Available memory: {memory_info['available_gb']:.2f} GB", "INFO")
            
            if memory_info['available_gb'] < 2.0:
                debug_print("Insufficient memory for SAM model", "ERROR")
                st.error("SAMモデルには最低2GB以上のメモリが必要です。")
                return None, None
            
            if checkpoint_path is None:
                checkpoint_urls = {
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                }
                
                checkpoint_path = f"sam_{model_type}.pth"
                
                if not os.path.exists(checkpoint_path):
                    debug_print(f"Downloading SAM checkpoint: {model_type}", "INFO")
                    
                    try:
                        with st.spinner(f"SAM {model_type} チェックポイントをダウンロードしています..."):
                            urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
                        debug_print(f"SAM checkpoint downloaded successfully", "SUCCESS")
                    except Exception as e:
                        debug_print(f"SAM checkpoint download error: {str(e)}", "ERROR")
                        st.error(f"SAMチェックポイントのダウンロードに失敗しました: {str(e)}")
                        return None, None
            
            if not os.path.exists(checkpoint_path):
                debug_print(f"SAM checkpoint not found: {checkpoint_path}", "ERROR")
                st.error(f"SAMチェックポイントが見つかりません: {checkpoint_path}")
                return None, None
            
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
            debug_print(f"SAM checkpoint file size: {file_size:.1f} MB", "INFO")
            
            if file_size < 100:
                debug_print("SAM checkpoint file seems incomplete", "WARNING")
                st.warning("SAMチェックポイントファイルが不完全な可能性があります。")
                return None, None
            
            with st.spinner(f"SAM {model_type} を読み込んでいます..."):
                cleanup_temp_files()
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                debug_print("Creating SAM model instance", "INFO")
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                debug_print(f"SAM using device: {device}", "INFO")
                
                debug_print("Moving SAM model to device", "INFO")
                sam.to(device=device)
                
                debug_print("Creating SAM predictor", "INFO")
                predictor = SamPredictor(sam)
                
                memory_info_after = check_memory_usage()
                debug_print(f"Memory after SAM loading: {memory_info_after['available_gb']:.2f} GB", "INFO")
                
                debug_print(f"SAM model loaded successfully", "SUCCESS")
                return sam, predictor
                
        except Exception as e:
            debug_print(f"SAM loading error: {str(e)}", "ERROR")
            st.error(f"SAM読み込みエラー: {str(e)}")
            return None, None
    
    def detect_objects_yolo(self, image: np.ndarray, confidence: float = 0.5, 
                           selected_classes: List[str] = None) -> List[Dict]:
        """YOLOv8によるオブジェクト検出"""
        try:
            debug_print(f"Starting YOLOv8 detection with confidence: {confidence}", "INFO")
            
            if self.yolo_model is None:
                debug_print("YOLOv8 model not loaded, loading now", "INFO")
                self.yolo_model = self.load_yolo_model()
                if self.yolo_model is None:
                    debug_print("YOLOv8 model loading failed", "ERROR")
                    return []
            
            debug_print(f"Input image shape: {image.shape}", "DEBUG")
            
            debug_print("Running YOLO inference", "INFO")
            results = self.yolo_model(image, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    debug_print(f"Found {len(boxes)} detections", "INFO")
                    
                    for i, box in enumerate(boxes):
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.yolo_model.names[class_id]
                            
                            debug_print(f"Detection {i}: {class_name} ({conf:.3f})", "DEBUG")
                            
                            if selected_classes is None or class_name in selected_classes:
                                detections.append({
                                    'x': int(x1),
                                    'y': int(y1),
                                    'width': int(x2 - x1),
                                    'height': int(y2 - y1),
                                    'confidence': float(conf),
                                    'class_name': class_name,
                                    'class_id': class_id,
                                    'detection_type': 'yolo',
                                    'center_x': int((x1 + x2) / 2),
                                    'center_y': int((y1 + y2) / 2)
                                })
                        except Exception as e:
                            debug_print(f"Error processing detection {i}: {str(e)}", "WARNING")
                else:
                    debug_print("No detections found", "INFO")
            
            debug_print(f"YOLOv8 detection completed: {len(detections)} objects", "SUCCESS")
            return detections
            
        except Exception as e:
            debug_print(f"YOLOv8 detection error: {str(e)}", "ERROR")
            st.error(f"YOLOv8検出エラー: {str(e)}")
            return []
    
    def detect_objects_sam_with_area(self, image: np.ndarray, input_points: List[Tuple[int, int]] = None,
                                    input_labels: List[int] = None, input_boxes: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """SAMによるオブジェクト検出（エリア指定対応）"""
        if not SAM_AVAILABLE:
            debug_print("SAM not available", "WARNING")
            st.warning("SAMライブラリが利用できません。")
            return []
        
        try:
            debug_print(f"Starting SAM detection with area support", "INFO")
            
            if self.sam_model is None or self.sam_predictor is None:
                debug_print("SAM model not loaded, loading now", "INFO")
                self.sam_model, self.sam_predictor = self.load_sam_model()
                if self.sam_model is None:
                    debug_print("SAM model loading failed", "ERROR")
                    return []
            
            debug_print(f"Input image shape: {image.shape}", "DEBUG")
            
            if image is None or image.size == 0:
                debug_print("Invalid image provided to SAM", "ERROR")
                return []
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                debug_print(f"Invalid image shape for SAM: {image.shape}", "ERROR")
                return []
            
            memory_info = check_memory_usage()
            if memory_info['available_gb'] < 1.0:
                debug_print("Insufficient memory for SAM processing", "ERROR")
                st.error("メモリ不足のため、SAM処理をスキップします。")
                return []
            
            debug_print("Setting image for SAM predictor", "INFO")
            
            try:
                self.sam_predictor.set_image(image)
                debug_print("Image set successfully for SAM", "SUCCESS")
            except Exception as e:
                debug_print(f"Failed to set image for SAM: {str(e)}", "ERROR")
                st.error(f"SAMの画像設定に失敗しました: {str(e)}")
                return []
            
            detections = []
            
            if (input_points is not None and input_labels is not None) or input_boxes is not None:
                try:
                    point_coords = np.array(input_points) if input_points else None
                    point_labels = np.array(input_labels) if input_labels else None
                    box_coords = np.array(input_boxes) if input_boxes else None
                    
                    debug_print(f"Using prompts: points={len(input_points) if input_points else 0}, boxes={len(input_boxes) if input_boxes else 0}", "INFO")
                    
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_coords,
                        multimask_output=True,
                    )
                    
                    debug_print(f"SAM produced {len(masks)} masks with scores: {scores}", "INFO")
                    
                    if len(masks) == 0 or len(scores) == 0:
                        debug_print("SAM produced no valid masks", "WARNING")
                        return []
                    
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    
                    if mask is None or mask.size == 0:
                        debug_print("Invalid mask produced by SAM", "ERROR")
                        return []
                    
                    coords = np.column_stack(np.where(mask))
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        
                        if x_max > x_min and y_max > y_min:
                            detection = {
                                'x': int(x_min),
                                'y': int(y_min),
                                'width': int(x_max - x_min),
                                'height': int(y_max - y_min),
                                'confidence': float(scores[best_mask_idx]),
                                'class_name': 'segment',
                                'class_id': -1,
                                'detection_type': 'sam',
                                'center_x': int((x_min + x_max) / 2),
                                'center_y': int((y_min + y_max) / 2),
                                'mask': mask,
                                'input_type': 'area' if input_boxes else 'point'
                            }
                            detections.append(detection)
                            debug_print(f"SAM segment created: {detection['width']}x{detection['height']}, confidence: {detection['confidence']:.3f}", "SUCCESS")
                        else:
                            debug_print("Invalid bounding box from SAM mask", "WARNING")
                    else:
                        debug_print("No coordinates found in SAM mask", "WARNING")
                        
                except Exception as e:
                    debug_print(f"SAM prediction error: {str(e)}", "ERROR")
                    st.error(f"SAM予測エラー: {str(e)}")
                    return []
            
            debug_print(f"SAM detection completed: {len(detections)} objects", "SUCCESS")
            return detections
            
        except Exception as e:
            debug_print(f"SAM detection error: {str(e)}", "ERROR")
            st.error(f"SAM検出エラー: {str(e)}")
            return []

class FixationDurationPredictor:
    """視線滞留時間予測クラス（差が顕著に出るよう改良版）"""
    
    def __init__(self):
        self.base_duration = 200
        self.saliency_weight = 300
        self.complexity_weight = 200
        self.semantic_weight = 150
        self.position_weight = 100
        
    def calculate_local_complexity(self, image_patch: np.ndarray) -> float:
        """局所的な視覚的複雑度を計算"""
        try:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY) if len(image_patch.shape) == 3 else image_patch
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            texture_complexity = np.std(gray) / 255.0
            
            if len(image_patch.shape) == 3:
                color_diversity = np.mean([np.std(image_patch[:,:,i]) for i in range(3)]) / 255.0
            else:
                color_diversity = 0
            
            complexity = (edge_density * 0.4 + texture_complexity * 0.3 + color_diversity * 0.3)
            return min(1.0, complexity)
            
        except Exception as e:
            debug_print(f"Complexity calculation error: {str(e)}", "WARNING")
            return 0.5
    
    def calculate_semantic_importance(self, roi: Dict, image: np.ndarray) -> float:
        """意味的重要度を計算（物体検出情報を活用）"""
        try:
            importance = 0.5
            
            high_importance_classes = ['person', 'face', 'text', 'car', 'animal']
            medium_importance_classes = ['furniture', 'electronics', 'food']
            
            class_name = roi.get('class_name', '').lower()
            
            if any(cls in class_name for cls in high_importance_classes):
                importance = 0.9
            elif any(cls in class_name for cls in medium_importance_classes):
                importance = 0.7
            
            img_h, img_w = image.shape[:2]
            center_x, center_y = img_w / 2, img_h / 2
            roi_center_x = roi.get('center_x', center_x)
            roi_center_y = roi.get('center_y', center_y)
            
            distance_from_center = np.sqrt((roi_center_x - center_x)**2 + (roi_center_y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            centrality = 1.0 - (distance_from_center / max_distance)
            
            importance = importance * 0.7 + centrality * 0.3
            
            return min(1.0, importance)
            
        except Exception as e:
            debug_print(f"Semantic importance calculation error: {str(e)}", "WARNING")
            return 0.5
    
    def predict_fixation_duration(self, fixation_point: Tuple[int, int], 
                                 saliency_map: np.ndarray, 
                                 image: np.ndarray,
                                 roi_info: Dict = None,
                                 fixation_index: int = 0) -> float:
        """特定の注視点における滞留時間を予測（ミリ秒）- 差を強調する改良版"""
        try:
            x, y = fixation_point
            h, w = saliency_map.shape
            
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            
            saliency_value = saliency_map[y, x]
            saliency_norm = (saliency_value - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            saliency_enhanced = np.power(saliency_norm, 0.5) if saliency_norm > 0.5 else np.power(saliency_norm, 2)
            
            patch_size = 64
            y1 = max(0, y - patch_size // 2)
            y2 = min(h, y + patch_size // 2)
            x1 = max(0, x - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            
            image_patch = image[y1:y2, x1:x2]
            complexity = self.calculate_local_complexity(image_patch)
            complexity_enhanced = np.power(complexity, 0.7)
            
            semantic_importance = 0.3
            if roi_info:
                semantic_importance = self.calculate_semantic_importance(roi_info, image)
                semantic_importance = min(1.0, semantic_importance * 1.5)
            
            center_x, center_y = w / 2, h / 2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            position_factor = 1.0 - (distance_from_center / max_distance) * 0.5
            
            order_decay = 1.0 - (fixation_index * 0.03)
            order_decay = max(0.5, order_decay)
            
            duration = (self.base_duration + 
                       self.saliency_weight * saliency_enhanced +
                       self.complexity_weight * complexity_enhanced +
                       self.semantic_weight * semantic_importance +
                       self.position_weight * position_factor) * order_decay
            
            duration = max(50, min(1200, duration))
            
            noise = np.random.uniform(0.9, 1.1)
            duration = duration * noise
            
            return duration
            
        except Exception as e:
            debug_print(f"Fixation duration prediction error: {str(e)}", "WARNING")
            return self.base_duration
    
    def predict_scanpath_durations(self, fixation_points: List[Tuple[int, int]],
                                  saliency_map: np.ndarray,
                                  image: np.ndarray,
                                  rois: List[Dict] = None) -> List[Dict]:
        """スキャンパス全体の滞留時間を予測（差を強調する改良版）"""
        try:
            scanpath_data = []
            cumulative_time = 0
            
            # ROIがない場合は空のリストとして扱う
            if rois is None:
                rois = []
            
            for i, point in enumerate(fixation_points):
                roi_info = None
                roi_id = 'None'
                roi_label = 'Background'
                
                if rois:
                    for roi in rois:
                        if self._point_in_roi(point, roi):
                            roi_info = roi
                            roi_id = roi.get('id', 'None')
                            roi_label = roi.get('label', 'Background')
                            break
                
                duration = self.predict_fixation_duration(
                    point, saliency_map, image, roi_info, fixation_index=i
                )
                
                saccade_time = 0
                if i > 0:
                    prev_point = fixation_points[i-1]
                    distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
                    saccade_time = 21 + 2.2 * np.log2(distance + 1)
                
                cumulative_time += saccade_time + duration
                
                scanpath_data.append({
                    'fixation_id': i + 1,
                    'x': point[0],
                    'y': point[1],
                    'duration_ms': duration,
                    'saccade_time_ms': saccade_time,
                    'cumulative_time_ms': cumulative_time,
                    'roi': roi_id,
                    'roi_label': roi_label
                })
            
            return scanpath_data
            
        except Exception as e:
            debug_print(f"Scanpath duration prediction error: {str(e)}", "ERROR")
            return []
    
    def _point_in_roi(self, point: Tuple[int, int], roi: Dict) -> bool:
        """点がROI内にあるかチェック"""
        x, y = point
        
        if roi.get('is_rotated', False) and 'rotated_corners' in roi:
            corners = np.array(roi['rotated_corners'], dtype=np.int32)
            result = cv2.pointPolygonTest(corners, (x, y), False)
            return result >= 0
        else:
            roi_x = roi['x']
            roi_y = roi['y']
            roi_w = roi['width']
            roi_h = roi['height']
            return (roi_x <= x <= roi_x + roi_w) and (roi_y <= y <= roi_y + roi_h)

class ROIDeepGazeAnalyzer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cpu')
        self.model_loaded = False
        self.auto_detector = AutoDetectionManager()
        self.coord_manager = ROICoordinateManager()
        self.duration_predictor = FixationDurationPredictor()
        
    def load_model(self):
        """DeepGaze IIIモデルの読み込み"""
        try:
            debug_print("Loading DeepGaze III model", "INFO")
            
            memory_info = check_memory_usage()
            debug_print(f"Available memory: {memory_info['available_gb']:.2f} GB", "INFO")
            
            with st.spinner("DeepGaze IIIモデルを読み込んでいます..."):
                cleanup_temp_files()
                gc.collect()
                
                model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(self.device).eval()
                self.model_loaded = True
                debug_print("DeepGaze III model loaded successfully", "SUCCESS")
                return model
                
        except Exception as e:
            debug_print(f"DeepGaze III loading error: {str(e)}", "ERROR")
            st.error(f"DeepGaze IIIモデル読み込みエラー: {str(e)}")
            return None
    
    def load_centerbias(self, h: int, w: int) -> np.ndarray:
        """MIT1003センターバイアスの読み込み"""
        try:
            debug_print(f"Loading center bias for {w}x{h}", "INFO")
            
            path = "centerbias_mit1003.npy"
            if not os.path.exists(path):
                debug_print("Center bias file not found, downloading", "INFO")
                
                disk_info = check_disk_space()
                if disk_info['free_gb'] < 0.1:
                    debug_print("Insufficient disk space for center bias", "ERROR")
                    return self.generate_center_bias(h, w, 0.6)
                
                url = ("https://github.com/matthias-k/DeepGaze/"
                       "releases/download/v1.0.0/centerbias_mit1003.npy")
                try:
                    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
                        shutil.copyfileobj(r, f)
                    debug_print("Center bias downloaded successfully", "SUCCESS")
                except Exception as e:
                    debug_print(f"Center bias download error: {str(e)}", "WARNING")
                    return self.generate_center_bias(h, w, 0.6)
            
            try:
                cb = np.load(path)
                cb = cv2.resize(cb, (w, h))
                cb -= logsumexp(cb)
                debug_print("Center bias loaded and processed", "SUCCESS")
                return cb
            except Exception as e:
                debug_print(f"Center bias processing error: {str(e)}", "WARNING")
                return self.generate_center_bias(h, w, 0.6)
                
        except Exception as e:
            debug_print(f"Center bias loading error: {str(e)}", "ERROR")
            return self.generate_center_bias(h, w, 0.6)
    
    def generate_center_bias(self, height: int, width: int, center_bias_weight: float = 0.6) -> np.ndarray:
        """シンプルなセンターバイアス生成"""
        try:
            debug_print(f"Generating simple center bias for {width}x{height}", "INFO")
            
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            if max_distance > 0:
                gaussian_bias = np.exp(-distance**2 / (2 * (max_distance / 3)**2))
            else:
                gaussian_bias = np.ones((height, width))
            
            center_bias = np.log(gaussian_bias + 1e-8)
            center_bias -= logsumexp(center_bias)
            
            debug_print("Simple center bias generated", "SUCCESS")
            return center_bias
            
        except Exception as e:
            debug_print(f"Center bias generation error: {str(e)}", "ERROR")
            return np.zeros((height, width))
    
    def run_deepgaze(self, img: np.ndarray, cb_log: np.ndarray, seed_xy: Tuple[int, int]) -> np.ndarray:
        """DeepGaze IIIの実行"""
        try:
            debug_print(f"Running DeepGaze with seed: {seed_xy}", "INFO")
            
            if self.model is None:
                debug_print("DeepGaze model not loaded, loading now", "INFO")
                self.model = self.load_model()
                if self.model is None:
                    debug_print("DeepGaze model loading failed", "ERROR")
                    return None
            
            fx, fy = seed_xy
            hx = [fx] + [np.nan] * (len(self.model.included_fixations) - 1)
            hy = [fy] + [np.nan] * (len(self.model.included_fixations) - 1)
            
            debug_print(f"Creating tensors with shapes: img={img.shape}, cb_log={cb_log.shape}", "DEBUG")
            
            it = torch.tensor([img.transpose(2, 0, 1)], dtype=torch.float32).to(self.device)
            ct = torch.tensor([cb_log], dtype=torch.float32).to(self.device)
            xt = torch.tensor([hx], dtype=torch.float32).to(self.device)
            yt = torch.tensor([hy], dtype=torch.float32).to(self.device)
            
            debug_print("Running DeepGaze inference", "INFO")
            with torch.no_grad():
                log_map = self.model(it, ct, xt, yt)[0, 0].cpu().numpy()
            
            result = np.exp(log_map)
            debug_print(f"DeepGaze inference completed, output shape: {result.shape}", "SUCCESS")
            return result
            
        except Exception as e:
            debug_print(f"DeepGaze execution error: {str(e)}", "ERROR")
            st.error(f"DeepGaze実行エラー: {str(e)}")
            return None
    
    def apply_ior(self, p: np.ndarray, rad: int, dec: float, n: int = 5) -> np.ndarray:
        """IOR (Inhibition of Return) の適用"""
        try:
            debug_print(f"Applying IOR with radius={rad}, decay={dec}, iterations={n}", "INFO")
            
            q = p.copy()
            for i in range(n):
                y, x = np.unravel_index(q.argmax(), q.shape)
                m = np.zeros_like(q)
                cv2.circle(m, (x, y), rad, 1, -1)
                m = cv2.GaussianBlur(m, (0, 0), rad / 2)
                q *= 1 - dec * m
                
                if i < 3:
                    debug_print(f"IOR iteration {i}: max at ({x}, {y})", "DEBUG")
            
            result = q / (q.max() + 1e-8)
            debug_print("IOR applied successfully", "SUCCESS")
            return result
            
        except Exception as e:
            debug_print(f"IOR application error: {str(e)}", "ERROR")
            return p
    
    def extract_fixations(self, p: np.ndarray, n: int, rad: int, dec: float) -> List[Tuple[int, int]]:
        """注視点の抽出"""
        try:
            debug_print(f"Extracting {n} fixations with radius={rad}, decay={dec}", "INFO")
            
            pts = []
            q = p.copy()
            for i in range(n):
                y, x = np.unravel_index(q.argmax(), q.shape)
                pts.append((x, y))
                m = np.zeros_like(q)
                cv2.circle(m, (x, y), rad, 1, -1)
                m = cv2.GaussianBlur(m, (0, 0), rad / 2)
                q *= 1 - dec * m
                
                if i < 5:
                    debug_print(f"Fixation {i}: ({x}, {y})", "DEBUG")
            
            debug_print(f"Extracted {len(pts)} fixations", "SUCCESS")
            return pts
            
        except Exception as e:
            debug_print(f"Fixation extraction error: {str(e)}", "ERROR")
            return []
    
    def resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """画像のリサイズ処理"""
        try:
            debug_print(f"Resizing image from {image.size} to max {max_size}", "INFO")
            
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                debug_print(f"Image resized to {new_size}", "SUCCESS")
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                debug_print("Image converted to RGB", "INFO")
            
            return image
            
        except Exception as e:
            debug_print(f"Image resize error: {str(e)}", "ERROR")
            return image
    
    def set_image_dimensions(self, image_width: int, image_height: int, 
                           canvas_width: int = None, canvas_height: int = None):
        """画像とCanvasの寸法を設定"""
        if canvas_width is None:
            canvas_width = image_width
        if canvas_height is None:
            canvas_height = image_height
            
        self.coord_manager.set_coordinate_system(canvas_width, canvas_height, image_width, image_height)
        debug_print(f"Image dimensions set: {image_width}x{image_height}, Canvas: {canvas_width}x{canvas_height}", "INFO")
    
    def calculate_rotated_corners(self, center_x: float, center_y: float, 
                                width: float, height: float, angle: float) -> List[Tuple[float, float]]:
        """回転した矩形の角座標を計算（数学的に正確な版）"""
        try:
            half_width = width / 2
            half_height = height / 2
            
            corners_relative = [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height)
            ]
            
            angle_rad = math.radians(angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            rotated_corners = []
            for rel_x, rel_y in corners_relative:
                rotated_x = rel_x * cos_angle - rel_y * sin_angle
                rotated_y = rel_x * sin_angle + rel_y * cos_angle
                
                abs_x = center_x + rotated_x
                abs_y = center_y + rotated_y
                
                rotated_corners.append((abs_x, abs_y))
            
            debug_print(f"Rotated corners calculated for angle {angle:.1f}°: center=({center_x:.1f}, {center_y:.1f})", "DEBUG")
            return rotated_corners
            
        except Exception as e:
            debug_print(f"Rotated corners calculation error: {str(e)}", "ERROR")
            return [(center_x, center_y)] * 4
    
    def create_rotated_mask(self, saliency_shape: Tuple[int, int], center_x: float, center_y: float,
                          width: float, height: float, angle: float) -> np.ndarray:
        """回転ROI用の正確なマスクを生成"""
        try:
            debug_print(f"Creating rotated mask: center=({center_x:.1f}, {center_y:.1f}), size=({width:.1f}, {height:.1f}), angle={angle:.1f}°", "DEBUG")
            
            corners = self.calculate_rotated_corners(center_x, center_y, width, height, angle)
            
            corner_points = np.array([(int(round(x)), int(round(y))) for x, y in corners], dtype=np.int32)
            
            mask = np.zeros(saliency_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corner_points], 1)
            
            debug_print(f"Rotated mask created: {np.sum(mask)} pixels filled", "DEBUG")
            return mask.astype(bool)
            
        except Exception as e:
            debug_print(f"Rotated mask creation error: {str(e)}", "ERROR")
            x1 = max(0, int(center_x - width/2))
            y1 = max(0, int(center_y - height/2))
            x2 = min(saliency_shape[1], int(center_x + width/2))
            y2 = min(saliency_shape[0], int(center_y + height/2))
            
            mask = np.zeros(saliency_shape, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            return mask.astype(bool)
    
    def extract_rois_from_canvas(self, canvas_data: dict, image_width: int, image_height: int) -> List[Dict]:
        """Canvas JSONからROI情報を抽出（回転ROI完全対応版）"""
        try:
            debug_print(f"Extracting ROIs from canvas (rotation fixed) for {image_width}x{image_height}", "INFO")
            
            self.coord_manager.set_coordinate_system(image_width, image_height, image_width, image_height)
            
            rois = []
            if canvas_data and 'objects' in canvas_data:
                debug_print(f"Found {len(canvas_data['objects'])} canvas objects", "INFO")
                
                for i, obj in enumerate(canvas_data['objects']):
                    if obj.get('type') == 'rect':
                        try:
                            image_rect = self.coord_manager.canvas_rect_to_image_rect(obj)
                            
                            angle = float(obj.get('angle', 0))
                            
                            roi = {
                                'id': i,
                                'x': image_rect['x'],
                                'y': image_rect['y'],
                                'width': image_rect['width'],
                                'height': image_rect['height'],
                                'center_x': image_rect['center_x'],
                                'center_y': image_rect['center_y'],
                                'angle': angle,
                                'source': 'manual',
                                'canvas_x': image_rect['canvas_x'],
                                'canvas_y': image_rect['canvas_y'],
                                'canvas_width': image_rect['canvas_width'],
                                'canvas_height': image_rect['canvas_height']
                            }
                            
                            if abs(angle) > 0.1:
                                corners = self.calculate_rotated_corners(
                                    image_rect['center_x'], image_rect['center_y'],
                                    image_rect['width'], image_rect['height'], angle
                                )
                                roi['rotated_corners'] = corners
                                roi['is_rotated'] = True
                                debug_print(f"ROI {i} rotated by {angle:.1f}°: corners calculated", "DEBUG")
                            else:
                                roi['is_rotated'] = False
                            
                            if (0 <= roi['center_x'] <= image_width and 
                                0 <= roi['center_y'] <= image_height):
                                rois.append(roi)
                                debug_print(f"ROI {i} added: center=({roi['center_x']:.1f}, {roi['center_y']:.1f}), angle={angle:.1f}°", "DEBUG")
                            else:
                                debug_print(f"ROI {i} center out of bounds, skipped", "WARNING")
                                
                        except Exception as e:
                            debug_print(f"Error processing ROI {i}: {str(e)}", "ERROR")
                            continue
            
            debug_print(f"Extracted {len(rois)} valid ROIs", "SUCCESS")
            return rois
            
        except Exception as e:
            debug_print(f"ROI extraction error: {str(e)}", "ERROR")
            return []
    
    def extract_seed_points_from_canvas(self, canvas_data: dict, image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """Canvas JSONからシードポイントを抽出（座標修正版）"""
        try:
            debug_print("Extracting seed points from canvas (fixed)", "INFO")
            
            self.coord_manager.set_coordinate_system(image_width, image_height, image_width, image_height)
            
            points = []
            if canvas_data and 'objects' in canvas_data:
                for obj in canvas_data['objects']:
                    if obj.get('type') == 'circle':
                        try:
                            canvas_x = float(obj.get('left', 0))
                            canvas_y = float(obj.get('top', 0))
                            radius = float(obj.get('radius', 5))
                            
                            center_canvas_x = canvas_x + radius
                            center_canvas_y = canvas_y + radius
                            
                            image_x, image_y = self.coord_manager.canvas_to_image_coords(
                                center_canvas_x, center_canvas_y
                            )
                            
                            points.append((image_x, image_y))
                            debug_print(f"Seed point: Canvas({center_canvas_x:.1f}, {center_canvas_y:.1f}) -> Image({image_x}, {image_y})", "DEBUG")
                            
                        except Exception as e:
                            debug_print(f"Error processing seed point: {str(e)}", "ERROR")
                            continue
            
            debug_print(f"Extracted {len(points)} seed points", "SUCCESS")
            return points
            
        except Exception as e:
            debug_print(f"Seed point extraction error: {str(e)}", "ERROR")
            return []
    
    def detections_to_rois(self, detections: List[Dict], source: str = 'auto') -> List[Dict]:
        """検出結果をROI形式に変換（SAMマスク対応）"""
        try:
            debug_print(f"Converting {len(detections)} detections to ROIs", "INFO")
            
            rois = []
            for i, detection in enumerate(detections):
                roi = {
                    'id': i,
                    'x': detection['x'],
                    'y': detection['y'],
                    'width': detection['width'],
                    'height': detection['height'],
                    'center_x': detection['center_x'],
                    'center_y': detection['center_y'],
                    'angle': 0,
                    'is_rotated': False,
                    'source': source,
                    'confidence': detection.get('confidence', 0.0),
                    'class_name': detection.get('class_name', 'unknown'),
                    'detection_type': detection.get('detection_type', 'unknown')
                }
                
                if 'mask' in detection and detection['mask'] is not None:
                    try:
                        mask = detection['mask']
                        debug_print(f"Processing SAM mask for detection {i}", "DEBUG")
                        
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            
                            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                            
                            contour_points = [(int(point[0][0]), int(point[0][1])) for point in simplified_contour]
                            
                            roi['has_mask'] = True
                            roi['mask'] = mask
                            roi['contour_points'] = contour_points
                            roi['contour_area'] = cv2.contourArea(largest_contour)
                            
                            debug_print(f"ROI {i}: contour with {len(contour_points)} points, area: {roi['contour_area']:.1f}", "DEBUG")
                        else:
                            debug_print(f"No contours found for detection {i}", "WARNING")
                            roi['has_mask'] = False
                            
                    except Exception as e:
                        debug_print(f"Error processing mask for detection {i}: {str(e)}", "WARNING")
                        roi['has_mask'] = False
                else:
                    roi['has_mask'] = False
                
                rois.append(roi)
                
            debug_print(f"Converted to {len(rois)} ROIs", "SUCCESS")
            return rois
            
        except Exception as e:
            debug_print(f"Detection to ROI conversion error: {str(e)}", "ERROR")
            return []
    
    def generate_saliency_map(self, image: Image.Image, seed_points: List[Tuple[int, int]], 
                            params: Dict) -> np.ndarray:
        """サリエンシーマップの生成"""
        try:
            debug_print("Generating saliency map", "INFO")
            
            img_array = np.array(image)
            if img_array is None or img_array.size == 0:
                debug_print("Invalid image array", "ERROR")
                return None
            
            h, w = img_array.shape[:2]
            debug_print(f"Image shape: {w}x{h}", "INFO")
            
            cb_log = self.load_centerbias(h, w) * params['center_bias_weight']
            debug_print(f"Center bias loaded with weight: {params['center_bias_weight']}", "INFO")
            
            seeds = seed_points if seed_points else [(w//2, h//2)]
            debug_print(f"Using {len(seeds)} seed points", "INFO")
            
            if params.get('add_corner_seeds', True):
                corner_seeds = [(w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]
                seeds += corner_seeds
                debug_print(f"Added corner seeds, total: {len(seeds)}", "INFO")
            
            saliency_maps = []
            for i, seed in enumerate(seeds):
                debug_print(f"Processing seed {i}: {seed}", "DEBUG")
                sal_map = self.run_deepgaze(img_array, cb_log, seed)
                if sal_map is not None:
                    saliency_maps.append(sal_map)
            
            if not saliency_maps:
                debug_print("No saliency maps generated", "ERROR")
                return None
            
            debug_print(f"Generated {len(saliency_maps)} saliency maps", "INFO")
            sal = np.mean(saliency_maps, axis=0)
            
            debug_print("Applying IOR", "INFO")
            sal = self.apply_ior(sal / sal.max(), params['saccade_radius'], params['ior_decay'])
            
            debug_print(f"Applying gamma correction: {params['gamma']}", "INFO")
            sal = np.power(sal, params['gamma'])
            
            if params['blur_sigma'] > 0:
                debug_print(f"Applying Gaussian blur: {params['blur_sigma']}", "INFO")
                k = int(params['blur_sigma'] * 3) // 2 * 2 + 1
                sal = cv2.GaussianBlur(sal, (k, k), params['blur_sigma'], 
                                     borderType=cv2.BORDER_REPLICATE)
                sal /= sal.max() + 1e-8
            
            debug_print("Saliency map generation completed", "SUCCESS")
            return sal
            
        except Exception as e:
            debug_print(f"Saliency map generation error: {str(e)}", "ERROR")
            return None
    
    def calculate_roi_attention(self, saliency_map: np.ndarray, rois: List[Dict], 
                              roi_labels: List[str]) -> pd.DataFrame:
        """ROI内の注目度を計算（回転ROI位置ずれ完全修正版）"""
        try:
            debug_print(f"Calculating attention for {len(rois)} ROIs (rotation position fixed)", "INFO")
            
            if saliency_map is None or len(saliency_map.shape) != 2:
                debug_print("Invalid saliency map", "ERROR")
                return pd.DataFrame()
            
            total_attention = np.sum(saliency_map)
            debug_print(f"Total attention: {total_attention:.3f}", "INFO")
            
            roi_data = []
            
            for i, (roi, label) in enumerate(zip(rois, roi_labels)):
                try:
                    debug_print(f"Processing ROI {i}: {label}", "DEBUG")
                    
                    if roi.get('has_mask', False) and 'mask' in roi:
                        mask = roi['mask']
                        if mask.shape != saliency_map.shape:
                            mask = cv2.resize(mask.astype(np.uint8), 
                                            (saliency_map.shape[1], saliency_map.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        roi_attention = np.sum(saliency_map[mask])
                        roi_max = np.max(saliency_map[mask]) if np.any(mask) else 0
                        roi_mean = np.mean(saliency_map[mask]) if np.any(mask) else 0
                        roi_pixels = np.sum(mask)
                        
                    elif roi.get('is_rotated', False):
                        mask = self.create_rotated_mask(
                            saliency_map.shape,
                            roi['center_x'], roi['center_y'],
                            roi['width'], roi['height'],
                            roi['angle']
                        )
                        
                        roi_attention = np.sum(saliency_map[mask])
                        roi_max = np.max(saliency_map[mask]) if np.any(mask) else 0
                        roi_mean = np.mean(saliency_map[mask]) if np.any(mask) else 0
                        roi_pixels = np.sum(mask)
                        
                    else:
                        x1, y1 = roi['x'], roi['y']
                        x2 = min(x1 + roi['width'], saliency_map.shape[1])
                        y2 = min(y1 + roi['height'], saliency_map.shape[0])
                        
                        roi_region = saliency_map[y1:y2, x1:x2]
                        roi_attention = np.sum(roi_region)
                        roi_max = np.max(roi_region) if roi_region.size > 0 else 0
                        roi_mean = np.mean(roi_region) if roi_region.size > 0 else 0
                        roi_pixels = roi_region.size
                    
                    roi_percentage = (roi_attention / total_attention * 100) if total_attention > 0 else 0
                    
                    roi_data.append({
                        'ROI_ID': roi['id'],
                        'Label': label,
                        'Source': roi.get('source', 'unknown'),
                        'Class': roi.get('class_name', 'N/A'),
                        'Confidence': roi.get('confidence', 0.0),
                        'X': roi['x'],
                        'Y': roi['y'],
                        'Width': roi['width'],
                        'Height': roi['height'],
                        'Angle': roi.get('angle', 0),
                        'Pixels': roi_pixels,
                        'Total_Attention': roi_attention,
                        'Mean_Attention': roi_mean,
                        'Max_Attention': roi_max,
                        'Attention_Percentage': roi_percentage
                    })
                    
                    debug_print(f"ROI {i}: attention={roi_attention:.3f}, percentage={roi_percentage:.2f}%", "DEBUG")
                    
                except Exception as e:
                    debug_print(f"Error processing ROI {i}: {str(e)}", "WARNING")
                    continue
            
            df = pd.DataFrame(roi_data)
            debug_print(f"ROI attention calculation completed for {len(df)} ROIs", "SUCCESS")
            return df
            
        except Exception as e:
            debug_print(f"ROI attention calculation error: {str(e)}", "ERROR")
            return pd.DataFrame()
    
    def analyze_with_duration(self, image: Image.Image, 
                             seed_points: List[Tuple[int, int]],
                             rois: List[Dict],
                             roi_labels: List[str],
                             params: Dict) -> Dict:
        """滞留時間を含む完全な分析を実行"""
        try:
            debug_print("Starting analysis with fixation duration prediction", "INFO")
            
            saliency_map = self.generate_saliency_map(image, seed_points, params)
            if saliency_map is None:
                return None
            
            num_fixations = params.get('num_fixations', 10)
            saccade_radius = params.get('saccade_radius', 30)
            ior_decay = params.get('ior_decay', 0.8)
            
            fixation_points = self.extract_fixations(
                saliency_map, num_fixations, saccade_radius, ior_decay
            )
            
            # ROIがある場合のみラベルを設定
            if rois and roi_labels:
                for roi, label in zip(rois, roi_labels):
                    roi['label'] = label
            
            image_array = np.array(image)
            scanpath_data = self.duration_predictor.predict_scanpath_durations(
                fixation_points, saliency_map, image_array, rois
            )
            
            # ROIがある場合のみROI注目度を計算
            if rois and roi_labels:
                roi_attention_df = self.calculate_roi_attention(saliency_map, rois, roi_labels)
                roi_duration_stats = self.calculate_roi_duration_stats(scanpath_data, rois)
            else:
                roi_attention_df = pd.DataFrame()
                roi_duration_stats = pd.DataFrame()
            
            return {
                'saliency_map': saliency_map,
                'fixation_points': fixation_points,
                'scanpath_data': scanpath_data,
                'roi_attention': roi_attention_df,
                'roi_duration_stats': roi_duration_stats,
                'total_viewing_time_ms': scanpath_data[-1]['cumulative_time_ms'] if scanpath_data else 0
            }
            
        except Exception as e:
            debug_print(f"Analysis with duration error: {str(e)}", "ERROR")
            return None
    
    def calculate_roi_duration_stats(self, scanpath_data: List[Dict], 
                                    rois: List[Dict]) -> pd.DataFrame:
        """ROIごとの滞留時間統計を計算"""
        try:
            roi_stats = []
            
            for roi in rois:
                roi_id = roi.get('id', 'unknown')
                roi_label = roi.get('label', f'ROI_{roi_id}')
                
                roi_fixations = [f for f in scanpath_data if f.get('roi') == roi_id]
                
                if roi_fixations:
                    total_duration = sum(f['duration_ms'] for f in roi_fixations)
                    avg_duration = total_duration / len(roi_fixations)
                    first_fixation_time = roi_fixations[0]['cumulative_time_ms'] - roi_fixations[0]['duration_ms']
                    
                    roi_stats.append({
                        'ROI_ID': roi_id,
                        'ROI_Label': roi_label,
                        'Total_Duration_ms': total_duration,
                        'Avg_Duration_ms': avg_duration,
                        'Fixation_Count': len(roi_fixations),
                        'First_Fixation_Time_ms': first_fixation_time,
                        'Duration_Percentage': 0
                    })
                else:
                    roi_stats.append({
                        'ROI_ID': roi_id,
                        'ROI_Label': roi_label,
                        'Total_Duration_ms': 0,
                        'Avg_Duration_ms': 0,
                        'Fixation_Count': 0,
                        'First_Fixation_Time_ms': -1,
                        'Duration_Percentage': 0
                    })
            
            total_duration_all = sum(r['Total_Duration_ms'] for r in roi_stats)
            if total_duration_all > 0:
                for stat in roi_stats:
                    stat['Duration_Percentage'] = (stat['Total_Duration_ms'] / total_duration_all) * 100
            
            return pd.DataFrame(roi_stats)
            
        except Exception as e:
            debug_print(f"ROI duration stats calculation error: {str(e)}", "ERROR")
            return pd.DataFrame()

# ========== 可視化関数 ==========
def create_custom_colormap():
    """赤→黄色→青のカスタムカラーマップを作成"""
    colors = [(0, 0, 1),     # 青 (低い値)
              (1, 1, 0),     # 黄色 (中間値)  
              (1, 0, 0)]     # 赤 (高い値)
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_rby', colors, N=n_bins)
    return cmap

def apply_custom_colormap(saliency_map: np.ndarray) -> np.ndarray:
    """カスタムカラーマップを適用"""
    # 0-1に正規化
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # カスタムカラーマップを取得
    cmap = create_custom_colormap()
    
    # カラーマップを適用 (matplotlib使用)
    colored = cmap(saliency_norm)
    
    # RGBに変換 (0-255の範囲に)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_rgb

def create_heatmap_overlay_with_roi(image: np.ndarray, saliency_map: np.ndarray, 
                                   rois: List[Dict] = None, roi_labels: List[str] = None,
                                   alpha: float = 0.5, colormap: str = 'custom',
                                   show_roi: bool = True) -> np.ndarray:
    """グレースケール画像にヒートマップとROIをオーバーレイ"""
    try:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            gray_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if saliency_map.shape[:2] != image.shape[:2]:
            saliency_resized = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
        else:
            saliency_resized = saliency_map
        
        # カスタムカラーマップ（赤→黄色→青）を適用
        colored_saliency = apply_custom_colormap(saliency_resized)
        
        overlay = cv2.addWeighted(gray_rgb, 1 - alpha, colored_saliency, alpha, 0)
        
        if show_roi and rois is not None:
            from PIL import Image, ImageDraw, ImageFont
            
            overlay_pil = Image.fromarray(overlay)
            draw = ImageDraw.Draw(overlay_pil)
            
            try:
                if platform.system() == 'Windows':
                    font = ImageFont.truetype("meiryo.ttc", 12)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            for i, roi in enumerate(rois):
                colors_list = [
                    (255, 100, 100),
                    (100, 255, 100),
                    (100, 100, 255),
                    (255, 255, 100),
                    (255, 100, 255),
                    (100, 255, 255),
                ]
                roi_color = colors_list[i % len(colors_list)]
                
                if roi.get('is_rotated', False) and 'rotated_corners' in roi:
                    corners = roi['rotated_corners']
                    points = [(int(x), int(y)) for x, y in corners]
                    draw.polygon(points, outline=roi_color, width=2)
                else:
                    x1, y1 = roi['x'], roi['y']
                    x2 = x1 + roi['width']
                    y2 = y1 + roi['height']
                    draw.rectangle([x1, y1, x2, y2], outline=roi_color, width=2)
                
                if roi_labels and i < len(roi_labels):
                    label = roi_labels[i]
                    label_x = roi['x'] + 2
                    label_y = max(0, roi['y'] - 15)
                    
                    try:
                        bbox = draw.textbbox((label_x, label_y), label, font=font)
                        draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], 
                                     fill=(0, 0, 0, 180))
                    except:
                        pass
                    
                    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
                    
                    number_text = f"#{i+1}"
                    draw.text((roi['x'] + roi['width'] - 20, roi['y'] + 2), 
                            number_text, fill=roi_color, font=font)
            
            overlay = np.array(overlay_pil)
        
        return overlay
        
    except Exception as e:
        debug_print(f"Heatmap overlay with ROI error: {str(e)}", "ERROR")
        return image

def create_heatmap_overlay(image: np.ndarray, saliency_map: np.ndarray, 
                          alpha: float = 0.5, colormap: str = 'custom') -> np.ndarray:
    """グレースケール画像にヒートマップをオーバーレイ（後方互換性のため維持）"""
    return create_heatmap_overlay_with_roi(image, saliency_map, None, None, alpha, colormap, False)

def visualize_scanpath_skeleton(image: np.ndarray, 
                              scanpath_data: List[Dict],
                              circle_radius: int = 25,
                              alpha: float = 0.3,
                              show_duration: bool = True) -> np.ndarray:
    """スケルトンスタイルのスキャンパス可視化（凡例削除版）"""
    try:
        vis_image = image.copy()
        
        if not scanpath_data:
            return vis_image
        
        overlay = vis_image.copy()
        
        # まず矢印を描画（注視点の下に配置）
        for i in range(1, len(scanpath_data)):
            curr_fixation = scanpath_data[i]
            prev_fixation = scanpath_data[i - 1]
            curr_x, curr_y = int(curr_fixation['x']), int(curr_fixation['y'])
            prev_x, prev_y = int(prev_fixation['x']), int(prev_fixation['y'])
            
            # 矢印の描画（グラデーション効果）
            # 太い矢印（背景）
            cv2.arrowedLine(overlay, (prev_x, prev_y), (curr_x, curr_y),
                          (0, 180, 0), 4, tipLength=0.15)
            # 細い矢印（前景）
            cv2.arrowedLine(overlay, (prev_x, prev_y), (curr_x, curr_y),
                          (0, 255, 0), 2, tipLength=0.15)
        
        # 注視点と番号を描画
        for i, fixation in enumerate(scanpath_data):
            x, y = int(fixation['x']), int(fixation['y'])
            
            # 注視点の順番に応じて色を変化（早い：青→遅い：赤）
            color_ratio = i / max(len(scanpath_data) - 1, 1)
            b = int(255 * (1 - color_ratio))
            r = int(255 * color_ratio)
            g = 100
            circle_color = (b, g, r)
            
            # 円の描画（多層構造で立体感を演出）
            # 外側の影
            cv2.circle(overlay, (x, y), circle_radius + 4, (50, 50, 50), -1)
            # 外枠（グロー効果）
            cv2.circle(overlay, (x, y), circle_radius + 2, (255, 255, 255), -1)
            # メインの円（グラデーション色）
            cv2.circle(overlay, (x, y), circle_radius, circle_color, -1)
            # 内側のハイライト
            highlight_x = x - circle_radius // 3
            highlight_y = y - circle_radius // 3
            cv2.circle(overlay, (highlight_x, highlight_y), circle_radius // 3, 
                      (255, 255, 255), -1)
            
            # 円の境界線
            cv2.circle(vis_image, (x, y), circle_radius + 1, (0, 0, 0), 3)
            cv2.circle(vis_image, (x, y), circle_radius, (255, 255, 255), 2)
            
            # テキストの準備
            text = str(i + 1)
            
            # フォント設定（より大きく、太く）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 3
            
            # テキストサイズの取得
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            
            # テキストの背景（可読性向上）
            padding = 3
            bg_pt1 = (text_x - padding, text_y - text_size[1] - padding)
            bg_pt2 = (text_x + text_size[0] + padding, text_y + padding)
            
            # 半透明の黒い背景
            overlay_text = overlay.copy()
            cv2.rectangle(overlay_text, bg_pt1, bg_pt2, (0, 0, 0), -1)
            overlay = cv2.addWeighted(overlay, 0.7, overlay_text, 0.3, 0)
            
            # テキストの描画（多層構造）
            # 影（黒）
            shadow_offset = 2
            cv2.putText(vis_image, text, (text_x + shadow_offset, text_y + shadow_offset),
                       font, font_scale, (0, 0, 0), font_thickness + 2)
            
            # 外枠（白）
            cv2.putText(vis_image, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), font_thickness + 1)
            
            # メインテキスト（黒）
            cv2.putText(vis_image, text, (text_x, text_y),
                       font, font_scale, (0, 0, 0), font_thickness - 1)
            
            # 滞留時間の表示（オプション）
            if show_duration and 'duration_ms' in fixation:
                duration_text = f"{fixation['duration_ms']:.0f}ms"
                duration_font_scale = 0.4
                duration_thickness = 1
                
                duration_size = cv2.getTextSize(duration_text, font, 
                                               duration_font_scale, duration_thickness)[0]
                duration_x = x - duration_size[0] // 2
                duration_y = y + circle_radius + 15
                
                # 滞留時間の背景
                dur_bg_pt1 = (duration_x - 2, duration_y - duration_size[1] - 2)
                dur_bg_pt2 = (duration_x + duration_size[0] + 2, duration_y + 2)
                cv2.rectangle(vis_image, dur_bg_pt1, dur_bg_pt2, (255, 255, 200), -1)
                cv2.rectangle(vis_image, dur_bg_pt1, dur_bg_pt2, (100, 100, 100), 1)
                
                # 滞留時間テキスト
                cv2.putText(vis_image, duration_text, (duration_x, duration_y),
                           font, duration_font_scale, (50, 50, 50), duration_thickness)
        
        # 最初と最後の注視点にマーカーを追加
        if len(scanpath_data) > 0:
            # START マーカー
            start_x, start_y = int(scanpath_data[0]['x']), int(scanpath_data[0]['y'])
            cv2.putText(vis_image, "START", (start_x - 25, start_y - circle_radius - 10),
                       font, 0.5, (0, 200, 0), 2)
            
            # END マーカー
            if len(scanpath_data) > 1:
                end_x, end_y = int(scanpath_data[-1]['x']), int(scanpath_data[-1]['y'])
                cv2.putText(vis_image, "END", (end_x - 15, end_y - circle_radius - 10),
                           font, 0.5, (200, 0, 0), 2)
        
        # 凡例を削除（Fixation Order Early->Lateの表示を削除）
        
        result = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
        
        return result
        
    except Exception as e:
        debug_print(f"Scanpath skeleton visualization error: {str(e)}", "ERROR")
        return image

def display_duration_analysis(analysis_results: Dict):
    """滞留時間分析結果の表示（ROI別完全削除版）"""
    
    st.markdown('<div class="section-header">🕐 視線滞留時間分析</div>', unsafe_allow_html=True)
    
    if 'scanpath_data' in analysis_results and analysis_results['scanpath_data']:
        scanpath_df = pd.DataFrame(analysis_results['scanpath_data'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総視認時間", 
                     f"{analysis_results.get('total_viewing_time_ms', 0):.1f} ms")
        with col2:
            st.metric("平均滞留時間", 
                     f"{scanpath_df['duration_ms'].mean():.1f} ms")
        with col3:
            st.metric("最長滞留時間", 
                     f"{scanpath_df['duration_ms'].max():.1f} ms")
        with col4:
            st.metric("注視点数", 
                     len(scanpath_df))
        
        st.subheader("📊 注視点別滞留時間")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        fixation_ids = scanpath_df['fixation_id'].tolist()
        durations = scanpath_df['duration_ms'].tolist()
        cumulative_times = scanpath_df['cumulative_time_ms'].tolist()
        
        bars = ax1.bar(fixation_ids, durations, color='steelblue', alpha=0.7)
        
        max_idx = durations.index(max(durations))
        min_idx = durations.index(min(durations))
        bars[max_idx].set_color('red')
        bars[min_idx].set_color('blue')
        
        mean_duration = np.mean(durations)
        ax1.axhline(y=mean_duration, color='green', linestyle='--', 
                   linewidth=2, label=f'平均: {mean_duration:.1f}ms')
        
        ax1.set_xticks(fixation_ids)
        ax1.set_xticklabels(fixation_ids)
        ax1.set_xlabel('注視点番号', fontsize=11)
        ax1.set_ylabel('滞留時間 (ms)', fontsize=11)
        ax1.set_title('注視点別滞留時間', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.plot(fixation_ids, cumulative_times, color='green', marker='o', 
                linewidth=2, markersize=8)
        ax2.fill_between(fixation_ids, 0, cumulative_times, alpha=0.3, color='green')
        
        ax2.set_xticks(fixation_ids)
        ax2.set_xticklabels(fixation_ids)
        ax2.set_xlabel('注視点番号', fontsize=11)
        ax2.set_ylabel('累積時間 (ms)', fontsize=11)
        ax2.set_title('累積視認時間', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# ========== メインアプリケーション ==========
def main():
    st.markdown('<h1 class="main-header">👁️ ROI DeepGaze Analyzer Ultimate with Duration</h1>', unsafe_allow_html=True)
    st.markdown("### 視線予測と滞留時間分析システム")
    
    # セッション状態の初期化
    if 'show_debug' not in st.session_state:
        st.session_state['show_debug'] = False
    
    with st.sidebar:
        st.header("⚙️ 設定")
        
        st.subheader("分析パラメータ")
        params = {
            'center_bias_weight': st.slider("センターバイアス重み", 0.0, 1.0, 0.6, 0.1),
            'gamma': st.slider("ガンマ補正", 0.1, 3.0, 1.0, 0.1),
            'blur_sigma': st.slider("ブラーσ", 0.0, 10.0, 0.0, 0.5),
            'saccade_radius': st.slider("サッケード半径", 10, 100, 30, 5),
            'ior_decay': st.slider("IOR減衰", 0.1, 1.0, 0.8, 0.1),
            'num_fixations': st.slider("注視点数", 1, 20, 10, 1),
            'add_corner_seeds': st.checkbox("コーナーシード追加", True)
        }
        
        st.subheader("可視化設定")
        heatmap_alpha = st.slider("ヒートマップ透明度", 0.1, 0.9, 0.5, 0.05,
                                 help="ヒートマップオーバーレイの透明度を設定")
        skeleton_alpha = st.slider("ゲイズプロット透明度", 0.1, 0.5, 0.3, 0.05,
                                  help="ゲイズプロットの円の透明度を設定")
        circle_radius = st.slider("注視点サイズ", 15, 35, 25, 1,
                                 help="ゲイズプロットの円のサイズを設定")
        show_duration_labels = st.checkbox("滞留時間を表示", True,
                                          help="各注視点の滞留時間を表示します")
        
        st.subheader("自動検出設定")
        use_auto_detection = st.checkbox("自動オブジェクト検出を使用", False)
        
        if use_auto_detection:
            detection_method = st.selectbox(
                "検出方法",
                ["YOLOv8", "SAM (Segment Anything)"]
            )
            confidence_threshold = st.slider("信頼度しきい値", 0.1, 1.0, 0.5, 0.05)
        
        st.subheader("デバッグ")
        st.session_state['show_debug'] = st.checkbox("デバッグ情報を表示", False)
    
    uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        analyzer = ROIDeepGazeAnalyzer()
        image = analyzer.resize_image(image, max_size=1024)
        
        st.image(image, caption="アップロード画像", use_container_width=True)
        
        # ROI設定オプション
        st.markdown('<div class="section-header">🎯 ROI設定（オプション）</div>', unsafe_allow_html=True)
        
        # ROI設定なしで解析可能なことを明示
        st.info("💡 ROIを設定せずに画像全体の視線解析も可能です。ROIを設定する場合は以下のキャンバスで矩形を描いてください。")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#ff7f0e",
                background_image=image,
                update_streamlit=True,
                height=image.height,
                width=image.width,
                drawing_mode="rect",
                key="canvas",
            )
        
        with col2:
            st.markdown("### ROI設定方法")
            st.markdown("📐 **手動設定**: 矩形を描いてROIを設定")
            st.markdown("🤖 **自動検出**: 下のボタンで自動検出")
            st.markdown("✨ **設定なし**: そのまま分析実行可能")
            
            if use_auto_detection:
                if st.button("🔍 自動検出実行"):
                    with st.spinner("検出中..."):
                        img_array = np.array(image)
                        
                        if detection_method == "YOLOv8":
                            detections = analyzer.auto_detector.detect_objects_yolo(
                                img_array, confidence_threshold
                            )
                        else:
                            detections = analyzer.auto_detector.detect_objects_sam_with_area(
                                img_array
                            )
                        
                        if detections:
                            st.success(f"{len(detections)}個のオブジェクトを検出しました")
                            st.session_state['auto_detections'] = detections
                        else:
                            st.warning("オブジェクトが検出されませんでした")
        
        # 分析実行ボタン（ROI設定に関わらず常に表示）
        if st.button("🚀 分析実行（滞留時間予測付き）", type="primary", use_container_width=True):
            with st.spinner("分析中..."):
                rois = []
                roi_labels = []
                
                # Canvas からの手動ROI
                if canvas_result.json_data:
                    manual_rois = analyzer.extract_rois_from_canvas(
                        canvas_result.json_data, image.width, image.height
                    )
                    for i, roi in enumerate(manual_rois):
                        roi['label'] = f"ROI {i+1}"
                        rois.append(roi)
                        roi_labels.append(f"ROI {i+1}")
                
                # 自動検出ROI
                if 'auto_detections' in st.session_state:
                    auto_rois = analyzer.detections_to_rois(
                        st.session_state['auto_detections'], 'auto'
                    )
                    start_idx = len(rois)
                    for i, roi in enumerate(auto_rois):
                        label = f"ROI {start_idx + i + 1}"
                        roi['label'] = label
                        rois.append(roi)
                        roi_labels.append(label)
                
                # ROIがなくても解析を実行
                if not rois:
                    st.markdown('<div class="no-roi-info">ℹ️ ROI設定なしで画像全体の視線解析を実行します</div>', 
                              unsafe_allow_html=True)
                
                # 解析実行（ROIがなくても実行）
                results = analyzer.analyze_with_duration(
                    image, [], rois, roi_labels, params
                )
                
                if results:
                    st.markdown('<div class="section-header">📊 分析結果</div>', 
                              unsafe_allow_html=True)
                    
                    # ROIの有無でタブ構成を変更
                    if rois:
                        tab1, tab2, tab3 = st.tabs(["🎨 メイン分析", "📈 詳細分析", "💾 データ"])
                    else:
                        tab1, tab2, tab3 = st.tabs(["🎨 視線分析", "📈 滞留時間分析", "💾 データ"])
                    
                    with tab1:
                        if rois:
                            st.markdown("### 📊 視線分析メインビュー（ROI設定あり）")
                        else:
                            st.markdown("### 📊 視線分析メインビュー（画像全体）")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔥 ヒートマップオーバーレイ")
                            
                            if rois:
                                show_roi_on_heatmap = st.checkbox("ROIを表示", value=True, key="show_roi_heatmap")
                            else:
                                show_roi_on_heatmap = False
                            
                            heatmap_overlay = create_heatmap_overlay_with_roi(
                                np.array(image), 
                                results['saliency_map'],
                                rois if show_roi_on_heatmap else None,
                                roi_labels if show_roi_on_heatmap else None,
                                alpha=heatmap_alpha,
                                show_roi=show_roi_on_heatmap
                            )
                            
                            if rois:
                                caption = "ヒートマップ + ROI（グレースケール + 熱分布）"
                            else:
                                caption = "ヒートマップ（グレースケール + 熱分布）"
                            
                            st.image(heatmap_overlay, caption=caption, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 👁️ ゲイズプロット")
                            
                            vis_image = visualize_scanpath_skeleton(
                                np.array(image), 
                                results['scanpath_data'],
                                circle_radius=circle_radius,
                                alpha=skeleton_alpha,
                                show_duration=show_duration_labels
                            )
                            st.image(vis_image, caption="スキャンパス（スケルトン表示）", 
                                    use_container_width=True)
                    
                    with tab2:
                        # ROIがある場合のみROI注目度分析を表示
                        if rois and not results['roi_attention'].empty:
                            st.markdown('<div class="subsection-header">🎯 ROI注目度分析</div>', 
                                      unsafe_allow_html=True)
                            st.dataframe(results['roi_attention'], use_container_width=True)
                            
                            # ROI別滞留時間統計も表示
                            if not results['roi_duration_stats'].empty:
                                st.markdown('<div class="subsection-header">⏱️ ROI別滞留時間統計</div>', 
                                          unsafe_allow_html=True)
                                st.dataframe(results['roi_duration_stats'], use_container_width=True)
                        
                        # 滞留時間分析（ROIの有無に関わらず表示）
                        display_duration_analysis(results)
                        
                        # 注視点詳細テーブル
                        if results['scanpath_data']:
                            st.markdown('<div class="subsection-header">🔍 注視点詳細</div>', 
                                      unsafe_allow_html=True)
                            fixation_table = create_fixation_duration_table(results['scanpath_data'])
                            st.dataframe(fixation_table, use_container_width=True)
                    
                    with tab3:
                        st.markdown('<div class="subsection-header">💾 データダウンロード</div>', 
                                  unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 CSV形式")
                            
                            # スキャンパスデータ（常に存在）
                            if results['scanpath_data']:
                                scanpath_df = pd.DataFrame(results['scanpath_data'])
                                csv = scanpath_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 スキャンパスデータ (CSV)",
                                    data=csv,
                                    file_name='scanpath_with_duration.csv',
                                    mime='text/csv'
                                )
                            
                            # ROI注目度データ（ROIがある場合のみ）
                            if rois and not results['roi_attention'].empty:
                                csv = results['roi_attention'].to_csv(index=False, 
                                                                     encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 ROI注目度データ (CSV)",
                                    data=csv,
                                    file_name='roi_attention_data.csv',
                                    mime='text/csv'
                                )
                            
                            # ROI滞留時間統計（ROIがある場合のみ）
                            if rois and not results['roi_duration_stats'].empty:
                                csv = results['roi_duration_stats'].to_csv(index=False, 
                                                                          encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 ROI滞留時間統計 (CSV)",
                                    data=csv,
                                    file_name='roi_duration_stats.csv',
                                    mime='text/csv'
                                )
                        
                        with col2:
                            st.markdown("#### 📋 JSON形式")
                            
                            # ゲイズプロット滞留時間（常に存在）
                            gaze_duration_json = []
                            if results['scanpath_data']:
                                for fixation in results['scanpath_data']:
                                    gaze_duration_json.append({
                                        "注視点番号": fixation['fixation_id'],
                                        "X座標": int(fixation['x']),
                                        "Y座標": int(fixation['y']),
                                        "滞留時間(ms)": round(fixation['duration_ms'], 1),
                                        "ROI": fixation.get('roi_label', 'Background')
                                    })
                                
                                gaze_duration_json_str = json.dumps(
                                    gaze_duration_json,
                                    ensure_ascii=False,
                                    indent=2
                                )
                                
                                st.download_button(
                                    label="📥 ゲイズプロット滞留時間 (JSON)",
                                    data=gaze_duration_json_str,
                                    file_name='gaze_fixation_duration.json',
                                    mime='application/json'
                                )
                            
                            # ROI注目度データ（ROIがある場合のみ）
                            roi_attention_json = []
                            if rois and not results['roi_attention'].empty:
                                for _, row in results['roi_attention'].iterrows():
                                    roi_attention_json.append({
                                        "ROI番号": int(row['ROI_ID'] + 1),
                                        "ROIラベル": row['Label'],
                                        "注目度(%)": round(row['Attention_Percentage'], 1),
                                        "座標": {
                                            "X": int(row['X']),
                                            "Y": int(row['Y']),
                                            "Width": int(row['Width']),
                                            "Height": int(row['Height'])
                                        }
                                    })
                                
                                roi_attention_json = sorted(roi_attention_json, 
                                                           key=lambda x: x['注目度(%)'], 
                                                           reverse=True)
                                
                                roi_attention_json_str = json.dumps(
                                    roi_attention_json, 
                                    ensure_ascii=False, 
                                    indent=2
                                )
                                
                                st.download_button(
                                    label="📥 ROI注目度データ (JSON)",
                                    data=roi_attention_json_str,
                                    file_name='roi_attention.json',
                                    mime='application/json'
                                )
                            
                            # 統合分析データ
                            integrated_json = {
                                "分析日時": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "総視認時間(ms)": round(results.get('total_viewing_time_ms', 0), 1),
                                "注視点数": len(results['scanpath_data']) if results['scanpath_data'] else 0,
                                "ROI設定": "あり" if rois else "なし",
                                "ROI数": len(rois) if rois else 0
                            }
                            
                            if results['scanpath_data']:
                                integrated_json["ゲイズプロット滞留時間"] = gaze_duration_json
                                integrated_json["統計サマリー"] = {
                                    "平均滞留時間(ms)": round(
                                        sum(f['滞留時間(ms)'] for f in gaze_duration_json) / len(gaze_duration_json), 
                                        1
                                    ) if gaze_duration_json else 0
                                }
                            
                            if rois and roi_attention_json:
                                integrated_json["ROI注目度"] = roi_attention_json
                                integrated_json["統計サマリー"]["最高注目度ROI"] = roi_attention_json[0] if roi_attention_json else None
                            
                            integrated_json_str = json.dumps(
                                integrated_json,
                                ensure_ascii=False,
                                indent=2
                            )
                            
                            st.download_button(
                                label="📥 統合分析データ (JSON)",
                                data=integrated_json_str,
                                file_name='integrated_analysis_data.json',
                                mime='application/json',
                                help="すべての分析結果を含む統合データ"
                            )
                        
                        st.markdown("---")
                        
                        st.markdown("#### 🖼️ 画像ファイル")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            overlay_pil = Image.fromarray(heatmap_overlay)
                            buf1 = io.BytesIO()
                            overlay_pil.save(buf1, format='PNG')
                            
                            st.download_button(
                                label="📥 ヒートマップ画像",
                                data=buf1.getvalue(),
                                file_name='heatmap_overlay.png',
                                mime='image/png'
                            )
                        
                        with col4:
                            gaze_pil = Image.fromarray(vis_image)
                            buf2 = io.BytesIO()
                            gaze_pil.save(buf2, format='PNG')
                            
                            st.download_button(
                                label="📥 ゲイズプロット画像",
                                data=buf2.getvalue(),
                                file_name='gaze_plot.png',
                                mime='image/png'
                            )
                        
                        with st.expander("📋 データプレビュー"):
                            if results['scanpath_data']:
                                st.markdown("##### ゲイズプロット滞留時間データ (最初の5件)")
                                st.json(gaze_duration_json[:5])
                            
                            if rois and roi_attention_json:
                                st.markdown("##### ROI注目度データ (上位5件)")
                                st.json(roi_attention_json[:5])
                else:
                    st.error("分析に失敗しました。パラメータを調整して再度お試しください。")
    else:
        # 画像がアップロードされていない場合の説明
        st.markdown("---")
        st.markdown("### 📖 使い方")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1️⃣ 画像をアップロード")
            st.markdown("PNG、JPG、JPEG形式の画像をアップロードしてください。")
        
        with col2:
            st.markdown("#### 2️⃣ ROI設定（オプション）")
            st.markdown("- 手動：キャンバスで矩形を描画")
            st.markdown("- 自動：YOLOv8/SAMで検出")
            st.markdown("- なし：画像全体を解析")
        
        with col3:
            st.markdown("#### 3️⃣ 分析実行")
            st.markdown("「分析実行」ボタンをクリックして、視線解析を開始します。")
        
        st.markdown("---")
        st.markdown("### 🎯 主な機能")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("#### 視線解析")
            st.markdown("- DeepGaze IIIによる高精度予測")
            st.markdown("- ヒートマップ生成（赤→黄→青）")
            st.markdown("- スキャンパス可視化")
            st.markdown("- 滞留時間予測")
        
        with feature_col2:
            st.markdown("#### データ出力")
            st.markdown("- CSV形式でのエクスポート")
            st.markdown("- JSON形式での統合データ")
            st.markdown("- 画像ファイルの保存")
            st.markdown("- 詳細な統計情報")
        
        st.markdown("---")
        st.info("💡 **ヒント**: サイドバーでパラメータを調整することで、より精密な分析が可能です。")

if __name__ == "__main__":
    main()