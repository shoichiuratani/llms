"""
画像処理ユーティリティ

画像の前処理、変換、エンコードなどの機能を提供します。
"""

import numpy as np
import cv2
from PIL import Image
import base64
import io
from typing import Dict, Any, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    画像前処理クラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get('max_image_size', [1920, 1080])
        self.target_size = config.get('target_size', None)
        
    def process(self, image: np.ndarray, 
                processing_config: Dict[str, Any] = None) -> np.ndarray:
        """
        画像の前処理
        
        Args:
            image: 入力画像
            processing_config: 処理設定
            
        Returns:
            前処理済み画像
        """
        config = processing_config or {}
        processed_image = image.copy()
        
        # サイズ調整
        if config.get('resize', True):
            processed_image = self.resize_image(processed_image)
        
        # 色空間変換
        color_space = config.get('color_space', 'BGR')
        if color_space != 'BGR':
            processed_image = self.convert_color_space(processed_image, color_space)
        
        # 正規化
        if config.get('normalize', True):
            processed_image = self.normalize_image(processed_image)
        
        # ノイズ除去
        if config.get('denoise', False):
            processed_image = self.denoise_image(processed_image)
        
        return processed_image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """画像のリサイズ"""
        h, w = image.shape[:2]
        max_w, max_h = self.max_size
        
        if h <= max_h and w <= max_w:
            return image
        
        # アスペクト比を保持したリサイズ
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        logger.debug(f"Image resized from ({w}, {h}) to ({new_w}, new_h)")
        return resized
    
    def convert_color_space(self, image: np.ndarray, target_space: str) -> np.ndarray:
        """色空間変換"""
        if len(image.shape) != 3:
            return image
        
        if target_space.upper() == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target_space.upper() == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_space.upper() == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_space.upper() == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """画像の正規化"""
        if image.dtype == np.uint8:
            return image
        
        # 0-255の範囲に正規化
        if image.max() <= 1.0:
            normalized = (image * 255).astype(np.uint8)
        else:
            normalized = np.clip(image, 0, 255).astype(np.uint8)
        
        return normalized
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """ノイズ除去"""
        if len(image.shape) == 3:
            # カラー画像の場合
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # グレースケール画像の場合
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def encode_array_to_base64(self, array: np.ndarray, 
                              format: str = 'PNG') -> str:
        """
        NumPy配列をBase64エンコード
        
        Args:
            array: 入力配列
            format: 画像フォーマット
            
        Returns:
            Base64エンコード文字列
        """
        try:
            # 正規化（0-255範囲）
            if array.max() <= 1.0:
                normalized = (array * 255).astype(np.uint8)
            else:
                normalized = np.clip(array, 0, 255).astype(np.uint8)
            
            # グレースケールの場合はカラーマップ適用
            if len(normalized.shape) == 2:
                # ヒートマップとして可視化
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                pil_image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
            else:
                if normalized.shape[2] == 3:
                    # BGR to RGB
                    rgb_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                else:
                    pil_image = Image.fromarray(normalized)
            
            # Base64エンコード
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error encoding array to base64: {e}")
            # フォールバック: 空の画像
            fallback_img = Image.new('RGB', (100, 100), color='gray')
            buffer = io.BytesIO()
            fallback_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
    
    def decode_base64_to_array(self, b64_string: str) -> np.ndarray:
        """
        Base64文字列をNumPy配列に変換
        
        Args:
            b64_string: Base64エンコード文字列
            
        Returns:
            NumPy配列
        """
        try:
            # Base64デコード
            img_data = base64.b64decode(b64_string)
            
            # PIL Imageに変換
            pil_image = Image.open(io.BytesIO(img_data))
            
            # NumPy配列に変換
            image_array = np.array(pil_image)
            
            # BGRに変換（OpenCV形式）
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error decoding base64 to array: {e}")
            # フォールバック: ダミー画像
            return np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    def create_thumbnail(self, image: np.ndarray, 
                        size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """サムネイル画像の生成"""
        h, w = image.shape[:2]
        th, tw = size
        
        # アスペクト比を保持したリサイズ
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # パディングで目標サイズに調整
        pad_w = (tw - new_w) // 2
        pad_h = (th - new_h) // 2
        
        if len(image.shape) == 3:
            padded = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        else:
            padded = np.zeros((th, tw), dtype=image.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    
    def apply_gaussian_blur(self, image: np.ndarray, 
                           sigma: float = 1.0) -> np.ndarray:
        """ガウシアンブラーの適用"""
        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def enhance_contrast(self, image: np.ndarray, 
                        alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """コントラスト強化"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """ヒストグラム平坦化"""
        if len(image.shape) == 3:
            # カラー画像の場合（YUV色空間で処理）
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # グレースケール画像の場合
            equalized = cv2.equalizeHist(image)
        
        return equalized
    
    def get_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """画像統計の取得"""
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std())
        }
        
        if len(image.shape) == 3:
            stats['channels'] = image.shape[2]
            for i, channel in enumerate(['B', 'G', 'R']):
                channel_data = image[:, :, i]
                stats[f'channel_{channel}'] = {
                    'min': float(channel_data.min()),
                    'max': float(channel_data.max()),
                    'mean': float(channel_data.mean()),
                    'std': float(channel_data.std())
                }
        
        return stats