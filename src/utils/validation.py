"""
入力検証ユーティリティ

画像データや設定パラメータの検証機能を提供します。
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """検証エラー"""
    pass


class InputValidator:
    """
    入力データの検証クラス
    """
    
    def __init__(self):
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.min_image_size = (32, 32)
        self.max_image_dimensions = (8192, 8192)
        
    def validate_image(self, image: np.ndarray) -> bool:
        """
        画像データの検証
        
        Args:
            image: 検証対象の画像
            
        Returns:
            検証結果
            
        Raises:
            ValidationError: 検証に失敗した場合
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError("Image must be a numpy array")
        
        # 次元チェック
        if len(image.shape) not in [2, 3]:
            raise ValidationError(f"Image must be 2D or 3D array, got {len(image.shape)}D")
        
        # サイズチェック
        h, w = image.shape[:2]
        
        if h < self.min_image_size[0] or w < self.min_image_size[1]:
            raise ValidationError(
                f"Image too small: {(w, h)}, minimum: {self.min_image_size}"
            )
        
        if h > self.max_image_dimensions[0] or w > self.max_image_dimensions[1]:
            raise ValidationError(
                f"Image too large: {(w, h)}, maximum: {self.max_image_dimensions}"
            )
        
        # チャンネル数チェック
        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels not in [1, 3, 4]:
                raise ValidationError(f"Invalid channel count: {channels}")
        
        # データタイプチェック
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            logger.warning(f"Unusual image dtype: {image.dtype}")
        
        # 値範囲チェック
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                raise ValidationError("uint8 image values out of range [0, 255]")
        elif image.dtype in [np.float32, np.float64]:
            if image.min() < -1.0 or image.max() > 255.0:
                logger.warning("Float image values outside typical ranges")
        
        # メモリサイズチェック
        memory_size = image.nbytes
        if memory_size > self.max_image_size:
            raise ValidationError(
                f"Image too large: {memory_size} bytes, max: {self.max_image_size}"
            )
        
        logger.debug(f"Image validation passed: {image.shape}, {image.dtype}")
        return True
    
    def validate_saliency_map(self, saliency_map: np.ndarray) -> bool:
        """
        顕著性マップの検証
        
        Args:
            saliency_map: 検証対象の顕著性マップ
            
        Returns:
            検証結果
        """
        if not isinstance(saliency_map, np.ndarray):
            raise ValidationError("Saliency map must be a numpy array")
        
        # 2次元チェック
        if len(saliency_map.shape) != 2:
            raise ValidationError("Saliency map must be 2D array")
        
        # 値範囲チェック（0-1の範囲を推奨）
        if saliency_map.min() < -0.1 or saliency_map.max() > 1.1:
            logger.warning(
                f"Saliency values outside [0, 1]: [{saliency_map.min():.3f}, {saliency_map.max():.3f}]"
            )
        
        # NaN/Inf チェック
        if np.any(np.isnan(saliency_map)):
            raise ValidationError("Saliency map contains NaN values")
        
        if np.any(np.isinf(saliency_map)):
            raise ValidationError("Saliency map contains infinite values")
        
        return True
    
    def validate_fixation_sequence(self, fixations: List[Dict[str, Any]]) -> bool:
        """
        注視点シーケンスの検証
        
        Args:
            fixations: 注視点のリスト
            
        Returns:
            検証結果
        """
        if not isinstance(fixations, list):
            raise ValidationError("Fixation sequence must be a list")
        
        if len(fixations) == 0:
            logger.warning("Empty fixation sequence")
            return True
        
        required_fields = ['location', 'duration', 'timestamp']
        
        for i, fixation in enumerate(fixations):
            if not isinstance(fixation, dict):
                raise ValidationError(f"Fixation {i} must be a dictionary")
            
            # 必須フィールドチェック
            for field in required_fields:
                if field not in fixation:
                    raise ValidationError(f"Fixation {i} missing required field: {field}")
            
            # 位置の検証
            location = fixation['location']
            if not isinstance(location, (list, tuple)) or len(location) != 2:
                raise ValidationError(f"Fixation {i} location must be [x, y] coordinates")
            
            x, y = location
            if not (0 <= x <= 1 and 0 <= y <= 1):
                logger.warning(f"Fixation {i} location outside [0, 1]: ({x}, {y})")
            
            # 時間の検証
            duration = fixation['duration']
            timestamp = fixation['timestamp']
            
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValidationError(f"Fixation {i} duration must be non-negative number")
            
            if not isinstance(timestamp, (int, float)) or timestamp < 0:
                raise ValidationError(f"Fixation {i} timestamp must be non-negative number")
        
        # 時系列チェック
        timestamps = [fix['timestamp'] for fix in fixations]
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            logger.warning("Fixation timestamps not in ascending order")
        
        return True
    
    def validate_config(self, config: Dict[str, Any], 
                       schema: Dict[str, Any]) -> bool:
        """
        設定の検証
        
        Args:
            config: 検証対象の設定
            schema: 検証スキーマ
            
        Returns:
            検証結果
        """
        return self._validate_dict_recursive(config, schema, "config")
    
    def _validate_dict_recursive(self, data: Dict[str, Any], 
                               schema: Dict[str, Any], 
                               path: str = "") -> bool:
        """再帰的辞書検証"""
        for key, schema_value in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in data:
                if schema_value.get('required', False):
                    raise ValidationError(f"Missing required field: {current_path}")
                continue
            
            value = data[key]
            
            # データタイプチェック
            expected_type = schema_value.get('type')
            if expected_type and not isinstance(value, expected_type):
                raise ValidationError(
                    f"Invalid type for {current_path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            # 値範囲チェック
            if 'min' in schema_value and value < schema_value['min']:
                raise ValidationError(
                    f"Value too small for {current_path}: {value} < {schema_value['min']}"
                )
            
            if 'max' in schema_value and value > schema_value['max']:
                raise ValidationError(
                    f"Value too large for {current_path}: {value} > {schema_value['max']}"
                )
            
            # 選択肢チェック
            if 'choices' in schema_value and value not in schema_value['choices']:
                raise ValidationError(
                    f"Invalid choice for {current_path}: {value} not in {schema_value['choices']}"
                )
            
            # ネストした辞書の検証
            if isinstance(value, dict) and 'properties' in schema_value:
                self._validate_dict_recursive(
                    value, schema_value['properties'], current_path
                )
        
        return True
    
    def validate_analysis_config(self, config: Dict[str, Any]) -> bool:
        """
        分析設定の検証
        """
        schema = {
            'use_deepgaze': {'type': bool, 'required': False},
            'use_sam2': {'type': bool, 'required': False},
            'use_llm': {'type': bool, 'required': False},
            'output_resolution': {
                'type': str, 
                'choices': ['high', 'medium', 'low'],
                'required': False
            },
            'generate_scanpath': {'type': bool, 'required': False},
            'preprocessing': {
                'type': dict,
                'required': False,
                'properties': {
                    'resize': {'type': bool},
                    'normalize': {'type': bool},
                    'denoise': {'type': bool}
                }
            }
        }
        
        return self.validate_config(config, schema)
    
    def validate_user_context(self, context: Dict[str, Any]) -> bool:
        """
        ユーザーコンテキストの検証
        """
        schema = {
            'task': {
                'type': str,
                'choices': ['free_viewing', 'face_detection', 'text_reading', 
                           'navigation', 'visual_search'],
                'required': False
            },
            'expertise_level': {
                'type': str,
                'choices': ['novice', 'intermediate', 'expert', 'general'],
                'required': False
            },
            'language': {'type': str, 'required': False},
            'viewing_duration_ms': {
                'type': int,
                'min': 100,
                'max': 60000,
                'required': False
            }
        }
        
        return self.validate_config(context, schema)
    
    def sanitize_filename(self, filename: str) -> str:
        """
        ファイル名のサニタイズ
        """
        import re
        
        # 危険な文字を除去
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # 制御文字を除去
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        # 長さ制限
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized
    
    def validate_api_request(self, request_data: Dict[str, Any]) -> bool:
        """
        API リクエストの検証
        """
        # 必須フィールド
        if 'image' not in request_data:
            raise ValidationError("Missing required field: image")
        
        # 画像データの検証（Base64エンコード文字列の場合）
        image_data = request_data['image']
        if isinstance(image_data, str):
            try:
                import base64
                # Base64デコードテスト
                decoded = base64.b64decode(image_data)
                if len(decoded) == 0:
                    raise ValidationError("Empty image data")
            except Exception as e:
                raise ValidationError(f"Invalid base64 image data: {e}")
        
        # オプションフィールドの検証
        if 'analysis_config' in request_data:
            self.validate_analysis_config(request_data['analysis_config'])
        
        if 'user_context' in request_data:
            self.validate_user_context(request_data['user_context'])
        
        return True