"""
DeepGaze III ボトムアップ処理モジュール

人間の視覚的注意の基盤となるDeepGaze IIIの特徴抽出と
脳の視覚処理階層へのマッピングを実装します。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import logging

# DeepGaze関連のインポート（実際の実装では適切なライブラリを使用）
try:
    import deepgaze_pytorch
except ImportError:
    logging.warning("DeepGaze PyTorch not found. Using mock implementation.")
    deepgaze_pytorch = None

logger = logging.getLogger(__name__)


class DeepGazeIII:
    """
    DeepGaze III モデルのラッパークラス
    """
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = self._load_model()
        self.neural_feature_extractor = NeuralFeatureExtractor()
        
    def _get_device(self, device: str) -> torch.device:
        """デバイスの自動選択"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """DeepGaze IIIモデルのロード"""
        if deepgaze_pytorch is not None:
            try:
                model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load DeepGaze III: {e}")
        
        # フォールバック: モックモデル
        logger.info("Using mock DeepGaze III model")
        return MockDeepGazeIII(self.device)
    
    def extract_features(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """階層的特徴抽出"""
        return self.model.extract_features(image)
    
    def predict_saliency(self, image: np.ndarray) -> np.ndarray:
        """顕著性予測"""
        return self.model.predict_saliency(image)
    
    def get_center_bias(self, shape: Tuple[int, int]) -> np.ndarray:
        """センターバイアスの計算"""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # ガウシアンセンターバイアス
        sigma_h, sigma_w = h * 0.3, w * 0.3
        center_bias = np.exp(-((y - center_y)**2 / (2 * sigma_h**2) + 
                             (x - center_x)**2 / (2 * sigma_w**2)))
        
        return center_bias / center_bias.max()
    
    def compute_temporal_priority(self, saliency_map: np.ndarray) -> np.ndarray:
        """時間的優先度の計算"""
        # 時間依存の顕著性調整
        temporal_decay = 0.95
        enhanced_saliency = saliency_map * temporal_decay
        
        # 新奇性に基づく増強
        novelty_boost = self._compute_novelty_map(saliency_map)
        
        return enhanced_saliency + 0.1 * novelty_boost
    
    def _compute_novelty_map(self, saliency_map: np.ndarray) -> np.ndarray:
        """新奇性マップの計算"""
        # エッジ検出による新奇性推定
        edges = cv2.Canny((saliency_map * 255).astype(np.uint8), 50, 150)
        novelty = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        
        return novelty.astype(np.float32) / 255.0


class NeuralFeatureExtractor:
    """
    脳の視覚野に対応する特徴抽出器
    """
    
    def __init__(self):
        self.gabor_filters = self._create_gabor_filters()
    
    def _create_gabor_filters(self):
        """ガボールフィルタの生成（V1/V2相当）"""
        filters = []
        for theta in np.arange(0, np.pi, np.pi/4):  # 4方向
            for freq in [0.1, 0.3, 0.5]:  # 3周波数
                kernel = cv2.getGaborKernel((21, 21), 3, theta, 
                                          2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kernel)
        return filters
    
    def extract_early_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """初期視覚特徴の抽出（V1/V2相当）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        features = {}
        
        # エッジ方向特徴
        gabor_responses = []
        for kernel in self.gabor_filters:
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(response)
        
        features['gabor_like_filters'] = np.array(gabor_responses)
        features['edge_orientation'] = np.std(gabor_responses, axis=0)
        
        return features
    
    def extract_intermediate_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """中間視覚特徴の抽出（V4相当）"""
        features = {}
        
        # 色特徴
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            features['color_distribution'] = np.histogram(hsv[:,:,0], bins=16)[0]
            features['saturation_map'] = hsv[:,:,1]
        
        # 形状特徴
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_map = np.zeros_like(gray)
        cv2.drawContours(shape_map, contours, -1, 255, 2)
        features['shape_boundaries'] = shape_map
        
        return features


class MockDeepGazeIII:
    """
    DeepGaze IIIのモック実装（開発・テスト用）
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.neural_extractor = NeuralFeatureExtractor()
        
    def extract_features(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """特徴抽出のモック実装"""
        h, w = image.shape[:2]
        
        # モック特徴の生成
        features = {
            'conv1_features': torch.randn(1, 64, h//2, w//2).to(self.device),
            'conv3_features': torch.randn(1, 256, h//4, w//4).to(self.device),
            'conv5_features': torch.randn(1, 512, h//8, w//8).to(self.device),
            'fc_features': torch.randn(1, 1024).to(self.device)
        }
        
        # 実際の特徴も組み込み
        early_features = self.neural_extractor.extract_early_features(image)
        intermediate_features = self.neural_extractor.extract_intermediate_features(image)
        
        features.update({
            'gabor_like_filters': torch.from_numpy(early_features['gabor_like_filters']).to(self.device),
            'mid_level_features': torch.from_numpy(intermediate_features['shape_boundaries']).unsqueeze(0).to(self.device),
        })
        
        return features
    
    def predict_saliency(self, image: np.ndarray) -> np.ndarray:
        """顕著性予測のモック実装"""
        h, w = image.shape[:2]
        
        # 簡単な特徴ベース顕著性
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # エッジベース顕著性
        edges = cv2.Canny(gray, 50, 150)
        edge_saliency = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        # 色ベース顕著性（カラー画像の場合）
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            color_saliency = np.std(lab, axis=2)
        else:
            color_saliency = np.zeros_like(gray, dtype=np.float32)
        
        # 統合顕著性
        combined_saliency = 0.6 * edge_saliency + 0.4 * color_saliency
        combined_saliency = cv2.GaussianBlur(combined_saliency, (21, 21), 0)
        
        # 正規化
        if combined_saliency.max() > 0:
            combined_saliency = combined_saliency / combined_saliency.max()
        
        return combined_saliency


class DeepGazeBottomUpProcessor:
    """
    DeepGaze IIIの特徴抽出を脳の視覚処理階層にマッピングする処理クラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deepgaze = DeepGazeIII(config.get('device', 'auto'))
        self.neural_extractor = NeuralFeatureExtractor()
        
        # パラメータ設定
        self.center_bias_enabled = config.get('center_bias_enabled', True)
        self.temporal_dynamics_enabled = config.get('temporal_dynamics_enabled', True)
        self.gaussian_sigma = config.get('gaussian_sigma', 16)
        
        logger.info("DeepGaze BottomUp Processor initialized")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        DeepGaze IIIによるボトムアップ処理
        
        Args:
            image: 入力画像 (H, W, C) または (H, W)
            
        Returns:
            処理結果辞書
        """
        try:
            # 前処理
            processed_image = self._preprocess_image(image)
            
            # DeepGaze IIIの階層的特徴抽出
            deepgaze_features = self.deepgaze.extract_features(processed_image)
            
            # 脳の視覚野に対応する特徴マッピング
            neural_mapping = self._map_to_visual_areas(deepgaze_features, processed_image)
            
            # DeepGaze IIIの顕著性予測
            saliency_prediction = self.deepgaze.predict_saliency(processed_image)
            
            # センターバイアスの計算
            center_bias = None
            if self.center_bias_enabled:
                center_bias = self.deepgaze.get_center_bias(saliency_prediction.shape)
            
            # 時間的ダイナミクスの考慮
            temporal_saliency = None
            if self.temporal_dynamics_enabled:
                temporal_saliency = self.deepgaze.compute_temporal_priority(saliency_prediction)
            
            return {
                'features': neural_mapping,
                'raw_saliency': saliency_prediction,
                'temporal_saliency': temporal_saliency,
                'center_bias': center_bias,
                'metadata': {
                    'image_shape': processed_image.shape,
                    'processing_time': 0,  # 実際の実装では時間測定
                    'model_version': 'deepgaze_III'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in DeepGaze processing: {e}")
            return self._create_fallback_result(image.shape)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """画像の前処理"""
        # サイズ制限
        max_size = self.config.get('max_image_size', [1920, 1080])
        h, w = image.shape[:2]
        
        if h > max_size[1] or w > max_size[0]:
            scale = min(max_size[0]/w, max_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 正規化
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def _map_to_visual_areas(self, features: Dict[str, torch.Tensor], 
                           image: np.ndarray) -> Dict[str, np.ndarray]:
        """脳の視覚野への特徴マッピング"""
        mapping = {}
        
        # V1/V2相当: 初期層の特徴
        early_features = self.neural_extractor.extract_early_features(image)
        mapping['early_visual'] = early_features['gabor_like_filters']
        mapping['edge_orientation'] = early_features['edge_orientation']
        
        # V4相当: 中間層の特徴  
        intermediate_features = self.neural_extractor.extract_intermediate_features(image)
        mapping['intermediate'] = intermediate_features['shape_boundaries']
        mapping['color_shape'] = intermediate_features.get('saturation_map', 
                                                         np.zeros_like(image[:,:,0]))
        
        # IT/FFA相当: 高次特徴
        if 'conv5_features' in features:
            # テンソルをNumPy配列に変換
            high_level_tensor = features['conv5_features'].detach().cpu().numpy()
            mapping['high_level'] = high_level_tensor[0]  # バッチ次元削除
        
        if 'fc_features' in features:
            semantic_tensor = features['fc_features'].detach().cpu().numpy()
            mapping['semantic_features'] = semantic_tensor[0]
        
        return mapping
    
    def _create_fallback_result(self, image_shape: Tuple) -> Dict[str, Any]:
        """エラー時のフォールバック結果"""
        h, w = image_shape[:2]
        
        return {
            'features': {
                'early_visual': np.zeros((4, h, w)),
                'edge_orientation': np.zeros((h, w)),
                'intermediate': np.zeros((h, w)),
                'color_shape': np.zeros((h, w)),
                'high_level': np.zeros((512, h//8, w//8)),
                'semantic_features': np.zeros(1024)
            },
            'raw_saliency': np.ones((h, w)) * 0.5,
            'temporal_saliency': np.ones((h, w)) * 0.5,
            'center_bias': np.ones((h, w)) * 0.5,
            'metadata': {
                'image_shape': image_shape,
                'processing_time': 0,
                'model_version': 'fallback',
                'error': True
            }
        }