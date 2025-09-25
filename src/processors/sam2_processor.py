"""
SAM2 物体ベース注意メカニズム

SAM2を使用した精密な物体セグメンテーションと
物体単位の視覚的注意制御を実装します。
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import cv2
from PIL import Image
import logging
from dataclasses import dataclass

# SAM2関連のインポート（実際の実装では適切なライブラリを使用）
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    logging.warning("Segment Anything not found. Using mock implementation.")
    sam_model_registry = None
    SamAutomaticMaskGenerator = None

logger = logging.getLogger(__name__)


@dataclass
class ObjectSegment:
    """セグメント情報を格納するデータクラス"""
    mask: np.ndarray
    area: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    eccentricity: float
    contrast: float
    saliency_score: float
    object_class: Optional[str] = None
    confidence: float = 1.0


class SAM2Model:
    """
    SAM2 (Segment Anything Model 2) のラッパークラス
    """
    
    def __init__(self, model_checkpoint: str, model_cfg: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = self._load_model(model_checkpoint, model_cfg)
        self.mask_generator = self._setup_mask_generator()
        
    def _get_device(self, device: str) -> torch.device:
        """デバイスの自動選択"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self, model_checkpoint: str, model_cfg: str):
        """SAM2モデルのロード"""
        if sam_model_registry is not None:
            try:
                # 実際のSAM2実装
                model_type = "vit_h"  # または設定から取得
                sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
                sam.to(device=self.device)
                return sam
            except Exception as e:
                logger.warning(f"Failed to load SAM2: {e}")
        
        # フォールバック: モックモデル
        logger.info("Using mock SAM2 model")
        return MockSAM2(self.device)
    
    def _setup_mask_generator(self):
        """自動マスク生成器の設定"""
        if SamAutomaticMaskGenerator is not None and hasattr(self.model, 'generate'):
            return SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=0,
            )
        else:
            return MockMaskGenerator()
    
    def segment_image(self, image: np.ndarray) -> List[Dict]:
        """画像の自動セグメンテーション"""
        try:
            masks = self.mask_generator.generate(image)
            return masks
        except Exception as e:
            logger.error(f"Error in SAM2 segmentation: {e}")
            return []


class MockSAM2:
    """SAM2のモック実装（開発・テスト用）"""
    
    def __init__(self, device):
        self.device = device


class MockMaskGenerator:
    """マスク生成器のモック実装"""
    
    def generate(self, image: np.ndarray) -> List[Dict]:
        """簡易セグメンテーション"""
        h, w = image.shape[:2]
        masks = []
        
        # 簡単な閾値ベースセグメンテーション
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 複数の閾値でセグメント生成
        thresholds = [64, 128, 192]
        for i, thresh in enumerate(thresholds):
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, contour in enumerate(contours[:5]):  # 最大5個まで
                if cv2.contourArea(contour) < 100:  # 小さすぎるものは除外
                    continue
                    
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                x, y, w_bbox, h_bbox = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                masks.append({
                    'segmentation': mask > 0,
                    'area': area,
                    'bbox': [x, y, w_bbox, h_bbox],
                    'predicted_iou': 0.8 + 0.1 * np.random.random(),
                    'stability_score': 0.9 + 0.05 * np.random.random(),
                })
        
        return masks


class ObjectGeometryAnalyzer:
    """物体の幾何学的特性を分析するクラス"""
    
    @staticmethod
    def compute_eccentricity(mask: np.ndarray) -> float:
        """物体の離心率を計算"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # 最大の輪郭を使用
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) >= 5:  # 楕円フィッティングには最低5点必要
            ellipse = cv2.fitEllipse(largest_contour)
            (_, _), (ma, MA), _ = ellipse
            
            if MA > 0:
                eccentricity = np.sqrt(1 - (min(ma, MA) / max(ma, MA))**2)
                return eccentricity
        
        return 0.0
    
    @staticmethod
    def compute_contrast(image: np.ndarray, mask: np.ndarray) -> float:
        """物体と背景のコントラストを計算"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # マスク領域の平均輝度
        object_pixels = gray[mask > 0]
        if len(object_pixels) == 0:
            return 0.0
        object_mean = np.mean(object_pixels)
        
        # 背景領域の平均輝度
        background_pixels = gray[mask == 0]
        if len(background_pixels) == 0:
            return 0.0
        background_mean = np.mean(background_pixels)
        
        # マイケルソンコントラスト
        contrast = abs(object_mean - background_mean) / (object_mean + background_mean + 1e-8)
        return contrast
    
    @staticmethod
    def compute_spatial_frequency(image: np.ndarray, mask: np.ndarray) -> float:
        """物体領域の空間周波数を計算"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # マスク領域にフィルタを適用
        masked_region = gray * (mask > 0)
        
        # ソーベルフィルタで勾配計算
        grad_x = cv2.Sobel(masked_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # 勾配強度
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # マスク領域での平均勾配
        mask_area = np.sum(mask > 0)
        if mask_area > 0:
            spatial_freq = np.sum(gradient_magnitude * (mask > 0)) / mask_area
            return spatial_freq
        
        return 0.0


class ObjectImportanceCalculator:
    """物体の重要度を計算するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.area_weight = config.get('area_weight', 0.3)
        self.contrast_weight = config.get('contrast_weight', 0.4)
        self.eccentricity_weight = config.get('eccentricity_weight', 0.3)
        
        self.geometry_analyzer = ObjectGeometryAnalyzer()
    
    def compute_importance(self, segment: ObjectSegment, image: np.ndarray, 
                         image_area: int) -> float:
        """物体の重要度スコアを計算"""
        
        # 面積に基づく重要度 (相対的サイズ)
        relative_area = segment.area / image_area
        area_score = min(relative_area * 10, 1.0)  # 10%以上で最大スコア
        
        # コントラストに基づる重要度
        contrast_score = min(segment.contrast * 2, 1.0)  # 0.5以上で最大スコア
        
        # 形状の複雑さに基づく重要度 (離心率の逆数)
        eccentricity_score = 1.0 - segment.eccentricity
        
        # 重み付き合計
        importance = (
            self.area_weight * area_score +
            self.contrast_weight * contrast_score +
            self.eccentricity_weight * eccentricity_score
        )
        
        return importance
    
    def compute_spatial_bias(self, bbox: Tuple[int, int, int, int], 
                           image_shape: Tuple[int, int]) -> float:
        """空間的位置による重要度バイアス"""
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        
        # 物体の中心
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 画像中心からの距離
        img_center_x = img_w // 2
        img_center_y = img_h // 2
        
        distance_from_center = np.sqrt((center_x - img_center_x)**2 + 
                                     (center_y - img_center_y)**2)
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        
        # 中心に近いほど高スコア
        spatial_bias = 1.0 - (distance_from_center / max_distance)
        
        return spatial_bias


class ObjectBasedAttention:
    """
    SAM2を使用した物体単位の注意メカニズム
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sam2 = SAM2Model(
            config.get('model_checkpoint', 'sam2_hiera_large.pt'),
            config.get('model_cfg', 'sam2_hiera_l.yaml'),
            config.get('device', 'auto')
        )
        
        self.geometry_analyzer = ObjectGeometryAnalyzer()
        self.importance_calculator = ObjectImportanceCalculator(
            config.get('object_importance', {})
        )
        
        logger.info("Object-based Attention initialized with SAM2")
    
    def enhance_with_segmentation(self, image: np.ndarray, 
                                deepgaze_saliency: np.ndarray) -> np.ndarray:
        """
        SAM2セグメンテーションによる顕著性の強化
        
        Args:
            image: 入力画像 (H, W, C)
            deepgaze_saliency: DeepGazeによる顕著性マップ (H, W)
            
        Returns:
            物体ベース顕著性マップ (H, W)
        """
        try:
            # SAM2によるセグメンテーション
            raw_segments = self.sam2.segment_image(image)
            
            # セグメント情報の構造化
            segments = self._process_segments(raw_segments, image)
            
            # セグメントごとの顕著性統合
            segment_saliencies = self._compute_segment_saliencies(
                segments, deepgaze_saliency, image
            )
            
            # 物体ベース顕著性マップの生成
            object_saliency_map = self._create_object_saliency_map(
                segment_saliencies, image.shape[:2]
            )
            
            return object_saliency_map
            
        except Exception as e:
            logger.error(f"Error in object-based attention: {e}")
            return deepgaze_saliency.copy()  # フォールバック
    
    def _process_segments(self, raw_segments: List[Dict], 
                         image: np.ndarray) -> List[ObjectSegment]:
        """生セグメントを構造化データに変換"""
        segments = []
        
        for seg_data in raw_segments:
            mask = seg_data['segmentation'].astype(np.uint8)
            area = seg_data['area']
            bbox = tuple(seg_data['bbox'])
            
            # 幾何学的特性の計算
            eccentricity = self.geometry_analyzer.compute_eccentricity(mask)
            contrast = self.geometry_analyzer.compute_contrast(image, mask)
            
            segment = ObjectSegment(
                mask=mask,
                area=area,
                bbox=bbox,
                eccentricity=eccentricity,
                contrast=contrast,
                saliency_score=0.0,  # 後で計算
                confidence=seg_data.get('stability_score', 0.9)
            )
            
            segments.append(segment)
        
        return segments
    
    def _compute_segment_saliencies(self, segments: List[ObjectSegment],
                                   deepgaze_saliency: np.ndarray,
                                   image: np.ndarray) -> List[ObjectSegment]:
        """各セグメントの顕著性スコアを計算"""
        image_area = image.shape[0] * image.shape[1]
        
        for segment in segments:
            # DeepGaze予測値をセグメント内で統合
            segment_saliency = self._aggregate_saliency_in_segment(
                deepgaze_saliency, segment.mask
            )
            
            # 物体の重要度による重み付け
            object_importance = self.importance_calculator.compute_importance(
                segment, image, image_area
            )
            
            # 空間的バイアス
            spatial_bias = self.importance_calculator.compute_spatial_bias(
                segment.bbox, image.shape
            )
            
            # 統合スコア
            segment.saliency_score = (
                segment_saliency * object_importance * spatial_bias * segment.confidence
            )
        
        return segments
    
    def _aggregate_saliency_in_segment(self, saliency_map: np.ndarray, 
                                     mask: np.ndarray) -> float:
        """セグメント内の顕著性を集計"""
        if np.sum(mask) == 0:
            return 0.0
        
        # マスク領域の顕著性統計
        masked_saliency = saliency_map * mask
        
        # 複数の統計を組み合わせ
        mean_saliency = np.sum(masked_saliency) / np.sum(mask)
        max_saliency = np.max(masked_saliency)
        
        # 重み付き組み合わせ
        aggregated_saliency = 0.7 * mean_saliency + 0.3 * max_saliency
        
        return aggregated_saliency
    
    def _create_object_saliency_map(self, segments: List[ObjectSegment],
                                   shape: Tuple[int, int]) -> np.ndarray:
        """物体ベース顕著性マップの生成"""
        h, w = shape
        saliency_map = np.zeros((h, w), dtype=np.float32)
        
        # 各セグメントの顕著性をマップに描画
        for segment in segments:
            mask_area = segment.mask.astype(np.float32)
            saliency_map += mask_area * segment.saliency_score
        
        # 正規化
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        
        # ガウシアンスムージング
        saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
        
        return saliency_map
    
    def get_segment_information(self, image: np.ndarray) -> Dict[str, Any]:
        """セグメント情報の詳細分析"""
        try:
            raw_segments = self.sam2.segment_image(image)
            segments = self._process_segments(raw_segments, image)
            
            # 統計情報の計算
            num_segments = len(segments)
            total_area = sum(seg.area for seg in segments)
            avg_contrast = np.mean([seg.contrast for seg in segments]) if segments else 0
            
            # セグメント品質評価
            quality_scores = [seg.confidence for seg in segments]
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            return {
                'num_segments': num_segments,
                'total_segmented_area': total_area,
                'average_contrast': avg_contrast,
                'average_quality': avg_quality,
                'segments': segments,
                'segmentation_success': num_segments > 0
            }
            
        except Exception as e:
            logger.error(f"Error in segment analysis: {e}")
            return {
                'num_segments': 0,
                'total_segmented_area': 0,
                'average_contrast': 0,
                'average_quality': 0,
                'segments': [],
                'segmentation_success': False,
                'error': str(e)
            }