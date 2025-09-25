"""
拡張注視点シーケンス生成器

DeepGaze IIIの予測を基に人間らしい注視パターンを生成します。
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class FixationPoint:
    """注視点情報"""
    location: Tuple[float, float]  # (x, y) 正規化座標
    duration: float  # ミリ秒
    timestamp: float  # ミリ秒
    information_gain: float
    confidence: float
    saliency_value: float


class InhibitionOfReturnManager:
    """Inhibition of Return (IOR) 管理クラス"""
    
    def __init__(self, decay_rate: float = 0.02, sigma: float = 50):
        self.decay_rate = decay_rate
        self.sigma = sigma
        self.ior_map = None
    
    def initialize(self, shape: Tuple[int, int]):
        """IORマップの初期化"""
        self.ior_map = np.zeros(shape, dtype=np.float32)
    
    def update(self, location: Tuple[float, float], shape: Tuple[int, int]):
        """IORマップの更新"""
        if self.ior_map is None:
            self.initialize(shape)
        
        h, w = shape
        x, y = int(location[0] * w), int(location[1] * h)
        
        # ガウシアン形状でIOR強度を追加
        y_grid, x_grid = np.ogrid[:h, :w]
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * self.sigma**2))
        
        # IOR強度を追加
        self.ior_map += gaussian * 0.5
        
        # 全体的な減衰
        self.ior_map *= (1 - self.decay_rate)
        
        # クリップ
        self.ior_map = np.clip(self.ior_map, 0, 1)
    
    def get_map(self) -> np.ndarray:
        """現在のIORマップを取得"""
        return self.ior_map if self.ior_map is not None else np.zeros((100, 100))


class FixationDurationCalculator:
    """注視時間計算クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_duration = config.get('min_fixation_duration_ms', 100)
        self.max_duration = config.get('max_fixation_duration_ms', 800)
        self.base_duration = config.get('base_duration_ms', 250)
    
    def calculate_duration(self, information_content: float, 
                         saliency_value: float,
                         eccentricity: float = 0.0) -> float:
        """
        情報理論的アプローチに基づく注視時間計算
        
        Args:
            information_content: 情報量
            saliency_value: 顕著性値
            eccentricity: 中心からの離心率
            
        Returns:
            注視時間（ミリ秒）
        """
        # 基本時間
        duration = self.base_duration
        
        # 情報量による調整（多いほど長く）
        info_factor = 1.0 + information_content * 0.8
        duration *= info_factor
        
        # 顕著性による調整（高いほど長く）
        saliency_factor = 0.8 + saliency_value * 0.4
        duration *= saliency_factor
        
        # 周辺視による調整（中心から遠いほど短く）
        eccentricity_factor = 1.0 - eccentricity * 0.3
        duration *= max(0.5, eccentricity_factor)\n        \n        # ランダム性の追加（個人差のシミュレーション）\n        noise_factor = np.random.normal(1.0, 0.15)\n        duration *= max(0.5, min(2.0, noise_factor))\n        \n        # 範囲内にクリップ\n        duration = max(self.min_duration, min(self.max_duration, duration))\n        \n        return duration


class SaccadeTargetSelector:
    \"\"\"サッカード目標選択クラス\"\"\"\n    \n    def __init__(self, config: Dict[str, Any]):\n        self.temperature = config.get('temperature', 0.8)\n        self.top_k = config.get('top_k_candidates', 5)\n        self.max_saccade_distance = config.get('max_saccade_distance', 200)\n    \n    def select_next_target(self, saliency_map: np.ndarray,\n                          current_location: Tuple[float, float],\n                          ior_map: np.ndarray) -> Tuple[Tuple[float, float], float]:\n        \"\"\"\n        次のサッカード目標の確率的選択\n        \n        Args:\n            saliency_map: 顕著性マップ\n            current_location: 現在位置（正規化座標）\n            ior_map: IORマップ\n            \n        Returns:\n            (次の位置, 顕著性値)\n        \"\"\"\n        h, w = saliency_map.shape\n        \n        # IOR適用\n        adjusted_saliency = saliency_map * (1 - ior_map)\n        \n        # 距離制約の適用\n        distance_mask = self._create_distance_mask(\n            current_location, (h, w), self.max_saccade_distance\n        )\n        adjusted_saliency *= distance_mask\n        \n        # 候補点の抽出\n        candidates = self._extract_candidates(adjusted_saliency)\n        \n        if not candidates:\n            # フォールバック: ランダム選択\n            return self._random_selection(saliency_map.shape), 0.5\n        \n        # 確率的選択\n        selected_idx = self._stochastic_selection(candidates)\n        y, x = candidates[selected_idx]\n        \n        # 正規化座標に変換\n        norm_x, norm_y = x / w, y / h\n        saliency_value = saliency_map[y, x]\n        \n        return (norm_x, norm_y), saliency_value\n    \n    def _create_distance_mask(self, center: Tuple[float, float], \n                            shape: Tuple[int, int],\n                            max_distance: int) -> np.ndarray:\n        \"\"\"距離制約マスクの作成\"\"\"\n        h, w = shape\n        center_x, center_y = int(center[0] * w), int(center[1] * h)\n        \n        y, x = np.ogrid[:h, :w]\n        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)\n        \n        # ソフト距離制約（ガウシアン減衰）\n        mask = np.exp(-(distances / max_distance)**2)\n        \n        return mask\n    \n    def _extract_candidates(self, saliency_map: np.ndarray) -> List[Tuple[int, int]]:\n        \"\"\"候補点の抽出\"\"\"\n        # 局所最大値の検出\n        local_maxima = self._find_local_maxima(saliency_map)\n        \n        # 閾値による フィルタリング\n        threshold = saliency_map.mean() + saliency_map.std()\n        candidates = [(y, x) for y, x in local_maxima \n                     if saliency_map[y, x] > threshold]\n        \n        # 上位K個を選択\n        candidates.sort(key=lambda pos: saliency_map[pos[0], pos[1]], reverse=True)\n        \n        return candidates[:self.top_k]\n    \n    def _find_local_maxima(self, saliency_map: np.ndarray) -> List[Tuple[int, int]]:\n        \"\"\"局所最大値の検出\"\"\"\n        # モルフォロジー演算による局所最大値検出\n        kernel = np.ones((5, 5))\n        dilated = cv2.dilate(saliency_map, kernel)\n        local_maxima = (saliency_map == dilated) & (saliency_map > 0.1)\n        \n        # 座標の抽出\n        coords = np.where(local_maxima)\n        return list(zip(coords[0], coords[1]))\n    \n    def _stochastic_selection(self, candidates: List[Tuple[int, int]]) -> int:\n        \"\"\"確率的選択（ボルツマン分布）\"\"\"\n        if len(candidates) == 1:\n            return 0\n        \n        # スコアの計算（ここでは単純にインデックスの逆順）\n        scores = np.array([len(candidates) - i for i in range(len(candidates))])\n        \n        # ボルツマン分布\n        exp_scores = np.exp(scores / self.temperature)\n        probabilities = exp_scores / exp_scores.sum()\n        \n        # 確率的選択\n        return np.random.choice(len(candidates), p=probabilities)\n    \n    def _random_selection(self, shape: Tuple[int, int]) -> Tuple[float, float]:\n        \"\"\"ランダム選択（フォールバック）\"\"\"\n        return (np.random.random(), np.random.random())


class InformationCalculator:\n    \"\"\"情報量計算クラス\"\"\"\n    \n    def __init__(self, config: Dict[str, Any]):\n        self.surprise_weight = config.get('surprise_weight', 0.6)\n        self.entropy_weight = config.get('entropy_weight', 0.4)\n    \n    def estimate_information(self, location: Tuple[float, float],\n                           saliency_map: np.ndarray,\n                           local_features: Optional[np.ndarray] = None) -> float:\n        \"\"\"\n        指定位置での情報量推定\n        \n        Args:\n            location: 位置（正規化座標）\n            saliency_map: 顕著性マップ\n            local_features: 局所特徴（オプション）\n            \n        Returns:\n            情報量\n        \"\"\"\n        h, w = saliency_map.shape\n        x, y = int(location[0] * w), int(location[1] * h)\n        \n        # 境界チェック\n        x = max(0, min(w-1, x))\n        y = max(0, min(h-1, y))\n        \n        # 驚き度の計算（顕著性値の逆数）\n        saliency_value = saliency_map[y, x]\n        surprise = -np.log(saliency_value + 1e-8)\n        \n        # 局所エントロピーの計算\n        patch_size = 15\n        y_start = max(0, y - patch_size//2)\n        y_end = min(h, y + patch_size//2 + 1)\n        x_start = max(0, x - patch_size//2)\n        x_end = min(w, x + patch_size//2 + 1)\n        \n        local_patch = saliency_map[y_start:y_end, x_start:x_end]\n        \n        # エントロピー計算\n        normalized_patch = local_patch / (local_patch.sum() + 1e-8)\n        entropy = -np.sum(normalized_patch * np.log(normalized_patch + 1e-8))\n        \n        # 重み付き合計\n        information = self.surprise_weight * surprise + self.entropy_weight * entropy\n        \n        return information


class EnhancedFixationSequenceGenerator:\n    \"\"\"\n    DeepGaze IIIの予測を基に人間らしい注視パターンを生成する拡張クラス\n    \"\"\"\n    \n    def __init__(self, config: Dict[str, Any]):\n        self.config = config\n        \n        # サブコンポーネントの初期化\n        self.ior_manager = InhibitionOfReturnManager(\n            decay_rate=config.get('ior_decay_rate', 0.02)\n        )\n        \n        self.duration_calculator = FixationDurationCalculator(config)\n        self.target_selector = SaccadeTargetSelector(\n            config.get('selection', {})\n        )\n        self.info_calculator = InformationCalculator(\n            config.get('information_theory', {})\n        )\n        \n        # パラメータ\n        self.default_duration_ms = config.get('default_duration_ms', 5000)\n        self.saccade_velocity = config.get('saccade_velocity_deg_per_sec', 300)\n        \n        logger.info(\"Enhanced Fixation Sequence Generator initialized\")\n    \n    def generate_scanpath(self, saliency_map: np.ndarray, \n                         duration_ms: int = None) -> List[Dict[str, Any]]:\n        \"\"\"\n        人間らしいスキャンパス生成\n        \n        Args:\n            saliency_map: 統合顕著性マップ\n            duration_ms: 総閲覧時間\n            \n        Returns:\n            注視点シーケンス\n        \"\"\"\n        if duration_ms is None:\n            duration_ms = self.default_duration_ms\n        \n        try:\n            # 初期化\n            self.ior_manager.initialize(saliency_map.shape)\n            scanpath = []\n            current_time = 0\n            \n            # 初期位置（中央付近または最高顕著性位置）\n            current_location = self._get_initial_location(saliency_map)\n            \n            while current_time < duration_ms:\n                # 情報量の推定\n                information_content = self.info_calculator.estimate_information(\n                    current_location, saliency_map\n                )\n                \n                # 現在位置の顕著性値\n                h, w = saliency_map.shape\n                x, y = int(current_location[0] * w), int(current_location[1] * h)\n                x, y = max(0, min(w-1, x)), max(0, min(h-1, y))\n                saliency_value = saliency_map[y, x]\n                \n                # 注視時間の計算\n                eccentricity = self._calculate_eccentricity(current_location)\n                fixation_duration = self.duration_calculator.calculate_duration(\n                    information_content, saliency_value, eccentricity\n                )\n                \n                # 注視点の記録\n                fixation = {\n                    'location': current_location,\n                    'duration': fixation_duration,\n                    'timestamp': current_time,\n                    'information_gain': information_content,\n                    'saliency_value': saliency_value,\n                    'confidence': min(1.0, saliency_value * 2),\n                    'eccentricity': eccentricity\n                }\n                scanpath.append(fixation)\n                \n                # IORの更新\n                self.ior_manager.update(current_location, saliency_map.shape)\n                \n                # 次の目標選択\n                ior_map = self.ior_manager.get_map()\n                next_location, next_saliency = self.target_selector.select_next_target(\n                    saliency_map, current_location, ior_map\n                )\n                \n                # サッカード時間の計算\n                saccade_time = self._calculate_saccade_time(\n                    current_location, next_location\n                )\n                \n                # 時間更新\n                current_time += fixation_duration + saccade_time\n                current_location = next_location\n                \n                # 無限ループ防止\n                if len(scanpath) > 50:\n                    break\n            \n            logger.info(f\"Generated scanpath with {len(scanpath)} fixations\")\n            return scanpath\n            \n        except Exception as e:\n            logger.error(f\"Error generating scanpath: {e}\")\n            return self._create_fallback_scanpath(saliency_map.shape, duration_ms)\n    \n    def _get_initial_location(self, saliency_map: np.ndarray) -> Tuple[float, float]:\n        \"\"\"初期注視位置の決定\"\"\"\n        h, w = saliency_map.shape\n        \n        # 最高顕著性位置を探す\n        max_pos = np.unravel_index(saliency_map.argmax(), saliency_map.shape)\n        max_y, max_x = max_pos\n        \n        # 中央バイアスとの重み付き平均\n        center_x, center_y = w // 2, h // 2\n        center_bias = 0.3  # 中央バイアス強度\n        \n        init_x = (1 - center_bias) * max_x + center_bias * center_x\n        init_y = (1 - center_bias) * max_y + center_bias * center_y\n        \n        # 正規化\n        norm_x = init_x / w\n        norm_y = init_y / h\n        \n        return (norm_x, norm_y)\n    \n    def _calculate_eccentricity(self, location: Tuple[float, float]) -> float:\n        \"\"\"中心からの離心率計算\"\"\"\n        center_x, center_y = 0.5, 0.5\n        distance = np.sqrt((location[0] - center_x)**2 + (location[1] - center_y)**2)\n        \n        # 最大距離で正規化（対角線の半分）\n        max_distance = np.sqrt(0.5**2 + 0.5**2)\n        eccentricity = distance / max_distance\n        \n        return eccentricity\n    \n    def _calculate_saccade_time(self, start: Tuple[float, float], \n                              end: Tuple[float, float]) -> float:\n        \"\"\"サッカード時間の計算\"\"\"\n        # 画面上の距離（ピクセル単位での近似）\n        distance_norm = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)\n        \n        # 視角への変換（近似：画面幅を40度と仮定）\n        distance_deg = distance_norm * 40\n        \n        # サッカード時間の計算（主系列関係）\n        # Duration = a + b * amplitude\n        a = 21  # ms (基本時間)\n        b = 2.2  # ms/deg (勾配)\n        \n        saccade_duration = a + b * distance_deg\n        \n        # 制限（生理学的妥当性）\n        saccade_duration = max(15, min(100, saccade_duration))\n        \n        return saccade_duration\n    \n    def _create_fallback_scanpath(self, shape: Tuple[int, int], \n                                duration_ms: int) -> List[Dict[str, Any]]:\n        \"\"\"フォールバック用スキャンパス\"\"\"\n        scanpath = []\n        num_fixations = max(3, duration_ms // 1000)  # 1秒に1注視点程度\n        \n        for i in range(num_fixations):\n            location = (np.random.random(), np.random.random())\n            timestamp = i * (duration_ms / num_fixations)\n            \n            fixation = {\n                'location': location,\n                'duration': 250,  # デフォルト時間\n                'timestamp': timestamp,\n                'information_gain': 0.5,\n                'saliency_value': 0.5,\n                'confidence': 0.3,\n                'eccentricity': self._calculate_eccentricity(location)\n            }\n            scanpath.append(fixation)\n        \n        return scanpath"