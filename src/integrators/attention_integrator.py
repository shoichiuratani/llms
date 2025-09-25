"""
注意統合器

複数のモダリティからの注意情報を神経科学的に妥当な方法で統合します。
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModalityWeights:
    """モダリティ間の重み設定"""
    deepgaze_base: float = 0.45
    sam2_object: float = 0.25
    llm_semantic: float = 0.30


@dataclass 
class TemporalWeights:
    """時間依存の重み設定"""
    bottom_up_dominance_duration: int = 150  # ms
    top_down_emergence_time: int = 200       # ms
    full_integration_time: int = 500         # ms


@dataclass
class BiologicalConstraints:
    """生物学的制約パラメータ"""
    center_bias_strength: float = 0.1
    ior_decay_rate: float = 0.02
    max_saccade_distance: int = 200  # pixels


class AttentionIntegrator:
    """
    複数モダリティの注意情報を統合する中核クラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 重み設定の初期化
        weight_config = config.get('modality_weights', {})
        self.modality_weights = ModalityWeights(
            deepgaze_base=weight_config.get('deepgaze_base', 0.45),
            sam2_object=weight_config.get('sam2_object', 0.25),
            llm_semantic=weight_config.get('llm_semantic', 0.30)
        )
        
        temporal_config = config.get('temporal_weights', {})
        self.temporal_weights = TemporalWeights(
            bottom_up_dominance_duration=temporal_config.get('bottom_up_dominance_duration', 150),
            top_down_emergence_time=temporal_config.get('top_down_emergence_time', 200),
            full_integration_time=temporal_config.get('full_integration_time', 500)
        )
        
        bio_config = config.get('biological_constraints', {})
        self.bio_constraints = BiologicalConstraints(
            center_bias_strength=bio_config.get('center_bias_strength', 0.1),
            ior_decay_rate=bio_config.get('ior_decay_rate', 0.02),
            max_saccade_distance=bio_config.get('max_saccade_distance', 200)
        )
        
        logger.info("Attention Integrator initialized")
    
    def compute_temporal_weights(self, time_ms: int = 0) -> Dict[str, float]:
        """
        時間に応じた重み計算
        
        Args:
            time_ms: 刺激提示からの経過時間（ミリ秒）
            
        Returns:
            時間依存の重み辞書
        """
        weights = {'bottom_up': 1.0, 'top_down': 0.0}
        
        if time_ms <= self.temporal_weights.bottom_up_dominance_duration:
            # 初期段階: ボトムアップ優位
            weights['bottom_up'] = 1.0
            weights['top_down'] = 0.0
            
        elif time_ms <= self.temporal_weights.top_down_emergence_time:\n            # 移行段階: トップダウンの出現\n            progress = (time_ms - self.temporal_weights.bottom_up_dominance_duration) / \\\n                      (self.temporal_weights.top_down_emergence_time - self.temporal_weights.bottom_up_dominance_duration)\n            weights['bottom_up'] = 1.0 - progress * 0.4  # 60%まで減少\n            weights['top_down'] = progress * 0.3         # 30%まで増加\n            \n        elif time_ms <= self.temporal_weights.full_integration_time:\n            # 統合段階: 両方のバランス\n            progress = (time_ms - self.temporal_weights.top_down_emergence_time) / \\\n                      (self.temporal_weights.full_integration_time - self.temporal_weights.top_down_emergence_time)\n            weights['bottom_up'] = 0.6 - progress * 0.1  # 50%まで減少\n            weights['top_down'] = 0.3 + progress * 0.2   # 50%まで増加\n            \n        else:\n            # 完全統合段階: 均衡\n            weights['bottom_up'] = 0.5\n            weights['top_down'] = 0.5\n            \n        return weights\n    \n    def integrate_modalities(self, deepgaze_output: Dict[str, Any],\n                           sam2_output: np.ndarray,\n                           llm_output: Dict[str, Any],\n                           time_ms: int = 0) -> np.ndarray:\n        \"\"\"\n        複数モダリティの統合\n        \n        Args:\n            deepgaze_output: DeepGazeの出力\n            sam2_output: SAM2の強化顕著性\n            llm_output: LLMの出力\n            time_ms: 経過時間\n            \n        Returns:\n            統合された顕著性マップ\n        \"\"\"\n        try:\n            base_saliency = deepgaze_output['raw_saliency']\n            semantic_weights = llm_output['semantic_weights']\n            \n            # 時間依存重みの計算\n            time_weights = self.compute_temporal_weights(time_ms)\n            \n            # ボトムアップ成分の統合\n            bottom_up_component = self._integrate_bottom_up(\n                base_saliency, sam2_output\n            )\n            \n            # トップダウン成分の適用\n            top_down_component = self._apply_top_down_modulation(\n                bottom_up_component, semantic_weights\n            )\n            \n            # 時間重み付き統合\n            integrated_saliency = (\n                time_weights['bottom_up'] * bottom_up_component +\n                time_weights['top_down'] * top_down_component\n            )\n            \n            # 予測的コーディングの組み込み\n            if 'predictive_map' in llm_output:\n                predictive_influence = 0.2 * time_weights['top_down']\n                integrated_saliency = (\n                    (1 - predictive_influence) * integrated_saliency +\n                    predictive_influence * llm_output['predictive_map']\n                )\n            \n            return integrated_saliency\n            \n        except Exception as e:\n            logger.error(f\"Error in modality integration: {e}\")\n            return deepgaze_output.get('raw_saliency', \n                                     np.ones(sam2_output.shape) * 0.5)\n    \n    def _integrate_bottom_up(self, base_saliency: np.ndarray, \n                           sam2_saliency: np.ndarray) -> np.ndarray:\n        \"\"\"ボトムアップ成分の統合\"\"\"\n        # DeepGazeとSAM2の重み付き統合\n        bottom_up = (\n            self.modality_weights.deepgaze_base * base_saliency +\n            self.modality_weights.sam2_object * sam2_saliency\n        )\n        \n        # 正規化\n        if bottom_up.max() > bottom_up.min():\n            bottom_up = (bottom_up - bottom_up.min()) / \\\n                       (bottom_up.max() - bottom_up.min())\n        \n        return bottom_up\n    \n    def _apply_top_down_modulation(self, bottom_up: np.ndarray,\n                                 semantic_weights: np.ndarray) -> np.ndarray:\n        \"\"\"トップダウン変調の適用\"\"\"\n        # セマンティック重みによる変調\n        modulated = bottom_up * semantic_weights\n        \n        # 正規化\n        if modulated.max() > modulated.min():\n            modulated = (modulated - modulated.min()) / \\\n                       (modulated.max() - modulated.min())\n        \n        return modulated\n    \n    def apply_biological_constraints(self, saliency_map: np.ndarray,\n                                   center_bias: Optional[np.ndarray] = None,\n                                   ior_map: Optional[np.ndarray] = None) -> np.ndarray:\n        \"\"\"\n        生物学的制約の適用\n        \n        Args:\n            saliency_map: 入力顕著性マップ\n            center_bias: センターバイアス\n            ior_map: Inhibition of Return マップ\n            \n        Returns:\n            制約適用後の顕著性マップ\n        \"\"\"\n        constrained_map = saliency_map.copy()\n        \n        # センターバイアスの適用\n        if center_bias is not None:\n            center_influence = self.bio_constraints.center_bias_strength\n            constrained_map = (\n                (1 - center_influence) * constrained_map +\n                center_influence * center_bias\n            )\n        \n        # Inhibition of Returnの適用\n        if ior_map is not None:\n            constrained_map = constrained_map * (1 - ior_map)\n        \n        # ガウシアンスムージング（神経的拡散のシミュレーション）\n        constrained_map = cv2.GaussianBlur(constrained_map, (5, 5), 0)\n        \n        # 最終正規化\n        if constrained_map.max() > constrained_map.min():\n            constrained_map = (constrained_map - constrained_map.min()) / \\\n                            (constrained_map.max() - constrained_map.min())\n        \n        return constrained_map\n    \n    def compute_attention_distribution(self, saliency_map: np.ndarray) -> Dict[str, float]:\n        \"\"\"\n        注意分布の統計計算\n        \n        Args:\n            saliency_map: 顕著性マップ\n            \n        Returns:\n            注意分布の統計情報\n        \"\"\"\n        # エントロピーの計算\n        normalized_map = saliency_map / (saliency_map.sum() + 1e-8)\n        entropy = -np.sum(normalized_map * np.log(normalized_map + 1e-8))\n        \n        # 集中度の計算（最大値/平均値比）\n        max_saliency = saliency_map.max()\n        mean_saliency = saliency_map.mean()\n        concentration = max_saliency / (mean_saliency + 1e-8)\n        \n        # 空間的拡散度\n        moments = cv2.moments(saliency_map)\n        if moments['m00'] > 0:\n            cx = int(moments['m10'] / moments['m00'])\n            cy = int(moments['m01'] / moments['m00'])\n            \n            # 中心からの平均距離\n            h, w = saliency_map.shape\n            y, x = np.ogrid[:h, :w]\n            distances = np.sqrt((x - cx)**2 + (y - cy)**2)\n            avg_distance = np.average(distances, weights=saliency_map)\n            \n            # 正規化された拡散度\n            max_distance = np.sqrt(h**2 + w**2) / 2\n            spread = avg_distance / max_distance\n        else:\n            spread = 0.0\n        \n        return {\n            'entropy': entropy,\n            'concentration': concentration,\n            'spatial_spread': spread,\n            'peak_saliency': max_saliency,\n            'mean_saliency': mean_saliency\n        }\n    \n    def update_weights_adaptive(self, performance_feedback: Dict[str, float]):\n        \"\"\"\n        パフォーマンスフィードバックに基づく適応的重み更新\n        \n        Args:\n            performance_feedback: 各モダリティのパフォーマンス評価\n        \"\"\"\n        # 簡易的な適応メカニズム\n        learning_rate = 0.01\n        \n        # DeepGazeの重み更新\n        if 'deepgaze_performance' in performance_feedback:\n            perf = performance_feedback['deepgaze_performance']\n            adjustment = (perf - 0.5) * learning_rate  # 0.5を基準とした調整\n            self.modality_weights.deepgaze_base = np.clip(\n                self.modality_weights.deepgaze_base + adjustment, 0.1, 0.8\n            )\n        \n        # SAM2の重み更新\n        if 'sam2_performance' in performance_feedback:\n            perf = performance_feedback['sam2_performance']\n            adjustment = (perf - 0.5) * learning_rate\n            self.modality_weights.sam2_object = np.clip(\n                self.modality_weights.sam2_object + adjustment, 0.1, 0.5\n            )\n        \n        # LLMの重み更新\n        if 'llm_performance' in performance_feedback:\n            perf = performance_feedback['llm_performance']\n            adjustment = (perf - 0.5) * learning_rate\n            self.modality_weights.llm_semantic = np.clip(\n                self.modality_weights.llm_semantic + adjustment, 0.1, 0.6\n            )\n        \n        # 重みの正規化\n        total = (self.modality_weights.deepgaze_base + \n                self.modality_weights.sam2_object + \n                self.modality_weights.llm_semantic)\n        \n        if total > 0:\n            self.modality_weights.deepgaze_base /= total\n            self.modality_weights.sam2_object /= total\n            self.modality_weights.llm_semantic /= total\n        \n        logger.info(f\"Updated weights - DeepGaze: {self.modality_weights.deepgaze_base:.3f}, \"\n                   f\"SAM2: {self.modality_weights.sam2_object:.3f}, \"\n                   f\"LLM: {self.modality_weights.llm_semantic:.3f}\")\n    \n    def get_integration_metadata(self) -> Dict[str, Any]:\n        \"\"\"統合処理のメタデータ取得\"\"\"\n        return {\n            'modality_weights': {\n                'deepgaze_base': self.modality_weights.deepgaze_base,\n                'sam2_object': self.modality_weights.sam2_object,\n                'llm_semantic': self.modality_weights.llm_semantic\n            },\n            'temporal_parameters': {\n                'bottom_up_dominance_ms': self.temporal_weights.bottom_up_dominance_duration,\n                'top_down_emergence_ms': self.temporal_weights.top_down_emergence_time,\n                'full_integration_ms': self.temporal_weights.full_integration_time\n            },\n            'biological_constraints': {\n                'center_bias_strength': self.bio_constraints.center_bias_strength,\n                'ior_decay_rate': self.bio_constraints.ior_decay_rate,\n                'max_saccade_distance': self.bio_constraints.max_saccade_distance\n            }\n        }"