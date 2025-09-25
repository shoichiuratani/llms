"""
検証メトリクス

DeepGaze IIIベンチマークとの比較や各種評価指標を計算します。
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """
    視覚的注意予測の評価メトリクスを計算するクラス
    """
    
    def __init__(self):
        self.eps = 1e-8  # 数値安定性のための小さな値
    
    def evaluate_against_deepgaze_baseline(self, predictions: np.ndarray, 
                                         ground_truth: np.ndarray) -> Dict[str, float]:
        """
        DeepGaze III ベースラインとの比較評価
        
        Args:
            predictions: 予測顕著性マップ
            ground_truth: 真の注視点データまたは顕著性マップ
            
        Returns:
            評価メトリクス辞書
        """
        try:
            # サイズ調整
            if predictions.shape != ground_truth.shape:
                predictions = cv2.resize(predictions, 
                                       (ground_truth.shape[1], ground_truth.shape[0]))
            
            # 正規化
            pred_norm = self._normalize_map(predictions)
            gt_norm = self._normalize_map(ground_truth)
            
            metrics = {}
            
            # DeepGaze III標準メトリクス
            metrics['LL'] = self.log_likelihood(pred_norm, gt_norm)
            metrics['NSS'] = self.normalized_scanpath_saliency(pred_norm, gt_norm)
            metrics['AUC'] = self.area_under_curve(pred_norm, gt_norm)
            metrics['CC'] = self.correlation_coefficient(pred_norm, gt_norm)
            
            # 拡張メトリクス
            metrics['KLD'] = self.kullback_leibler_divergence(pred_norm, gt_norm)
            metrics['SIM'] = self.similarity_metric(pred_norm, gt_norm)
            metrics['EMD'] = self.earth_movers_distance(pred_norm, gt_norm)
            
            # 新規メトリクス
            metrics['object_alignment'] = self.measure_object_alignment(predictions)
            metrics['semantic_coherence'] = self.measure_semantic_coherence(predictions)
            metrics['temporal_plausibility'] = self.measure_temporal_plausibility(predictions)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in baseline evaluation: {e}")
            return self._get_default_metrics()
    
    def log_likelihood(self, predictions: np.ndarray, 
                      ground_truth: np.ndarray) -> float:
        """
        対数尤度 (Log Likelihood) の計算
        """
        try:
            # 正規化（確率分布として扱う）
            pred_prob = predictions / (predictions.sum() + self.eps)
            
            # 真値が注視点の場合の処理
            if ground_truth.max() <= 1.0 and np.unique(ground_truth).size <= 10:
                # バイナリ注視点マップ
                fixation_points = ground_truth > 0
                if fixation_points.sum() == 0:
                    return 0.0
                
                ll = np.mean(np.log(pred_prob[fixation_points] + self.eps))
            else:
                # 連続値顕著性マップ
                gt_prob = ground_truth / (ground_truth.sum() + self.eps)
                ll = np.sum(gt_prob * np.log(pred_prob + self.eps))
            
            return float(ll)
            
        except Exception as e:
            logger.error(f"Error calculating log likelihood: {e}")
            return 0.0
    
    def normalized_scanpath_saliency(self, predictions: np.ndarray,
                                   fixation_points: np.ndarray) -> float:
        """
        正規化スキャンパス顕著性 (NSS) の計算
        """
        try:
            # 予測値の標準化
            pred_mean = predictions.mean()
            pred_std = predictions.std()
            
            if pred_std == 0:
                return 0.0
            
            normalized_pred = (predictions - pred_mean) / pred_std
            
            # 注視点での顕著性の平均
            if fixation_points.max() <= 1.0 and np.unique(fixation_points).size <= 10:
                # バイナリ注視点マップ
                fixation_mask = fixation_points > 0
                if fixation_mask.sum() == 0:
                    return 0.0
                
                nss = normalized_pred[fixation_mask].mean()
            else:
                # 重み付き平均（連続値の場合）
                weights = fixation_points / (fixation_points.sum() + self.eps)
                nss = np.sum(normalized_pred * weights)
            
            return float(nss)
            
        except Exception as e:
            logger.error(f"Error calculating NSS: {e}")
            return 0.0
    
    def area_under_curve(self, predictions: np.ndarray,
                        ground_truth: np.ndarray) -> float:
        """
        ROC曲線下面積 (AUC) の計算
        """
        try:
            # バイナリ分類問題として扱う
            pred_flat = predictions.flatten()
            
            if ground_truth.max() <= 1.0 and np.unique(ground_truth).size <= 10:
                # バイナリ真値
                gt_flat = ground_truth.flatten()
            else:
                # 連続値を閾値でバイナリ化
                threshold = np.percentile(ground_truth, 90)  # 上位10%を正例
                gt_flat = (ground_truth > threshold).astype(int).flatten()
            
            if len(np.unique(gt_flat)) < 2:
                return 0.5  # 単一クラスの場合
            
            auc = roc_auc_score(gt_flat, pred_flat)
            return float(auc)
            
        except Exception as e:
            logger.error(f"Error calculating AUC: {e}")
            return 0.5
    
    def correlation_coefficient(self, predictions: np.ndarray,
                              ground_truth: np.ndarray) -> float:
        """
        相関係数の計算
        """
        try:
            pred_flat = predictions.flatten()
            gt_flat = ground_truth.flatten()
            
            # ピアソン相関係数
            correlation, _ = pearsonr(pred_flat, gt_flat)
            
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def kullback_leibler_divergence(self, predictions: np.ndarray,
                                   ground_truth: np.ndarray) -> float:
        """
        KLダイバージェンス（カルバック・ライブラー情報量）の計算
        """
        try:
            # 確率分布として正規化
            pred_prob = predictions / (predictions.sum() + self.eps) + self.eps
            gt_prob = ground_truth / (ground_truth.sum() + self.eps) + self.eps
            
            # KLダイバージェンス
            kld = np.sum(gt_prob * np.log(gt_prob / pred_prob))
            
            return float(kld)
            
        except Exception as e:
            logger.error(f"Error calculating KLD: {e}")
            return float('inf')
    
    def similarity_metric(self, predictions: np.ndarray,
                         ground_truth: np.ndarray) -> float:
        """
        類似度メトリクス（最小値の合計）の計算
        """
        try:
            # 確率分布として正規化
            pred_prob = predictions / (predictions.sum() + self.eps)
            gt_prob = ground_truth / (ground_truth.sum() + self.eps)
            
            # 各位置での最小値の合計
            similarity = np.sum(np.minimum(pred_prob, gt_prob))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def earth_movers_distance(self, predictions: np.ndarray,
                            ground_truth: np.ndarray) -> float:
        """
        Earth Mover's Distance (Wasserstein距離) の近似計算
        """
        try:
            # 簡易版EMD（重心間距離で近似）
            h, w = predictions.shape
            
            # 予測の重心計算
            pred_prob = predictions / (predictions.sum() + self.eps)
            y_pred, x_pred = np.ogrid[:h, :w]
            cx_pred = np.sum(x_pred * pred_prob)
            cy_pred = np.sum(y_pred * pred_prob)
            
            # 真値の重心計算
            gt_prob = ground_truth / (ground_truth.sum() + self.eps)
            cx_gt = np.sum(x_pred * gt_prob)
            cy_gt = np.sum(y_pred * gt_prob)
            
            # 重心間距離
            emd = np.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)
            
            # 正規化（対角線長で除算）
            diagonal = np.sqrt(h**2 + w**2)
            emd_normalized = emd / diagonal
            
            return float(emd_normalized)
            
        except Exception as e:
            logger.error(f"Error calculating EMD: {e}")
            return 1.0
    
    def measure_object_alignment(self, predictions: np.ndarray) -> float:
        """
        物体境界との整合性測定
        """
        try:
            # エッジ検出
            edges = cv2.Canny((predictions * 255).astype(np.uint8), 50, 150)
            
            # 顕著性ピークの検出
            local_maxima = self._detect_local_maxima(predictions)
            
            # エッジ近傍でのピーク密度
            edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8))
            alignment = np.sum(local_maxima * (edge_dilated > 0)) / (np.sum(local_maxima) + 1)
            
            return float(alignment)
            
        except Exception as e:
            logger.error(f"Error measuring object alignment: {e}")
            return 0.0
    
    def measure_semantic_coherence(self, predictions: np.ndarray) -> float:
        """
        意味的一貫性の測定
        """
        try:
            # 空間的滑らかさの評価
            grad_x = np.abs(np.diff(predictions, axis=1))
            grad_y = np.abs(np.diff(predictions, axis=0))
            
            # 勾配の平均（低いほど滑らか）
            smoothness = 1.0 / (1.0 + np.mean(grad_x) + np.mean(grad_y))
            
            # クラスタリング評価（連結成分解析）
            binary_map = predictions > np.percentile(predictions, 75)
            num_components, _ = cv2.connectedComponents(binary_map.astype(np.uint8))
            
            # 適度な数の連結成分が望ましい
            ideal_components = 5
            component_score = 1.0 / (1.0 + abs(num_components - ideal_components) / ideal_components)
            
            coherence = 0.6 * smoothness + 0.4 * component_score
            
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Error measuring semantic coherence: {e}")
            return 0.5
    
    def measure_temporal_plausibility(self, predictions: np.ndarray) -> float:
        """
        時間的妥当性の測定
        """
        try:
            # 中央バイアスの評価
            h, w = predictions.shape
            center_y, center_x = h // 2, w // 2
            
            # 中央領域の定義
            center_region = np.zeros_like(predictions)
            radius = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            center_saliency = np.mean(predictions[center_mask])
            peripheral_saliency = np.mean(predictions[~center_mask])
            
            # 中央バイアススコア（中央が高い方が妥当）
            center_bias_score = center_saliency / (center_saliency + peripheral_saliency + self.eps)
            
            # 顕著性の分散（適度な集中が望ましい）
            concentration = predictions.std() / (predictions.mean() + self.eps)
            concentration_score = min(1.0, concentration / 2.0)  # 2.0を理想とする
            
            temporal_plausibility = 0.7 * center_bias_score + 0.3 * concentration_score
            
            return float(temporal_plausibility)
            
        except Exception as e:
            logger.error(f"Error measuring temporal plausibility: {e}")
            return 0.5
    
    def _normalize_map(self, saliency_map: np.ndarray) -> np.ndarray:
        """顕著性マップの正規化"""
        min_val = saliency_map.min()
        max_val = saliency_map.max()
        
        if max_val == min_val:
            return np.ones_like(saliency_map) / saliency_map.size
        
        normalized = (saliency_map - min_val) / (max_val - min_val)
        return normalized
    
    def _detect_local_maxima(self, saliency_map: np.ndarray) -> np.ndarray:
        """局所最大値の検出"""
        kernel = np.ones((3, 3))
        dilated = ndimage.maximum_filter(saliency_map, footprint=kernel)
        local_maxima = (saliency_map == dilated) & (saliency_map > np.mean(saliency_map))
        return local_maxima.astype(np.float32)
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """デフォルトメトリクス"""
        return {
            'LL': 0.0,
            'NSS': 0.0,
            'AUC': 0.5,
            'CC': 0.0,
            'KLD': float('inf'),
            'SIM': 0.0,
            'EMD': 1.0,
            'object_alignment': 0.0,
            'semantic_coherence': 0.5,
            'temporal_plausibility': 0.5
        }
    
    def compare_with_human_data(self, predictions: np.ndarray,
                               human_fixations: List[Tuple[float, float]],
                               image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        人間の注視データとの比較
        
        Args:
            predictions: 予測顕著性マップ
            human_fixations: 人間の注視点リスト [(x, y), ...]
            image_shape: 画像サイズ (height, width)
            
        Returns:
            比較メトリクス
        """
        try:
            h, w = image_shape
            
            # 注視点からバイナリマップを生成
            fixation_map = np.zeros((h, w), dtype=np.float32)
            
            for x, y in human_fixations:
                # 正規化座標を画像座標に変換
                img_x = int(x * w)
                img_y = int(y * h)
                
                # 境界チェック
                if 0 <= img_x < w and 0 <= img_y < h:
                    fixation_map[img_y, img_x] = 1.0
            
            # ガウシアンスムージング（人間の注視の広がりを考慮）
            sigma = min(h, w) * 0.02  # 画像サイズの2%程度
            fixation_map = cv2.GaussianBlur(fixation_map, (0, 0), sigma)
            
            # 予測マップのサイズ調整
            if predictions.shape != (h, w):
                predictions_resized = cv2.resize(predictions, (w, h))
            else:
                predictions_resized = predictions
            
            # メトリクス計算
            metrics = self.evaluate_against_deepgaze_baseline(
                predictions_resized, fixation_map
            )
            
            # 追加の人間データ特有メトリクス
            metrics['fixation_coverage'] = self._calculate_fixation_coverage(
                predictions_resized, human_fixations, image_shape
            )
            
            metrics['scanpath_similarity'] = self._calculate_scanpath_similarity(
                predictions_resized, human_fixations, image_shape
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error comparing with human data: {e}")
            return self._get_default_metrics()
    
    def _calculate_fixation_coverage(self, predictions: np.ndarray,
                                   fixations: List[Tuple[float, float]],
                                   image_shape: Tuple[int, int]) -> float:
        """注視点カバレッジの計算"""
        try:
            h, w = image_shape
            
            # 上位20%の顕著性領域
            threshold = np.percentile(predictions, 80)
            salient_region = predictions > threshold
            
            # 注視点のうち顕著性領域内にあるものの割合
            covered_fixations = 0
            total_fixations = len(fixations)
            
            for x, y in fixations:
                img_x = int(x * w)
                img_y = int(y * h)
                
                if 0 <= img_x < w and 0 <= img_y < h:
                    if salient_region[img_y, img_x]:
                        covered_fixations += 1
            
            coverage = covered_fixations / max(1, total_fixations)
            return float(coverage)
            
        except Exception as e:
            logger.error(f"Error calculating fixation coverage: {e}")
            return 0.0
    
    def _calculate_scanpath_similarity(self, predictions: np.ndarray,
                                     fixations: List[Tuple[float, float]],
                                     image_shape: Tuple[int, int]]) -> float:
        """スキャンパス類似度の計算"""
        try:
            if len(fixations) < 2:
                return 0.0
            
            # 予測に基づく理想的なスキャンパス生成
            ideal_path = self._generate_ideal_scanpath(predictions, len(fixations))
            
            # 実際のスキャンパスとの距離計算
            total_distance = 0.0
            
            for i, (real_fix, ideal_fix) in enumerate(zip(fixations, ideal_path)):
                distance = np.sqrt((real_fix[0] - ideal_fix[0])**2 + 
                                 (real_fix[1] - ideal_fix[1])**2)
                total_distance += distance
            
            # 正規化（対角線長で除算）
            diagonal = np.sqrt(2)  # 正規化座標での対角線長
            avg_distance = total_distance / len(fixations)
            similarity = max(0.0, 1.0 - avg_distance / diagonal)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating scanpath similarity: {e}")
            return 0.0
    
    def _generate_ideal_scanpath(self, predictions: np.ndarray,
                               num_fixations: int) -> List[Tuple[float, float]]:
        """理想的なスキャンパスの生成"""
        h, w = predictions.shape
        path = []
        
        # 現在の顕著性マップをコピー
        current_map = predictions.copy()
        
        for _ in range(num_fixations):
            # 最大値位置を検索
            max_pos = np.unravel_index(current_map.argmax(), current_map.shape)
            y, x = max_pos
            
            # 正規化座標に変換
            norm_x = x / w
            norm_y = y / h
            path.append((norm_x, norm_y))
            
            # IOR効果をシミュレーション（周辺を減衰）
            sigma = min(h, w) * 0.1
            y_grid, x_grid = np.ogrid[:h, :w]
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            current_map *= (1 - 0.5 * gaussian)
        
        return path