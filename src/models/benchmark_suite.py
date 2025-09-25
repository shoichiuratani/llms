"""
ベンチマークスイート

システムの性能評価とベンチマークテストを実行します。
"""

import numpy as np
import cv2
import time
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from PIL import Image

from .validation_metrics import ValidationMetrics
from ..core import IntegratedAttentionSystem

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    統合視覚的注意システムのベンチマークテストスイート
    """
    
    def __init__(self, system: Optional[IntegratedAttentionSystem] = None):
        self.system = system or IntegratedAttentionSystem()
        self.metrics_calculator = ValidationMetrics()
        self.results_history = []
        
    async def run_full_benchmark(self, test_images: List[np.ndarray] = None,
                               ground_truth_data: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        完全ベンチマークテストの実行
        
        Args:
            test_images: テスト画像リスト
            ground_truth_data: 真値データリスト
            
        Returns:
            ベンチマーク結果
        """
        logger.info("Starting full benchmark test")
        
        # テスト画像の準備
        if test_images is None:
            test_images = self._generate_synthetic_test_images()
        
        if ground_truth_data is None:
            ground_truth_data = self._generate_synthetic_ground_truth(test_images)
        
        benchmark_results = {
            'system_info': self._get_system_info(),
            'test_configuration': self._get_test_config(),
            'performance_tests': {},
            'accuracy_tests': {},
            'robustness_tests': {},
            'scalability_tests': {},
            'summary': {}
        }
        
        # 1. パフォーマンステスト
        logger.info("Running performance tests")
        benchmark_results['performance_tests'] = await self._run_performance_tests(test_images)
        
        # 2. 精度テスト
        logger.info("Running accuracy tests")
        benchmark_results['accuracy_tests'] = await self._run_accuracy_tests(
            test_images, ground_truth_data
        )
        
        # 3. 頑健性テスト
        logger.info("Running robustness tests")
        benchmark_results['robustness_tests'] = await self._run_robustness_tests(test_images)
        
        # 4. スケーラビリティテスト
        logger.info("Running scalability tests")
        benchmark_results['scalability_tests'] = await self._run_scalability_tests(test_images)
        
        # 5. サマリー生成
        benchmark_results['summary'] = self._generate_benchmark_summary(benchmark_results)
        
        # 結果の保存
        self.results_history.append(benchmark_results)
        
        logger.info("Benchmark test completed")
        return benchmark_results
    
    async def _run_performance_tests(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """パフォーマンステストの実行"""
        results = {
            'processing_speed': {},
            'memory_usage': {},
            'component_timing': {}
        }
        
        # 処理速度テスト
        processing_times = []
        component_times = {'deepgaze': [], 'sam2': [], 'llm': [], 'integration': []}
        
        for i, image in enumerate(test_images[:10]):  # 最初の10枚でテスト
            start_time = time.time()
            
            try:
                result = await self.system.analyze_integrated(image)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time * 1000)  # ms
                
                # コンポーネント別時間の記録
                timings = result.get('performance_metrics', {}).get('component_timings', {})
                for component in component_times.keys():
                    if component in timings:
                        component_times[component].append(timings[component])
                
                logger.debug(f"Image {i+1} processed in {processing_time*1000:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                processing_times.append(float('inf'))
        
        # 統計計算
        results['processing_speed'] = {
            'mean_time_ms': np.mean([t for t in processing_times if t != float('inf')]),
            'std_time_ms': np.std([t for t in processing_times if t != float('inf')]),
            'min_time_ms': np.min([t for t in processing_times if t != float('inf')]),
            'max_time_ms': np.max([t for t in processing_times if t != float('inf')]),
            'success_rate': sum(1 for t in processing_times if t != float('inf')) / len(processing_times)
        }
        
        # コンポーネント別統計
        for component, times in component_times.items():
            if times:
                results['component_timing'][component] = {
                    'mean_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'contribution_ratio': np.mean(times) / results['processing_speed']['mean_time_ms']
                }
        
        return results
    
    async def _run_accuracy_tests(self, test_images: List[np.ndarray],
                                ground_truth_data: List[np.ndarray]) -> Dict[str, Any]:
        """精度テストの実行"""
        results = {
            'overall_metrics': {},
            'component_comparison': {},
            'per_image_results': []
        }
        
        all_metrics = []
        component_metrics = {'deepgaze': [], 'sam2': [], 'llm': [], 'integrated': []}
        
        for i, (image, gt) in enumerate(zip(test_images, ground_truth_data)):
            try:
                # 統合分析の実行
                analysis_result = await self.system.analyze_integrated(image)
                
                # 各コンポーネントの出力を取得
                components = analysis_result.get('results', {}).get('component_outputs', {})
                
                # 統合結果の評価
                integrated_saliency = self._decode_saliency_map(
                    analysis_result.get('results', {}).get('primary_saliency_map', '')
                )
                
                if integrated_saliency is not None:
                    integrated_metrics = self.metrics_calculator.evaluate_against_deepgaze_baseline(
                        integrated_saliency, gt
                    )
                    all_metrics.append(integrated_metrics)
                    component_metrics['integrated'].append(integrated_metrics)
                    
                    # 個別結果の記録
                    results['per_image_results'].append({
                        'image_index': i,
                        'metrics': integrated_metrics,
                        'confidence': analysis_result.get('performance_metrics', {}).get('confidence_score', 0)
                    })
                
                # 各コンポーネントの個別評価
                for comp_name, comp_key in [
                    ('deepgaze', 'deepgaze_saliency'),
                    ('sam2', 'sam2_enhanced'),
                    ('llm', 'llm_weights')
                ]:
                    if comp_key in components:
                        comp_saliency = self._decode_saliency_map(components[comp_key])
                        if comp_saliency is not None:
                            comp_metrics = self.metrics_calculator.evaluate_against_deepgaze_baseline(
                                comp_saliency, gt
                            )
                            component_metrics[comp_name].append(comp_metrics)
                
            except Exception as e:
                logger.error(f"Error in accuracy test for image {i}: {e}")
        
        # 全体統計の計算
        if all_metrics:
            overall_stats = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics if not np.isinf(m[metric_name])]
                if values:
                    overall_stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            results['overall_metrics'] = overall_stats
        
        # コンポーネント比較
        for comp_name, comp_metrics_list in component_metrics.items():
            if comp_metrics_list:
                comp_stats = {}
                for metric_name in comp_metrics_list[0].keys():
                    values = [m[metric_name] for m in comp_metrics_list if not np.isinf(m[metric_name])]
                    if values:
                        comp_stats[metric_name] = np.mean(values)
                
                results['component_comparison'][comp_name] = comp_stats
        
        return results
    
    async def _run_robustness_tests(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """頑健性テストの実行"""
        results = {
            'noise_robustness': {},
            'brightness_robustness': {},
            'contrast_robustness': {},
            'blur_robustness': {},
            'rotation_robustness': {}
        }
        
        # ベースライン結果の取得
        baseline_results = []
        for image in test_images[:5]:  # 最初の5枚でテスト
            try:
                result = await self.system.analyze_integrated(image)
                baseline_results.append(result)
            except:
                baseline_results.append(None)
        
        # ノイズ頑健性テスト
        noise_levels = [0.1, 0.2, 0.3, 0.5]
        noise_scores = []
        
        for noise_level in noise_levels:
            level_scores = []
            for i, (image, baseline) in enumerate(zip(test_images[:5], baseline_results)):
                if baseline is None:
                    continue
                
                # ノイズ追加
                noise = np.random.normal(0, noise_level, image.shape)
                noisy_image = np.clip(image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
                
                try:
                    noisy_result = await self.system.analyze_integrated(noisy_image)
                    
                    # ベースラインとの類似度計算
                    similarity = self._calculate_result_similarity(baseline, noisy_result)
                    level_scores.append(similarity)
                    
                except Exception as e:
                    logger.error(f"Error in noise robustness test: {e}")
                    level_scores.append(0.0)
            
            if level_scores:
                noise_scores.append(np.mean(level_scores))
        
        results['noise_robustness'] = {
            'noise_levels': noise_levels,
            'similarity_scores': noise_scores,
            'overall_robustness': np.mean(noise_scores) if noise_scores else 0.0
        }
        
        # 明度変更テスト
        brightness_factors = [0.5, 0.7, 1.3, 1.5]
        brightness_scores = []
        
        for factor in brightness_factors:
            level_scores = []
            for i, (image, baseline) in enumerate(zip(test_images[:5], baseline_results)):
                if baseline is None:
                    continue
                
                # 明度変更
                bright_image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                
                try:
                    bright_result = await self.system.analyze_integrated(bright_image)
                    similarity = self._calculate_result_similarity(baseline, bright_result)
                    level_scores.append(similarity)
                    
                except Exception as e:
                    logger.error(f"Error in brightness robustness test: {e}")
                    level_scores.append(0.0)
            
            if level_scores:
                brightness_scores.append(np.mean(level_scores))
        
        results['brightness_robustness'] = {
            'brightness_factors': brightness_factors,
            'similarity_scores': brightness_scores,
            'overall_robustness': np.mean(brightness_scores) if brightness_scores else 0.0
        }
        
        # ブラーテスト
        blur_sigmas = [1, 2, 3, 5]
        blur_scores = []
        
        for sigma in blur_sigmas:
            level_scores = []
            for i, (image, baseline) in enumerate(zip(test_images[:5], baseline_results)):
                if baseline is None:
                    continue
                
                # ガウシアンブラー
                blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
                
                try:
                    blur_result = await self.system.analyze_integrated(blurred_image)
                    similarity = self._calculate_result_similarity(baseline, blur_result)
                    level_scores.append(similarity)
                    
                except Exception as e:
                    logger.error(f"Error in blur robustness test: {e}")
                    level_scores.append(0.0)
            
            if level_scores:
                blur_scores.append(np.mean(level_scores))
        
        results['blur_robustness'] = {
            'blur_sigmas': blur_sigmas,
            'similarity_scores': blur_scores,
            'overall_robustness': np.mean(blur_scores) if blur_scores else 0.0
        }
        
        return results
    
    async def _run_scalability_tests(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """スケーラビリティテストの実行"""
        results = {
            'image_size_scaling': {},
            'batch_processing': {},
            'concurrent_requests': {}
        }
        
        # 画像サイズ別性能テスト
        if test_images:
            base_image = test_images[0]
            image_sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)]
            size_performance = {}
            
            for size in image_sizes:
                resized_image = cv2.resize(base_image, size)
                
                times = []
                for _ in range(3):  # 3回実行して平均
                    start_time = time.time()
                    try:
                        await self.system.analyze_integrated(resized_image)
                        processing_time = time.time() - start_time
                        times.append(processing_time * 1000)  # ms
                    except Exception as e:
                        logger.error(f"Error in scalability test for size {size}: {e}")
                        times.append(float('inf'))
                
                valid_times = [t for t in times if t != float('inf')]
                size_performance[f"{size[0]}x{size[1]}"] = {
                    'mean_time_ms': np.mean(valid_times) if valid_times else float('inf'),
                    'std_time_ms': np.std(valid_times) if len(valid_times) > 1 else 0,
                    'success_rate': len(valid_times) / len(times)
                }
            
            results['image_size_scaling'] = size_performance
        
        # 同時リクエスト処理テスト
        if len(test_images) >= 3:
            concurrent_tests = [1, 2, 3]  # 同時リクエスト数
            concurrency_performance = {}
            
            for num_concurrent in concurrent_tests:
                test_batch = test_images[:num_concurrent]
                
                start_time = time.time()
                try:
                    # 同時実行
                    tasks = [self.system.analyze_integrated(img) for img in test_batch]
                    await asyncio.gather(*tasks)
                    
                    total_time = time.time() - start_time
                    avg_time_per_request = (total_time * 1000) / num_concurrent
                    
                    concurrency_performance[f"concurrent_{num_concurrent}"] = {
                        'total_time_ms': total_time * 1000,
                        'avg_time_per_request_ms': avg_time_per_request,
                        'throughput_requests_per_sec': num_concurrent / total_time
                    }
                    
                except Exception as e:
                    logger.error(f"Error in concurrent processing test: {e}")
                    concurrency_performance[f"concurrent_{num_concurrent}"] = {
                        'total_time_ms': float('inf'),
                        'avg_time_per_request_ms': float('inf'),
                        'throughput_requests_per_sec': 0.0
                    }
            
            results['concurrent_requests'] = concurrency_performance
        
        return results
    
    def _generate_synthetic_test_images(self, num_images: int = 10) -> List[np.ndarray]:
        """合成テスト画像の生成"""
        test_images = []
        
        for i in range(num_images):
            # 基本サイズ
            h, w = 480, 640
            
            # 背景生成
            background = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            
            # 複数の幾何学的図形を追加
            num_shapes = np.random.randint(2, 6)
            
            for _ in range(num_shapes):
                # ランダムな図形
                shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                
                if shape_type == 'circle':
                    center = (np.random.randint(50, w-50), np.random.randint(50, h-50))
                    radius = np.random.randint(20, 80)
                    cv2.circle(background, center, radius, color, -1)
                    
                elif shape_type == 'rectangle':
                    pt1 = (np.random.randint(0, w//2), np.random.randint(0, h//2))
                    pt2 = (np.random.randint(w//2, w), np.random.randint(h//2, h))
                    cv2.rectangle(background, pt1, pt2, color, -1)
                    
                elif shape_type == 'triangle':
                    pts = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(3)], np.int32)
                    cv2.fillPoly(background, [pts], color)
            
            # ノイズ追加
            noise = np.random.normal(0, 10, background.shape)
            noisy_image = np.clip(background.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            test_images.append(noisy_image)
        
        return test_images
    
    def _generate_synthetic_ground_truth(self, test_images: List[np.ndarray]) -> List[np.ndarray]:
        """合成真値データの生成"""
        ground_truths = []
        
        for image in test_images:
            h, w = image.shape[:2]
            
            # エッジベースの顕著性
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_saliency = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
            
            # 色ベースの顕著性
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            color_saliency = np.std(lab, axis=2)
            
            # 統合
            combined = 0.7 * edge_saliency + 0.3 * color_saliency
            
            # 正規化
            if combined.max() > 0:
                combined = combined / combined.max()
            
            ground_truths.append(combined)
        
        return ground_truths
    
    def _decode_saliency_map(self, b64_string: str) -> Optional[np.ndarray]:
        """Base64エンコードされた顕著性マップのデコード"""
        if not b64_string:
            return None
        
        try:
            from ..utils.image_utils import ImagePreprocessor
            processor = ImagePreprocessor({})
            decoded_image = processor.decode_base64_to_array(b64_string)
            
            # グレースケールに変換
            if len(decoded_image.shape) == 3:
                gray = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = decoded_image
            
            # 正規化
            return gray.astype(np.float32) / 255.0
            
        except Exception as e:
            logger.error(f"Error decoding saliency map: {e}")
            return None
    
    def _calculate_result_similarity(self, result1: Dict[str, Any], 
                                   result2: Dict[str, Any]) -> float:
        """2つの結果間の類似度計算"""
        try:
            # 顕著性マップの類似度
            sal1 = self._decode_saliency_map(
                result1.get('results', {}).get('primary_saliency_map', '')
            )
            sal2 = self._decode_saliency_map(
                result2.get('results', {}).get('primary_saliency_map', '')
            )
            
            if sal1 is None or sal2 is None:
                return 0.0
            
            # サイズ調整
            if sal1.shape != sal2.shape:
                sal2 = cv2.resize(sal2, (sal1.shape[1], sal1.shape[0]))
            
            # 相関係数による類似度
            correlation = self.metrics_calculator.correlation_coefficient(sal1, sal2)
            
            return max(0.0, correlation)
            
        except Exception as e:
            logger.error(f"Error calculating result similarity: {e}")
            return 0.0
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報の取得"""
        return {
            'system_version': '2.0.0',
            'components': {
                'deepgaze': 'DeepGaze III',
                'sam2': 'Segment Anything Model 2',
                'llm': self.system.config.get('llm', {}).get('model_name', 'Unknown')
            },
            'configuration': self.system.config_manager.config
        }
    
    def _get_test_config(self) -> Dict[str, Any]:
        """テスト設定の取得"""
        return {
            'test_suite_version': '1.0',
            'metrics_used': [
                'LL', 'NSS', 'AUC', 'CC', 'KLD', 'SIM', 'EMD',
                'object_alignment', 'semantic_coherence', 'temporal_plausibility'
            ],
            'test_categories': [
                'performance', 'accuracy', 'robustness', 'scalability'
            ]
        }
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """ベンチマーク結果のサマリー生成"""
        summary = {
            'overall_score': 0.0,
            'performance_score': 0.0,
            'accuracy_score': 0.0,
            'robustness_score': 0.0,
            'scalability_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        scores = []
        
        # パフォーマンススコア
        perf_tests = results.get('performance_tests', {})
        if 'processing_speed' in perf_tests:
            speed_data = perf_tests['processing_speed']
            success_rate = speed_data.get('success_rate', 0)
            mean_time = speed_data.get('mean_time_ms', float('inf'))
            
            # 1秒以下を目標とした正規化スコア
            time_score = max(0, 1 - (mean_time / 1000)) if mean_time != float('inf') else 0
            perf_score = 0.7 * success_rate + 0.3 * time_score
            
            summary['performance_score'] = perf_score
            scores.append(perf_score)
            
            if mean_time < 500:  # 500ms以下
                summary['strengths'].append("Fast processing speed")
            elif mean_time > 2000:  # 2秒以上
                summary['weaknesses'].append("Slow processing speed")
        
        # 精度スコア
        acc_tests = results.get('accuracy_tests', {})
        if 'overall_metrics' in acc_tests:
            overall_metrics = acc_tests['overall_metrics']
            
            # 主要メトリクスの平均
            key_metrics = ['AUC', 'CC', 'NSS']
            metric_scores = []
            
            for metric in key_metrics:
                if metric in overall_metrics:
                    metric_data = overall_metrics[metric]
                    mean_val = metric_data.get('mean', 0)
                    
                    if metric == 'AUC':
                        # AUCは0.5-1.0を0-1にスケール
                        normalized = max(0, (mean_val - 0.5) * 2)
                    elif metric == 'CC':
                        # 相関係数は-1から1なので0-1にスケール
                        normalized = (mean_val + 1) / 2
                    else:  # NSS
                        # NSSは通常0-3程度なので正規化
                        normalized = min(1.0, max(0, mean_val / 3.0))
                    
                    metric_scores.append(normalized)
            
            if metric_scores:
                acc_score = np.mean(metric_scores)
                summary['accuracy_score'] = acc_score
                scores.append(acc_score)
                
                if acc_score > 0.8:
                    summary['strengths'].append("High prediction accuracy")
                elif acc_score < 0.5:
                    summary['weaknesses'].append("Low prediction accuracy")
        
        # 頑健性スコア
        rob_tests = results.get('robustness_tests', {})
        robustness_scores = []
        
        for test_name in ['noise_robustness', 'brightness_robustness', 'blur_robustness']:
            if test_name in rob_tests:
                rob_score = rob_tests[test_name].get('overall_robustness', 0)
                robustness_scores.append(rob_score)
        
        if robustness_scores:
            rob_score = np.mean(robustness_scores)
            summary['robustness_score'] = rob_score
            scores.append(rob_score)
            
            if rob_score > 0.8:
                summary['strengths'].append("High robustness to input variations")
            elif rob_score < 0.5:
                summary['weaknesses'].append("Low robustness to input variations")
        
        # 全体スコア
        if scores:
            summary['overall_score'] = np.mean(scores)
        
        # 推奨事項の生成
        if summary['performance_score'] < 0.6:
            summary['recommendations'].append("Consider optimizing processing pipeline")
        
        if summary['accuracy_score'] < 0.6:
            summary['recommendations'].append("Review model parameters and training data")
        
        if summary['robustness_score'] < 0.6:
            summary['recommendations'].append("Implement better input preprocessing and normalization")
        
        return summary
    
    def save_benchmark_results(self, results: Dict[str, Any], 
                             filepath: str = None) -> str:
        """ベンチマーク結果の保存"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_results_{timestamp}.json"
        
        try:
            # NumPy配列を変換
            serializable_results = self._make_json_serializable(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Benchmark results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
            return ""
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """JSON シリアライズ可能な形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif obj is np.nan or obj == float('inf') or obj == float('-inf'):
            return None
        else:
            return obj