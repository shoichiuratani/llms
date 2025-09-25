"""
統合視覚的注意システム

DeepGaze III、SAM2、LLMを統合した人間の視覚的注意メカニズムの
再現システムのメインクラスです。
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import cv2
import time

from .config_manager import ConfigManager
from ..processors import (
    DeepGazeBottomUpProcessor,
    ObjectBasedAttention,
    LLMTopDownController
)
from ..integrators import (
    TripleIntegrationAlgorithm,
    EnhancedFixationSequenceGenerator
)
from ..utils.image_utils import ImagePreprocessor
from ..utils.validation import InputValidator

logger = logging.getLogger(__name__)


class IntegratedAttentionSystem:
    """
    DeepGaze III、SAM2、LLMを統合した注意予測システム
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        システムの初期化
        
        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルト設定を使用）
        """
        # 設定管理
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # ログ設定
        self._setup_logging()
        
        # バリデーター
        self.validator = InputValidator()
        self.image_preprocessor = ImagePreprocessor(
            self.config.get('system', {})
        )
        
        # コアコンポーネントの初期化
        self._initialize_components()
        
        # パフォーマンス追跡
        self.processing_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'last_update': time.time()
        }
        
        logger.info("Integrated Attention System v2.0 initialized successfully")
    
    def _setup_logging(self):
        """ログ設定"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # 既存のハンドラーをクリア
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 新しいハンドラー設定
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(level)
    
    def _initialize_components(self):
        """各コンポーネントの初期化"""
        try:
            # DeepGaze III ボトムアップ処理
            deepgaze_config = self.config.get('deepgaze', {})
            self.deepgaze_processor = DeepGazeBottomUpProcessor(deepgaze_config)
            
            # SAM2 物体ベース注意
            sam2_config = self.config.get('sam2', {})
            self.sam2_processor = ObjectBasedAttention(sam2_config)
            
            # LLM トップダウン制御
            llm_config = self.config.get('llm', {})
            self.llm_processor = LLMTopDownController(llm_config)
            
            # 統合アルゴリズム
            self.integration_algorithm = TripleIntegrationAlgorithm(
                self.deepgaze_processor,
                self.sam2_processor, 
                self.llm_processor,
                self.config
            )
            
            # 注視点シーケンス生成器
            fixation_config = self.config.get('fixation_sequence', {})
            self.fixation_generator = EnhancedFixationSequenceGenerator(
                fixation_config
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise RuntimeError(f"Failed to initialize system components: {e}")
    
    async def analyze_integrated(self, image: Union[np.ndarray, Image.Image, str], 
                               analysis_config: Optional[Dict[str, Any]] = None,
                               user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        統合分析の実行
        
        Args:
            image: 入力画像（numpy配列、PILイメージ、またはパス）
            analysis_config: 分析設定
            user_context: ユーザーコンテキスト
            
        Returns:
            分析結果辞書
        """
        start_time = time.time()
        
        try:
            # 入力検証と前処理
            processed_image = self._preprocess_input(
                image, analysis_config or {}
            )
            
            # ユーザーコンテキストの解析
            task_context = self._extract_task_context(user_context or {})
            
            # 統合分析の実行
            integration_result = await self.integration_algorithm.integrate_all_modalities(
                processed_image,
                task_context.get('task'),
                0  # 初期時間
            )
            
            # 注視点シーケンスの生成
            scanpath = None
            if analysis_config and analysis_config.get('generate_scanpath', True):
                duration_ms = user_context.get('viewing_duration_ms', 5000)
                scanpath = self.fixation_generator.generate_scanpath(
                    integration_result.saliency_map,
                    duration_ms
                )
            
            # 結果の構築
            result = self._build_analysis_result(
                integration_result,
                scanpath,
                processed_image.shape,
                time.time() - start_time
            )
            
            # 統計更新
            self._update_processing_stats(time.time() - start_time)
            
            logger.info(f"Analysis completed in {(time.time() - start_time)*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in integrated analysis: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _preprocess_input(self, image: Union[np.ndarray, Image.Image, str],
                         config: Dict[str, Any]) -> np.ndarray:
        """入力の前処理"""
        # 画像の読み込み・変換
        if isinstance(image, str):
            # ファイルパスの場合
            pil_image = Image.open(image)
            image_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            # PILイメージの場合
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            # NumPy配列の場合
            image_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 入力検証
        self.validator.validate_image(image_array)
        
        # 前処理
        processed_image = self.image_preprocessor.process(
            image_array, config.get('preprocessing', {})
        )
        
        return processed_image
    
    def _extract_task_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """ユーザーコンテキストからタスク情報を抽出"""
        task_mapping = {
            'free_viewing': 'free_viewing',
            'face_detection': 'face_detection',
            'text_reading': 'text_reading',
            'navigation': 'navigation',
            'search': 'visual_search'
        }
        
        task_type = user_context.get('task', 'free_viewing')
        mapped_task = task_mapping.get(task_type, task_type)
        
        return {
            'task': mapped_task,
            'expertise_level': user_context.get('expertise_level', 'general'),
            'language': user_context.get('language', 'en'),
            'viewing_duration_ms': user_context.get('viewing_duration_ms', 5000)
        }
    
    def _build_analysis_result(self, integration_result,
                              scanpath: Optional[List[Dict]],
                              image_shape: Tuple[int, int, int],
                              processing_time: float) -> Dict[str, Any]:
        """分析結果の構築"""
        # 顕著性マップのエンコード
        saliency_b64 = self.image_preprocessor.encode_array_to_base64(
            integration_result.saliency_map
        )
        
        # コンポーネント貢献度の計算
        contribution = self._calculate_component_contribution(
            integration_result.components
        )
        
        # 意味的解釈の生成
        semantic_interpretation = self._generate_semantic_interpretation(
            integration_result.components.get('llm_scene_context', {}),
            scanpath
        )
        
        return {
            'results': {
                'primary_saliency_map': saliency_b64,
                'fixation_sequence': scanpath or [],
                'attention_distribution': contribution,
                'semantic_interpretation': semantic_interpretation,
                'component_outputs': {
                    'deepgaze_saliency': self.image_preprocessor.encode_array_to_base64(
                        integration_result.components.get('deepgaze', np.array([]))
                    ),
                    'sam2_enhanced': self.image_preprocessor.encode_array_to_base64(
                        integration_result.components.get('sam2', np.array([]))
                    ),
                    'llm_weights': self.image_preprocessor.encode_array_to_base64(
                        integration_result.components.get('llm', {}).get('semantic_weights', np.array([]))
                    )
                }
            },
            'performance_metrics': {
                'processing_time_ms': processing_time * 1000,
                'confidence_score': integration_result.confidence_score,
                'component_timings': integration_result.metadata.get('processing_timings', {})
            },
            'metadata': {
                'image_shape': image_shape,
                'algorithm_version': '2.0',
                'timestamp': time.time(),
                'system_stats': self._get_system_stats()
            }
        }
    
    def _calculate_component_contribution(self, components: Dict[str, Any]) -> Dict[str, float]:
        """各コンポーネントの貢献度計算"""
        try:
            deepgaze_sal = components.get('deepgaze', np.array([]))
            sam2_sal = components.get('sam2', np.array([]))
            llm_weights = components.get('llm', {}).get('semantic_weights', np.array([]))
            
            # 各コンポーネントの影響度を計算
            contributions = {
                'deepgaze_contribution': 0.45,  # デフォルト値
                'sam2_contribution': 0.25,
                'llm_contribution': 0.30
            }
            
            # 実際の統合重みを使用（利用可能な場合）
            if hasattr(self.integration_algorithm.attention_integrator, 'modality_weights'):
                weights = self.integration_algorithm.attention_integrator.modality_weights
                contributions = {
                    'deepgaze_contribution': weights.deepgaze_base,
                    'sam2_contribution': weights.sam2_object,
                    'llm_contribution': weights.llm_semantic
                }
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Error calculating component contribution: {e}")
            return {
                'deepgaze_contribution': 0.45,
                'sam2_contribution': 0.25,
                'llm_contribution': 0.30
            }
    
    def _generate_semantic_interpretation(self, scene_context: Dict[str, Any],
                                        scanpath: Optional[List[Dict]]) -> Dict[str, Any]:
        """意味的解釈の生成"""
        main_objects = []
        gaze_narrative = "視線パターンの分析中..."
        
        # シーン要素の抽出
        elements = scene_context.get('main_elements', [])
        for element in elements[:5]:  # 上位5要素
            main_objects.append({
                'name': element.get('name', 'Unknown'),
                'importance': element.get('importance', 5),
                'attention_priority': element.get('attention_priority', 1)
            })
        
        # 注視シーケンスからの物語生成
        if scanpath and len(scanpath) > 0:
            first_fixation = scanpath[0]
            narrative_parts = ["最初に画像の"]
            
            # 最初の注視点の位置解析
            x, y = first_fixation.get('location', [0, 0])
            if x < 0.3:  # 画像の左側
                narrative_parts.append("左側に注目し、")
            elif x > 0.7:  # 画像の右側
                narrative_parts.append("右側に注目し、")
            else:
                narrative_parts.append("中央部分に注目し、")
            
            # 注視パターンの分析
            if len(scanpath) > 3:
                narrative_parts.append("その後複数の重要な領域を順次探索している。")
            else:
                narrative_parts.append("限定的な領域に集中している。")
            
            gaze_narrative = "".join(narrative_parts)
        
        return {
            'main_objects': main_objects,
            'scene_category': scene_context.get('scene_category', 'unknown'),
            'emotional_tone': scene_context.get('emotional_tone', 'neutral'),
            'complexity_level': scene_context.get('complexity_level', 1),
            'gaze_narrative': gaze_narrative,
            'attention_summary': f"{len(main_objects)}個の主要要素が検出され、" +
                               f"{scene_context.get('complexity_level', 1)}レベルの複雑度を持つシーンです。"
        }
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """システム統計の取得"""
        performance_stats = self.integration_algorithm.get_performance_statistics()
        
        return {
            'total_processed_images': self.processing_stats['total_processed'],
            'average_processing_time_ms': performance_stats.get('avg_processing_time', 0),
            'system_confidence': performance_stats.get('avg_confidence', 0),
            'success_rate': performance_stats.get('success_rate', 0),
            'uptime_hours': (time.time() - self.processing_stats['last_update']) / 3600
        }
    
    def _update_processing_stats(self, processing_time: float):
        """処理統計の更新"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['last_update'] = time.time()
    
    def _create_error_result(self, error_message: str, processing_time: float) -> Dict[str, Any]:
        """エラー結果の生成"""
        return {
            'results': {
                'primary_saliency_map': None,
                'fixation_sequence': [],
                'attention_distribution': {
                    'deepgaze_contribution': 0,
                    'sam2_contribution': 0,
                    'llm_contribution': 0
                },
                'semantic_interpretation': {
                    'main_objects': [],
                    'gaze_narrative': "エラーが発生しました",
                    'error_message': error_message
                }
            },
            'performance_metrics': {
                'processing_time_ms': processing_time * 1000,
                'confidence_score': 0.0
            },
            'metadata': {
                'success': False,
                'error': error_message,
                'timestamp': time.time()
            }
        }
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """実行時設定更新"""
        self.config_manager.update(config_updates)
        self.config = self.config_manager.config
        
        # 統合アルゴリズムの設定更新
        self.integration_algorithm.update_configuration(config_updates)
        
        logger.info("System configuration updated")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータスの取得"""
        return {
            'system_version': '2.0',
            'components_status': {
                'deepgaze': 'active',
                'sam2': 'active', 
                'llm': 'active',
                'integration': 'active'
            },
            'performance_stats': self._get_system_stats(),
            'configuration': {
                'debug_mode': self.config_manager.is_debug_mode(),
                'devices': {
                    'deepgaze': self.config_manager.get_device_config('deepgaze'),
                    'sam2': self.config_manager.get_device_config('sam2'),
                    'llm': self.config_manager.get_device_config('llm')
                }
            },
            'last_update': time.time()
        }
    
    async def cleanup(self):
        """システムのクリーンアップ"""
        try:
            # 各コンポーネントのクリーンアップ（実装により異なる）
            logger.info("System cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")