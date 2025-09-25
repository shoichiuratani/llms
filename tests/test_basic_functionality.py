#!/usr/bin/env python3
"""
基本機能テスト

統合視覚的注意システムの基本機能をテストします。
"""

import sys
import os
import pytest
import numpy as np
import asyncio
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.utils.validation import InputValidator, ValidationError
from src.utils.image_utils import ImagePreprocessor


class TestConfigManager:
    """設定管理のテスト"""
    
    def test_config_initialization(self):
        """設定の初期化テスト"""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert isinstance(config_manager.config, dict)
    
    def test_config_get(self):
        """設定取得のテスト"""
        config_manager = ConfigManager()
        
        # 既存キーの取得
        system_config = config_manager.get('system')
        assert isinstance(system_config, dict)
        
        # ドット記法での取得
        version = config_manager.get('system.version')
        assert version is not None
        
        # デフォルト値の取得
        nonexistent = config_manager.get('nonexistent.key', 'default')
        assert nonexistent == 'default'
    
    def test_config_update(self):
        """設定更新のテスト"""
        config_manager = ConfigManager()
        
        original_value = config_manager.get('system.debug', False)
        
        # 設定の更新
        config_manager.update({
            'system': {
                'debug': not original_value
            }
        })
        
        # 更新の確認
        updated_value = config_manager.get('system.debug')
        assert updated_value == (not original_value)
    
    def test_device_config(self):
        """デバイス設定のテスト"""
        config_manager = ConfigManager()
        
        # デバイス設定の取得
        deepgaze_device = config_manager.get_device_config('deepgaze')
        assert deepgaze_device in ['cpu', 'cuda']
        
        sam2_device = config_manager.get_device_config('sam2')
        assert sam2_device in ['cpu', 'cuda']


class TestInputValidator:
    """入力検証のテスト"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.validator = InputValidator()
    
    def test_valid_image_validation(self):
        """正常な画像の検証テスト"""
        # 正常なカラー画像
        valid_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert self.validator.validate_image(valid_image) == True
        
        # 正常なグレースケール画像
        valid_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        assert self.validator.validate_image(valid_gray) == True
    
    def test_invalid_image_validation(self):
        """不正な画像の検証テスト"""
        # 小さすぎる画像
        with pytest.raises(ValidationError):
            small_image = np.zeros((10, 10, 3), dtype=np.uint8)
            self.validator.validate_image(small_image)
        
        # 不正な次元
        with pytest.raises(ValidationError):
            invalid_dims = np.zeros((100, 100, 100, 3), dtype=np.uint8)
            self.validator.validate_image(invalid_dims)
        
        # 不正なチャンネル数
        with pytest.raises(ValidationError):
            invalid_channels = np.zeros((100, 100, 5), dtype=np.uint8)
            self.validator.validate_image(invalid_channels)
    
    def test_saliency_map_validation(self):
        """顕著性マップの検証テスト"""
        # 正常な顕著性マップ
        valid_saliency = np.random.rand(480, 640).astype(np.float32)
        assert self.validator.validate_saliency_map(valid_saliency) == True
        
        # 不正な次元
        with pytest.raises(ValidationError):
            invalid_saliency = np.random.rand(480, 640, 3)
            self.validator.validate_saliency_map(invalid_saliency)
        
        # NaN値を含む
        with pytest.raises(ValidationError):
            nan_saliency = np.full((100, 100), np.nan)
            self.validator.validate_saliency_map(nan_saliency)
    
    def test_fixation_sequence_validation(self):
        """注視点シーケンスの検証テスト"""
        # 正常な注視点シーケンス
        valid_fixations = [
            {'location': [0.5, 0.5], 'duration': 250, 'timestamp': 0},
            {'location': [0.3, 0.7], 'duration': 300, 'timestamp': 250}
        ]
        assert self.validator.validate_fixation_sequence(valid_fixations) == True
        
        # 空のシーケンス
        assert self.validator.validate_fixation_sequence([]) == True
        
        # 不正なフォーマット
        with pytest.raises(ValidationError):
            invalid_fixations = [
                {'location': [0.5], 'duration': 250, 'timestamp': 0}  # 位置が1次元
            ]
            self.validator.validate_fixation_sequence(invalid_fixations)
    
    def test_analysis_config_validation(self):
        """分析設定の検証テスト"""
        # 正常な設定
        valid_config = {
            'use_deepgaze': True,
            'use_sam2': True,
            'use_llm': False,
            'output_resolution': 'high'
        }
        assert self.validator.validate_analysis_config(valid_config) == True
        
        # 不正な解像度設定
        with pytest.raises(ValidationError):
            invalid_config = {'output_resolution': 'ultra_high'}
            self.validator.validate_analysis_config(invalid_config)
    
    def test_user_context_validation(self):
        """ユーザーコンテキストの検証テスト"""
        # 正常なコンテキスト
        valid_context = {
            'task': 'free_viewing',
            'expertise_level': 'general',
            'language': 'ja',
            'viewing_duration_ms': 5000
        }
        assert self.validator.validate_user_context(valid_context) == True
        
        # 不正なタスク
        with pytest.raises(ValidationError):
            invalid_context = {'task': 'invalid_task'}
            self.validator.validate_user_context(invalid_context)


class TestImagePreprocessor:
    """画像前処理のテスト"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.preprocessor = ImagePreprocessor({'max_image_size': [800, 600]})
    
    def test_image_resize(self):
        """画像リサイズのテスト"""
        # 大きな画像
        large_image = np.random.randint(0, 255, (1200, 1600, 3), dtype=np.uint8)
        resized = self.preprocessor.resize_image(large_image)
        
        # サイズが制限内に収まっているか確認
        assert resized.shape[0] <= 600
        assert resized.shape[1] <= 800
        
        # 小さな画像はそのまま
        small_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        unchanged = self.preprocessor.resize_image(small_image)
        assert np.array_equal(small_image, unchanged)
    
    def test_color_space_conversion(self):
        """色空間変換のテスト"""
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # RGB変換
        rgb_image = self.preprocessor.convert_color_space(color_image, 'RGB')
        assert rgb_image.shape == color_image.shape
        
        # グレースケール変換
        gray_image = self.preprocessor.convert_color_space(color_image, 'GRAY')
        assert len(gray_image.shape) == 2
        assert gray_image.shape[:2] == color_image.shape[:2]
    
    def test_image_normalization(self):
        """画像正規化のテスト"""
        # float値の画像（0-1範囲）
        float_image = np.random.rand(100, 100, 3).astype(np.float32)
        normalized = self.preprocessor.normalize_image(float_image)
        assert normalized.dtype == np.uint8
        assert 0 <= normalized.min() and normalized.max() <= 255
        
        # すでにuint8の画像
        uint8_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        unchanged = self.preprocessor.normalize_image(uint8_image)
        assert np.array_equal(uint8_image, unchanged)
    
    def test_base64_encoding_decoding(self):
        """Base64エンコード・デコードのテスト"""
        # テスト画像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # エンコード
        encoded = self.preprocessor.encode_array_to_base64(test_image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # デコード
        decoded = self.preprocessor.decode_base64_to_array(encoded)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == test_image.shape
    
    def test_thumbnail_creation(self):
        """サムネイル生成のテスト"""
        # 大きな画像
        large_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        thumbnail = self.preprocessor.create_thumbnail(large_image, (128, 128))
        
        assert thumbnail.shape == (128, 128, 3)
    
    def test_image_statistics(self):
        """画像統計の取得テスト"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        stats = self.preprocessor.get_image_statistics(test_image)
        
        assert 'shape' in stats
        assert 'dtype' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'channels' in stats
        
        assert stats['shape'] == (100, 100, 3)
        assert stats['channels'] == 3


class TestSystemIntegration:
    """システム統合テスト"""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """システム初期化のテスト"""
        try:
            from src.core.integrated_attention_system import IntegratedAttentionSystem
            
            # システムの初期化
            system = IntegratedAttentionSystem()
            assert system is not None
            
            # 設定の確認
            assert system.config_manager is not None
            assert system.validator is not None
            assert system.image_preprocessor is not None
            
        except ImportError:
            pytest.skip("System components not available")
    
    @pytest.mark.asyncio
    async def test_system_status(self):
        """システムステータスの取得テスト"""
        try:
            from src.core.integrated_attention_system import IntegratedAttentionSystem
            
            system = IntegratedAttentionSystem()
            status = system.get_system_status()
            
            assert isinstance(status, dict)
            assert 'system_version' in status
            assert 'components_status' in status
            
        except ImportError:
            pytest.skip("System components not available")


def create_test_image(size=(480, 640), color=True):
    """テスト用画像の生成"""
    if color:
        return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    else:
        return np.random.randint(0, 255, size, dtype=np.uint8)


def create_test_saliency_map(size=(480, 640)):
    """テスト用顕著性マップの生成"""
    return np.random.rand(*size).astype(np.float32)


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])