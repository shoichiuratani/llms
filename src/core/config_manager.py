"""
設定マネージャー

システム設定の読み込み、管理、検証を行います。
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    システム設定の管理クラス
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = None
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパスの取得"""
        # プロジェクトルートを基準にした相対パス
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "system_config.yaml"
        return str(config_path)
    
    def _load_config(self):
        """設定ファイルの読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self._config = self._get_default_config()
        
        # 環境変数による上書き
        self._override_with_env_vars()
        
        # 設定の検証
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定の取得"""
        return {
            'system': {
                'version': '2.0',
                'debug': False,
                'max_image_size': [1920, 1080],
                'default_output_resolution': 'high'
            },
            'deepgaze': {
                'model_name': 'deepgaze_III',
                'device': 'auto',
                'batch_size': 1,
                'center_bias_enabled': True,
                'temporal_dynamics_enabled': True,
                'gaussian_sigma': 16
            },
            'sam2': {
                'model_checkpoint': 'sam2_hiera_large.pt',
                'model_cfg': 'sam2_hiera_l.yaml',
                'device': 'auto'
            },
            'llm': {
                'provider': 'openai',
                'model_name': 'gpt-4-vision-preview',
                'max_tokens': 4096,
                'temperature': 0.1
            },
            'integration': {
                'modality_weights': {
                    'deepgaze_base': 0.45,
                    'sam2_object': 0.25,
                    'llm_semantic': 0.30
                },
                'temporal_weights': {
                    'bottom_up_dominance_duration': 150,
                    'top_down_emergence_time': 200,
                    'full_integration_time': 500
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'reload': False
            }
        }
    
    def _override_with_env_vars(self):
        """環境変数による設定上書き"""
        # LLM API キー
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self._config.setdefault('llm', {})['api_key'] = openai_key
        
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            if self._config.get('llm', {}).get('provider') == 'anthropic':
                self._config['llm']['api_key'] = anthropic_key
        
        # デバッグモード
        debug_mode = os.getenv('DEBUG', '').lower()
        if debug_mode in ['true', '1', 'yes']:
            self._config.setdefault('system', {})['debug'] = True
        
        # ポート設定
        port = os.getenv('PORT')
        if port and port.isdigit():
            self._config.setdefault('api', {})['port'] = int(port)
    
    def _validate_config(self):
        """設定の検証"""
        # 必須フィールドの確認
        required_sections = ['system', 'deepgaze', 'sam2', 'llm', 'integration']
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing config section: {section}")
                self._config[section] = {}
        
        # 数値範囲の検証
        integration = self._config.get('integration', {})
        modality_weights = integration.get('modality_weights', {})
        
        # 重みの合計確認
        total_weight = sum([
            modality_weights.get('deepgaze_base', 0.45),
            modality_weights.get('sam2_object', 0.25),
            modality_weights.get('llm_semantic', 0.30)
        ])
        
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Modality weights sum to {total_weight}, not 1.0. Will be normalized.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値の取得（ドット記法対応）"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """セクション全体の取得"""
        return self._config.get(section, {})
    
    def update(self, updates: Dict[str, Any]):
        """設定の更新"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
        self._validate_config()
        logger.info("Configuration updated")
    
    def save(self, path: Optional[str] = None):
        """設定の保存"""
        save_path = path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """全設定の取得"""
        return self._config.copy()
    
    def reload(self):
        """設定の再読み込み"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def is_debug_mode(self) -> bool:
        """デバッグモードの確認"""
        return self.get('system.debug', False)
    
    def get_device_config(self, component: str) -> str:
        """コンポーネント別デバイス設定取得"""
        device = self.get(f'{component}.device', 'auto')
        
        if device == 'auto':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        
        return device