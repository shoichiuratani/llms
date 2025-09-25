"""
人間の視覚的注意機構に基づく顕著性予測システム v2.0
DeepGaze III・SAM2・LLM統合版

このパッケージは、人間の視覚的注意メカニズムを多層的にモデル化し、
DeepGaze III、SAM2、LLMを統合した高度な顕著性予測システムを提供します。
"""

__version__ = "2.0.0"
__author__ = "Visual Attention Research Team"
__email__ = "team@visual-attention.ai"

from src.core.integrated_attention_system import IntegratedAttentionSystem

__all__ = [
    "IntegratedAttentionSystem",
]