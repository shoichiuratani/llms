#!/usr/bin/env python3
"""
統合視覚的注意システム v2.0 - 基本デモンストレーション

このデモは、システムのアーキテクチャと主要コンポーネントを
特別な依存関係なしで実演します。
"""

import asyncio
import logging
import numpy as np
from PIL import Image
import yaml
import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_image(width: int = 224, height: int = 224) -> np.ndarray:
    """サンプル画像を作成"""
    # カラフルなグラデーション画像を作成
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 水平グラデーション
    for x in range(width):
        for y in range(height):
            image[y, x, 0] = int(255 * (x / width))  # Red gradient
            image[y, x, 1] = int(255 * (y / height))  # Green gradient
            image[y, x, 2] = 128  # Constant blue
    
    # 中央に円を描画
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 6
    
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    image[mask] = [255, 255, 0]  # Yellow circle
    
    return image

def create_mock_saliency_map(image: np.ndarray) -> np.ndarray:
    """模擬的な顕著性マップを作成"""
    h, w = image.shape[:2]
    
    # 中央バイアス + ガウシアンノイズ
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    
    # 中央からの距離ベースの顕著性
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    
    # 中央バイアス（距離が近いほど高い値）
    center_bias = 1.0 - (distance_from_center / max_distance)
    
    # 色の変化に基づく顕著性
    gray = np.mean(image, axis=2)
    gradient_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gradient_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    
    # 統合顕著性マップ
    saliency = 0.6 * center_bias + 0.4 * (gradient_magnitude / np.max(gradient_magnitude))
    
    # ガウシアンフィルタでスムーシング
    return saliency

def generate_mock_fixation_sequence(saliency_map: np.ndarray, duration_ms: int = 5000) -> list:
    """模擬的な注視点シーケンスを生成"""
    fixations = []
    h, w = saliency_map.shape
    
    # 注視点数の計算（250ms間隔を仮定）
    num_fixations = max(1, duration_ms // 250)
    
    for i in range(num_fixations):
        # 顕著性の高い領域から確率的に選択
        saliency_flat = saliency_map.flatten()
        probabilities = saliency_flat / np.sum(saliency_flat)
        
        # 確率に基づいてピクセルを選択
        chosen_idx = np.random.choice(len(saliency_flat), p=probabilities)
        y_coord, x_coord = np.unravel_index(chosen_idx, saliency_map.shape)
        
        # 正規化座標に変換
        x_norm = x_coord / w
        y_norm = y_coord / h
        
        fixation = {
            'location': [float(x_norm), float(y_norm)],
            'duration_ms': 250 + np.random.randint(-50, 50),
            'timestamp_ms': i * 250,
            'confidence': float(saliency_map[y_coord, x_coord])
        }
        fixations.append(fixation)
    
    return fixations

async def demonstrate_system_architecture():
    """システムアーキテクチャのデモンストレーション"""
    print("=== 統合視覚的注意システム v2.0 デモンストレーション ===\n")
    
    # 1. 設定読み込み
    print("1. システム設定の読み込み")
    config_path = Path("config/system_config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ 設定ファイル読み込み完了: {config_path}")
        print(f"   ✓ DeepGaze設定: {config['deepgaze']['model_name']}")
        print(f"   ✓ SAM2設定: {config['sam2']['model_checkpoint']}")
        print(f"   ✓ LLM設定: {config['llm']['provider']}")
    else:
        print(f"   ⚠ 設定ファイルが見つかりません: {config_path}")
    
    # 2. サンプル画像作成
    print("\n2. サンプル画像の準備")
    sample_image = create_sample_image(224, 224)
    print(f"   ✓ サンプル画像生成完了: {sample_image.shape}")
    
    # サンプル画像を保存
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    pil_image = Image.fromarray(sample_image)
    sample_path = output_dir / "demo_sample.jpg"
    pil_image.save(sample_path)
    print(f"   ✓ サンプル画像保存: {sample_path}")
    
    # 3. 模擬的な処理パイプラインの実行
    print("\n3. 視覚的注意処理パイプラインの実行")
    
    # DeepGaze III (模擬処理)
    print("   🧠 DeepGaze III ボトムアップ処理...")
    await asyncio.sleep(0.1)  # 処理時間をシミュレート
    deepgaze_saliency = create_mock_saliency_map(sample_image)
    print(f"      ✓ 基礎顕著性マップ生成: {deepgaze_saliency.shape}")
    
    # SAM2 オブジェクトベース処理 (模擬処理)
    print("   🎯 SAM2 オブジェクトベース注意...")
    await asyncio.sleep(0.1)
    sam2_enhanced = deepgaze_saliency * 1.2  # 簡単な強化
    sam2_enhanced = np.clip(sam2_enhanced, 0, 1)
    print(f"      ✓ オブジェクト強化顕著性: {sam2_enhanced.shape}")
    
    # LLM トップダウン処理 (模擬処理)
    print("   🤖 LLM トップダウン制御...")
    await asyncio.sleep(0.1)
    task_weights = np.ones_like(deepgaze_saliency) * 0.8
    print(f"      ✓ 意味的重み生成: {task_weights.shape}")
    
    # 4. 統合処理
    print("\n4. 三重統合アルゴリズム")
    # 重み付き統合
    integrated_saliency = (
        0.45 * deepgaze_saliency +
        0.35 * sam2_enhanced +
        0.20 * task_weights
    )
    integrated_saliency = integrated_saliency / np.max(integrated_saliency)
    print(f"   ✓ 統合顕著性マップ: {integrated_saliency.shape}")
    print(f"   ✓ 最大顕著性値: {np.max(integrated_saliency):.3f}")
    print(f"   ✓ 平均顕著性値: {np.mean(integrated_saliency):.3f}")
    
    # 5. 注視点シーケンス生成
    print("\n5. 注視点シーケンス生成")
    fixation_sequence = generate_mock_fixation_sequence(integrated_saliency, 3000)
    print(f"   ✓ 生成された注視点数: {len(fixation_sequence)}")
    print(f"   ✓ 総視聴時間: 3000ms")
    
    # 最初の3つの注視点を表示
    for i, fixation in enumerate(fixation_sequence[:3]):
        x, y = fixation['location']
        duration = fixation['duration_ms']
        confidence = fixation['confidence']
        print(f"      注視点{i+1}: ({x:.3f}, {y:.3f}), {duration}ms, 信頼度{confidence:.3f}")
    
    # 6. 結果の保存と統計
    print("\n6. 結果の統計と保存")
    
    # 統計情報
    stats = {
        'processing_components': {
            'deepgaze_contribution': 0.45,
            'sam2_contribution': 0.35,
            'llm_contribution': 0.20
        },
        'saliency_statistics': {
            'max_value': float(np.max(integrated_saliency)),
            'mean_value': float(np.mean(integrated_saliency)),
            'std_value': float(np.std(integrated_saliency))
        },
        'fixation_statistics': {
            'num_fixations': len(fixation_sequence),
            'total_duration_ms': sum(f['duration_ms'] for f in fixation_sequence),
            'average_confidence': sum(f['confidence'] for f in fixation_sequence) / len(fixation_sequence)
        }
    }
    
    print(f"   ✓ DeepGaze貢献度: {stats['processing_components']['deepgaze_contribution']}")
    print(f"   ✓ SAM2貢献度: {stats['processing_components']['sam2_contribution']}")
    print(f"   ✓ LLM貢献度: {stats['processing_components']['llm_contribution']}")
    print(f"   ✓ 平均注視点信頼度: {stats['fixation_statistics']['average_confidence']:.3f}")
    
    # 結果をファイルに保存
    result_path = output_dir / "demo_results.yaml"
    full_results = {
        'metadata': {
            'system_version': '2.0',
            'demo_timestamp': str(asyncio.get_event_loop().time()),
            'image_shape': list(sample_image.shape),
            'processing_mode': 'mock_demonstration'
        },
        'statistics': stats,
        'fixation_sequence': fixation_sequence[:10]  # 最初の10個のみ保存
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        yaml.dump(full_results, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   ✓ 結果保存完了: {result_path}")
    
    print("\n=== デモンストレーション完了 ===")
    print("\n📊 システム概要:")
    print("   • DeepGaze III: 人間の視覚皮質をモデル化した階層的特徴抽出")
    print("   • SAM2: セグメンテーション強化によるオブジェクトベース注意")
    print("   • LLM: 意味的理解に基づくトップダウン制御")
    print("   • Triple Integration: 3つのモダリティの統合アルゴリズム")
    print("   • Temporal Dynamics: IOR（Inhibition of Return）メカニズム")
    print("\n🎯 実用的な応用:")
    print("   • ウェブページ・UIのユーザビリティ評価")
    print("   • 広告・マーケティング素材の効果分析")
    print("   • 医療画像・放射線画像の読影支援")
    print("   • 自動車・ロボットの視覚システム")
    print("   • 教育・学習効果の測定と改善")

if __name__ == "__main__":
    # デモ実行
    asyncio.run(demonstrate_system_architecture())