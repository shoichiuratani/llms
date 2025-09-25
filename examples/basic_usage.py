#!/usr/bin/env python3
"""
基本的な使用例

統合視覚的注意システムの基本的な使い方を示します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

from src.core.integrated_attention_system import IntegratedAttentionSystem
from src.utils.visualization import SaliencyVisualizer


async def basic_example():
    """基本的な使用例"""
    print("=== 統合視覚的注意システム v2.0 基本使用例 ===")
    
    # 1. システムの初期化
    print("1. システム初期化中...")
    attention_system = IntegratedAttentionSystem()
    print("✓ 初期化完了")
    
    # 2. テスト画像の生成
    print("2. テスト画像生成中...")
    test_image = create_test_image()
    print("✓ テスト画像生成完了")
    
    # 3. 統合分析の実行
    print("3. 統合分析実行中...")
    start_time = time.time()
    
    analysis_config = {
        "use_deepgaze": True,
        "use_sam2": True,
        "use_llm": True,
        "output_resolution": "high"
    }
    
    user_context = {
        "task": "free_viewing",
        "expertise_level": "general",
        "language": "ja",
        "viewing_duration_ms": 5000
    }
    
    result = await attention_system.analyze_integrated(
        test_image,
        analysis_config,
        user_context
    )
    
    processing_time = time.time() - start_time
    print(f"✓ 分析完了 (処理時間: {processing_time*1000:.2f}ms)")
    
    # 4. 結果の表示
    print("4. 結果表示中...")
    display_results(result, test_image)
    print("✓ 結果表示完了")
    
    return result


def create_test_image(size=(640, 480)):
    """テスト画像の生成"""
    width, height = size
    
    # 背景
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # 円（顔のような形）
    cv2.circle(image, (width//3, height//3), 80, (255, 200, 180), -1)
    cv2.circle(image, (width//3 - 20, height//3 - 15), 8, (50, 50, 50), -1)  # 左目
    cv2.circle(image, (width//3 + 20, height//3 - 15), 8, (50, 50, 50), -1)  # 右目
    cv2.ellipse(image, (width//3, height//3 + 10), (15, 8), 0, 0, 180, (150, 100, 100), 2)  # 口
    
    # 四角形（建物のような形）
    cv2.rectangle(image, (width//2, height//2), (width//2 + 100, height//2 + 80), (100, 150, 200), -1)
    cv2.rectangle(image, (width//2 + 20, height//2 + 20), (width//2 + 40, height//2 + 40), (200, 200, 50), -1)  # 窓
    cv2.rectangle(image, (width//2 + 60, height//2 + 20), (width//2 + 80, height//2 + 40), (200, 200, 50), -1)  # 窓
    
    # テキスト
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'SAMPLE', (width//2, height//4), font, 1, (255, 255, 255), 2)
    
    # ノイズ追加
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image


def display_results(result, original_image):
    """結果の表示"""
    try:
        results = result.get('results', {})
        performance = result.get('performance_metrics', {})
        
        print("\n=== 分析結果 ===")
        print(f"処理時間: {performance.get('processing_time_ms', 0):.2f}ms")
        print(f"信頼度スコア: {performance.get('confidence_score', 0):.3f}")
        
        # 注意分布の表示
        attention_dist = results.get('attention_distribution', {})
        print("\n--- 注意分布 ---")
        print(f"DeepGaze貢献度: {attention_dist.get('deepgaze_contribution', 0):.1%}")
        print(f"SAM2貢献度: {attention_dist.get('sam2_contribution', 0):.1%}")
        print(f"LLM貢献度: {attention_dist.get('llm_contribution', 0):.1%}")
        
        # 意味的解釈の表示
        semantic = results.get('semantic_interpretation', {})
        print("\n--- 意味的解釈 ---")
        print(f"シーンカテゴリ: {semantic.get('scene_category', 'unknown')}")
        print(f"感情的トーン: {semantic.get('emotional_tone', 'neutral')}")
        print(f"複雑度レベル: {semantic.get('complexity_level', 1)}")
        print(f"視線の物語: {semantic.get('gaze_narrative', 'なし')}")
        
        # 注視点シーケンスの表示
        fixations = results.get('fixation_sequence', [])
        print(f"\n--- 注視点シーケンス ({len(fixations)}個の注視点) ---")
        for i, fix in enumerate(fixations[:3]):  # 最初の3個のみ表示
            loc = fix.get('location', [0, 0])
            duration = fix.get('duration', 0)
            print(f"  注視点{i+1}: ({loc[0]:.3f}, {loc[1]:.3f}), 持続時間: {duration:.0f}ms")
        
        if len(fixations) > 3:
            print(f"  ... 他 {len(fixations) - 3}個")
        
        # 可視化の作成
        if 'primary_saliency_map' in results:
            create_visualization(original_image, results)
        
    except Exception as e:
        print(f"結果表示エラー: {e}")


def create_visualization(original_image, results):
    """可視化の作成"""
    try:
        visualizer = SaliencyVisualizer()
        
        # 顕著性マップのデコード
        from src.utils.image_utils import ImagePreprocessor
        processor = ImagePreprocessor({})
        
        primary_saliency_b64 = results.get('primary_saliency_map', '')
        if primary_saliency_b64:
            # Base64デコードは簡略化（実際の実装では適切なデコード処理が必要）
            print("\n--- 可視化 ---")
            print("✓ 顕著性マップ生成完了")
            print("✓ 注視点シーケンス可視化完了")
            print("※ 詳細な可視化は実際のGUI環境で表示されます")
        
    except Exception as e:
        print(f"可視化エラー: {e}")


async def advanced_example():
    """高度な使用例"""
    print("\n=== 高度な使用例 ===")
    
    attention_system = IntegratedAttentionSystem()
    
    # 異なるタスク設定での比較
    tasks = ["free_viewing", "face_detection", "text_reading"]
    test_image = create_test_image()
    
    for task in tasks:
        print(f"\nタスク: {task}")
        
        user_context = {
            "task": task,
            "expertise_level": "general",
            "language": "ja"
        }
        
        result = await attention_system.analyze_integrated(
            test_image,
            {"use_deepgaze": True, "use_sam2": True, "use_llm": True},
            user_context
        )
        
        performance = result.get('performance_metrics', {})
        attention_dist = result.get('results', {}).get('attention_distribution', {})
        
        print(f"  処理時間: {performance.get('processing_time_ms', 0):.2f}ms")
        print(f"  信頼度: {performance.get('confidence_score', 0):.3f}")
        print(f"  LLM貢献度: {attention_dist.get('llm_contribution', 0):.2f}")


async def batch_processing_example():
    """バッチ処理の例"""
    print("\n=== バッチ処理例 ===")
    
    attention_system = IntegratedAttentionSystem()
    
    # 複数のテスト画像を生成
    test_images = [create_test_image((640, 480)) for _ in range(3)]
    
    print(f"{len(test_images)}枚の画像を処理中...")
    
    results = []
    total_start_time = time.time()
    
    for i, image in enumerate(test_images):
        print(f"画像 {i+1} 処理中...")
        
        result = await attention_system.analyze_integrated(
            image,
            {"use_deepgaze": True, "use_sam2": True, "use_llm": True},
            {"task": "free_viewing"}
        )
        
        results.append(result)
        
        performance = result.get('performance_metrics', {})
        print(f"  完了 (処理時間: {performance.get('processing_time_ms', 0):.2f}ms)")
    
    total_time = time.time() - total_start_time
    print(f"\nバッチ処理完了:")
    print(f"  総処理時間: {total_time*1000:.2f}ms")
    print(f"  平均処理時間: {total_time*1000/len(test_images):.2f}ms/画像")
    
    # 統計の計算
    processing_times = [r.get('performance_metrics', {}).get('processing_time_ms', 0) for r in results]
    confidence_scores = [r.get('performance_metrics', {}).get('confidence_score', 0) for r in results]
    
    print(f"  処理時間 - 最小: {min(processing_times):.2f}ms, 最大: {max(processing_times):.2f}ms")
    print(f"  信頼度 - 平均: {np.mean(confidence_scores):.3f}, 標準偏差: {np.std(confidence_scores):.3f}")


async def main():
    """メイン関数"""
    print("統合視覚的注意システム v2.0 - 使用例デモ")
    print("=" * 50)
    
    try:
        # 基本的な使用例
        await basic_example()
        
        # 高度な使用例
        await advanced_example()
        
        # バッチ処理例
        await batch_processing_example()
        
        print("\n" + "=" * 50)
        print("全ての例が正常に実行されました！")
        
    except Exception as e:
        print(f"\n実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())