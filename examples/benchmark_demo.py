#!/usr/bin/env python3
"""
ベンチマークデモ

システムの性能評価とベンチマークテストのデモを実行します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import numpy as np
import time
import json
from typing import List

from src.core.integrated_attention_system import IntegratedAttentionSystem
from src.models.benchmark_suite import BenchmarkSuite
from src.models.validation_metrics import ValidationMetrics


async def benchmark_demo():
    """ベンチマークデモの実行"""
    print("=== 統合視覚的注意システム ベンチマークデモ ===")
    
    # システムの初期化
    print("1. システム初期化中...")
    attention_system = IntegratedAttentionSystem()
    benchmark_suite = BenchmarkSuite(attention_system)
    print("✓ 初期化完了")
    
    # テストデータの準備
    print("2. テストデータ準備中...")
    test_images = generate_diverse_test_images()
    ground_truth_data = generate_ground_truth_data(test_images)
    print(f"✓ {len(test_images)}枚のテスト画像と真値データを準備完了")
    
    # フルベンチマークの実行
    print("3. フルベンチマーク実行中...")
    print("   (この処理には数分かかる場合があります)")
    
    benchmark_results = await benchmark_suite.run_full_benchmark(
        test_images[:5],  # 時間短縮のため5枚のみ
        ground_truth_data[:5]
    )
    
    print("✓ ベンチマーク完了")
    
    # 結果の表示
    print("4. 結果表示中...")
    display_benchmark_results(benchmark_results)
    
    # 結果の保存
    print("5. 結果保存中...")
    saved_path = benchmark_suite.save_benchmark_results(benchmark_results)
    if saved_path:
        print(f"✓ 結果を {saved_path} に保存しました")
    
    return benchmark_results


def generate_diverse_test_images() -> List[np.ndarray]:
    """多様なテスト画像の生成"""
    import cv2
    
    images = []
    
    # 1. 顔画像（簡易版）
    face_img = np.ones((400, 400, 3), dtype=np.uint8) * 220
    cv2.circle(face_img, (200, 200), 120, (255, 200, 180), -1)  # 顔
    cv2.circle(face_img, (170, 170), 15, (50, 50, 50), -1)    # 左目
    cv2.circle(face_img, (230, 170), 15, (50, 50, 50), -1)    # 右目
    cv2.ellipse(face_img, (200, 220), (25, 15), 0, 0, 180, (150, 100, 100), 3)  # 口
    images.append(face_img)
    
    # 2. 自然風景（簡易版）
    nature_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # 空
    nature_img[:200, :] = [135, 206, 235]
    # 地面
    nature_img[200:, :] = [34, 139, 34]
    # 太陽
    cv2.circle(nature_img, (500, 80), 40, (255, 255, 0), -1)
    # 木（簡易版）
    cv2.rectangle(nature_img, (150, 250), (170, 400), (139, 69, 19), -1)  # 幹
    cv2.circle(nature_img, (160, 230), 50, (0, 100, 0), -1)  # 葉
    images.append(nature_img)
    
    # 3. テキスト画像
    text_img = np.ones((300, 500, 3), dtype=np.uint8) * 240
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(text_img, 'VISUAL ATTENTION', (50, 100), font, 1.2, (0, 0, 0), 2)
    cv2.putText(text_img, 'SYSTEM BENCHMARK', (30, 150), font, 1.0, (0, 0, 0), 2)
    cv2.putText(text_img, 'Testing Performance', (60, 200), font, 0.8, (100, 100, 100), 2)
    images.append(text_img)
    
    # 4. 複雑な幾何学模様
    geometry_img = np.zeros((400, 400, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i in range(5):
        center = (80 + i * 60, 100 + i * 40)
        cv2.circle(geometry_img, center, 30 + i * 5, colors[i], -1)
        cv2.rectangle(geometry_img, (center[0]-20, center[1]+50), 
                     (center[0]+20, center[1]+90), colors[i], -1)
    images.append(geometry_img)
    
    # 5. ノイズ画像
    noise_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    # 中央に明確な図形を追加
    cv2.circle(noise_img, (150, 150), 50, (255, 255, 255), -1)
    cv2.circle(noise_img, (150, 150), 25, (0, 0, 0), -1)
    images.append(noise_img)
    
    return images


def generate_ground_truth_data(test_images: List[np.ndarray]) -> List[np.ndarray]:
    """真値データの生成"""
    ground_truths = []
    
    for i, image in enumerate(test_images):
        h, w = image.shape[:2]
        
        if i == 0:  # 顔画像：目と口に高い顕著性
            gt = np.zeros((h, w), dtype=np.float32)
            cv2.circle(gt, (170, 170), 20, 1.0, -1)  # 左目
            cv2.circle(gt, (230, 170), 20, 1.0, -1)  # 右目
            cv2.ellipse(gt, (200, 220), (30, 20), 0, 0, 180, 0.8, -1)  # 口
            
        elif i == 1:  # 自然風景：太陽と木に高い顕著性
            gt = np.zeros((h, w), dtype=np.float32)
            cv2.circle(gt, (500, 80), 50, 0.9, -1)   # 太陽
            cv2.circle(gt, (160, 230), 60, 0.7, -1)  # 木
            
        elif i == 2:  # テキスト：文字領域に高い顕著性
            gt = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(gt, (30, 70), (470, 220), 0.8, -1)
            
        elif i == 3:  # 幾何学：各図形に顕著性
            gt = np.zeros((h, w), dtype=np.float32)
            for j in range(5):
                center = (80 + j * 60, 100 + j * 40)
                cv2.circle(gt, center, 40, 0.6 + j * 0.1, -1)
                
        else:  # ノイズ画像：中央の図形に高い顕著性
            gt = np.zeros((h, w), dtype=np.float32)
            cv2.circle(gt, (150, 150), 60, 0.9, -1)
        
        # ガウシアンスムージング
        gt = cv2.GaussianBlur(gt, (15, 15), 0)
        ground_truths.append(gt)
    
    return ground_truths


def display_benchmark_results(results: dict):
    """ベンチマーク結果の表示"""
    print("\n=== ベンチマーク結果 ===")
    
    # システム情報
    system_info = results.get('system_info', {})
    print(f"システムバージョン: {system_info.get('system_version', 'Unknown')}")
    
    # サマリー
    summary = results.get('summary', {})
    print(f"\n--- 総合評価 ---")
    print(f"全体スコア: {summary.get('overall_score', 0):.3f}")
    print(f"パフォーマンス: {summary.get('performance_score', 0):.3f}")
    print(f"精度: {summary.get('accuracy_score', 0):.3f}")
    print(f"頑健性: {summary.get('robustness_score', 0):.3f}")
    
    # パフォーマンステスト結果
    performance_tests = results.get('performance_tests', {})
    if 'processing_speed' in performance_tests:
        speed = performance_tests['processing_speed']
        print(f"\n--- パフォーマンス ---")
        print(f"平均処理時間: {speed.get('mean_time_ms', 0):.2f} ± {speed.get('std_time_ms', 0):.2f} ms")
        print(f"最小処理時間: {speed.get('min_time_ms', 0):.2f} ms")
        print(f"最大処理時間: {speed.get('max_time_ms', 0):.2f} ms")
        print(f"成功率: {speed.get('success_rate', 0):.1%}")
    
    # コンポーネント別時間
    if 'component_timing' in performance_tests:
        print(f"\n--- コンポーネント別時間 ---")
        for component, timing in performance_tests['component_timing'].items():
            print(f"{component}: {timing.get('mean_time_ms', 0):.2f} ms "
                  f"({timing.get('contribution_ratio', 0):.1%})")
    
    # 精度テスト結果
    accuracy_tests = results.get('accuracy_tests', {})
    if 'overall_metrics' in accuracy_tests:
        print(f"\n--- 精度メトリクス ---")
        metrics = accuracy_tests['overall_metrics']
        
        key_metrics = ['AUC', 'CC', 'NSS', 'KLD']
        for metric in key_metrics:
            if metric in metrics:
                metric_data = metrics[metric]
                print(f"{metric}: {metric_data.get('mean', 0):.4f} "
                      f"± {metric_data.get('std', 0):.4f}")
    
    # コンポーネント比較
    if 'component_comparison' in accuracy_tests:
        print(f"\n--- コンポーネント比較 (AUC) ---")
        comp_comparison = accuracy_tests['component_comparison']
        for component, metrics in comp_comparison.items():
            auc = metrics.get('AUC', 0)
            print(f"{component}: {auc:.4f}")
    
    # 頑健性テスト結果
    robustness_tests = results.get('robustness_tests', {})
    print(f"\n--- 頑健性テスト ---")
    
    robustness_categories = ['noise_robustness', 'brightness_robustness', 'blur_robustness']
    for category in robustness_categories:
        if category in robustness_tests:
            rob_data = robustness_tests[category]
            overall_rob = rob_data.get('overall_robustness', 0)
            print(f"{category.replace('_', ' ').title()}: {overall_rob:.3f}")
    
    # スケーラビリティテスト結果
    scalability_tests = results.get('scalability_tests', {})
    if 'image_size_scaling' in scalability_tests:
        print(f"\n--- 画像サイズ別性能 ---")
        size_scaling = scalability_tests['image_size_scaling']
        
        for size_key, perf_data in size_scaling.items():
            mean_time = perf_data.get('mean_time_ms', float('inf'))
            success_rate = perf_data.get('success_rate', 0)
            
            if mean_time != float('inf'):
                print(f"{size_key}: {mean_time:.2f} ms (成功率: {success_rate:.1%})")
            else:
                print(f"{size_key}: 処理失敗")
    
    # 推奨事項
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\n--- 推奨事項 ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # 強み・弱み
    strengths = summary.get('strengths', [])
    weaknesses = summary.get('weaknesses', [])
    
    if strengths:
        print(f"\n--- システムの強み ---")
        for strength in strengths:
            print(f"✓ {strength}")
    
    if weaknesses:
        print(f"\n--- 改善点 ---")
        for weakness in weaknesses:
            print(f"⚠ {weakness}")


async def quick_performance_test():
    """クイックパフォーマンステスト"""
    print("\n=== クイックパフォーマンステスト ===")
    
    attention_system = IntegratedAttentionSystem()
    
    # シンプルなテスト画像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 100, (255, 0, 0), -1)
    
    # 複数回実行して統計を取る
    times = []
    confidences = []
    
    print("10回の分析を実行中...")
    for i in range(10):
        start_time = time.time()
        
        try:
            result = await attention_system.analyze_integrated(test_image)
            
            processing_time = time.time() - start_time
            times.append(processing_time * 1000)  # ms
            
            performance = result.get('performance_metrics', {})
            confidence = performance.get('confidence_score', 0)
            confidences.append(confidence)
            
            print(f"  実行 {i+1}: {processing_time*1000:.2f}ms, 信頼度: {confidence:.3f}")
            
        except Exception as e:
            print(f"  実行 {i+1}: エラー - {e}")
            times.append(float('inf'))
            confidences.append(0.0)
    
    # 統計計算
    valid_times = [t for t in times if t != float('inf')]
    valid_confidences = [c for c in confidences if c > 0]
    
    if valid_times:
        print(f"\n--- 統計結果 ---")
        print(f"成功率: {len(valid_times)/10:.1%}")
        print(f"平均処理時間: {np.mean(valid_times):.2f} ± {np.std(valid_times):.2f} ms")
        print(f"最小時間: {np.min(valid_times):.2f} ms")
        print(f"最大時間: {np.max(valid_times):.2f} ms")
        
        if valid_confidences:
            print(f"平均信頼度: {np.mean(valid_confidences):.3f} ± {np.std(valid_confidences):.3f}")


async def component_comparison_test():
    """コンポーネント比較テスト"""
    print("\n=== コンポーネント比較テスト ===")
    
    attention_system = IntegratedAttentionSystem()
    
    # テスト画像
    test_image = generate_diverse_test_images()[0]  # 顔画像
    
    # 異なるコンポーネント組み合わせでテスト
    configurations = [
        {"use_deepgaze": True, "use_sam2": False, "use_llm": False, "name": "DeepGaze のみ"},
        {"use_deepgaze": True, "use_sam2": True, "use_llm": False, "name": "DeepGaze + SAM2"},
        {"use_deepgaze": True, "use_sam2": False, "use_llm": True, "name": "DeepGaze + LLM"},
        {"use_deepgaze": True, "use_sam2": True, "use_llm": True, "name": "全統合"}
    ]
    
    results = []
    
    for config in configurations:
        name = config.pop("name")
        print(f"\n{name} をテスト中...")
        
        try:
            start_time = time.time()
            result = await attention_system.analyze_integrated(test_image, config)
            processing_time = time.time() - start_time
            
            performance = result.get('performance_metrics', {})
            confidence = performance.get('confidence_score', 0)
            
            print(f"  処理時間: {processing_time*1000:.2f} ms")
            print(f"  信頼度: {confidence:.3f}")
            
            results.append({
                'name': name,
                'time_ms': processing_time * 1000,
                'confidence': confidence,
                'success': True
            })
            
        except Exception as e:
            print(f"  エラー: {e}")
            results.append({
                'name': name,
                'time_ms': float('inf'),
                'confidence': 0.0,
                'success': False
            })
    
    # 比較結果の表示
    print(f"\n--- 比較結果 ---")
    print(f"{'設定':<20} {'時間(ms)':<10} {'信頼度':<8} {'成功'}")
    print("-" * 50)
    
    for result in results:
        time_str = f"{result['time_ms']:.1f}" if result['time_ms'] != float('inf') else "失敗"
        success_str = "✓" if result['success'] else "✗"
        print(f"{result['name']:<20} {time_str:<10} {result['confidence']:.3f}  {success_str}")


async def main():
    """メイン関数"""
    print("統合視覚的注意システム - ベンチマークデモ")
    print("=" * 60)
    
    try:
        # クイックパフォーマンステスト
        await quick_performance_test()
        
        # コンポーネント比較テスト
        await component_comparison_test()
        
        # フルベンチマーク（オプション）
        print("\n" + "=" * 60)
        response = input("フルベンチマーク（数分かかります）を実行しますか？ (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            await benchmark_demo()
        else:
            print("フルベンチマークをスキップしました。")
        
        print("\n" + "=" * 60)
        print("ベンチマークデモが完了しました！")
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")
    except Exception as e:
        print(f"\nベンチマーク実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())