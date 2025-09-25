#!/usr/bin/env python3
"""
API クライアントデモ

REST API を使用したクライアント側の実装例を示します。
"""

import requests
import base64
import json
import time
import numpy as np
import cv2
from typing import Dict, Any
from PIL import Image
import io


class AttentionSystemAPIClient:
    """視覚的注意システム API クライアント"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def encode_image(self, image: np.ndarray) -> str:
        """画像をBase64エンコード"""
        # numpy配列をPIL Imageに変換
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            response = self.session.get(f"{self.base_url}/api/v2/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        try:
            response = self.session.get(f"{self.base_url}/api/v2/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_image(self, image: np.ndarray,
                     analysis_config: Dict[str, Any] = None,
                     user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """画像分析の実行"""
        try:
            # 画像エンコード
            image_b64 = self.encode_image(image)
            
            # リクエストデータの構築
            request_data = {
                "image": image_b64
            }
            
            if analysis_config:
                request_data["analysis_config"] = analysis_config
            
            if user_context:
                request_data["user_context"] = user_context
            
            # API呼び出し
            response = self.session.post(
                f"{self.base_url}/api/v2/analyze/integrated",
                json=request_data,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {e}",
                "status_code": getattr(e.response, 'status_code', None)
            }
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
    
    def batch_analyze(self, images: list, 
                     analysis_configs: list = None,
                     user_contexts: list = None) -> Dict[str, Any]:
        """バッチ分析"""
        try:
            requests_data = []
            
            for i, image in enumerate(images):
                image_b64 = self.encode_image(image)
                
                request_item = {"image": image_b64}
                
                if analysis_configs and i < len(analysis_configs):
                    request_item["analysis_config"] = analysis_configs[i]
                
                if user_contexts and i < len(user_contexts):
                    request_item["user_context"] = user_contexts[i]
                
                requests_data.append(request_item)
            
            # バッチAPI呼び出し
            response = self.session.post(
                f"{self.base_url}/api/v2/analyze/batch",
                json=requests_data,
                timeout=300  # 5分
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": f"Batch analysis failed: {e}"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        try:
            response = self.session.get(f"{self.base_url}/api/v2/performance/statistics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """設定更新"""
        try:
            request_data = {"config_updates": config_updates}
            
            response = self.session.post(
                f"{self.base_url}/api/v2/config/update",
                json=request_data
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": f"Configuration update failed: {e}"}


def create_sample_image() -> np.ndarray:
    """サンプル画像の生成"""
    # カラフルな幾何学模様
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 背景グラデーション
    for y in range(480):
        for x in range(640):
            image[y, x] = [x // 3, y // 2, (x + y) // 4]
    
    # 図形の追加
    cv2.circle(image, (160, 120), 60, (255, 100, 100), -1)
    cv2.rectangle(image, (300, 200), (500, 350), (100, 255, 100), -1)
    cv2.ellipse(image, (480, 120), (80, 40), 45, 0, 360, (100, 100, 255), -1)
    
    # テキスト
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'ATTENTION DEMO', (200, 400), font, 1, (255, 255, 255), 2)
    
    return image


def demo_basic_api_usage():
    """基本的なAPI使用法のデモ"""
    print("=== 基本的なAPI使用法デモ ===")
    
    # クライアント初期化
    client = AttentionSystemAPIClient()
    
    # ヘルスチェック
    print("1. ヘルスチェック...")
    health = client.health_check()
    print(f"   ステータス: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("   ⚠️ サーバーが利用できません。サーバーを起動してください:")
        print("   python -m src.api.main")
        return
    
    # システムステータス取得
    print("2. システムステータス取得...")
    status = client.get_system_status()
    if 'error' not in status:
        print(f"   システムバージョン: {status.get('system_version', 'unknown')}")
        components = status.get('components_status', {})
        print(f"   コンポーネント: {list(components.keys())}")
    else:
        print(f"   エラー: {status['error']}")
    
    # 画像分析
    print("3. 画像分析実行...")
    test_image = create_sample_image()
    
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
        "viewing_duration_ms": 3000
    }
    
    start_time = time.time()
    result = client.analyze_image(test_image, analysis_config, user_context)
    api_time = time.time() - start_time
    
    if 'error' not in result:
        print(f"   ✓ 分析完了 (API応答時間: {api_time*1000:.2f}ms)")
        
        # 結果の表示
        results = result.get('results', {})
        performance = result.get('performance_metrics', {})
        
        print(f"   処理時間: {performance.get('processing_time_ms', 0):.2f}ms")
        print(f"   信頼度: {performance.get('confidence_score', 0):.3f}")
        
        attention_dist = results.get('attention_distribution', {})
        print("   注意分布:")
        print(f"     DeepGaze: {attention_dist.get('deepgaze_contribution', 0):.1%}")
        print(f"     SAM2: {attention_dist.get('sam2_contribution', 0):.1%}")
        print(f"     LLM: {attention_dist.get('llm_contribution', 0):.1%}")
        
        fixations = results.get('fixation_sequence', [])
        print(f"   注視点数: {len(fixations)}")
        
    else:
        print(f"   エラー: {result['error']}")


def demo_batch_processing():
    """バッチ処理のデモ"""
    print("\n=== バッチ処理デモ ===")
    
    client = AttentionSystemAPIClient()
    
    # 複数の画像を生成
    print("1. テスト画像生成中...")
    images = []
    for i in range(3):
        img = create_sample_image()
        # 各画像を少し変更
        img = cv2.addWeighted(img, 1.0, np.ones_like(img) * (i * 30), 0.3, 0)
        images.append(img)
    
    print(f"   {len(images)}枚の画像を生成")
    
    # バッチ分析設定
    analysis_configs = [
        {"use_deepgaze": True, "use_sam2": True, "use_llm": True},
        {"use_deepgaze": True, "use_sam2": False, "use_llm": True},
        {"use_deepgaze": True, "use_sam2": True, "use_llm": False}
    ]
    
    user_contexts = [
        {"task": "free_viewing"},
        {"task": "face_detection"}, 
        {"task": "text_reading"}
    ]
    
    # バッチ分析実行
    print("2. バッチ分析実行中...")
    start_time = time.time()
    batch_result = client.batch_analyze(images, analysis_configs, user_contexts)
    batch_time = time.time() - start_time
    
    if 'error' not in batch_result:
        print(f"   ✓ バッチ処理完了 (総時間: {batch_time*1000:.2f}ms)")
        
        results = batch_result.get('batch_results', [])
        errors = batch_result.get('errors', [])
        
        print(f"   成功: {len(results)}, エラー: {len(errors)}")
        
        # 各結果の要約
        for i, item in enumerate(results):
            result = item.get('result', {})
            performance = result.get('performance_metrics', {})
            print(f"   画像{i+1}: {performance.get('processing_time_ms', 0):.2f}ms")
        
    else:
        print(f"   エラー: {batch_result['error']}")


def demo_performance_monitoring():
    """パフォーマンス監視のデモ"""
    print("\n=== パフォーマンス監視デモ ===")
    
    client = AttentionSystemAPIClient()
    
    # パフォーマンス統計取得
    print("1. パフォーマンス統計取得...")
    stats = client.get_performance_stats()
    
    if 'error' not in stats:
        print("   ✓ 統計取得完了")
        print(f"   処理済み画像数: {stats.get('num_processed', 0)}")
        print(f"   平均処理時間: {stats.get('avg_processing_time', 0):.2f}ms")
        print(f"   システム信頼度: {stats.get('avg_confidence', 0):.3f}")
        print(f"   成功率: {stats.get('success_rate', 0):.1%}")
        
    else:
        print(f"   エラー: {stats['error']}")


def demo_configuration_update():
    """設定更新のデモ"""
    print("\n=== 設定更新デモ ===")
    
    client = AttentionSystemAPIClient()
    
    # 設定更新
    print("1. 設定更新中...")
    config_updates = {
        "integration": {
            "modality_weights": {
                "deepgaze_base": 0.5,
                "sam2_object": 0.3,
                "llm_semantic": 0.2
            }
        }
    }
    
    update_result = client.update_configuration(config_updates)
    
    if 'error' not in update_result:
        print("   ✓ 設定更新完了")
        print(f"   メッセージ: {update_result.get('message', '')}")
        
    else:
        print(f"   エラー: {update_result['error']}")


def demo_error_handling():
    """エラーハンドリングのデモ"""
    print("\n=== エラーハンドリングデモ ===")
    
    client = AttentionSystemAPIClient()
    
    # 不正な画像データでテスト
    print("1. 不正データテスト...")
    invalid_image = np.zeros((10, 10), dtype=np.uint8)  # 小さすぎる画像
    
    result = client.analyze_image(invalid_image)
    
    if 'error' in result:
        print(f"   ✓ エラー正常検出: {result['error']}")
    else:
        print("   ⚠️ エラーが検出されませんでした")
    
    # 存在しないエンドポイントのテスト
    print("2. 存在しないエンドポイントテスト...")
    try:
        response = client.session.get(f"{client.base_url}/api/v2/nonexistent")
        print(f"   レスポンス: {response.status_code}")
    except Exception as e:
        print(f"   ✓ 期待通りのエラー: {e}")


def main():
    """メイン関数"""
    print("視覚的注意システム API クライアントデモ")
    print("=" * 50)
    
    print("\n注意: このデモを実行する前に、APIサーバーを起動してください:")
    print("python -m src.api.main")
    print("\nサーバー起動後、Enter キーを押して続行...")
    input()
    
    try:
        # 基本的な使用法
        demo_basic_api_usage()
        
        # バッチ処理
        demo_batch_processing()
        
        # パフォーマンス監視
        demo_performance_monitoring()
        
        # 設定更新
        demo_configuration_update()
        
        # エラーハンドリング
        demo_error_handling()
        
        print("\n" + "=" * 50)
        print("全てのデモが完了しました！")
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")
    except Exception as e:
        print(f"\nデモ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()