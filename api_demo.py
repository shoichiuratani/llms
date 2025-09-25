#!/usr/bin/env python3
"""
統合視覚的注意システム v2.0 - APIデモサーバー

ML依存関係なしで動作する簡化版APIサーバー
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import numpy as np
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Any, Optional
import json

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="統合視覚的注意システム v2.0 API",
    description="DeepGaze III + SAM2 + LLM による統合視覚的注意予測システム",
    version="2.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_mock_saliency_map(image: np.ndarray) -> np.ndarray:
    """模擬的な顕著性マップを作成"""
    h, w = image.shape[:2]
    
    # 中央バイアス + エッジ検出
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    
    # 中央からの距離ベースの顕著性
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    center_bias = 1.0 - (distance_from_center / max_distance)
    
    # 色の変化に基づく顕著性
    gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
    gradient_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gradient_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    
    # 統合顕著性マップ
    saliency = 0.6 * center_bias + 0.4 * (gradient_magnitude / np.max(gradient_magnitude))
    
    return saliency

def generate_fixation_sequence(saliency_map: np.ndarray, duration_ms: int = 5000) -> List[Dict]:
    """注視点シーケンスを生成"""
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

def encode_numpy_to_base64(array: np.ndarray) -> str:
    """NumPy配列をBase64エンコード"""
    # 0-255の範囲にスケール
    scaled_array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    
    # PIL Imageに変換
    pil_image = Image.fromarray(scaled_array, mode='L')
    
    # Base64エンコード
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "統合視覚的注意システム v2.0 API",
        "version": "2.0.0",
        "status": "active",
        "components": {
            "deepgaze": "active (mock)",
            "sam2": "active (mock)",
            "llm": "active (mock)",
            "integration": "active"
        }
    }

@app.get("/api/v2/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system_version": "2.0",
        "components_status": {
            "deepgaze": "operational",
            "sam2": "operational",
            "llm": "operational",
            "integration": "operational"
        }
    }

@app.post("/api/v2/analyze/integrated")
async def analyze_integrated(
    image: UploadFile = File(...),
    analysis_config: Optional[str] = Form(None),
    user_context: Optional[str] = Form(None)
):
    """
    統合視覚的注意分析エンドポイント
    
    DeepGaze III、SAM2、LLMを統合した視覚的注意予測を実行
    """
    start_time = time.time()
    
    try:
        # 画像の読み込み
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # RGB変換
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # NumPy配列に変換
        image_array = np.array(pil_image)
        h, w, c = image_array.shape
        
        # 設定の解析
        config = {}
        if analysis_config:
            config = json.loads(analysis_config)
        
        context = {}
        if user_context:
            context = json.loads(user_context)
        
        # 模擬的な統合処理
        await asyncio.sleep(0.1)  # 処理時間をシミュレート
        
        # 1. DeepGaze III処理 (模擬)
        deepgaze_saliency = create_mock_saliency_map(image_array)
        
        # 2. SAM2処理 (模擬)
        sam2_enhanced = deepgaze_saliency * 1.2
        sam2_enhanced = np.clip(sam2_enhanced, 0, 1)
        
        # 3. LLM処理 (模擬)
        task = context.get('task', 'free_viewing')
        if task == 'face_detection':
            llm_weights = np.ones_like(deepgaze_saliency) * 1.1
        elif task == 'text_reading':
            llm_weights = np.ones_like(deepgaze_saliency) * 0.9
        else:
            llm_weights = np.ones_like(deepgaze_saliency)
        
        # 4. 統合処理
        integrated_saliency = (
            0.45 * deepgaze_saliency +
            0.35 * sam2_enhanced +
            0.20 * llm_weights
        )
        integrated_saliency = integrated_saliency / np.max(integrated_saliency)
        
        # 5. 注視点シーケンス生成
        generate_scanpath = config.get('generate_scanpath', True)
        scanpath = None
        if generate_scanpath:
            duration_ms = context.get('viewing_duration_ms', 5000)
            scanpath = generate_fixation_sequence(integrated_saliency, duration_ms)
        
        # 結果の構築
        processing_time = time.time() - start_time
        
        result = {
            "results": {
                "primary_saliency_map": encode_numpy_to_base64(integrated_saliency),
                "fixation_sequence": scanpath or [],
                "attention_distribution": {
                    "deepgaze_contribution": 0.45,
                    "sam2_contribution": 0.35,
                    "llm_contribution": 0.20
                },
                "semantic_interpretation": {
                    "main_objects": [
                        {"name": "Center Region", "importance": 8, "attention_priority": 1},
                        {"name": "Edge Features", "importance": 6, "attention_priority": 2}
                    ],
                    "scene_category": "general",
                    "emotional_tone": "neutral",
                    "complexity_level": 3,
                    "gaze_narrative": f"画像の解析により{len(scanpath) if scanpath else 0}個の注視点が生成されました。",
                    "attention_summary": f"統合アルゴリズムにより画像の視覚的注意パターンを予測しました。"
                },
                "component_outputs": {
                    "deepgaze_saliency": encode_numpy_to_base64(deepgaze_saliency),
                    "sam2_enhanced": encode_numpy_to_base64(sam2_enhanced),
                    "llm_weights": encode_numpy_to_base64(llm_weights)
                }
            },
            "performance_metrics": {
                "processing_time_ms": processing_time * 1000,
                "confidence_score": 0.85,
                "component_timings": {
                    "deepgaze_ms": 30,
                    "sam2_ms": 25,
                    "llm_ms": 20,
                    "integration_ms": 15
                }
            },
            "metadata": {
                "image_shape": [h, w, c],
                "algorithm_version": "2.0",
                "timestamp": time.time(),
                "system_stats": {
                    "total_processed_images": 1,
                    "average_processing_time_ms": processing_time * 1000,
                    "system_confidence": 0.85,
                    "success_rate": 1.0,
                    "uptime_hours": 0.1
                }
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")

@app.get("/api/v2/system/status")
async def get_system_status():
    """システムステータスの取得"""
    return {
        "system_version": "2.0",
        "components_status": {
            "deepgaze": "active",
            "sam2": "active",
            "llm": "active",
            "integration": "active"
        },
        "performance_stats": {
            "total_processed_images": 0,
            "average_processing_time_ms": 0,
            "system_confidence": 0.85,
            "success_rate": 1.0,
            "uptime_hours": 0.1
        },
        "configuration": {
            "debug_mode": False,
            "devices": {
                "deepgaze": "cpu",
                "sam2": "cpu",
                "llm": "api"
            }
        },
        "last_update": time.time()
    }

@app.post("/api/v2/system/config")
async def update_system_config(config_updates: Dict[str, Any]):
    """システム設定の更新"""
    return {
        "message": "設定更新完了",
        "updated_keys": list(config_updates.keys()),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    print("=== 統合視覚的注意システム v2.0 APIサーバー起動 ===")
    print("📡 サーバーを開始しています...")
    print("🌐 アクセス URL: http://localhost:8000")
    print("📖 API ドキュメント: http://localhost:8000/docs")
    print("💡 ヘルスチェック: http://localhost:8000/api/v2/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)