#!/usr/bin/env python3
"""
統合視覚的注意システム v2.0 - スタイリッシュWebアプリケーション

顕著性ヒートマップとゲイズプロット分析アプリ
"""

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import time
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI バックエンドを使用しない
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Any, Optional
import cv2

# ディレクトリ設定
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "uploads"

# ディレクトリ作成
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True) 
UPLOADS_DIR.mkdir(exist_ok=True)

# FastAPIアプリケーション
app = FastAPI(title="Visual Attention Analysis", version="2.0.0")

# 静的ファイルとテンプレート
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# カスタムカラーマップ
def create_custom_colormap():
    """カスタム顕著性ヒートマップカラーマップ"""
    colors = ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffcc00', '#ff6b35', '#f7931e', '#ff0000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('saliency', colors, N=n_bins)
    return cmap

def create_saliency_map(image: np.ndarray) -> np.ndarray:
    """高品質な顕著性マップ生成"""
    h, w = image.shape[:2]
    
    # ガウシアン中央バイアス
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    
    # 複数スケールの中央バイアス
    sigma_1 = min(w, h) * 0.3
    sigma_2 = min(w, h) * 0.15
    
    gauss_1 = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma_1**2))
    gauss_2 = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma_2**2))
    
    center_bias = 0.6 * gauss_1 + 0.4 * gauss_2
    
    # 色・エッジ顕著性
    if len(image.shape) == 3:
        # LAB色空間での色顕著性
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        
        # 各チャンネルでの顕著性
        l_saliency = np.abs(l_channel - np.mean(l_channel))
        a_saliency = np.abs(a_channel - np.mean(a_channel))
        b_saliency = np.abs(b_channel - np.mean(b_channel))
        
        color_saliency = (l_saliency + a_saliency + b_saliency) / 3
    else:
        color_saliency = np.abs(image.astype(np.float32) - np.mean(image))
    
    # エッジ検出
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Sobelエッジ検出
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)
    
    # 統合顕著性マップ
    # 正規化
    center_bias = center_bias / np.max(center_bias)
    color_saliency = color_saliency / np.max(color_saliency) if np.max(color_saliency) > 0 else color_saliency
    edge_magnitude = edge_magnitude / np.max(edge_magnitude) if np.max(edge_magnitude) > 0 else edge_magnitude
    edges_norm = edges / 255.0
    
    # 重み付き統合
    saliency = (0.4 * center_bias + 
                0.3 * color_saliency + 
                0.2 * edge_magnitude + 
                0.1 * edges_norm)
    
    # ガウシアンスムージング
    saliency = cv2.GaussianBlur(saliency, (11, 11), 2)
    
    # 最終正規化
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    
    return saliency

def generate_gaze_points(saliency_map: np.ndarray, num_fixations: int = 15) -> List[Dict]:
    """高品質な視線プロット生成"""
    h, w = saliency_map.shape
    fixations = []
    
    # IOR (Inhibition of Return) マスク
    ior_mask = np.ones_like(saliency_map)
    ior_sigma = min(w, h) * 0.05  # IOR影響範囲
    
    for i in range(num_fixations):
        # 現在の顕著性分布（IOR考慮）
        current_saliency = saliency_map * ior_mask
        
        # 確率分布として正規化
        saliency_flat = current_saliency.flatten()
        if np.sum(saliency_flat) == 0:
            probabilities = np.ones_like(saliency_flat) / len(saliency_flat)
        else:
            probabilities = saliency_flat / np.sum(saliency_flat)
        
        # 確率的サンプリング
        chosen_idx = np.random.choice(len(saliency_flat), p=probabilities)
        y_coord, x_coord = np.unravel_index(chosen_idx, saliency_map.shape)
        
        # 注視点詳細
        fixation_duration = np.random.normal(250, 50)  # ms
        fixation_duration = max(150, min(400, fixation_duration))
        
        fixation = {
            'x': int(x_coord),
            'y': int(y_coord), 
            'x_norm': float(x_coord / w),
            'y_norm': float(y_coord / h),
            'duration_ms': int(fixation_duration),
            'timestamp_ms': i * 250,
            'confidence': float(current_saliency[y_coord, x_coord]),
            'order': i + 1
        }
        fixations.append(fixation)
        
        # IOR更新：注視した場所の顕著性を抑制
        y_grid, x_grid = np.ogrid[:h, :w]
        ior_gaussian = np.exp(-((x_grid - x_coord)**2 + (y_grid - y_coord)**2) / (2 * ior_sigma**2))
        ior_mask *= (1 - 0.7 * ior_gaussian)  # 70%抑制
        ior_mask = np.clip(ior_mask, 0.1, 1.0)  # 最小10%残す
    
    return fixations

def create_overlay_visualization(original_image: np.ndarray, saliency_map: np.ndarray, 
                               gaze_points: List[Dict], output_path: str) -> str:
    """スタイリッシュなオーバーレイ画像生成"""
    
    # 高解像度プロット設定
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # オリジナル画像表示
    ax.imshow(original_image)
    
    # 顕著性ヒートマップオーバーレイ
    custom_cmap = create_custom_colormap()
    heatmap = ax.imshow(saliency_map, alpha=0.6, cmap=custom_cmap, 
                       interpolation='gaussian')
    
    # 視線プロット
    if gaze_points:
        # 軌跡線（サッケード）
        x_coords = [p['x'] for p in gaze_points]
        y_coords = [p['y'] for p in gaze_points]
        
        # グラデーション軌跡
        for i in range(len(gaze_points) - 1):
            alpha = 0.3 + 0.4 * (i / len(gaze_points))
            width = 1 + 2 * (i / len(gaze_points))
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   color='cyan', alpha=alpha, linewidth=width, 
                   linestyle='-', zorder=10)
        
        # 注視点円（サイズは注視時間に比例）
        for i, point in enumerate(gaze_points):
            # 外側の輝度リング
            circle_size = 15 + (point['duration_ms'] - 150) / 250 * 20
            outer_circle = patches.Circle((point['x'], point['y']), 
                                        circle_size + 3, 
                                        color='white', alpha=0.8, 
                                        fill=False, linewidth=2, zorder=15)
            ax.add_patch(outer_circle)
            
            # メインの注視点
            main_circle = patches.Circle((point['x'], point['y']), 
                                       circle_size, 
                                       color='red', alpha=0.9, 
                                       fill=True, zorder=20)
            ax.add_patch(main_circle)
            
            # 注視順序番号
            ax.text(point['x'], point['y'], str(point['order']), 
                   ha='center', va='center', fontsize=10, 
                   color='white', weight='bold', zorder=25)
    
    # スタイリング
    ax.set_xlim(0, original_image.shape[1])
    ax.set_ylim(original_image.shape[0], 0)
    ax.axis('off')
    
    # タイトル
    plt.suptitle('Visual Attention Analysis - Saliency & Gaze Overlay', 
                fontsize=16, color='white', y=0.95)
    
    # カラーバー
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, 
                       orientation='vertical')
    cbar.set_label('Saliency Intensity', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none', 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    return output_path

def create_analysis_dashboard(original_image: np.ndarray, saliency_map: np.ndarray, 
                            gaze_points: List[Dict], output_path: str) -> str:
    """分析ダッシュボード画像生成"""
    
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
    fig.patch.set_facecolor('black')
    
    # 1. オリジナル画像
    ax1.imshow(original_image)
    ax1.set_title('Original Image', color='white', fontsize=14)
    ax1.axis('off')
    
    # 2. 顕著性ヒートマップ
    custom_cmap = create_custom_colormap()
    im2 = ax2.imshow(saliency_map, cmap=custom_cmap, interpolation='gaussian')
    ax2.set_title('Saliency Heatmap', color='white', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. 視線パターン
    ax3.imshow(original_image, alpha=0.7)
    if gaze_points:
        x_coords = [p['x'] for p in gaze_points]
        y_coords = [p['y'] for p in gaze_points]
        
        # 軌跡
        ax3.plot(x_coords, y_coords, color='cyan', linewidth=2, alpha=0.8)
        
        # 注視点
        for i, point in enumerate(gaze_points):
            size = 50 + point['duration_ms'] / 10
            ax3.scatter(point['x'], point['y'], s=size, c='red', 
                       alpha=0.8, edgecolors='white', linewidth=2)
            ax3.text(point['x'] + 10, point['y'] - 10, str(point['order']), 
                    color='white', fontsize=10, weight='bold')
    
    ax3.set_title('Gaze Pattern Analysis', color='white', fontsize=14)
    ax3.axis('off')
    
    # 4. 統計グラフ
    if gaze_points:
        durations = [p['duration_ms'] for p in gaze_points]
        confidences = [p['confidence'] for p in gaze_points]
        orders = [p['order'] for p in gaze_points]
        
        # 注視時間の分布
        ax4_twin = ax4.twinx()
        
        bars = ax4.bar(orders, durations, alpha=0.7, color='skyblue', label='Duration (ms)')
        line = ax4_twin.plot(orders, confidences, color='orange', marker='o', 
                            linewidth=2, label='Confidence')
        
        ax4.set_xlabel('Fixation Order', color='white')
        ax4.set_ylabel('Duration (ms)', color='skyblue')
        ax4_twin.set_ylabel('Confidence', color='orange')
        ax4.set_title('Fixation Statistics', color='white', fontsize=14)
        
        # 軸の色設定
        ax4.tick_params(colors='white')
        ax4_twin.tick_params(colors='white')
        ax4.spines['bottom'].set_color('white')
        ax4.spines['left'].set_color('white')
        ax4_twin.spines['right'].set_color('white')
    else:
        ax4.text(0.5, 0.5, 'No Gaze Data', ha='center', va='center', 
                transform=ax4.transAxes, color='white', fontsize=16)
        ax4.axis('off')
    
    plt.suptitle('Visual Attention Analysis Dashboard', 
                fontsize=18, color='white', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none', 
               bbox_inches='tight', dpi=150)
    plt.close()
    
    return output_path

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """メインページ"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    """画像分析エンドポイント"""
    try:
        # 画像読み込み
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # リサイズ（処理速度向上）
        max_size = 800
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        image_array = np.array(pil_image)
        
        # 分析実行
        start_time = time.time()
        
        # 顕著性マップ生成
        saliency_map = create_saliency_map(image_array)
        
        # 視線プロット生成
        gaze_points = generate_gaze_points(saliency_map, num_fixations=12)
        
        # 結果画像生成
        timestamp = int(time.time() * 1000)
        overlay_path = UPLOADS_DIR / f"overlay_{timestamp}.png"
        dashboard_path = UPLOADS_DIR / f"dashboard_{timestamp}.png"
        
        create_overlay_visualization(image_array, saliency_map, gaze_points, 
                                   str(overlay_path))
        create_analysis_dashboard(image_array, saliency_map, gaze_points, 
                                str(dashboard_path))
        
        processing_time = time.time() - start_time
        
        # 統計計算
        avg_duration = np.mean([p['duration_ms'] for p in gaze_points])
        avg_confidence = np.mean([p['confidence'] for p in gaze_points])
        max_saliency = np.max(saliency_map)
        
        return {
            "success": True,
            "processing_time_ms": processing_time * 1000,
            "overlay_image": f"/uploads/{overlay_path.name}",
            "dashboard_image": f"/uploads/{dashboard_path.name}",
            "statistics": {
                "num_fixations": len(gaze_points),
                "avg_fixation_duration_ms": float(avg_duration),
                "avg_confidence": float(avg_confidence),
                "max_saliency": float(max_saliency),
                "image_dimensions": list(image_array.shape)
            },
            "gaze_points": gaze_points[:5]  # 最初の5点のみ返す
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析エラー: {str(e)}")

@app.get("/uploads/{filename}")
async def get_upload_file(filename: str):
    """アップロード画像取得"""
    file_path = UPLOADS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    print("=== 視覚的注意分析アプリケーション起動 ===")
    print("🎨 スタイリッシュWebアプリを開始します...")
    print("🌐 アクセス URL: http://localhost:5000")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)