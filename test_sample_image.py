#!/usr/bin/env python3
"""
サンプル画像を作成してアプリをテスト
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import os

def create_test_image():
    """テスト用のカラフルな画像を生成"""
    width, height = 600, 400
    
    # 基本画像作成
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 背景グラデーション
    for y in range(height):
        color = int(255 * (1 - y / height))
        draw.line([(0, y), (width, y)], fill=(color, color + 20, 255 - color))
    
    # 図形を描画
    # 大きな円（中央）
    center_x, center_y = width // 2, height // 2
    circle_radius = 80
    draw.ellipse([center_x - circle_radius, center_y - circle_radius,
                  center_x + circle_radius, center_y + circle_radius], 
                 fill='red', outline='darkred', width=3)
    
    # 三角形（左上）
    triangle_points = [(50, 50), (150, 50), (100, 150)]
    draw.polygon(triangle_points, fill='green', outline='darkgreen')
    
    # 長方形（右上）
    draw.rectangle([width - 150, 50, width - 50, 100], 
                   fill='blue', outline='darkblue', width=2)
    
    # 星形（右下）
    star_center = (width - 100, height - 100)
    star_points = []
    for i in range(10):
        angle = i * np.pi / 5
        if i % 2 == 0:
            radius = 40
        else:
            radius = 20
        x = star_center[0] + radius * np.cos(angle - np.pi/2)
        y = star_center[1] + radius * np.sin(angle - np.pi/2)
        star_points.append((x, y))
    
    draw.polygon(star_points, fill='yellow', outline='orange')
    
    # テキスト（左下）
    try:
        draw.text((50, height - 80), "Visual Attention", fill='black', 
                 anchor='ls')
        draw.text((50, height - 50), "Test Image", fill='black', 
                 anchor='ls')
    except:
        # フォントがない場合はスキップ
        pass
    
    return image

def save_test_image():
    """テスト画像を保存"""
    image = create_test_image()
    
    # uploadsディレクトリに保存
    os.makedirs('uploads', exist_ok=True)
    path = 'uploads/test_sample.jpg'
    image.save(path, 'JPEG', quality=95)
    
    print(f"✓ テスト画像を作成: {path}")
    print(f"  サイズ: {image.size}")
    return path

def test_api_endpoint(image_path):
    """API エンドポイントをテスト"""
    url = "https://5000-iibdv88pco3xhi6ujndg1-6532622b.e2b.dev/analyze"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("\n🎉 API テスト成功!")
            print(f"  処理時間: {result.get('processing_time_ms', 0):.1f}ms")
            print(f"  注視点数: {result.get('statistics', {}).get('num_fixations', 0)}")
            print(f"  平均注視時間: {result.get('statistics', {}).get('avg_fixation_duration_ms', 0):.1f}ms")
            print(f"  信頼度: {result.get('statistics', {}).get('avg_confidence', 0):.3f}")
            return True
        else:
            print(f"❌ API エラー: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API テストエラー: {e}")
        return False

if __name__ == "__main__":
    print("=== Visual Attention Analyzer テスト ===\n")
    
    # テスト画像作成
    image_path = save_test_image()
    
    # API テスト
    print("\nAPI エンドポイントをテスト中...")
    success = test_api_endpoint(image_path)
    
    if success:
        print("\n🌐 アプリURL: https://5000-iibdv88pco3xhi6ujndg1-6532622b.e2b.dev")
        print("📱 ブラウザでアクセスして画像をアップロードしてください！")
    else:
        print("\n⚠️  API テストに失敗しました。アプリの起動を確認してください。")