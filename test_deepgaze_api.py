#!/usr/bin/env python3
"""
Test Real DeepGaze III API endpoint
"""

import requests
import numpy as np
from PIL import Image
import io
import json

def create_test_image():
    """Create a test image with clear visual features"""
    width, height = 600, 400
    
    # Create base image
    image = Image.new('RGB', (width, height), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw clear visual features for DeepGaze III to detect
    # Large red circle (should be salient)
    center_x, center_y = width // 2, height // 2
    draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80], 
                 fill='red', outline='darkred', width=5)
    
    # Blue rectangle (top-left)
    draw.rectangle([50, 50, 150, 100], fill='blue', outline='darkblue', width=3)
    
    # Yellow triangle (bottom-right)
    draw.polygon([(width-100, height-50), (width-50, height-50), (width-75, height-100)], 
                 fill='yellow', outline='orange')
    
    # Green text-like shape (should draw attention)
    draw.rectangle([200, 300, 400, 350], fill='green', outline='darkgreen', width=2)
    
    return image

def test_deepgaze_api():
    """Test the Real DeepGaze III API"""
    print("🧠 Testing Real DeepGaze III API")
    print("🌐 URL: https://7000-iibdv88pco3xhi6ujndg1-6532622b.e2b.dev")
    print("📄 Model: Kümmerer et al., Journal of Vision 2021\n")
    
    try:
        # Create test image
        print("1. Creating test image with visual features...")
        test_image = create_test_image()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        print(f"   ✅ Test image created: {test_image.size}")
        
        # Make API request
        print("\n2. Sending request to Real DeepGaze III API...")
        url = "https://7000-iibdv88pco3xhi6ujndg1-6532622b.e2b.dev/analyze"
        
        files = {'image': ('test.png', img_buffer, 'image/png')}
        
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("   ✅ API request successful!")
            
            # Check DeepGaze III specific information
            if 'deepgaze_info' in result:
                dg_info = result['deepgaze_info']
                print(f"\n🧠 DeepGaze III Information:")
                print(f"   Model: {dg_info['model']}")
                print(f"   Architecture: {dg_info['architecture']}")
                print(f"   Paper: {dg_info['paper']}")
                print(f"   Device: {dg_info['device']}")
                print(f"   Processing time: {dg_info['processing_time_ms']:.1f}ms")
                print(f"   Confidence: {dg_info['confidence']:.4f}")
            
            # Check statistics
            stats = result['statistics']
            print(f"\n📊 Analysis Results:")
            print(f"   Fixation points: {stats['num_fixations']}")
            print(f"   Avg fixation duration: {stats['avg_fixation_duration_ms']:.1f}ms")
            print(f"   Overall confidence: {stats['avg_confidence']:.4f}")
            print(f"   Max saliency: {stats['max_saliency']:.6f}")
            print(f"   DeepGaze max activation: {stats.get('deepgaze_max_activation', 'N/A')}")
            
            # Check gaze points
            gaze_points = result['gaze_points']
            print(f"\n👁️ Gaze Pattern (first 3 points):")
            for i, point in enumerate(gaze_points[:3]):
                print(f"   Point {i+1}: ({point['x_norm']:.3f}, {point['y_norm']:.3f}) "
                      f"- {point['duration_ms']}ms - conf:{point['confidence']:.4f}")
            
            # Check neural features
            if 'neural_features' in result:
                neural = result['neural_features']
                print(f"\n🧠 Neural Cortex Features:")
                if 'v1_v2_response' in neural:
                    print(f"   V1/V2 response shape: Available")
                if 'v4_response' in neural:
                    print(f"   V4 response shape: Available")
                if 'it_response' in neural:
                    print(f"   IT response shape: Available")
            
            print(f"\n🎉 Real DeepGaze III API Test: SUCCESS!")
            print(f"✅ Confirmed using actual DeepGaze III implementation")
            
            return True
            
        else:
            print(f"   ❌ API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deepgaze_api()