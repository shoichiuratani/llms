"""
可視化ユーティリティ

顕著性マップや注視点シーケンスの可視化機能を提供します。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from PIL import Image, ImageDraw, ImageFont
import base64
import io

logger = logging.getLogger(__name__)


class SaliencyVisualizer:
    """
    顕著性マップと注視点の可視化クラス
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.colormap = self.config.get('colormap', 'jet')
        self.alpha = self.config.get('alpha', 0.6)
        self.figsize = self.config.get('figsize', (12, 8))
        
        # フォント設定
        try:
            self.font = ImageFont.load_default()
        except:
            self.font = None
    
    def visualize_saliency_overlay(self, image: np.ndarray, 
                                 saliency_map: np.ndarray,
                                 title: str = "Saliency Map Overlay") -> np.ndarray:
        """
        画像上に顕著性マップをオーバーレイ
        
        Args:
            image: 背景画像
            saliency_map: 顕著性マップ
            title: タイトル
            
        Returns:
            オーバーレイ画像
        """
        try:
            # 画像サイズの調整
            h, w = image.shape[:2]
            if saliency_map.shape != (h, w):
                saliency_resized = cv2.resize(saliency_map, (w, h))
            else:
                saliency_resized = saliency_map
            
            # 顕著性マップの正規化
            saliency_norm = (saliency_resized - saliency_resized.min()) / \
                          (saliency_resized.max() - saliency_resized.min() + 1e-8)
            
            # カラーマップの適用
            saliency_colored = cv2.applyColorMap(
                (saliency_norm * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            
            # 背景画像の準備
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image.copy()
            
            # オーバーレイ
            overlay = cv2.addWeighted(
                image_bgr, 1 - self.alpha, 
                saliency_colored, self.alpha, 0
            )
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating saliency overlay: {e}")
            return image.copy()
    
    def visualize_fixation_sequence(self, image: np.ndarray,
                                  fixations: List[Dict[str, Any]],
                                  show_order: bool = True,
                                  show_duration: bool = True) -> np.ndarray:
        """
        注視点シーケンスの可視化
        
        Args:
            image: 背景画像
            fixations: 注視点リスト
            show_order: 順番の表示
            show_duration: 持続時間の表示
            
        Returns:
            可視化画像
        """
        try:
            h, w = image.shape[:2]
            
            # PIL Imageに変換
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            draw = ImageDraw.Draw(pil_image)
            
            # 注視点の描画
            for i, fixation in enumerate(fixations):
                location = fixation['location']
                duration = fixation.get('duration', 250)
                
                # 画像座標に変換
                x = int(location[0] * w)
                y = int(location[1] * h)
                
                # 円のサイズ（持続時間に比例）
                if show_duration:
                    radius = max(5, min(30, int(duration / 50)))
                else:
                    radius = 10
                
                # 色（時間経過で変化）
                color_intensity = int(255 * (i / max(1, len(fixations) - 1)))
                color = (255 - color_intensity, color_intensity, 100)
                
                # 注視点の描画
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=(255, 255, 255), width=2)
                
                # 順番の表示
                if show_order and self.font:
                    draw.text((x+radius+5, y-radius), str(i+1), 
                            fill=(255, 255, 255), font=self.font)
            
            # サッカードの描画
            if len(fixations) > 1:
                for i in range(len(fixations) - 1):
                    start_loc = fixations[i]['location']
                    end_loc = fixations[i + 1]['location']
                    
                    start_x = int(start_loc[0] * w)
                    start_y = int(start_loc[1] * h)
                    end_x = int(end_loc[0] * w)
                    end_y = int(end_loc[1] * h)
                    
                    # 矢印の描画
                    draw.line([(start_x, start_y), (end_x, end_y)], 
                            fill=(255, 255, 0), width=2)
            
            # NumPy配列に変換して返す
            result_array = np.array(pil_image)
            if len(image.shape) == 3:
                result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            return result_array
            
        except Exception as e:
            logger.error(f"Error visualizing fixation sequence: {e}")
            return image.copy()
    
    def create_comparison_plot(self, components: Dict[str, np.ndarray],
                             titles: List[str] = None) -> np.ndarray:
        """
        複数の顕著性マップの比較プロット
        
        Args:
            components: 各コンポーネントの顕著性マップ辞書
            titles: 各サブプロットのタイトル
            
        Returns:
            比較プロット画像
        """
        try:
            n_components = len(components)
            if n_components == 0:
                return np.zeros((400, 400, 3), dtype=np.uint8)
            
            # サブプロットの配置計算
            cols = min(3, n_components)
            rows = (n_components + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
            if n_components == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            # 各コンポーネントの描画
            for i, (name, saliency_map) in enumerate(components.items()):
                ax = axes[i]
                
                # 正規化
                normalized = (saliency_map - saliency_map.min()) / \
                           (saliency_map.max() - saliency_map.min() + 1e-8)
                
                # ヒートマップ表示
                im = ax.imshow(normalized, cmap=self.colormap, aspect='auto')
                
                # タイトル設定
                title = titles[i] if titles and i < len(titles) else name
                ax.set_title(title)
                ax.axis('off')
                
                # カラーバー
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            # 余分なサブプロットを非表示
            for i in range(n_components, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # 画像として保存
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # NumPy配列に変換
            pil_image = Image.open(buffer)
            result_array = np.array(pil_image)
            
            plt.close(fig)
            
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def create_attention_statistics_plot(self, stats: Dict[str, Any]) -> np.ndarray:
        """
        注意統計のプロット
        
        Args:
            stats: 注意統計情報
            
        Returns:
            統計プロット画像
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
            
            # 1. コンポーネント貢献度（円グラフ）
            if 'attention_distribution' in stats:
                distribution = stats['attention_distribution']
                labels = ['DeepGaze', 'SAM2', 'LLM']
                values = [
                    distribution.get('deepgaze_contribution', 0),
                    distribution.get('sam2_contribution', 0),
                    distribution.get('llm_contribution', 0)
                ]
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                
                ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
                ax1.set_title('Component Contributions')
            
            # 2. 処理時間（棒グラフ）
            if 'component_timings' in stats:
                timings = stats['component_timings']
                components = list(timings.keys())
                times = list(timings.values())
                
                ax2.bar(components, times, color=['#ff9999', '#66b3ff', '#99ff99'])
                ax2.set_title('Processing Times (ms)')
                ax2.set_ylabel('Time (ms)')
                plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # 3. 注視点統計
            if 'fixation_sequence' in stats and stats['fixation_sequence']:
                fixations = stats['fixation_sequence']
                durations = [f.get('duration', 0) for f in fixations]
                
                ax3.hist(durations, bins=10, color='skyblue', alpha=0.7)
                ax3.set_title('Fixation Duration Distribution')
                ax3.set_xlabel('Duration (ms)')
                ax3.set_ylabel('Frequency')
            
            # 4. システムステータス
            if 'system_stats' in stats:
                system_stats = stats['system_stats']
                metrics = ['Confidence', 'Success Rate', 'Avg Time']
                values = [
                    system_stats.get('system_confidence', 0) * 100,
                    system_stats.get('success_rate', 0) * 100,
                    system_stats.get('average_processing_time_ms', 0) / 10  # スケール調整
                ]
                
                bars = ax4.bar(metrics, values, color=['green', 'blue', 'orange'])
                ax4.set_title('System Performance')
                ax4.set_ylabel('Score / Time (scaled)')
                
                # 値をバーの上に表示
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,\n                            f'{value:.1f}', ha='center', va='bottom')\n            \n            plt.tight_layout()\n            \n            # 画像として保存\n            buffer = io.BytesIO()\n            plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')\n            buffer.seek(0)\n            \n            # NumPy配列に変換\n            pil_image = Image.open(buffer)\n            result_array = np.array(pil_image)\n            \n            plt.close(fig)\n            \n            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)\n            \n        except Exception as e:\n            logger.error(f\"Error creating statistics plot: {e}\")\n            return np.zeros((400, 400, 3), dtype=np.uint8)\n    \n    def create_scanpath_heatmap(self, fixations: List[Dict[str, Any]], \n                              image_shape: Tuple[int, int]) -> np.ndarray:\n        \"\"\"\n        注視点から密度ヒートマップを生成\n        \n        Args:\n            fixations: 注視点リスト\n            image_shape: 画像サイズ (height, width)\n            \n        Returns:\n            密度ヒートマップ\n        \"\"\"\n        try:\n            h, w = image_shape\n            heatmap = np.zeros((h, w), dtype=np.float32)\n            \n            for fixation in fixations:\n                location = fixation['location']\n                duration = fixation.get('duration', 250)\n                \n                # 画像座標に変換\n                x = int(location[0] * w)\n                y = int(location[1] * h)\n                \n                # 境界チェック\n                x = max(0, min(w-1, x))\n                y = max(0, min(h-1, y))\n                \n                # ガウシアン分布で密度を追加\n                sigma = max(10, duration / 50)  # 持続時間に応じたサイズ\n                \n                # ガウシアンカーネルの生成\n                kernel_size = int(sigma * 3)\n                if kernel_size % 2 == 0:\n                    kernel_size += 1\n                \n                y_start = max(0, y - kernel_size // 2)\n                y_end = min(h, y + kernel_size // 2 + 1)\n                x_start = max(0, x - kernel_size // 2)\n                x_end = min(w, x + kernel_size // 2 + 1)\n                \n                # ガウシアン重みの計算\n                yy, xx = np.ogrid[y_start:y_end, x_start:x_end]\n                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))\n                \n                # 密度の加算（持続時間による重み付け）\n                weight = duration / 250.0  # 正規化\n                heatmap[y_start:y_end, x_start:x_end] += gaussian * weight\n            \n            # 正規化\n            if heatmap.max() > 0:\n                heatmap = heatmap / heatmap.max()\n            \n            return heatmap\n            \n        except Exception as e:\n            logger.error(f\"Error creating scanpath heatmap: {e}\")\n            return np.zeros(image_shape, dtype=np.float32)\n    \n    def export_visualization_report(self, analysis_result: Dict[str, Any],\n                                  output_path: str = None) -> str:\n        \"\"\"\n        可視化レポートの生成\n        \n        Args:\n            analysis_result: 分析結果\n            output_path: 出力パス\n            \n        Returns:\n            レポートのBase64エンコード文字列\n        \"\"\"\n        try:\n            # レポート用のHTML生成\n            html_content = self._generate_html_report(analysis_result)\n            \n            if output_path:\n                with open(output_path, 'w', encoding='utf-8') as f:\n                    f.write(html_content)\n            \n            # Base64エンコード\n            html_bytes = html_content.encode('utf-8')\n            b64_content = base64.b64encode(html_bytes).decode('utf-8')\n            \n            return b64_content\n            \n        except Exception as e:\n            logger.error(f\"Error exporting visualization report: {e}\")\n            return \"\"\n    \n    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:\n        \"\"\"HTML レポートの生成\"\"\"\n        results = analysis_result.get('results', {})\n        performance = analysis_result.get('performance_metrics', {})\n        metadata = analysis_result.get('metadata', {})\n        \n        html = f\"\"\"\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>Visual Attention Analysis Report</title>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}\n                .metric {{ margin: 10px 0; }}\n                .image {{ max-width: 500px; margin: 10px; }}\n            </style>\n        </head>\n        <body>\n            <h1>Visual Attention Analysis Report</h1>\n            \n            <div class=\"section\">\n                <h2>Analysis Summary</h2>\n                <div class=\"metric\">Processing Time: {performance.get('processing_time_ms', 0):.2f} ms</div>\n                <div class=\"metric\">Confidence Score: {performance.get('confidence_score', 0):.3f}</div>\n                <div class=\"metric\">Image Shape: {metadata.get('image_shape', 'Unknown')}</div>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Component Contributions</h2>\n                <div class=\"metric\">DeepGaze: {results.get('attention_distribution', {}).get('deepgaze_contribution', 0):.1%}</div>\n                <div class=\"metric\">SAM2: {results.get('attention_distribution', {}).get('sam2_contribution', 0):.1%}</div>\n                <div class=\"metric\">LLM: {results.get('attention_distribution', {}).get('llm_contribution', 0):.1%}</div>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Semantic Interpretation</h2>\n                <p>{results.get('semantic_interpretation', {}).get('gaze_narrative', 'No narrative available')}</p>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Fixation Sequence</h2>\n                <p>Number of Fixations: {len(results.get('fixation_sequence', []))}</p>\n            </div>\n        </body>\n        </html>\n        \"\"\"\n        \n        return html"