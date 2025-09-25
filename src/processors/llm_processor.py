"""
LLM トップダウン処理コントローラ

大規模言語モデル（LLM）を使用した高次認知機能のモデル化と
意味的文脈に基づくトップダウン注意制御を実装します。
"""

import numpy as np
import base64
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
import logging
from dataclasses import dataclass
from PIL import Image
import io
import cv2

# LLM関連のインポート
try:
    import openai
except ImportError:
    logging.warning("OpenAI not found. Using mock implementation.")
    openai = None

try:
    import anthropic
except ImportError:
    logging.warning("Anthropic not found.")
    anthropic = None

logger = logging.getLogger(__name__)


@dataclass
class SemanticElement:
    """意味的要素の情報"""
    name: str
    importance: float  # 1-10のスケール
    bbox: Optional[Tuple[int, int, int, int]]
    relationships: List[str]
    emotional_valence: float  # -1 to 1
    attention_priority: int


@dataclass
class SceneContext:
    """シーンの文脈情報"""
    main_elements: List[SemanticElement]
    scene_category: str
    emotional_tone: str
    complexity_level: int  # 1-5
    cultural_context: Optional[str]
    narrative_flow: List[str]


class LLMInterface:
    """
    マルチモーダルLLMとのインターフェース
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.model_name = config.get('model_name', 'gpt-4-vision-preview')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.1)
        
        self._setup_client()
        
    def _setup_client(self):
        """LLMクライアントの設定"""
        if self.provider == 'openai' and openai is not None:
            api_key = self.config.get('api_key') or self._get_env_key('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                self.client = openai
            else:
                logger.warning("OpenAI API key not found. Using mock client.")
                self.client = MockLLMClient()
        elif self.provider == 'anthropic' and anthropic is not None:
            api_key = self.config.get('api_key') or self._get_env_key('ANTHROPIC_API_KEY')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                logger.warning("Anthropic API key not found. Using mock client.")
                self.client = MockLLMClient()
        else:
            logger.info("Using mock LLM client")
            self.client = MockLLMClient()
    
    def _get_env_key(self, env_var: str) -> Optional[str]:
        """環境変数からAPIキーを取得"""
        import os
        return os.getenv(env_var)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """画像をBase64エンコード"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        
        # PNGとしてエンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    async def analyze_scene(self, image: np.ndarray, 
                          deepgaze_features: Dict[str, Any],
                          prompt: str) -> Dict[str, Any]:
        """シーンの意味的分析"""
        try:
            image_b64 = self._encode_image(image)
            
            if hasattr(self.client, 'analyze_scene'):
                # モッククライアントの場合
                return await self.client.analyze_scene(image, prompt)
            
            # 実際のLLM API呼び出し
            if self.provider == 'openai':
                return await self._openai_analyze_scene(image_b64, prompt)
            elif self.provider == 'anthropic':
                return await self._anthropic_analyze_scene(image_b64, prompt)
            else:
                # フォールバック分析
                return self._fallback_scene_analysis(image)
                
        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return self._fallback_scene_analysis(image)
    
    async def _openai_analyze_scene(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        """OpenAI GPT-4 Vision による分析"""
        try:
            response = await self.client.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }\n                        ]\n                    }\n                ],\n                max_tokens=self.max_tokens,\n                temperature=self.temperature\n            )\n            \n            content = response.choices[0].message.content\n            return self._parse_llm_response(content)\n            \n        except Exception as e:\n            logger.error(f\"OpenAI API error: {e}\")\n            return self._fallback_scene_analysis(None)\n    \n    async def _anthropic_analyze_scene(self, image_b64: str, prompt: str) -> Dict[str, Any]:\n        \"\"\"Anthropic Claude による分析\"\"\"\n        try:\n            message = await self.client.messages.create(\n                model=\"claude-3-sonnet-20240229\",\n                max_tokens=self.max_tokens,\n                messages=[\n                    {\n                        \"role\": \"user\",\n                        \"content\": [\n                            {\n                                \"type\": \"image\",\n                                \"source\": {\n                                    \"type\": \"base64\",\n                                    \"media_type\": \"image/png\",\n                                    \"data\": image_b64\n                                }\n                            },\n                            {\"type\": \"text\", \"text\": prompt}\n                        ]\n                    }\n                ]\n            )\n            \n            content = message.content[0].text\n            return self._parse_llm_response(content)\n            \n        except Exception as e:\n            logger.error(f\"Anthropic API error: {e}\")\n            return self._fallback_scene_analysis(None)\n    \n    def _parse_llm_response(self, content: str) -> Dict[str, Any]:\n        \"\"\"LLMレスポンスの解析\"\"\"\n        # 実際の実装では、構造化された応答を期待\n        # ここでは簡略化した解析を実装\n        \n        lines = content.split('\\n')\n        elements = []\n        scene_category = \"general\"\n        emotional_tone = \"neutral\"\n        \n        for line in lines:\n            line = line.strip()\n            if \"要素:\" in line or \"Element:\" in line:\n                # 要素の抽出（簡略化）\n                element_name = line.split(\":\")[1].strip()\n                elements.append({\n                    \"name\": element_name,\n                    \"importance\": 7,  # デフォルト値\n                    \"bbox\": None,\n                    \"relationships\": [],\n                    \"emotional_valence\": 0.0,\n                    \"attention_priority\": 1\n                })\n            elif \"カテゴリ:\" in line or \"Category:\" in line:\n                scene_category = line.split(\":\")[1].strip()\n            elif \"感情:\" in line or \"Emotion:\" in line:\n                emotional_tone = line.split(\":\")[1].strip()\n        \n        return {\n            \"main_elements\": elements,\n            \"scene_category\": scene_category,\n            \"emotional_tone\": emotional_tone,\n            \"complexity_level\": min(len(elements), 5),\n            \"cultural_context\": None,\n            \"narrative_flow\": [\"初期注目\", \"詳細探索\", \"統合理解\"]\n        }\n    \n    def _fallback_scene_analysis(self, image: Optional[np.ndarray]) -> Dict[str, Any]:\n        \"\"\"フォールバック分析（画像処理ベース）\"\"\"\n        elements = [\n            {\n                \"name\": \"中央領域\",\n                \"importance\": 8,\n                \"bbox\": None,\n                \"relationships\": [],\n                \"emotional_valence\": 0.0,\n                \"attention_priority\": 1\n            },\n            {\n                \"name\": \"周辺領域\", \n                \"importance\": 4,\n                \"bbox\": None,\n                \"relationships\": [],\n                \"emotional_valence\": 0.0,\n                \"attention_priority\": 2\n            }\n        ]\n        \n        return {\n            \"main_elements\": elements,\n            \"scene_category\": \"unknown\",\n            \"emotional_tone\": \"neutral\",\n            \"complexity_level\": 2,\n            \"cultural_context\": None,\n            \"narrative_flow\": [\"basic_attention\"]\n        }\n\n\nclass MockLLMClient:\n    \"\"\"LLMクライアントのモック実装\"\"\"\n    \n    async def analyze_scene(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:\n        \"\"\"シーン分析のモック\"\"\"\n        # 画像の基本的な解析\n        h, w = image.shape[:2]\n        \n        # 簡単な特徴抽出\n        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image\n        edges = cv2.Canny(gray, 50, 150)\n        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n        \n        # 要素の生成\n        elements = []\n        for i, contour in enumerate(contours[:5]):  # 最大5要素\n            area = cv2.contourArea(contour)\n            if area > 100:  # 十分な大きさの要素のみ\n                x, y, w_bbox, h_bbox = cv2.boundingRect(contour)\n                \n                elements.append({\n                    \"name\": f\"物体_{i+1}\",\n                    \"importance\": max(1, min(10, int(area / 1000) + 3)),\n                    \"bbox\": [x, y, w_bbox, h_bbox],\n                    \"relationships\": [],\n                    \"emotional_valence\": 0.0,\n                    \"attention_priority\": i + 1\n                })\n        \n        # 複雑度の推定\n        complexity = min(5, len(contours) // 10 + 1)\n        \n        return {\n            \"main_elements\": elements,\n            \"scene_category\": \"complex\" if complexity > 3 else \"simple\",\n            \"emotional_tone\": \"neutral\",\n            \"complexity_level\": complexity,\n            \"cultural_context\": None,\n            \"narrative_flow\": [\"初期探索\", \"要素認識\", \"関係理解\"]\n        }\n\n\nclass TaskSpecificBiasGenerator:\n    \"\"\"タスク特有の注意バイアス生成器\"\"\"\n    \n    def __init__(self, llm_interface: LLMInterface):\n        self.llm = llm_interface\n        \n        # 定義済みタスクパターン\n        self.task_patterns = {\n            \"face_detection\": {\n                \"keywords\": [\"顔\", \"人\", \"表情\", \"eye\", \"face\"],\n                \"bias_weight\": 2.0,\n                \"spatial_preference\": \"center\"\n            },\n            \"text_reading\": {\n                \"keywords\": [\"文字\", \"テキスト\", \"看板\", \"text\"],\n                \"bias_weight\": 1.8,\n                \"spatial_preference\": \"left_to_right\"\n            },\n            \"navigation\": {\n                \"keywords\": [\"道路\", \"標識\", \"建物\", \"道\"],\n                \"bias_weight\": 1.5,\n                \"spatial_preference\": \"horizon\"\n            },\n            \"free_viewing\": {\n                \"keywords\": [],\n                \"bias_weight\": 1.0,\n                \"spatial_preference\": \"balanced\"\n            }\n        }\n    \n    async def generate_task_bias(self, scene_understanding: Dict[str, Any],\n                               task_context: str,\n                               image_shape: Tuple[int, int]) -> np.ndarray:\n        \"\"\"タスク特有の注意バイアスを生成\"\"\"\n        h, w = image_shape\n        bias_map = np.ones((h, w), dtype=np.float32)\n        \n        # タスクパターンの取得\n        task_pattern = self.task_patterns.get(task_context, \n                                            self.task_patterns[\"free_viewing\"])\n        \n        # 要素ベースのバイアス\n        elements = scene_understanding.get(\"main_elements\", [])\n        for element in elements:\n            if self._element_matches_task(element, task_pattern):\n                bbox = element.get(\"bbox\")\n                if bbox:\n                    x, y, w_bbox, h_bbox = bbox\n                    # 要素領域の強化\n                    y_end = min(h, y + h_bbox)\n                    x_end = min(w, x + w_bbox)\n                    bias_map[y:y_end, x:x_end] *= task_pattern[\"bias_weight\"]\n        \n        # 空間的選好の適用\n        spatial_bias = self._generate_spatial_preference(\n            task_pattern[\"spatial_preference\"], (h, w)\n        )\n        bias_map *= spatial_bias\n        \n        return bias_map\n    \n    def _element_matches_task(self, element: Dict[str, Any], \n                            task_pattern: Dict[str, Any]) -> bool:\n        \"\"\"要素がタスクにマッチするかチェック\"\"\"\n        element_name = element.get(\"name\", \"\").lower()\n        keywords = task_pattern.get(\"keywords\", [])\n        \n        for keyword in keywords:\n            if keyword.lower() in element_name:\n                return True\n        \n        return False\n    \n    def _generate_spatial_preference(self, preference: str, \n                                   shape: Tuple[int, int]) -> np.ndarray:\n        \"\"\"空間的選好バイアスの生成\"\"\"\n        h, w = shape\n        y, x = np.ogrid[:h, :w]\n        \n        if preference == \"center\":\n            # 中央優先\n            center_y, center_x = h // 2, w // 2\n            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)\n            max_distance = np.sqrt(center_y**2 + center_x**2)\n            bias = 1.0 + 0.5 * (1 - distance / max_distance)\n            \n        elif preference == \"left_to_right\":\n            # 左から右への選好\n            bias = np.ones((h, w))\n            bias *= np.linspace(0.8, 1.2, w)\n            \n        elif preference == \"horizon\":\n            # 水平線付近への選好\n            horizon_y = h // 2\n            distance_from_horizon = np.abs(y - horizon_y) / h\n            bias = 1.0 + 0.3 * (1 - distance_from_horizon * 2)\n            \n        else:  # balanced\n            bias = np.ones((h, w))\n        \n        return bias\n\n\nclass PredictiveCodingGenerator:\n    \"\"\"予測的コーディングマップ生成器\"\"\"\n    \n    def __init__(self):\n        pass\n    \n    def generate_predictive_map(self, scene_understanding: Dict[str, Any],\n                              deepgaze_features: Dict[str, Any],\n                              image_shape: Tuple[int, int]) -> np.ndarray:\n        \"\"\"予測的コーディングに基づく注意マップ生成\"\"\"\n        h, w = image_shape\n        predictive_map = np.ones((h, w), dtype=np.float32)\n        \n        # シーンの複雑度に基づく予測\n        complexity = scene_understanding.get(\"complexity_level\", 1)\n        complexity_factor = min(1.5, 1.0 + complexity * 0.1)\n        \n        # 要素間の関係に基づく予測\n        elements = scene_understanding.get(\"main_elements\", [])\n        for element in elements:\n            importance = element.get(\"importance\", 5)\n            bbox = element.get(\"bbox\")\n            \n            if bbox and importance > 6:  # 重要な要素\n                x, y, w_bbox, h_bbox = bbox\n                # 予測的注意の拡散\n                center_x, center_y = x + w_bbox // 2, y + h_bbox // 2\n                \n                # ガウシアン拡散\n                sigma = max(w_bbox, h_bbox) * 0.5\n                y_grid, x_grid = np.ogrid[:h, :w]\n                gaussian = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))\n                \n                predictive_map += gaussian * (importance / 10.0) * 0.3\n        \n        # 感情的文脈の影響\n        emotional_tone = scene_understanding.get(\"emotional_tone\", \"neutral\")\n        if emotional_tone in [\"positive\", \"exciting\"]:\n            predictive_map *= 1.1\n        elif emotional_tone in [\"negative\", \"threatening\"]:\n            predictive_map *= 1.2  # より強い注意\n        \n        # 正規化\n        if predictive_map.max() > predictive_map.min():\n            predictive_map = (predictive_map - predictive_map.min()) / \\\n                           (predictive_map.max() - predictive_map.min())\n        \n        return predictive_map\n\n\nclass LLMTopDownController:\n    \"\"\"\n    LLMによる高次認知機能のモデル化とトップダウン注意制御\n    \"\"\"\n    \n    def __init__(self, config: Dict[str, Any]):\n        self.config = config\n        self.llm_interface = LLMInterface(config)\n        self.bias_generator = TaskSpecificBiasGenerator(self.llm_interface)\n        self.predictive_generator = PredictiveCodingGenerator()\n        \n        # プロンプトテンプレート\n        self.scene_analysis_prompt = config.get('prompts', {}).get(\n            'scene_analysis',\n            \"画像を分析し、主要な要素、重要度、関係性を説明してください。\"\n        )\n        \n        logger.info(\"LLM TopDown Controller initialized\")\n    \n    async def process_with_context(self, image: np.ndarray, \n                                 deepgaze_features: Dict[str, Any],\n                                 task_context: Optional[str] = None) -> Dict[str, Any]:\n        \"\"\"\n        LLMによる文脈処理とトップダウン制御\n        \n        Args:\n            image: 入力画像 (H, W, C)\n            deepgaze_features: DeepGazeの特徴量\n            task_context: タスクコンテキスト\n            \n        Returns:\n            トップダウン処理結果\n        \"\"\"\n        try:\n            # シーンの意味的理解\n            scene_understanding = await self.llm_interface.analyze_scene(\n                image, deepgaze_features, self.scene_analysis_prompt\n            )\n            \n            # タスク依存の注意調整\n            attention_bias = np.ones(image.shape[:2], dtype=np.float32)\n            if task_context:\n                attention_bias = await self.bias_generator.generate_task_bias(\n                    scene_understanding, task_context, image.shape[:2]\n                )\n            \n            # 期待に基づく予測的コーディング\n            predictive_map = self.predictive_generator.generate_predictive_map(\n                scene_understanding, deepgaze_features, image.shape[:2]\n            )\n            \n            # メタデータの生成\n            metadata = self._generate_processing_metadata(\n                scene_understanding, task_context\n            )\n            \n            return {\n                'semantic_weights': attention_bias,\n                'predictive_map': predictive_map,\n                'scene_context': scene_understanding,\n                'metadata': metadata\n            }\n            \n        except Exception as e:\n            logger.error(f\"Error in LLM top-down processing: {e}\")\n            return self._create_fallback_result(image.shape[:2])\n    \n    def _generate_processing_metadata(self, scene_understanding: Dict[str, Any],\n                                    task_context: Optional[str]) -> Dict[str, Any]:\n        \"\"\"処理メタデータの生成\"\"\"\n        return {\n            'num_elements': len(scene_understanding.get('main_elements', [])),\n            'scene_complexity': scene_understanding.get('complexity_level', 1),\n            'emotional_context': scene_understanding.get('emotional_tone', 'neutral'),\n            'task_applied': task_context is not None,\n            'task_type': task_context,\n            'processing_success': True\n        }\n    \n    def _create_fallback_result(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:\n        \"\"\"フォールバック結果の生成\"\"\"\n        h, w = image_shape\n        \n        return {\n            'semantic_weights': np.ones((h, w), dtype=np.float32),\n            'predictive_map': np.ones((h, w), dtype=np.float32),\n            'scene_context': {\n                'main_elements': [],\n                'scene_category': 'unknown',\n                'emotional_tone': 'neutral',\n                'complexity_level': 1,\n                'cultural_context': None,\n                'narrative_flow': ['fallback']\n            },\n            'metadata': {\n                'num_elements': 0,\n                'scene_complexity': 1,\n                'emotional_context': 'neutral',\n                'task_applied': False,\n                'task_type': None,\n                'processing_success': False,\n                'error': True\n            }\n        }"