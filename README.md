# 人間の視覚的注意機構に基づく顕著性予測システム v2.0
## DeepGaze III・SAM2・LLM統合版

### システム概要
本システムは、人間の視覚的注意メカニズムを多層的にモデル化し、以下の3つの主要コンポーネントを統合します：

- **DeepGaze III**: 人間の視線予測の基盤となるボトムアップ処理
- **SAM2**: 精密な物体セグメンテーションによる物体ベース注意
- **LLM**: 意味的理解と文脈処理によるトップダウン制御

### アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                統合システム                          │
├─────────────────┬─────────────────┬─────────────────┤
│   DeepGaze III  │      SAM2       │       LLM       │
│  ボトムアップ    │  物体ベース注意  │  トップダウン    │
│     処理        │     メカニズム   │     制御        │
└─────────────────┴─────────────────┴─────────────────┘
```

### 主要機能

1. **階層的視覚処理**: 脳の視覚野（V1/V2/V4/IT）に対応する特徴抽出
2. **物体ベース注意**: SAM2による精密なセグメンテーションを活用した注意制御
3. **意味的文脈処理**: LLMによる高次認知機能のモデル化
4. **統合顕著性予測**: 3つのモダリティを神経科学的に妥当な方法で統合
5. **注視点シーケンス生成**: 人間らしい視線移動パターンの生成

### インストール

```bash
pip install -r requirements.txt
```

### 使用方法

#### 基本的な使用例

```python
from src.core.integrated_attention_system import IntegratedAttentionSystem
import numpy as np
from PIL import Image

# システムの初期化
attention_system = IntegratedAttentionSystem()

# 画像の読み込み
image = Image.open("path/to/image.jpg")

# 統合分析の実行
result = attention_system.analyze_integrated(
    image,
    analysis_config={
        "use_deepgaze": True,
        "use_sam2": True,
        "use_llm": True,
        "output_resolution": "high"
    }
)

# 結果の取得
saliency_map = result["saliency_map"]
fixation_sequence = result["fixation_sequence"]
semantic_interpretation = result["semantic_interpretation"]
```

#### API使用例

```bash
# サーバー起動
python -m src.api.main

# 分析リクエスト
curl -X POST "http://localhost:8000/api/v2/analyze/integrated" \
     -H "Content-Type: application/json" \
     -d @request.json
```

### プロジェクト構造

```
webapp/
├── src/
│   ├── core/                 # コアシステム
│   ├── models/               # モデル定義
│   ├── processors/           # 各種処理モジュール
│   ├── integrators/          # 統合アルゴリズム
│   ├── api/                  # API エンドポイント
│   └── utils/                # ユーティリティ
├── tests/                    # テストスイート
├── docs/                     # ドキュメント
├── config/                   # 設定ファイル
├── data/                     # データとモデル
├── notebooks/                # Jupyter notebooks
└── scripts/                  # スクリプト
```

### 技術仕様

- **Python**: 3.8+
- **深層学習フレームワーク**: PyTorch
- **画像処理**: OpenCV, PIL
- **数値計算**: NumPy, SciPy
- **API フレームワーク**: FastAPI
- **可視化**: Matplotlib, Seaborn

### ライセンス

MIT License

### 貢献

プロジェクトへの貢献は歓迎します。詳細は [CONTRIBUTING.md](docs/CONTRIBUTING.md) をご覧ください。

### 引用

本システムを研究で使用される場合は、以下を引用してください：

```bibtex
@software{integrated_attention_system,
  title={人間の視覚的注意機構に基づく顕著性予測システム},
  author={Your Name},
  version={2.0},
  year={2024},
  url={https://github.com/yourname/integrated-attention-system}
}
```