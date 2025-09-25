#!/usr/bin/env python3
"""
API サーバー起動スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        from src.api.main import run_development_server
        print("統合視覚的注意システム API サーバーを起動中...")
        print("サーバーURL: http://localhost:8000")
        print("API ドキュメント: http://localhost:8000/docs")
        print("停止するには Ctrl+C を押してください")
        run_development_server()
        
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")
    except Exception as e:
        print(f"サーバー起動エラー: {e}")
        sys.exit(1)
