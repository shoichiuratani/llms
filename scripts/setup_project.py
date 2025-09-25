#!/usr/bin/env python3
"""
プロジェクトセットアップスクリプト

統合視覚的注意システムの初期設定と環境構築を行います。
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def check_python_version():
    """Python バージョンチェック"""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 以上が必要です")
        return False
    
    logger.info(f"Python バージョン: {sys.version}")
    return True


def install_dependencies():
    """依存関係のインストール"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("依存パッケージをインストール中...")
        
        # requirements.txt の存在確認
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt が見つかりません")
            return False
        
        # pip install の実行
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"依存パッケージのインストールに失敗: {result.stderr}")
            return False
        
        logger.info("依存パッケージのインストール完了")
        return True
        
    except Exception as e:
        logger.error(f"インストールエラー: {e}")
        return False


def create_directories():
    """必要なディレクトリの作成"""
    logger = logging.getLogger(__name__)
    
    project_root = Path(__file__).parent.parent
    
    directories = [
        "logs",
        "data/samples",
        "data/models", 
        "data/cache",
        "outputs",
        "temp"
    ]
    
    try:
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"ディレクトリ作成: {directory}")
        
        return True
        
    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        return False


def setup_environment_variables():
    """環境変数の設定"""
    logger = logging.getLogger(__name__)
    
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env.example"
    
    try:
        with open(env_file, 'w') as f:
            f.write("""# 統合視覚的注意システム環境変数設定例

# OpenAI API Key (LLM使用時に必要)
# OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (Claude使用時に必要)  
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# デバッグモード
# DEBUG=false

# APIサーバー設定
# PORT=8000
# HOST=0.0.0.0

# ログレベル
# LOG_LEVEL=INFO

# データベース設定 (将来的な拡張用)
# DATABASE_URL=sqlite:///./app.db

# キャッシュ設定
# REDIS_URL=redis://localhost:6379

# プロファイリング
# ENABLE_PROFILING=false
""")
        
        logger.info("環境変数テンプレート (.env.example) を作成しました")
        logger.info("必要に応じて .env ファイルを作成し、API キーを設定してください")
        
        return True
        
    except Exception as e:
        logger.error(f"環境変数設定エラー: {e}")
        return False


def run_basic_tests():
    """基本テストの実行"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("基本インポートテストを実行中...")
        
        # 基本インポートのテスト
        import numpy as np
        import cv2
        from PIL import Image
        logger.info("✓ 基本パッケージのインポート成功")
        
        # システムインポートのテスト
        sys.path.append(str(Path(__file__).parent.parent))
        
        try:
            from src.core import IntegratedAttentionSystem
            logger.info("✓ システムコアのインポート成功")
        except ImportError as e:
            logger.warning(f"システムコアのインポートに失敗（正常、依存関係による）: {e}")
        
        # 設定ファイルのテスト
        try:
            from src.core.config_manager import ConfigManager
            config_manager = ConfigManager()
            logger.info("✓ 設定管理システムの初期化成功")
        except Exception as e:
            logger.warning(f"設定管理システムの初期化に失敗: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return False


def create_startup_script():
    """起動スクリプトの作成"""
    logger = logging.getLogger(__name__)
    
    project_root = Path(__file__).parent.parent
    
    # API サーバー起動スクリプト
    api_script = project_root / "start_api_server.py"
    
    try:
        with open(api_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
API サーバー起動スクリプト
\"\"\"

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
        print("\\nサーバーを停止しました")
    except Exception as e:
        print(f"サーバー起動エラー: {e}")
        sys.exit(1)
""")
        
        # 実行権限を付与
        api_script.chmod(0o755)
        logger.info("API サーバー起動スクリプトを作成しました: start_api_server.py")
        
        # デモ実行スクリプト
        demo_script = project_root / "run_demos.py"
        
        with open(demo_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
デモ実行スクリプト
\"\"\"

import sys
import os
import asyncio
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def show_menu():
    print("=== 統合視覚的注意システム デモメニュー ===")
    print("1. 基本使用例")
    print("2. API クライアントデモ")
    print("3. ベンチマークデモ")
    print("4. 全て実行")
    print("0. 終了")
    return input("選択してください (0-4): ")

async def run_basic_demo():
    from examples.basic_usage import main
    await main()

def run_api_demo():
    from examples.api_client_demo import main
    main()

async def run_benchmark_demo():
    from examples.benchmark_demo import main
    await main()

async def main():
    while True:
        choice = show_menu()
        
        if choice == "0":
            print("終了します")
            break
        elif choice == "1":
            await run_basic_demo()
        elif choice == "2":
            run_api_demo()
        elif choice == "3":
            await run_benchmark_demo()
        elif choice == "4":
            print("全てのデモを実行します...")
            await run_basic_demo()
            run_api_demo()
            await run_benchmark_demo()
        else:
            print("無効な選択です")
        
        input("\\nEnter キーを押して続行...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n終了しました")
""")
        
        demo_script.chmod(0o755)
        logger.info("デモ実行スクリプトを作成しました: run_demos.py")
        
        return True
        
    except Exception as e:
        logger.error(f"起動スクリプト作成エラー: {e}")
        return False


def print_setup_summary():
    """セットアップ完了メッセージ"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🎉 統合視覚的注意システム セットアップ完了！")
    print("="*60)
    
    print("\n📋 次のステップ:")
    print("1. API キーの設定（オプション）:")
    print("   - .env ファイルを作成")
    print("   - OPENAI_API_KEY または ANTHROPIC_API_KEY を設定")
    
    print("\n2. システムの起動:")
    print("   - API サーバー: python start_api_server.py")
    print("   - デモ実行: python run_demos.py")
    
    print("\n3. 使用例:")
    print("   - 基本使用: python examples/basic_usage.py")
    print("   - API テスト: python examples/api_client_demo.py")
    print("   - ベンチマーク: python examples/benchmark_demo.py")
    
    print("\n📚 ドキュメント:")
    print("   - README.md: プロジェクト概要")
    print("   - API ドキュメント: http://localhost:8000/docs (サーバー起動後)")
    
    print("\n🔧 設定:")
    print("   - システム設定: config/system_config.yaml")
    print("   - 環境変数: .env (テンプレート: .env.example)")
    
    print("\n" + "="*60)


def main():
    """メイン実行関数"""
    logger = setup_logging()
    
    print("統合視覚的注意システム セットアップを開始します...")
    
    steps = [
        ("Python バージョンチェック", check_python_version),
        ("ディレクトリ作成", create_directories),
        ("依存パッケージインストール", install_dependencies),
        ("環境変数設定", setup_environment_variables),
        ("基本テスト実行", run_basic_tests),
        ("起動スクリプト作成", create_startup_script),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"実行中: {step_name}")
        
        try:
            if not step_func():
                failed_steps.append(step_name)
                logger.error(f"失敗: {step_name}")
            else:
                logger.info(f"完了: {step_name}")
                
        except Exception as e:
            failed_steps.append(step_name)
            logger.error(f"エラー in {step_name}: {e}")
    
    if failed_steps:
        print(f"\n⚠️  以下のステップでエラーが発生しました:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n手動で問題を解決してください。")
        sys.exit(1)
    else:
        print_setup_summary()


if __name__ == "__main__":
    main()