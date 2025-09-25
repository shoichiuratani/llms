#!/usr/bin/env python3
"""
デモ実行スクリプト
"""

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
        
        input("\nEnter キーを押して続行...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n終了しました")
