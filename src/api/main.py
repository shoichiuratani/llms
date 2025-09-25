"""
FastAPI メインアプリケーション

視覚的注意システムのAPIサーバーのエントリーポイント
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from contextlib import asynccontextmanager

from .endpoints import router
from ..core import ConfigManager
from ..utils.validation import ValidationError


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    logger.info("Starting Visual Attention System API v2.0")
    
    # システム初期化（必要に応じて）
    try:
        config_manager = ConfigManager()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
    
    yield
    
    # 終了時の処理
    logger.info("Shutting down Visual Attention System API")


# FastAPIアプリケーションの作成
app = FastAPI(
    title="Visual Attention System API",
    description="人間の視覚的注意機構に基づく顕著性予測システム v2.0 - DeepGaze III・SAM2・LLM統合版",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に設定する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 信頼できるホストの設定（本番環境用）
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["localhost", "127.0.0.1", "*.your-domain.com"]
# )

# ルーターの登録
app.include_router(router, tags=["Visual Attention Analysis"])


# グローバル例外ハンドラー
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """バリデーションエラーのハンドリング"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": {"message": str(exc)},
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP例外のハンドリング"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "error_code": f"HTTP_{exc.status_code}",
            "details": exc.detail if isinstance(exc.detail, dict) else {},
            "timestamp": time.time()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """内部サーバーエラーのハンドリング"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "details": {"message": "An unexpected error occurred"},
            "timestamp": time.time()
        }
    )


# リクエスト処理時間の計測ミドルウェア
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """処理時間をヘッダーに追加"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# レート制限ミドルウェア（簡易版）
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """簡易レート制限"""
    # 実装: IPアドレス別のリクエスト頻度制限
    # 本番環境では Redis などを使用した本格的な実装を推奨
    
    client_ip = request.client.host
    # 現在は通過させるのみ（実装は省略）
    
    response = await call_next(request)
    return response


# ルートエンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Visual Attention System API v2.0",
        "description": "DeepGaze III・SAM2・LLM統合版",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v2/health",
        "status_endpoint": "/api/v2/status"
    }


# 開発用サーバー起動関数
def run_development_server():
    """開発用サーバーの起動"""
    config_manager = ConfigManager()
    api_config = config_manager.get_section('api')
    
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', True),
        workers=api_config.get('workers', 1),
        log_level=api_config.get('log_level', 'info')
    )


if __name__ == "__main__":
    run_development_server()