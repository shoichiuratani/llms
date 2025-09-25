"""
API エンドポイント

各種 REST API エンドポイントの実装
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import asyncio
import logging
import time
from typing import Dict, Any

from .models import (
    AnalysisRequest, AnalysisResponse, ErrorResponse,
    SystemStatus, ConfigUpdateRequest, ConfigUpdateResponse
)
from ..core import IntegratedAttentionSystem
from ..utils.validation import InputValidator, ValidationError

logger = logging.getLogger(__name__)

# ルーターの作成
router = APIRouter()

# グローバルシステムインスタンス
attention_system = None
validator = InputValidator()


async def get_attention_system():
    """依存性注入: 注意システムの取得"""
    global attention_system
    if attention_system is None:
        attention_system = IntegratedAttentionSystem()
    return attention_system


@router.post("/api/v2/analyze/integrated",
            response_model=AnalysisResponse,
            responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze_integrated(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    統合分析エンドポイント
    
    DeepGaze III、SAM2、LLMを統合した視覚的注意分析を実行します。
    """
    start_time = time.time()
    
    try:
        # リクエスト検証
        try:
            validator.validate_api_request(request.dict())
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid request",
                    "error_code": "VALIDATION_ERROR",
                    "details": {"validation_error": str(e)},
                    "timestamp": time.time()
                }
            )
        
        # 画像データのデコード
        try:
            import base64
            image_data = base64.b64decode(request.image)
            
            if len(image_data) == 0:
                raise ValueError("Empty image data")
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid image data",
                    "error_code": "IMAGE_DECODE_ERROR", 
                    "details": {"decode_error": str(e)},
                    "timestamp": time.time()
                }
            )
        
        # 分析の実行
        try:
            result = await system.analyze_integrated(
                request.image,  # Base64文字列を直接渡す
                request.analysis_config.dict() if request.analysis_config else None,
                request.user_context.dict() if request.user_context else None
            )
            
            # 成功レスポンスの構築
            if result.get('metadata', {}).get('success', True):
                return JSONResponse(
                    content=result,
                    status_code=200
                )
            else:
                # 分析は完了したが何らかの問題があった場合
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": result.get('metadata', {}).get('error', 'Analysis failed'),
                        "error_code": "ANALYSIS_FAILED",
                        "details": result.get('metadata', {}),
                        "timestamp": time.time()
                    }
                )
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Analysis processing failed",
                    "error_code": "ANALYSIS_ERROR",
                    "details": {"analysis_error": str(e)},
                    "timestamp": time.time()
                }
            )
        
    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        # 予期しないエラー
        logger.error(f"Unexpected error in analyze_integrated: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "details": {"unexpected_error": str(e)},
                "timestamp": time.time()
            }
        )


@router.get("/api/v2/status", response_model=SystemStatus)
async def get_system_status(
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    システムステータス取得エンドポイント
    
    システムの現在の状態と統計情報を返します。
    """
    try:
        status = system.get_system_status()
        return JSONResponse(content=status, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get system status",
                "error_code": "STATUS_ERROR",
                "details": {"status_error": str(e)},
                "timestamp": time.time()
            }
        )


@router.post("/api/v2/config/update", response_model=ConfigUpdateResponse)
async def update_configuration(
    request: ConfigUpdateRequest,
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    設定更新エンドポイント
    
    システムの実行時設定を更新します。
    """
    try:
        # 設定の更新
        system.update_configuration(request.config_updates)
        
        # 更新後の設定を取得
        updated_config = system.config_manager.config
        
        response = {
            "success": True,
            "message": "Configuration updated successfully",
            "updated_config": updated_config
        }
        
        return JSONResponse(content=response, status_code=200)
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update configuration",
                "error_code": "CONFIG_UPDATE_ERROR",
                "details": {"config_error": str(e)},
                "timestamp": time.time()
            }
        )


@router.get("/api/v2/health")
async def health_check():
    """
    ヘルスチェックエンドポイント
    
    システムの基本的な動作確認を行います。
    """
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0",
            "uptime": time.time() - start_time if 'start_time' in globals() else 0
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/api/v2/models/info")
async def get_models_info(
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    モデル情報取得エンドポイント
    
    使用中のモデルの詳細情報を返します。
    """
    try:
        models_info = {
            "deepgaze": {
                "model_name": "DeepGaze III",
                "version": "3.0",
                "description": "Human visual attention prediction model",
                "status": "active"
            },
            "sam2": {
                "model_name": "Segment Anything Model 2",
                "version": "2.0",
                "description": "Object segmentation model",
                "status": "active"
            },
            "llm": {
                "model_name": system.config.get('llm', {}).get('model_name', 'GPT-4 Vision'),
                "provider": system.config.get('llm', {}).get('provider', 'openai'),
                "description": "Large language model for semantic understanding",
                "status": "active"
            }
        }
        
        return JSONResponse(content=models_info, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get models information",
                "error_code": "MODELS_INFO_ERROR",
                "timestamp": time.time()
            }
        )


@router.post("/api/v2/analyze/batch")
async def analyze_batch(
    requests: list[AnalysisRequest],
    background_tasks: BackgroundTasks,
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    バッチ分析エンドポイント
    
    複数の画像を一度に分析します。
    """
    if len(requests) > 10:  # バッチサイズ制限
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Batch size too large",
                "error_code": "BATCH_SIZE_ERROR",
                "details": {"max_batch_size": 10, "requested": len(requests)},
                "timestamp": time.time()
            }
        )
    
    results = []
    errors = []
    
    for i, request in enumerate(requests):
        try:
            result = await system.analyze_integrated(
                request.image,
                request.analysis_config.dict() if request.analysis_config else None,
                request.user_context.dict() if request.user_context else None
            )
            results.append({"index": i, "result": result})
            
        except Exception as e:
            logger.error(f"Batch analysis error for request {i}: {e}")
            errors.append({
                "index": i,
                "error": str(e),
                "error_code": "BATCH_ANALYSIS_ERROR"
            })
    
    return {
        "batch_results": results,
        "errors": errors,
        "total_processed": len(results),
        "total_errors": len(errors),
        "timestamp": time.time()
    }


@router.get("/api/v2/performance/statistics")
async def get_performance_statistics(
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    パフォーマンス統計取得エンドポイント
    
    システムのパフォーマンス統計を返します。
    """
    try:
        stats = system.integration_algorithm.get_performance_statistics()
        return JSONResponse(content=stats, status_code=200)
        
    except Exception as e:
        logger.error(f"Error getting performance statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get performance statistics",
                "error_code": "PERF_STATS_ERROR",
                "timestamp": time.time()
            }
        )


@router.post("/api/v2/system/cleanup")
async def cleanup_system(
    background_tasks: BackgroundTasks,
    system: IntegratedAttentionSystem = Depends(get_attention_system)
):
    """
    システムクリーンアップエンドポイント
    
    システムリソースのクリーンアップを実行します。
    """
    try:
        # バックグラウンドでクリーンアップ実行
        background_tasks.add_task(system.cleanup)
        
        return {
            "message": "Cleanup initiated",
            "status": "success",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error initiating cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to initiate cleanup",
                "error_code": "CLEANUP_ERROR",
                "timestamp": time.time()
            }
        )


# 起動時の初期化を記録
start_time = time.time()