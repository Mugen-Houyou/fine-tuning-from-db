"""Model Management Routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.response import (
    CurrentModelResponse, ModelInfo, ModelsListResponse, ModelsListData,
    AvailableModel, ErrorDetail, MetaInfo
)
from api.dependencies import get_model_manager
from api.middleware.auth import verify_api_key
from korean_predictor.models.model_manager import ModelManager
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/current", response_model=CurrentModelResponse)
async def get_current_model(
    model_manager: ModelManager = Depends(get_model_manager),
    api_key: str = Depends(verify_api_key)
):
    """현재 로드된 모델 정보 조회"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        info = model_manager.get_model_info()

        if not info.get('loaded'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "MODEL_NOT_LOADED",
                    "message": "Model not loaded",
                    "details": None
                }
            )

        # 모델 정보 생성
        model_info = ModelInfo(
            model_name=info['model_name'],
            model_id=info.get('model_id', 'unknown'),
            vocab_size=info['vocab_size'],
            max_length=info.get('max_length', 0),
            parameters=info['parameters'],
            device=info['device'],
            supports_temperature=info.get('supports_temperature', True),
            loaded_at=datetime.utcnow(),  # 실제로는 로드 시간을 저장해야 함
            memory_usage_mb=None  # 실제 메모리 사용량 계산 필요
        )

        response = CurrentModelResponse(
            success=True,
            data=model_info,
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Model info retrieved: {info['model_name']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )


@router.get("", response_model=ModelsListResponse)
async def list_models(
    api_key: str = Depends(verify_api_key)
):
    """사용 가능한 모델 목록 조회"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        models_dict = ModelManager.list_models()
        models_list = []

        for model_id, model_data in models_dict.items():
            models_list.append(AvailableModel(
                id=model_id,
                name=model_data.get('description', model_id),
                description=model_data.get('description', ''),
                params=model_data.get('params', 'unknown'),
                memory=model_data.get('memory', 'unknown'),
                supports_temperature=not model_data.get('reasoning_model', False),
                reasoning_model=model_data.get('reasoning_model', False),
                supported_modes=model_data.get('supported_modes', ['cpu', 'nvidia-gpu'])
            ))

        response = ModelsListResponse(
            success=True,
            data=ModelsListData(models=models_list),
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Models list retrieved: {len(models_list)} models")
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )
