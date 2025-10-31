"""Health Check Routes"""
from fastapi import APIRouter, Depends
from api.schemas.response import HealthStatus, DetailedHealthStatus, ComponentHealth
from api.dependencies import get_model_manager, get_cache_service, get_start_time
from korean_predictor.models.model_manager import ModelManager
from korean_predictor.cache.cache_service import CacheService
import time
import logging
import torch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthStatus)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    start_time: float = Depends(get_start_time)
):
    """기본 헬스체크"""
    try:
        # 모델 로드 여부 확인
        model_info = model_manager.get_model_info()
        is_healthy = model_info.get('loaded', False)

        uptime = int(time.time() - start_time)

        if is_healthy:
            return HealthStatus(
                status="healthy",
                uptime_seconds=uptime,
                version="1.0.0"
            )
        else:
            return HealthStatus(
                status="unhealthy",
                reason="Model not loaded",
                uptime_seconds=uptime,
                version="1.0.0"
            )

    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return HealthStatus(
            status="unhealthy",
            reason=f"Error: {str(e)}",
            uptime_seconds=0,
            version="1.0.0"
        )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    cache_service: CacheService = Depends(get_cache_service),
    start_time: float = Depends(get_start_time)
):
    """상세 헬스체크"""
    try:
        uptime = int(time.time() - start_time)

        # 모델 상태
        model_info = model_manager.get_model_info()
        model_status = ComponentHealth(
            status="healthy" if model_info.get('loaded') else "unhealthy",
            loaded=model_info.get('loaded', False),
            model_name=model_info.get('model_name', None) if model_info.get('loaded') else None
        )

        # 캐시 상태
        if cache_service:
            cache_stats = cache_service.get_stats()
            cache_status = ComponentHealth(
                status="healthy",
                enabled=True,
                size_kb=cache_stats.get('size', 0) / 1024
            )
        else:
            cache_status = ComponentHealth(
                status="healthy",
                enabled=False
            )

        # 디바이스 상태
        device = model_manager.device
        device_type = str(device).split(':')[0]

        device_status = ComponentHealth(
            status="healthy",
            type=device_type
        )

        # GPU 메모리 정보 (CUDA 사용 시)
        if torch.cuda.is_available() and device_type == 'cuda':
            device_status.memory_used_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            device_status.memory_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

        # 전체 상태 판단
        overall_status = "healthy" if model_info.get('loaded') else "unhealthy"

        return DetailedHealthStatus(
            status=overall_status,
            components={
                "model": model_status,
                "cache": cache_status,
                "device": device_status
            },
            uptime_seconds=uptime,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Detailed health check error: {str(e)}", exc_info=True)
        return DetailedHealthStatus(
            status="unhealthy",
            components={
                "error": ComponentHealth(status="unhealthy")
            },
            uptime_seconds=0,
            version="1.0.0"
        )
