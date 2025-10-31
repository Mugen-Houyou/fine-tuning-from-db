"""Cache Management Routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.response import (
    CacheStatsResponse, CacheStats, CacheClearResponse,
    CacheClearData, ErrorDetail, MetaInfo
)
from api.dependencies import get_cache_service
from api.middleware.auth import verify_api_key
from korean_predictor.cache.cache_service import CacheService
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service),
    api_key: str = Depends(verify_api_key)
):
    """캐시 통계 조회"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        if cache_service is None:
            # 캐시 비활성화
            stats = CacheStats(
                enabled=False,
                size_kb=0.0,
                items=0,
                hits=0,
                misses=0,
                hit_rate=0.0,
                ttl_seconds=0
            )
        else:
            # 캐시 통계 가져오기
            cache_stats = cache_service.get_stats()
            total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            hit_rate = cache_stats.get('hits', 0) / total_requests if total_requests > 0 else 0.0

            stats = CacheStats(
                enabled=True,
                size_kb=cache_stats.get('size', 0) / 1024,
                items=cache_stats.get('items', 0),
                hits=cache_stats.get('hits', 0),
                misses=cache_stats.get('misses', 0),
                hit_rate=hit_rate,
                ttl_seconds=cache_stats.get('ttl_seconds', 300)
            )

        response = CacheStatsResponse(
            success=True,
            data=stats,
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Cache stats retrieved: {stats.items} items")
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Error getting cache stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )


@router.delete("", response_model=CacheClearResponse)
async def clear_cache(
    cache_service: CacheService = Depends(get_cache_service),
    api_key: str = Depends(verify_api_key)
):
    """캐시 삭제 (관리자 전용)"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        if cache_service is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "CACHE_DISABLED",
                    "message": "Cache is disabled",
                    "details": None
                }
            )

        # 삭제 전 항목 수 기록
        stats_before = cache_service.get_stats()
        items_before = stats_before.get('items', 0)

        # 캐시 삭제
        cache_service.clear()

        response = CacheClearResponse(
            success=True,
            data=CacheClearData(
                message="Cache cleared successfully",
                items_deleted=items_before
            ),
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Cache cleared: {items_before} items deleted")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )
