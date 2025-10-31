"""
Cache Service - 예측 결과 캐싱
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
from diskcache import Cache

logger = logging.getLogger(__name__)


class CacheService:
    """디스크 기반 캐시 서비스"""

    def __init__(self, cache_dir: Optional[str] = None, size_limit: int = 1024 * 1024 * 100):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            size_limit: 캐시 크기 제한 (바이트, 기본값 100MB)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "korean_predictor" / "predictions"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # DiskCache 초기화
        self.cache = Cache(
            str(self.cache_dir),
            size_limit=size_limit,
            eviction_policy='least-recently-used'
        )

        logger.info(f"캐시 초기화: {self.cache_dir} (크기 제한: {size_limit / 1024 / 1024:.1f}MB)")

    def _generate_key(self, key: str) -> str:
        """캐시 키 해싱"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 가져오기

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None
        """
        try:
            hashed_key = self._generate_key(key)
            value = self.cache.get(hashed_key)
            if value:
                logger.debug(f"캐시 히트: {key[:30]}...")
            return value
        except Exception as e:
            logger.error(f"캐시 읽기 오류: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        캐시에 값 저장

        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: Time To Live (초, 기본값 5분)

        Returns:
            성공 여부
        """
        try:
            hashed_key = self._generate_key(key)
            success = self.cache.set(hashed_key, value, expire=ttl)
            if success:
                logger.debug(f"캐시 저장: {key[:30]}... (TTL: {ttl}초)")
            return success
        except Exception as e:
            logger.error(f"캐시 저장 오류: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        캐시에서 특정 키 삭제

        Args:
            key: 삭제할 키

        Returns:
            성공 여부
        """
        try:
            hashed_key = self._generate_key(key)
            success = self.cache.delete(hashed_key)
            if success:
                logger.debug(f"캐시 삭제: {key[:30]}...")
            return success
        except Exception as e:
            logger.error(f"캐시 삭제 오류: {e}")
            return False

    def clear(self):
        """전체 캐시 클리어"""
        try:
            self.cache.clear()
            logger.info("전체 캐시 클리어 완료")
        except Exception as e:
            logger.error(f"캐시 클리어 오류: {e}")

    def get_stats(self) -> dict:
        """캐시 통계 반환"""
        try:
            return {
                "size": self.cache.volume(),
                "items": len(self.cache),
                "hits": self.cache.stats(enable=True)['hits'],
                "misses": self.cache.stats(enable=True)['misses'],
                "directory": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"캐시 통계 조회 오류: {e}")
            return {
                "size": 0,
                "items": 0,
                "hits": 0,
                "misses": 0,
                "directory": str(self.cache_dir),
                "error": str(e)
            }

    def close(self):
        """캐시 종료"""
        try:
            self.cache.close()
            logger.info("캐시 서비스 종료")
        except Exception as e:
            logger.error(f"캐시 종료 오류: {e}")