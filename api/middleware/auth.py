"""Authentication Middleware"""
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

# API 키 관리 (환경 변수 또는 설정 파일에서 로드)
# 프로덕션에서는 데이터베이스나 별도 키 관리 시스템 사용 권장
VALID_API_KEYS = set()

# 환경 변수에서 API 키 로드
_env_keys = os.getenv("API_KEYS", "")
if _env_keys:
    VALID_API_KEYS.update(key.strip() for key in _env_keys.split(",") if key.strip())

# 개발 환경용 테스트 키 (프로덕션에서는 제거)
if os.getenv("ENVIRONMENT", "development") == "development":
    VALID_API_KEYS.add("kp_test_development_key_12345")
    logger.warning("Development mode: Using test API key")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    API Key 검증

    Args:
        credentials: HTTP Authorization 헤더의 Bearer 토큰

    Returns:
        str: 검증된 API Key

    Raises:
        HTTPException: 인증 실패 시
    """
    api_key = credentials.credentials

    # API 키가 설정되어 있지 않으면 모든 요청 허용 (개발 모드)
    if not VALID_API_KEYS:
        logger.warning("No API keys configured. Allowing all requests (development mode)")
        return "dev_mode"

    # API 키 검증
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid API key",
                "details": None
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug(f"API key validated: {api_key[:10]}...")
    return api_key


def add_api_key(api_key: str):
    """API 키 추가 (관리자용)"""
    VALID_API_KEYS.add(api_key)
    logger.info(f"API key added: {api_key[:10]}...")


def remove_api_key(api_key: str):
    """API 키 제거 (관리자용)"""
    if api_key in VALID_API_KEYS:
        VALID_API_KEYS.remove(api_key)
        logger.info(f"API key removed: {api_key[:10]}...")


def list_api_keys_masked() -> list:
    """API 키 목록 (마스킹)"""
    return [f"{key[:10]}...{key[-4:]}" for key in VALID_API_KEYS]
