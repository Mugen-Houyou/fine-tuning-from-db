"""API Dependencies"""
from korean_predictor.models.model_manager import ModelManager
from korean_predictor.models.predictor import PredictionService
from korean_predictor.cache.cache_service import CacheService
from korean_predictor.utils.config import Config
import logging

logger = logging.getLogger(__name__)

# 전역 인스턴스 (서버 시작 시 초기화)
_model_manager = None
_predictor = None
_cache_service = None
_start_time = None


def init_services(run_mode: str = "auto", model_name: str = "kogpt2"):
    """서비스 초기화"""
    global _model_manager, _predictor, _cache_service, _start_time
    import time

    logger.info(f"Initializing services with run_mode={run_mode}, model={model_name}")

    # Config 디렉토리 설정
    Config.setup_directories()

    # 모델 매니저 초기화
    _model_manager = ModelManager()
    _model_manager.set_run_mode(run_mode)

    # 모델 로드
    success, message = _model_manager.load_model(model_name)
    if not success:
        logger.error(f"Failed to load model: {message}")
        raise RuntimeError(f"Failed to load model: {message}")

    logger.info(f"Model loaded: {message}")

    # 캐시 서비스 초기화
    if Config.CACHE_ENABLED:
        _cache_service = CacheService(
            cache_dir=Config.CACHE_DIR,
            max_size_mb=Config.CACHE_SIZE_MB,
            ttl_seconds=Config.CACHE_TTL_SECONDS
        )
        logger.info("Cache service initialized")
    else:
        _cache_service = None
        logger.info("Cache service disabled")

    # Predictor 초기화
    _predictor = PredictionService(_model_manager, _cache_service)
    logger.info("Predictor service initialized")

    # 시작 시간 기록
    _start_time = time.time()
    logger.info("Services initialization completed")


def get_predictor() -> PredictionService:
    """Predictor 의존성"""
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call init_services() first.")
    return _predictor


def get_model_manager() -> ModelManager:
    """ModelManager 의존성"""
    if _model_manager is None:
        raise RuntimeError("ModelManager not initialized. Call init_services() first.")
    return _model_manager


def get_cache_service() -> CacheService:
    """CacheService 의존성"""
    return _cache_service


def get_start_time() -> float:
    """서버 시작 시간 의존성"""
    if _start_time is None:
        raise RuntimeError("Start time not initialized. Call init_services() first.")
    return _start_time
