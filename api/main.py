"""
Korean Predictor REST API

FastAPI 서버 메인 애플리케이션
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from api.routes import predict_router, models_router, cache_router, health_router
from api.dependencies import init_services
import logging
import time
import uuid
import sys
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Korean Predictor API",
    description="한국어 다음 토큰/어절 예측 및 End-of-Turn 확률 분석 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)


# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 및 request_id 추가"""
    request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    start_time = time.time()

    # 요청 로깅
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # 응답 헤더에 정보 추가
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        # 응답 로깅
        logger.info(
            f"[{request_id}] Completed in {process_time:.3f}s - "
            f"Status: {response.status_code}"
        )

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] Error after {process_time:.3f}s: {str(e)}", exc_info=True)
        raise


# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    logger.error(f"[{request_id}] Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(exc)}
            },
            "meta": {
                "api_version": "1.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id
            }
        }
    )


# 라우터 등록
app.include_router(predict_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(cache_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")


# 루트 경로
@app.get("/")
async def root():
    """API 루트 경로"""
    return {
        "name": "Korean Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health"
    }


# 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    logger.info("=" * 60)
    logger.info("Korean Predictor API Server Starting...")
    logger.info("=" * 60)

    # 환경 변수에서 설정 로드
    run_mode = os.getenv("RUN_MODE", "auto")
    model_name = os.getenv("MODEL_NAME", "kogpt2")

    logger.info(f"Configuration:")
    logger.info(f"  - RUN_MODE: {run_mode}")
    logger.info(f"  - MODEL_NAME: {model_name}")
    logger.info(f"  - ENVIRONMENT: {os.getenv('ENVIRONMENT', 'development')}")

    try:
        # 서비스 초기화
        init_services(run_mode=run_mode, model_name=model_name)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
        logger.error("Server startup failed")
        raise

    logger.info("=" * 60)
    logger.info("Korean Predictor API Server Started")
    logger.info("=" * 60)


# 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    logger.info("=" * 60)
    logger.info("Korean Predictor API Server Shutting Down...")
    logger.info("=" * 60)

    # 정리 작업 수행 (필요시)
    # 예: 모델 언로드, 캐시 저장 등

    logger.info("Server shutdown completed")


if __name__ == "__main__":
    import uvicorn

    # 개발 서버 실행
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드에서만 사용
        log_level="info"
    )
