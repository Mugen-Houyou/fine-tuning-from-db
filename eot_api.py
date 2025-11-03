#!/usr/bin/env python3
"""
EOT (End-of-Turn) Prediction REST API Server
한국어 채팅 문맥에서 발화 종료 확률을 예측하는 REST API 서비스
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import hashlib
import json

from fastapi import FastAPI, HTTPException, Request, status, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import click

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from eot_predictor import EOTPredictor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eot_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API 설정
API_VERSION = "1.0.0"
DEFAULT_PORT = 8177
DEFAULT_HOST = "0.0.0.0"
DEFAULT_RUN_MODE = "auto"

# Rate limiting 설정
RATE_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PREMIUM_PER_MINUTE = 1000
RATE_LIMIT_BATCH_PER_MINUTE = 10

# 전역 예측기 인스턴스
predictor: Optional[EOTPredictor] = None
current_model: str = ""  # 현재 로드된 모델 이름

# Rate limiting storage (실제 환경에서는 Redis 사용 권장)
rate_limit_storage = {}


# ========== Pydantic 모델 정의 ==========

class EOTPredictRequest(BaseModel):
    """EOT 예측 요청 모델"""
    text: str = Field(..., min_length=1, max_length=1000, description="입력 텍스트")
    model: str = Field(default="polyglot", description="사용할 모델")
    run_mode: Optional[str] = Field(default=None, description="실행 모드 (auto, cpu, nvidia-gpu, amd-gpu)")
    top_k: int = Field(default=10, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=0.5, ge=0.1, le=2.0, description="샘플링 온도")
    timeout: int = Field(default=60, ge=0, le=300, description="타임아웃 (초)")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("텍스트는 공백만으로 이루어질 수 없습니다")
        return v

    @validator('run_mode')
    def validate_run_mode(cls, v):
        if v is not None and v not in ['auto', 'cpu', 'nvidia-gpu', 'amd-gpu']:
            raise ValueError("run_mode는 auto, cpu, nvidia-gpu, amd-gpu 중 하나여야 합니다")
        return v


class BatchEOTPredictRequest(BaseModel):
    """배치 EOT 예측 요청 모델"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="입력 텍스트 리스트")
    model: str = Field(default="polyglot", description="사용할 모델")
    run_mode: Optional[str] = Field(default=None, description="실행 모드 (auto, cpu, nvidia-gpu, amd-gpu)")
    top_k: int = Field(default=10, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=0.5, ge=0.1, le=2.0, description="샘플링 온도")
    timeout: int = Field(default=120, ge=0, le=600, description="타임아웃 (초)")

    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError("텍스트는 공백만으로 이루어질 수 없습니다")
        return v


class ContextEOTPredictRequest(BaseModel):
    """컨텍스트 기반 EOT 예측 요청 모델"""
    context: List[str] = Field(..., min_items=1, max_items=20, description="대화 컨텍스트")
    model: str = Field(default="polyglot", description="사용할 모델")
    run_mode: Optional[str] = Field(default=None, description="실행 모드 (auto, cpu, nvidia-gpu, amd-gpu)")
    top_k: int = Field(default=10, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=0.5, ge=0.1, le=2.0, description="샘플링 온도")
    timeout: int = Field(default=60, ge=0, le=300, description="타임아웃 (초)")


class TokenPrediction(BaseModel):
    """토큰 예측 결과"""
    token: str
    probability: float
    rank: int
    is_eot: bool
    type: str


class EOTPredictResponse(BaseModel):
    """EOT 예측 응답 모델"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any]


# ========== FastAPI 앱 초기화 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # 시작 시
    global predictor, current_model

    # 환경 변수에서 설정 읽기
    run_mode = os.getenv("EOT_API_RUN_MODE", DEFAULT_RUN_MODE)
    model_name = os.getenv("EOT_API_MODEL", "polyglot")

    logger.info(f"EOT API 서버 시작 중... (포트: {DEFAULT_PORT}, 모델: {model_name}, 실행 모드: {run_mode})")

    try:
        # EOT 예측기 초기화
        predictor = EOTPredictor(model_name=model_name, run_mode=run_mode)
        current_model = model_name
        logger.info("EOT 예측기 초기화 완료")

        # 워밍업 (첫 예측은 느릴 수 있음)
        logger.info("모델 워밍업 중...")
        _, _ = predictor.predict_eot("테스트", top_k=5, temperature=0.5)
        logger.info("모델 워밍업 완료")

    except Exception as e:
        logger.error(f"EOT 예측기 초기화 실패: {e}")
        raise

    yield  # 앱 실행

    # 종료 시
    logger.info("EOT API 서버 종료 중...")
    if predictor and hasattr(predictor, 'cache_service'):
        predictor.cache_service.close()
    logger.info("EOT API 서버 종료 완료")


app = FastAPI(
    title="EOT Prediction API",
    description="한국어 채팅 발화 종료(End-of-Turn) 확률 예측 API",
    version=API_VERSION,
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 유틸리티 함수 ==========

def get_request_id():
    """요청 ID 생성"""
    import uuid
    return f"req_{uuid.uuid4().hex[:12]}"


def get_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Authorization 헤더에서 API 키 추출"""
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


async def check_rate_limit(api_key: str, is_batch: bool = False) -> bool:
    """Rate limiting 체크"""
    current_time = time.time()
    window_start = current_time - 60  # 1분 윈도우

    if api_key not in rate_limit_storage:
        rate_limit_storage[api_key] = []

    # 오래된 기록 제거
    rate_limit_storage[api_key] = [
        t for t in rate_limit_storage[api_key] if t > window_start
    ]

    # Rate limit 확인
    if is_batch:
        limit = RATE_LIMIT_BATCH_PER_MINUTE
    elif api_key.startswith("eot_premium_"):
        limit = RATE_LIMIT_PREMIUM_PER_MINUTE
    else:
        limit = RATE_LIMIT_PER_MINUTE

    if len(rate_limit_storage[api_key]) >= limit:
        return False

    rate_limit_storage[api_key].append(current_time)
    return True


def create_response(
    success: bool,
    data: Optional[Dict] = None,
    error: Optional[Dict] = None,
    request_id: Optional[str] = None
) -> EOTPredictResponse:
    """표준 응답 생성"""
    return EOTPredictResponse(
        success=success,
        data=data,
        error=error,
        meta={
            "api_version": API_VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id or get_request_id()
        }
    )


# ========== API 엔드포인트 ==========

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "name": "EOT Prediction API",
        "version": API_VERSION,
        "description": "한국어 채팅 발화 종료 확률 예측 서비스",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "reason": "EOT predictor not loaded",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": API_VERSION,
        "model_loaded": predictor is not None
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """상세 헬스 체크"""
    components = {}

    # EOT 예측기 상태
    if predictor:
        components["predictor"] = {
            "status": "healthy",
            "loaded": True,
            "eot_tokens": len(predictor.eot_tokens),
            "user_eot_tokens": len(predictor.user_eot_tokens),
            "punctuation_count": len(predictor.punctuation)
        }

        # 캐시 상태
        if hasattr(predictor, 'cache_service'):
            cache_stats = predictor.cache_service.get_stats()
            components["cache"] = {
                "status": "healthy",
                "enabled": True,
                "size_kb": cache_stats.get("size_kb", 0),
                "items": cache_stats.get("items", 0)
            }
    else:
        components["predictor"] = {
            "status": "unhealthy",
            "loaded": False
        }

    overall_status = "healthy" if all(
        c.get("status") == "healthy" for c in components.values()
    ) else "unhealthy"

    return {
        "status": overall_status,
        "components": components,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": API_VERSION
    }


@app.post("/predict/eot")
async def predict_eot(
    request: EOTPredictRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """EOT 확률 예측"""
    request_id = get_request_id()

    # API 키 확인 (개발 환경에서는 선택적)
    if api_key is None:
        api_key = "anonymous"

    # Rate limiting
    if not await check_rate_limit(api_key):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=create_response(
                success=False,
                error={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "요청 제한 초과",
                    "details": {
                        "limit": RATE_LIMIT_PER_MINUTE,
                        "reset_at": datetime.utcnow().isoformat() + "Z"
                    }
                },
                request_id=request_id
            ).dict()
        )

    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_NOT_LOADED",
                    "message": "모델이 로드되지 않음"
                },
                request_id=request_id
            ).dict()
        )

    # 요청된 모델과 현재 로드된 모델이 다른 경우 에러 반환
    if request.model != current_model:
        logger.warning(f"모델 불일치: 요청={request.model}, 현재={current_model}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_MISMATCH",
                    "message": f"요청된 모델({request.model})이 서버에 로드된 모델({current_model})과 다릅니다",
                    "details": {
                        "requested_model": request.model,
                        "current_model": current_model,
                        "suggestion": f"서버를 재시작하거나 model 파라미터를 '{current_model}'로 설정하세요"
                    }
                },
                request_id=request_id
            ).dict()
        )

    try:
        # 타임아웃 처리
        start_time = time.time()

        # EOT 예측
        eot_prob, details = predictor.predict_eot(
            text=request.text,
            top_k=request.top_k,
            temperature=request.temperature
        )

        elapsed_time = time.time() - start_time

        # 타임아웃 체크
        if request.timeout > 0 and elapsed_time > request.timeout:
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content=create_response(
                    success=False,
                    error={
                        "code": "TIMEOUT",
                        "message": f"예측 타임아웃 초과 ({request.timeout}초)",
                        "details": {
                            "timeout": request.timeout,
                            "elapsed": elapsed_time
                        }
                    },
                    request_id=request_id
                ).dict()
            )

        # 예측 결과 포맷팅
        predictions = []
        for i, (token, prob, is_eot) in enumerate(details, 1):
            # 토큰 타입 판별
            token_type = "general"
            if is_eot:
                if token in predictor.eot_tokens:
                    token_type = "eot_expression"
                elif token in predictor.user_eot_tokens:
                    token_type = "user_defined"
                elif any(p in token for p in predictor.punctuation):
                    token_type = "punctuation"
                elif len(token) > 10:
                    token_type = "abnormal"
                else:
                    token_type = "other_eot"

            predictions.append({
                "token": token,
                "probability": round(prob, 4),
                "rank": i,
                "is_eot": is_eot,
                "type": token_type
            })

        # 응답 생성
        return create_response(
            success=True,
            data={
                "eot_probability": round(eot_prob, 4),
                "predictions": predictions,
                "input_text": request.text,
                "model": current_model,  # 실제 사용된 모델
                "elapsed_time": round(elapsed_time, 3),
                "eot_assessment": (
                    "high" if eot_prob > 0.7 else
                    "medium" if eot_prob > 0.3 else
                    "low"
                )
            },
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"예측 오류: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_response(
                success=False,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "내부 서버 오류",
                    "details": str(e)
                },
                request_id=request_id
            ).dict()
        )


@app.post("/predict/batch")
async def predict_batch(
    request: BatchEOTPredictRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """배치 EOT 예측"""
    request_id = get_request_id()

    if api_key is None:
        api_key = "anonymous"

    # Rate limiting (배치는 더 엄격)
    if not await check_rate_limit(api_key, is_batch=True):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=create_response(
                success=False,
                error={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "배치 요청 제한 초과",
                    "details": {
                        "limit": RATE_LIMIT_BATCH_PER_MINUTE,
                        "reset_at": datetime.utcnow().isoformat() + "Z"
                    }
                },
                request_id=request_id
            ).dict()
        )

    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_NOT_LOADED",
                    "message": "모델이 로드되지 않음"
                },
                request_id=request_id
            ).dict()
        )

    # 요청된 모델과 현재 로드된 모델이 다른 경우 에러 반환
    if request.model != current_model:
        logger.warning(f"모델 불일치 (배치): 요청={request.model}, 현재={current_model}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_MISMATCH",
                    "message": f"요청된 모델({request.model})이 서버에 로드된 모델({current_model})과 다릅니다",
                    "details": {
                        "requested_model": request.model,
                        "current_model": current_model,
                        "suggestion": f"model 파라미터를 '{current_model}'로 설정하세요"
                    }
                },
                request_id=request_id
            ).dict()
        )

    try:
        results = []
        total_start = time.time()
        success_count = 0
        failure_count = 0

        for text in request.texts:
            try:
                start_time = time.time()

                # EOT 예측
                eot_prob, details = predictor.predict_eot(
                    text=text,
                    top_k=request.top_k,
                    temperature=request.temperature
                )

                elapsed = time.time() - start_time

                # 간단한 결과만 포함
                results.append({
                    "text": text,
                    "eot_probability": round(eot_prob, 4),
                    "eot_assessment": (
                        "high" if eot_prob > 0.7 else
                        "medium" if eot_prob > 0.3 else
                        "low"
                    ),
                    "top_prediction": details[0][0] if details else None,
                    "elapsed_time": round(elapsed, 3),
                    "success": True
                })
                success_count += 1

            except Exception as e:
                results.append({
                    "text": text,
                    "success": False,
                    "error": str(e)
                })
                failure_count += 1

        total_elapsed = time.time() - total_start

        return create_response(
            success=True,
            data={
                "results": results,
                "total_count": len(request.texts),
                "success_count": success_count,
                "failure_count": failure_count,
                "total_elapsed_time": round(total_elapsed, 3),
                "model": current_model  # 실제 사용된 모델
            },
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"배치 예측 오류: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_response(
                success=False,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "내부 서버 오류",
                    "details": str(e)
                },
                request_id=request_id
            ).dict()
        )


@app.post("/predict/context")
async def predict_context(
    request: ContextEOTPredictRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """컨텍스트 기반 EOT 예측"""
    request_id = get_request_id()

    if api_key is None:
        api_key = "anonymous"

    # Rate limiting
    if not await check_rate_limit(api_key):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=create_response(
                success=False,
                error={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "요청 제한 초과",
                    "details": {
                        "limit": RATE_LIMIT_PER_MINUTE,
                        "reset_at": datetime.utcnow().isoformat() + "Z"
                    }
                },
                request_id=request_id
            ).dict()
        )

    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_NOT_LOADED",
                    "message": "모델이 로드되지 않음"
                },
                request_id=request_id
            ).dict()
        )

    # 요청된 모델과 현재 로드된 모델이 다른 경우 에러 반환
    if request.model != current_model:
        logger.warning(f"모델 불일치 (컨텍스트): 요청={request.model}, 현재={current_model}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=create_response(
                success=False,
                error={
                    "code": "MODEL_MISMATCH",
                    "message": f"요청된 모델({request.model})이 서버에 로드된 모델({current_model})과 다릅니다",
                    "details": {
                        "requested_model": request.model,
                        "current_model": current_model,
                        "suggestion": f"model 파라미터를 '{current_model}'로 설정하세요"
                    }
                },
                request_id=request_id
            ).dict()
        )

    try:
        # 컨텍스트를 하나의 텍스트로 결합
        combined_text = " ".join(request.context)

        start_time = time.time()

        # EOT 예측
        eot_prob, details = predictor.predict_eot(
            text=combined_text,
            top_k=request.top_k,
            temperature=request.temperature
        )

        elapsed_time = time.time() - start_time

        # 예측 결과 포맷팅
        predictions = []
        for i, (token, prob, is_eot) in enumerate(details[:5], 1):  # 상위 5개만
            predictions.append({
                "token": token,
                "probability": round(prob, 4),
                "rank": i,
                "is_eot": is_eot
            })

        return create_response(
            success=True,
            data={
                "eot_probability": round(eot_prob, 4),
                "predictions": predictions,
                "context_length": len(request.context),
                "combined_length": len(combined_text),
                "model": current_model,  # 실제 사용된 모델
                "elapsed_time": round(elapsed_time, 3),
                "eot_assessment": (
                    "high" if eot_prob > 0.7 else
                    "medium" if eot_prob > 0.3 else
                    "low"
                ),
                "recommendation": (
                    "대화를 종료해도 좋습니다" if eot_prob > 0.7 else
                    "상황에 따라 결정하세요" if eot_prob > 0.3 else
                    "대화를 계속하는 것이 좋습니다"
                )
            },
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"컨텍스트 예측 오류: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_response(
                success=False,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "내부 서버 오류",
                    "details": str(e)
                },
                request_id=request_id
            ).dict()
        )


@app.get("/stats")
async def get_stats():
    """API 통계"""
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Service not ready"}
        )

    cache_stats = {}
    if hasattr(predictor, 'cache_service'):
        cache_stats = predictor.cache_service.get_stats()

    return {
        "api_version": API_VERSION,
        "predictor": {
            "eot_tokens": len(predictor.eot_tokens),
            "user_eot_tokens": len(predictor.user_eot_tokens),
            "punctuation_marks": len(predictor.punctuation)
        },
        "cache": cache_stats,
        "rate_limits": {
            "standard": RATE_LIMIT_PER_MINUTE,
            "premium": RATE_LIMIT_PREMIUM_PER_MINUTE,
            "batch": RATE_LIMIT_BATCH_PER_MINUTE
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/models")
async def get_models():
    """사용 가능한 모델 목록"""
    return {
        "models": [
            {
                "id": "polyglot",
                "name": "EleutherAI Polyglot-Ko",
                "description": "한국어 특화 언어 모델",
                "params": "1.3B",
                "default": True,
                "loaded": current_model == "polyglot"
            },
            {
                "id": "kogpt2",
                "name": "SKT KoGPT2",
                "description": "가볍고 빠른 한국어 GPT",
                "params": "125M",
                "default": False,
                "loaded": current_model == "kogpt2"
            },
            {
                "id": "kanana-nano-2.1b-base",
                "name": "Kanana Nano",
                "description": "경량 한국어 모델",
                "params": "2.1B",
                "default": False,
                "loaded": current_model == "kanana-nano-2.1b-base"
            }
        ],
        "current": current_model
    }


# ========== 메인 실행 ==========

@click.command()
@click.option(
    '--model',
    '-m',
    default='polyglot',
    help='사용할 모델 (kogpt2, polyglot, kanana-nano-2.1b-base, kanana-nano-2.1b-instruct)'
)
@click.option(
    '--run-mode',
    default='auto',
    type=click.Choice(['auto', 'cpu', 'nvidia-gpu', 'amd-gpu']),
    help='실행 모드: auto(자동감지), cpu, nvidia-gpu, amd-gpu'
)
@click.option(
    '--host',
    default=DEFAULT_HOST,
    help=f'API 서버 호스트 (기본값: {DEFAULT_HOST})'
)
@click.option(
    '--port',
    '-p',
    default=DEFAULT_PORT,
    type=int,
    help=f'API 서버 포트 (기본값: {DEFAULT_PORT})'
)
@click.option(
    '--list-models',
    is_flag=True,
    help='사용 가능한 모델 목록 표시'
)
@click.option(
    '--reload',
    is_flag=True,
    default=False,
    help='개발 모드: 코드 변경 시 자동 재시작'
)
def main(model, run_mode, host, port, list_models, reload):
    """
    EOT (End-of-Turn) Prediction REST API Server

    한국어 채팅 문맥에서 발화 종료 확률을 예측하는 REST API 서비스

    사용 예시:
      python eot_api.py --model kogpt2 --port 8000
      python eot_api.py --model polyglot --run-mode cpu
      python eot_api.py --list-models
    """
    # 모델 목록 표시
    if list_models:
        print("\n사용 가능한 모델 목록:\n")
        print("  1. kogpt2                      - SKT KoGPT2 (125M, 빠름)")
        print("  2. polyglot                    - EleutherAI Polyglot-Ko (1.3B, 기본값)")
        print("  3. kanana-nano-2.1b-base       - Kakao Kanana Base (2.1B)")
        print("  4. kanana-nano-2.1b-instruct   - Kakao Kanana Instruct (2.1B)")
        print("\n사용 예시:")
        print("  python eot_api.py --model kogpt2")
        print("  python eot_api.py --model polyglot --port 8177")
        print("  python eot_api.py --model kanana-nano-2.1b-base --run-mode nvidia-gpu")
        return

    # 커맨드라인 인자를 환경 변수로 설정 (lifespan에서 읽을 수 있도록)
    os.environ["EOT_API_MODEL"] = model
    os.environ["EOT_API_RUN_MODE"] = run_mode
    os.environ["EOT_API_HOST"] = host
    os.environ["EOT_API_PORT"] = str(port)

    print(f"\n{'='*60}")
    print(f"  EOT Prediction API Server")
    print(f"  버전: {API_VERSION}")
    print(f"{'='*60}")
    print(f"  모델:       {model}")
    print(f"  실행 모드:  {run_mode}")
    print(f"  호스트:     {host}")
    print(f"  포트:       {port}")
    print(f"  API 문서:   http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"{'='*60}\n")

    # Uvicorn 서버 실행
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=reload
    )


if __name__ == "__main__":
    main()