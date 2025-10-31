"""Prediction Routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.request import PredictRequest, ContextPredictRequest, BatchPredictRequest
from api.schemas.response import (
    PredictResponse, PredictData, PredictionItem,
    ContextPredictResponse, ContextPredictData,
    BatchPredictResponse, BatchPredictData, BatchResultItem,
    ErrorDetail, MetaInfo
)
from api.dependencies import get_predictor, get_model_manager
from api.middleware.auth import verify_api_key
from korean_predictor.models.predictor import PredictionService
from korean_predictor.models.model_manager import ModelManager
import logging
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


def create_prediction_items(predictions: list) -> list[PredictionItem]:
    """예측 결과를 PredictionItem 리스트로 변환"""
    items = []
    for rank, (token, prob) in enumerate(predictions, 1):
        # 토큰 타입 판단
        if token in ['</s>', '<eos>', '</d>', '<|end_of_text|>']:
            token_type = "eos"
        elif token in ['<pad>', '<unk>', '<bos>']:
            token_type = "special"
        else:
            token_type = "normal"

        items.append(PredictionItem(
            token=token,
            probability=prob,
            rank=rank,
            type=token_type
        ))
    return items


@router.post("", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    predictor: PredictionService = Depends(get_predictor),
    model_manager: ModelManager = Depends(get_model_manager),
    api_key: str = Depends(verify_api_key)
):
    """
    다음 토큰/어절 예측 및 EOT 확률 반환

    - **text**: 입력 텍스트 (필수)
    - **model**: 사용할 모델 ID (기본값: kogpt2)
    - **top_k**: 예측할 토큰 개수 (1-20, 기본값: 10)
    - **temperature**: 샘플링 온도 (0.1-2.0, 기본값: 1.3)
    - **complete_word**: 완전한 어절까지 생성 (기본값: true)
    - **include_special_tokens**: 특수 토큰 포함 (기본값: true)
    - **timeout**: 타임아웃 초 (기본값: 60, 0=무제한)
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    logger.info(f"[{request_id}] Prediction request: model={request.model}, top_k={request.top_k}")

    try:
        # 모델 확인 및 필요시 전환 (간단 구현: 현재는 지원하지 않음)
        current_model = model_manager.get_model_info()
        if not current_model.get('loaded'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "MODEL_NOT_LOADED",
                    "message": "Model not loaded",
                    "details": None
                }
            )

        # 예측 수행
        try:
            timeout_val = request.timeout if request.timeout > 0 else None
            eot_prob, predictions = predictor.predict_next_tokens(
                text=request.text,
                top_k=request.top_k,
                temperature=request.temperature,
                complete_word=request.complete_word,
                include_special_tokens=request.include_special_tokens,
                timeout=timeout_val
            )
        except TimeoutError as e:
            elapsed = time.time() - start_time
            logger.warning(f"[{request_id}] Prediction timeout after {elapsed:.2f}s")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={
                    "code": "TIMEOUT",
                    "message": f"Prediction timeout exceeded ({request.timeout} seconds)",
                    "details": {
                        "timeout": request.timeout,
                        "elapsed": elapsed
                    }
                }
            )

        elapsed = time.time() - start_time

        # 응답 생성
        prediction_items = create_prediction_items(predictions)

        response = PredictResponse(
            success=True,
            data=PredictData(
                predictions=prediction_items,
                eot_probability=eot_prob,
                input_text=request.text,
                model=current_model.get('model_name', 'unknown'),
                elapsed_time=elapsed
            ),
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Prediction completed in {elapsed:.3f}s, {len(predictions)} results")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )


@router.post("/context", response_model=ContextPredictResponse)
async def predict_context(
    request: ContextPredictRequest,
    predictor: PredictionService = Depends(get_predictor),
    model_manager: ModelManager = Depends(get_model_manager),
    api_key: str = Depends(verify_api_key)
):
    """
    컨텍스트 기반 예측

    - **context**: 대화 컨텍스트 (1-10개 문장)
    - **model**: 사용할 모델 ID
    - **top_k**: 예측할 토큰 개수
    - **temperature**: 샘플링 온도
    - **timeout**: 타임아웃 초
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    logger.info(f"[{request_id}] Context prediction request: context_length={len(request.context)}")

    try:
        # 컨텍스트 결합
        combined_text = " ".join(request.context)

        # 예측 수행
        try:
            timeout_val = request.timeout if request.timeout > 0 else None
            eot_prob, predictions = predictor.predict_next_tokens(
                text=combined_text,
                top_k=request.top_k,
                temperature=request.temperature,
                complete_word=True,
                include_special_tokens=True,
                timeout=timeout_val
            )
        except TimeoutError:
            elapsed = time.time() - start_time
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={
                    "code": "TIMEOUT",
                    "message": f"Prediction timeout exceeded ({request.timeout} seconds)",
                    "details": {"timeout": request.timeout, "elapsed": elapsed}
                }
            )

        elapsed = time.time() - start_time
        prediction_items = create_prediction_items(predictions)

        current_model = model_manager.get_model_info()
        response = ContextPredictResponse(
            success=True,
            data=ContextPredictData(
                predictions=prediction_items,
                eot_probability=eot_prob,
                context_length=len(request.context),
                model=current_model.get('model_name', 'unknown'),
                elapsed_time=elapsed
            ),
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Context prediction completed in {elapsed:.3f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Context prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    predictor: PredictionService = Depends(get_predictor),
    model_manager: ModelManager = Depends(get_model_manager),
    api_key: str = Depends(verify_api_key)
):
    """
    배치 예측 (여러 텍스트 일괄 처리)

    - **texts**: 입력 텍스트 목록 (1-100개)
    - **model**: 사용할 모델 ID
    - **top_k**: 예측할 토큰 개수
    - **temperature**: 샘플링 온도
    - **timeout**: 타임아웃 초
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    logger.info(f"[{request_id}] Batch prediction request: {len(request.texts)} texts")

    try:
        results = []
        success_count = 0
        failure_count = 0

        for text in request.texts:
            item_start = time.time()
            try:
                timeout_val = request.timeout if request.timeout > 0 else None
                eot_prob, predictions = predictor.predict_next_tokens(
                    text=text,
                    top_k=request.top_k,
                    temperature=request.temperature,
                    complete_word=True,
                    include_special_tokens=True,
                    timeout=timeout_val
                )
                item_elapsed = time.time() - item_start
                prediction_items = create_prediction_items(predictions)

                results.append(BatchResultItem(
                    text=text,
                    predictions=prediction_items,
                    eot_probability=eot_prob,
                    elapsed_time=item_elapsed
                ))
                success_count += 1

            except Exception as e:
                logger.warning(f"[{request_id}] Batch item failed: {text[:30]}... - {str(e)}")
                failure_count += 1
                # 실패한 항목은 빈 결과로 추가
                results.append(BatchResultItem(
                    text=text,
                    predictions=[],
                    eot_probability=0.0,
                    elapsed_time=time.time() - item_start
                ))

        total_elapsed = time.time() - start_time

        response = BatchPredictResponse(
            success=True,
            data=BatchPredictData(
                results=results,
                total_count=len(request.texts),
                success_count=success_count,
                failure_count=failure_count,
                total_elapsed_time=total_elapsed
            ),
            meta=MetaInfo(request_id=request_id)
        )

        logger.info(f"[{request_id}] Batch prediction completed: {success_count}/{len(request.texts)} succeeded")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": {"error": str(e)}
            }
        )
