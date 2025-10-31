"""Response Schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionItem(BaseModel):
    """개별 예측 결과"""
    token: str = Field(..., description="예측된 토큰/어절")
    probability: float = Field(..., ge=0.0, le=1.0, description="확률")
    rank: int = Field(..., ge=1, description="순위")
    type: str = Field(default="normal", description="토큰 타입 (normal, eos, special)")


class MetaInfo(BaseModel):
    """응답 메타 정보"""
    api_version: str = Field(default="1.0.0", description="API 버전")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="응답 시각")
    request_id: str = Field(..., description="요청 ID")


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(default=None, description="추가 상세 정보")


class PredictData(BaseModel):
    """예측 응답 데이터"""
    predictions: List[PredictionItem] = Field(..., description="예측 결과 목록")
    eot_probability: float = Field(..., ge=0.0, le=1.0, description="End-of-Turn 확률")
    input_text: str = Field(..., description="입력 텍스트")
    model: str = Field(..., description="사용된 모델")
    elapsed_time: float = Field(..., ge=0.0, description="소요 시간 (초)")


class PredictResponse(BaseModel):
    """예측 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[PredictData] = Field(default=None, description="응답 데이터")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class ContextPredictData(BaseModel):
    """컨텍스트 예측 응답 데이터"""
    predictions: List[PredictionItem] = Field(..., description="예측 결과 목록")
    eot_probability: float = Field(..., ge=0.0, le=1.0, description="End-of-Turn 확률")
    context_length: int = Field(..., ge=1, description="컨텍스트 길이")
    model: str = Field(..., description="사용된 모델")
    elapsed_time: float = Field(..., ge=0.0, description="소요 시간 (초)")


class ContextPredictResponse(BaseModel):
    """컨텍스트 예측 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[ContextPredictData] = Field(default=None, description="응답 데이터")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class BatchResultItem(BaseModel):
    """배치 예측 개별 결과"""
    text: str = Field(..., description="입력 텍스트")
    predictions: List[PredictionItem] = Field(..., description="예측 결과")
    eot_probability: float = Field(..., ge=0.0, le=1.0, description="End-of-Turn 확률")
    elapsed_time: float = Field(..., ge=0.0, description="소요 시간 (초)")


class BatchPredictData(BaseModel):
    """배치 예측 응답 데이터"""
    results: List[BatchResultItem] = Field(..., description="배치 예측 결과")
    total_count: int = Field(..., ge=0, description="전체 개수")
    success_count: int = Field(..., ge=0, description="성공 개수")
    failure_count: int = Field(..., ge=0, description="실패 개수")
    total_elapsed_time: float = Field(..., ge=0.0, description="총 소요 시간 (초)")


class BatchPredictResponse(BaseModel):
    """배치 예측 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[BatchPredictData] = Field(default=None, description="응답 데이터")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class ModelInfo(BaseModel):
    """모델 정보"""
    model_name: str = Field(..., description="모델 전체 경로")
    model_id: str = Field(..., description="모델 ID")
    vocab_size: int = Field(..., description="어휘 크기")
    max_length: int = Field(..., description="최대 길이")
    parameters: float = Field(..., description="파라미터 수 (M)")
    device: str = Field(..., description="디바이스")
    supports_temperature: bool = Field(..., description="Temperature 지원 여부")
    loaded_at: datetime = Field(..., description="로드 시각")
    memory_usage_mb: Optional[float] = Field(default=None, description="메모리 사용량 (MB)")


class CurrentModelResponse(BaseModel):
    """현재 모델 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[ModelInfo] = Field(default=None, description="모델 정보")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class AvailableModel(BaseModel):
    """사용 가능한 모델 정보"""
    id: str = Field(..., description="모델 ID")
    name: str = Field(..., description="모델 이름")
    description: str = Field(..., description="설명")
    params: str = Field(..., description="파라미터 수")
    memory: str = Field(..., description="필요 메모리")
    supports_temperature: bool = Field(..., description="Temperature 지원 여부")
    reasoning_model: bool = Field(..., description="추론 특화 모델 여부")
    supported_modes: List[str] = Field(..., description="지원 모드")


class ModelsListData(BaseModel):
    """모델 목록 데이터"""
    models: List[AvailableModel] = Field(..., description="모델 목록")


class ModelsListResponse(BaseModel):
    """모델 목록 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[ModelsListData] = Field(default=None, description="모델 목록 데이터")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class CacheStats(BaseModel):
    """캐시 통계"""
    enabled: bool = Field(..., description="캐시 활성화 여부")
    size_kb: float = Field(..., description="캐시 크기 (KB)")
    items: int = Field(..., description="항목 수")
    hits: int = Field(..., description="히트 수")
    misses: int = Field(..., description="미스 수")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="히트율")
    ttl_seconds: int = Field(..., description="TTL (초)")


class CacheStatsResponse(BaseModel):
    """캐시 통계 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[CacheStats] = Field(default=None, description="캐시 통계")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class CacheClearData(BaseModel):
    """캐시 삭제 데이터"""
    message: str = Field(..., description="결과 메시지")
    items_deleted: int = Field(..., ge=0, description="삭제된 항목 수")


class CacheClearResponse(BaseModel):
    """캐시 삭제 응답"""
    success: bool = Field(..., description="성공 여부")
    data: Optional[CacheClearData] = Field(default=None, description="삭제 결과")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 정보")
    meta: MetaInfo = Field(..., description="메타 정보")


class HealthStatus(BaseModel):
    """헬스체크 상태"""
    status: str = Field(..., description="상태 (healthy, unhealthy)")
    reason: Optional[str] = Field(default=None, description="unhealthy일 경우 이유")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="체크 시각")
    uptime_seconds: int = Field(..., ge=0, description="가동 시간 (초)")
    version: str = Field(default="1.0.0", description="API 버전")


class ComponentHealth(BaseModel):
    """컴포넌트 헬스 상태"""
    status: str = Field(..., description="상태")
    loaded: Optional[bool] = Field(default=None, description="로드 여부")
    model_name: Optional[str] = Field(default=None, description="모델 이름")
    enabled: Optional[bool] = Field(default=None, description="활성화 여부")
    size_kb: Optional[float] = Field(default=None, description="크기 (KB)")
    type: Optional[str] = Field(default=None, description="타입")
    memory_used_mb: Optional[float] = Field(default=None, description="사용 메모리 (MB)")
    memory_total_mb: Optional[float] = Field(default=None, description="전체 메모리 (MB)")


class DetailedHealthStatus(BaseModel):
    """상세 헬스체크 상태"""
    status: str = Field(..., description="전체 상태")
    components: Dict[str, ComponentHealth] = Field(..., description="컴포넌트별 상태")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="체크 시각")
    uptime_seconds: int = Field(..., ge=0, description="가동 시간 (초)")
    version: str = Field(default="1.0.0", description="API 버전")
