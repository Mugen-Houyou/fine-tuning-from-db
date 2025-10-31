"""Request Schemas"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictRequest(BaseModel):
    """예측 요청"""
    text: str = Field(..., min_length=1, max_length=2000, description="입력 텍스트")
    model: str = Field(default="kogpt2", description="사용할 모델 ID")
    top_k: int = Field(default=10, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=1.3, ge=0.1, le=2.0, description="샘플링 온도")
    complete_word: bool = Field(default=True, description="완전한 어절까지 생성 여부")
    include_special_tokens: bool = Field(default=True, description="특수 토큰 포함 여부")
    timeout: int = Field(default=60, ge=0, description="타임아웃 (초, 0=무제한)")

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        valid_models = ['kogpt2', 'kanana', 'polyglot-ko-5.8b', 'dna-r1']
        if v not in valid_models:
            raise ValueError(f"Invalid model. Must be one of {valid_models}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요",
                "model": "kogpt2",
                "top_k": 10,
                "temperature": 1.3,
                "complete_word": True,
                "include_special_tokens": True,
                "timeout": 60
            }
        }


class ContextPredictRequest(BaseModel):
    """컨텍스트 기반 예측 요청"""
    context: List[str] = Field(..., min_length=1, max_length=10, description="대화 컨텍스트")
    model: str = Field(default="kogpt2", description="사용할 모델 ID")
    top_k: int = Field(default=10, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=1.3, ge=0.1, le=2.0, description="샘플링 온도")
    timeout: int = Field(default=60, ge=0, description="타임아웃 (초)")

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        valid_models = ['kogpt2', 'kanana', 'polyglot-ko-5.8b', 'dna-r1']
        if v not in valid_models:
            raise ValueError(f"Invalid model. Must be one of {valid_models}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "context": ["안녕하세요", "네 안녕하세요", "오늘 날씨가"],
                "model": "kogpt2",
                "top_k": 10,
                "temperature": 1.3,
                "timeout": 60
            }
        }


class BatchPredictRequest(BaseModel):
    """배치 예측 요청"""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="입력 텍스트 목록")
    model: str = Field(default="kogpt2", description="사용할 모델 ID")
    top_k: int = Field(default=5, ge=1, le=20, description="예측할 토큰 개수")
    temperature: float = Field(default=1.3, ge=0.1, le=2.0, description="샘플링 온도")
    timeout: int = Field(default=120, ge=0, description="타임아웃 (초)")

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        valid_models = ['kogpt2', 'kanana', 'polyglot-ko-5.8b', 'dna-r1']
        if v not in valid_models:
            raise ValueError(f"Invalid model. Must be one of {valid_models}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["안녕하세요", "오늘 날씨는", "저는 학생"],
                "model": "kogpt2",
                "top_k": 5,
                "temperature": 1.3,
                "timeout": 120
            }
        }
