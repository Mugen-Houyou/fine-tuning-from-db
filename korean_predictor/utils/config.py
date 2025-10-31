"""
Configuration - 설정 관리
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """애플리케이션 설정"""

    # 모델 설정
    DEFAULT_MODEL = 'kogpt2'  # 기본 모델
    MODEL_CACHE_DIR = os.getenv(
        'MODEL_CACHE_DIR',
        str(Path.home() / '.cache' / 'korean_predictor' / 'models')
    )

    # 예측 설정
    DEFAULT_TOP_K = 10  # 기본 예측 개수
    DEFAULT_TEMPERATURE = 1.3  # 기본 온도
    DEFAULT_TIMEOUT = 60  # 기본 타임아웃 (초)
    MAX_INPUT_LENGTH = 128  # 최대 입력 길이
    MAX_WORD_LENGTH = 15  # 어절 최대 길이
    COMPLETE_WORD = True  # 완전한 어절 생성
    INCLUDE_SPECIAL_TOKENS = True  # 특수 토큰 포함

    # 캐시 설정
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_DIR = os.getenv(
        'CACHE_DIR',
        str(Path.home() / '.cache' / 'korean_predictor' / 'cache')
    )
    CACHE_SIZE_MB = int(os.getenv('CACHE_SIZE_MB', '100'))
    CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '300'))

    # 로깅 설정
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', None)

    # CLI 설정
    INTERACTIVE_MODE = True  # 대화형 모드 기본값
    SHOW_CONFIDENCE = True  # 신뢰도 점수 표시
    COLOR_OUTPUT = True  # 컬러 출력

    # 성능 설정
    USE_GPU = os.getenv('USE_GPU', 'auto')  # auto, true, false
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """모델 경로 반환"""
        model_paths = {
            'kogpt2': 'skt/kogpt2-base-v2',
            'kogpt2-small': 'skt/kogpt2-base-v2',
            'polyglot-small': 'EleutherAI/polyglot-ko-1.3b',
            'kcgpt2': 'beomi/KcGPT2',
        }
        return model_paths.get(model_name, model_name)

    @classmethod
    def setup_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            Path(cls.MODEL_CACHE_DIR),
            Path(cls.CACHE_DIR),
        ]

        if cls.LOG_FILE:
            dirs.append(Path(cls.LOG_FILE).parent)

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls) -> dict:
        """설정을 딕셔너리로 반환"""
        return {
            'model': {
                'default': cls.DEFAULT_MODEL,
                'cache_dir': cls.MODEL_CACHE_DIR,
            },
            'prediction': {
                'top_k': cls.DEFAULT_TOP_K,
                'temperature': cls.DEFAULT_TEMPERATURE,
                'max_input_length': cls.MAX_INPUT_LENGTH,
                'max_word_length': cls.MAX_WORD_LENGTH,
            },
            'cache': {
                'enabled': cls.CACHE_ENABLED,
                'dir': cls.CACHE_DIR,
                'size_mb': cls.CACHE_SIZE_MB,
                'ttl_seconds': cls.CACHE_TTL_SECONDS,
            },
            'logging': {
                'level': cls.LOG_LEVEL,
                'file': cls.LOG_FILE,
            },
            'cli': {
                'interactive': cls.INTERACTIVE_MODE,
                'show_confidence': cls.SHOW_CONFIDENCE,
                'color_output': cls.COLOR_OUTPUT,
            },
            'performance': {
                'use_gpu': cls.USE_GPU,
                'batch_size': cls.BATCH_SIZE,
            }
        }