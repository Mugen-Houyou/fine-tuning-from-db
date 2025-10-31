"""
Model Manager - 한국어 언어 모델 관리
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
from typing import Optional, Tuple
import logging
from pathlib import Path
from packaging import version

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """싱글톤 패턴으로 모델 관리"""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    # 지원하는 모델 목록
    SUPPORTED_MODELS = {
        # 경량 모델 (125M~1.3B)
        'kogpt2': {
            'path': 'skt/kogpt2-base-v2',
            'params': '125M',
            'description': 'SKT KoGPT2 - 가볍고 빠른 한국어 GPT',
            'memory': '~2GB',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },
        'kogpt2-small': {
            'path': 'skt/kogpt2-base-v2',
            'params': '125M',
            'description': 'KoGPT2 별칭',
            'memory': '~2GB',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },
        'polyglot': {
            'path': 'EleutherAI/polyglot-ko-1.3b',
            'params': '1.3B',
            'description': 'EleutherAI Polyglot-Ko - 중형 한국어 모델',
            'memory': '~4GB',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },
        'kcgpt2': {
            'path': 'beomi/KcGPT2',
            'params': '124M',
            'description': 'Beomi KcGPT2 - 경량 한국어 GPT',
            'memory': '~2GB',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },

        # 중대형 모델 (2.1B+)
        'kanana-nano-2.1b-base': {
            'path': 'kakaocorp/kanana-nano-2.1b-base',
            'params': '2.1B',
            'description': 'Kakao Kanana Base - 순수 언어 모델 (2025년, CPU/GPU)',
            'memory': '~5GB (CPU: ~8GB)',
            'use_bfloat16': True,  # GPU에서 BFloat16 권장
            'min_transformers_version': '4.45.0',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },
        'kanana-nano-2.1b-instruct': {
            'path': 'kakaocorp/kanana-nano-2.1b-instruct',
            'params': '2.1B',
            'description': 'Kakao Kanana Instruct - 대화 최적화 (2025년, CPU/GPU)',
            'memory': '~5GB (CPU: ~8GB)',
            'use_bfloat16': True,  # GPU에서 BFloat16 권장
            'min_transformers_version': '4.45.0',
            'supported_modes': ['cpu', 'nvidia-gpu']
        },

        # 대형 추론 모델 (14B+)
        'dna-r1': {
            'path': 'dnotitia/DNA-R1',
            'params': '14B',
            'description': 'DNA-R1 - 추론 특화 한국어 모델 (DeepSeek-R1 방식)',
            'memory': '~28GB (GPU 필수)',
            'reasoning_model': True,
            'supported_modes': ['nvidia-gpu']  # GPU 전용
        },
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """초기화는 한 번만 수행"""
        if self._device is None:
            self._setup_device('auto')  # 기본값은 auto

    def set_run_mode(self, run_mode='auto'):
        """실행 모드 설정

        Args:
            run_mode: 실행 모드 ('auto', 'cpu', 'nvidia-gpu', 'radeon-gpu')
        """
        # 장치를 다시 설정
        self._device = None
        self._setup_device(run_mode)

    def _setup_device(self, run_mode):
        """디바이스 설정

        Args:
            run_mode: 실행 모드 ('auto', 'cpu', 'nvidia-gpu', 'radeon-gpu')
        """
        self._run_mode = run_mode

        if run_mode == 'cpu':
            self._device = torch.device("cpu")
            logger.info("CPU 모드로 실행 (사용자 지정)")

        elif run_mode == 'nvidia-gpu':
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"NVIDIA GPU 사용: {torch.cuda.get_device_name(0)}")
            else:
                logger.error("NVIDIA GPU가 감지되지 않았습니다. CUDA가 설치되어 있는지 확인하세요.")
                raise RuntimeError("NVIDIA GPU를 사용할 수 없습니다.")

        elif run_mode == 'radeon-gpu':
            # ROCm 지원 확인
            try:
                import torch_directml  # Windows에서 DirectML 사용
                self._device = torch_directml.device()
                logger.info("AMD Radeon GPU 사용 (DirectML)")
            except ImportError:
                if hasattr(torch, 'hip') and torch.hip.is_available():  # Linux에서 ROCm 사용
                    self._device = torch.device("cuda")  # ROCm도 cuda 장치 사용
                    logger.info("AMD Radeon GPU 사용 (ROCm)")
                else:
                    logger.error("AMD GPU 지원이 설치되지 않았습니다. ROCm 또는 DirectML 설치가 필요합니다.")
                    raise RuntimeError("AMD GPU를 사용할 수 없습니다.")

        else:  # auto 모드
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"GPU 자동 감지: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("CPU 모드로 실행 (자동)")

    def load_model(
        self,
        model_name: str = 'kogpt2',
        cache_dir: Optional[str] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        모델 로드

        Args:
            model_name: 모델 이름 또는 HuggingFace 경로
            cache_dir: 모델 캐시 디렉토리
            force: 호환성 체크를 무시하고 강제 로드

        Returns:
            (성공 여부, 메시지)
        """
        try:
            # 모델 정보 가져오기
            model_info = self.SUPPORTED_MODELS.get(model_name)

            if model_info:
                # 등록된 모델
                model_path = model_info['path']

                # run-mode 호환성 체크
                supported_modes = model_info.get('supported_modes', ['cpu', 'nvidia-gpu'])
                actual_mode = self._run_mode if self._run_mode != 'auto' else ('nvidia-gpu' if self._device.type == 'cuda' else 'cpu')

                # Radeon GPU는 nvidia-gpu와 호환 가능 (둘 다 cuda 장치 사용)
                if actual_mode == 'radeon-gpu' and 'nvidia-gpu' in supported_modes:
                    pass  # 호환 가능
                elif actual_mode not in supported_modes:
                    if not force:
                        # force가 아니면 확인 필요
                        error_msg = (
                            f"모델 '{model_name}'은 {actual_mode} 모드를 지원하지 않습니다.\n"
                            f"지원 모드: {', '.join(supported_modes)}"
                        )
                        logger.warning(error_msg)
                        return False, error_msg  # 확인 필요 신호
                    else:
                        # force 모드: 경고만 하고 진행
                        logger.warning(f"경고: 모델 '{model_name}'은 {actual_mode} 모드를 공식적으로 지원하지 않습니다. 강제 실행 중...")
                        logger.warning(f"지원 모드: {', '.join(supported_modes)}")

                # transformers 버전 확인
                min_version = model_info.get('min_transformers_version')
                if min_version and version.parse(transformers_version) < version.parse(min_version):
                    error_msg = (
                        f"모델 '{model_name}'은 transformers >= {min_version}이 필요합니다.\n"
                        f"현재 버전: {transformers_version}\n"
                        f"업데이트: pip install --upgrade transformers"
                    )
                    logger.error(error_msg)
                    return False, error_msg

                # 기존 requires_gpu 체크 (호환성을 위해 유지)
                if model_info.get('requires_gpu', False) and not torch.cuda.is_available():
                    error_msg = (
                        f"모델 '{model_name}'은 GPU가 필요합니다.\n"
                        f"필요 메모리: {model_info['memory']}\n"
                        f"현재 GPU를 사용할 수 없습니다."
                    )
                    logger.error(error_msg)
                    return False, error_msg

                # 모델 정보 로깅
                logger.info(f"모델: {model_name}")
                logger.info(f"  - 설명: {model_info['description']}")
                logger.info(f"  - 파라미터: {model_info['params']}")
                logger.info(f"  - 메모리: {model_info['memory']}")
            else:
                # 직접 HuggingFace 경로 지정
                model_path = model_name
                logger.info(f"커스텀 모델 경로: {model_path}")

            # 캐시 디렉토리 설정
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "korean_predictor"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_dir = str(cache_dir)

            logger.info(f"모델 로딩 중: {model_path}")

            # 토크나이저 로드
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True  # DNA-R1 등 커스텀 모델 지원
            )

            # pad_token이 없는 경우 eos_token으로 설정
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 모델 로드 - 메모리 최적화
            model_kwargs = {
                'cache_dir': cache_dir,
                'low_cpu_mem_usage': True,
                'trust_remote_code': True,  # DNA-R1 등 커스텀 모델 지원
            }

            # dtype 설정 (GPU/CPU 구분)
            if self._device.type == 'cuda':
                # GPU: BFloat16 또는 Float16 사용
                if model_info and model_info.get('use_bfloat16', False):
                    model_kwargs['torch_dtype'] = torch.bfloat16
                    logger.info("Using BFloat16 for GPU")
                else:
                    model_kwargs['torch_dtype'] = torch.float16
                    logger.info("Using Float16 for GPU")

                # 대형 모델의 경우 device_map 자동 설정
                if model_info and model_info.get('requires_gpu', False):
                    model_kwargs['device_map'] = 'auto'
            else:
                # CPU: Float32 사용 (BFloat16은 CPU에서 느릴 수 있음)
                model_kwargs['torch_dtype'] = torch.float32
                logger.info("Using Float32 for CPU")

            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )

            # 디바이스로 이동 (device_map='auto'가 아닌 경우만)
            if 'device_map' not in model_kwargs:
                self._model = self._model.to(self._device)

            self._model.eval()  # 평가 모드

            # 메모리 정리
            if self._device.type == 'cuda':
                torch.cuda.empty_cache()

            model_size = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(f"모델 로드 완료: {model_size:.1f}M 파라미터")

            # 성공 메시지 구성
            if model_info:
                success_msg = (
                    f"모델 '{model_name}' 로드 성공\n"
                    f"  {model_info['description']} ({model_info['params']})"
                )
            else:
                success_msg = f"모델 '{model_path}' 로드 성공"

            return True, success_msg

        except Exception as e:
            error_msg = f"모델 로드 실패: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def is_loaded(self) -> bool:
        """모델 로드 여부 확인"""
        return self._model is not None and self._tokenizer is not None

    @property
    def model(self):
        """모델 반환"""
        if not self.is_loaded():
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        return self._model

    @property
    def tokenizer(self):
        """토크나이저 반환"""
        if not self.is_loaded():
            raise RuntimeError("토크나이저가 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        return self._tokenizer

    @property
    def device(self):
        """디바이스 반환"""
        return self._device

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        if not self.is_loaded():
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self._model.config._name_or_path,
            "vocab_size": self._tokenizer.vocab_size,
            "max_length": self._model.config.max_position_embeddings if hasattr(self._model.config, 'max_position_embeddings') else "unknown",
            "device": str(self._device),
            "parameters": sum(p.numel() for p in self._model.parameters()) / 1e6,
        }

    def unload_model(self):
        """모델 언로드 및 메모리 정리"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("모델 언로드 완료")

    @classmethod
    def list_models(cls) -> dict:
        """사용 가능한 모델 목록 반환"""
        return cls.SUPPORTED_MODELS

    @classmethod
    def get_model_metadata(cls, model_name: str) -> Optional[dict]:
        """특정 모델 메타데이터 반환"""
        return cls.SUPPORTED_MODELS.get(model_name)