# 빠른 시작 가이드 (Quick Start)

한국어 다음 토큰 예측기를 5분 안에 실행해보세요.

---

## 1. 파이썬 설치

### Windows
1. [Python 3.12 이상 다운로드](https://www.python.org/downloads/)
2. 설치 시 **"Add Python to PATH"** 체크 필수
3. 설치 확인:
   ```bash
   python --version
   ```
   출력: `Python 3.12.x` 이상

### Mac
```bash
# Homebrew 설치 (없다면)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python@3.12
python3 --version
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

---

## 2. 라이브러리 설치

### requirements.txt 사용

프로젝트 폴더로 이동 후:

```bash
# Windows
pip install -r requirements.txt

# Mac/Linux
pip3 install -r requirements.txt
```

### 설치 확인

```bash
python -c "import torch, transformers; print('설치 완료!')"
```

---

## 3. 실행 방법

### 3.1 기본 실행 (KoGPT2 모델)

**대화형 모드:**
```bash
python main.py
```

**단일 예측:**
```bash
python main.py -t "오늘 날씨가"
```

**출력 예시:**
```
다음 단어 예측:
1. 추      (7.0%)
2. 쌀      (7.4%)
3. 좋      (1.5%)
...
발화 종료 확률: 0.0%
```

### 3.2 Polyglot 모델 (1.3B)

```bash
# 대화형 모드
python main.py --model polyglot

# 단일 예측
python main.py --model polyglot -t "대화를 종료하겠습니다."
```

**출력 예시:**
```
다음 단어 예측:
1. <|end_of_text|>...  (0.0%)  [긴토큰(226)]
2. 2                   (1.2%)  [일반]
3. (I'm done...)       (0.0%)  [긴토큰(26)]
...
발화 종료 확률: 14.2%  ████░░░░░░░░░░░░░░░░░░░░░░░░░░
```

### 3.3 사용 가능한 모델 확인

```bash
python main.py --list-models
```

**주요 모델:**
- `kogpt2` - SKT KoGPT2 (125M, 빠름, 기본값)
- `polyglot` - EleutherAI Polyglot-Ko (1.3B)
- `kanana-nano-2.1b-base` - Kakao Kanana Base (2.1B, 대화 지속 경향 있음)
- `kanana-nano-2.1b-instruct` - Kakao Kanana Instruct (2.1B, 대화 지속 경향 강함)

### 3.4 추가 옵션

**실행 아키텍처 지정:**
```bash
# CPU 모드
python main.py --run-mode cpu --model kogpt2 -t "테스트"

# AMD Radeon GPU 모드
python main.py --run-mode radeon-gpu --model polyglot -t "테스트"

# NVIDIA GPU 모드
python main.py --run-mode nvidia-gpu --model polyglot -t "테스트"

# 자동 감지 (기본값)
python main.py --run-mode auto -t "테스트"
```

**예측 개수 조정:**
```bash
python main.py --top-k 10 -t "오늘 날씨가"
```

**온도 조절 (창의성):**
```bash
python main.py --temperature 1.2 -t "인공지능은"
```

**캐시 비활성화:**
```bash
python main.py --no-cache -t "테스트"
```

**배치 처리 (파일):**
```bash
# input.txt 파일의 각 줄을 처리
python main.py -f input.txt
```

### 3.5 대화형 모드 명령어

대화형 모드에서 사용 가능한 명령:

```
/help              - 도움말
/config            - 현재 설정
/model info        - 모델 정보
/model list        - 모델 목록
/set top_k 10      - 예측 개수 변경
/set temp 1.0      - 온도 변경
/cache             - 캐시 통계
/cache clear       - 캐시 비우기
quit, exit, q      - 종료
```

---

## 4. 테스트 스크립트 실행

간단한 테스트:

```bash
python test_simple.py
```

**테스트 내용:**
- 모델 로딩
- 기본 예측 (3개 예시)
- EOT 확률 계산
- 모델 정보 출력

---

## 주의사항

### 첫 실행 시

- **모델 다운로드**: 첫 실행 시 자동으로 모델을 다운로드합니다
  - KoGPT2: ~500MB (3-5분)
  - Kanana: ~5GB (10-30분)
- **인터넷 연결 필수**: 첫 실행에만 필요, 이후 오프라인 가능
- **디스크 공간**: 모델별 최소 2-8GB 필요

### 메모리 요구사항

| 모델 | CPU | GPU |
|------|-----|-----|
| KoGPT2 | ~2GB RAM | ~2GB VRAM |
| Kanana (Base/Instruct) | ~8GB RAM | ~5GB VRAM |
| Polyglot | ~4GB RAM | ~3GB VRAM |
| DNA-R1 | 불가능 | ~28GB VRAM |

### 모델별 지원 실행 모드

| 모델 | CPU | NVIDIA GPU | AMD GPU |
|------|-----|------------|---------|
| kogpt2 | ✅ | ✅ | ⚠️ |
| polyglot | ✅ | ✅ | ⚠️ |
| kanana-nano-2.1b-base | ✅ | ✅ | ⚠️ |
| kanana-nano-2.1b-instruct | ✅ | ✅ | ⚠️ |
| dna-r1 | ❌ | ✅ | ⚠️ |

- ✅: 완전 지원
- ❌: 지원 안함
- ⚠️: ROCm 또는 DirectML 설치 시 가능

### 성능

- **첫 예측**: 느림 (모델 워밍업)
- **이후 예측**: 빠름 (캐싱)
- **CPU 모드**: 느리지만 작동 (3-20초/예측)
- **GPU 모드**: 빠름 (<1초/예측)

### GPU 사용 (선택사항)

CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다:

```bash
# GPU 사용 확인
python -c "import torch; print('GPU 사용 가능:', torch.cuda.is_available())"
```

### transformers 버전

Kanana 모델 사용 시 transformers >= 4.45.0 필요:

```bash
# 업그레이드
pip install --upgrade transformers
```

### 문제 해결

**"No module named 'torch'"**
```bash
pip install -r requirements.txt
```

**메모리 부족 오류**
- 다른 프로그램 종료
- 더 작은 모델 사용 (kogpt2)

**모델 다운로드 실패**
- 인터넷 연결 확인
- VPN/방화벽 비활성화
- 수동 재시도

**느린 속도 (CPU)**
- 정상입니다. GPU를 이용하면 개선됩니다.
- 캐시가 활성화되면 이후 빨라집니다

**지원되지 않는 조합 시도**

예) DNA-R1을 CPU 모드로 실행
```
⚠️  경고: 모델 'dna-r1'은 cpu 모드를 지원하지 않습니다.
지원 모드: nvidia-gpu

이 조합은 공식적으로 지원되지 않으며, 오류가 발생할 수 있습니다.

강제로 시도하시겠습니까? (Y/N):
```

- **Y 입력**: 강제로 시도 (실패할 수 있음)
- **N 입력**: 취소하고 종료

---

**도움이 필요하신가요?**
- GitHub Issues: 버그 리포트 및 질문
- README.md: 상세 문서
