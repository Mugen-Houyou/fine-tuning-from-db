# 한국어 다음 토큰 예측기

한국어 텍스트의 다음 토큰을 예측하고 화자의 발화 종료(End-of-Turn) 확률을 분석하는 CLI 기반 한국어 언어 모델 도구입니다.

**주요 기능:**
- ✅ 다음 토큰/어절 예측 (Top-K)
- ✅ End-of-Turn (EOT) 확률 자동 계산
- ✅ 다양한 한국어 모델 지원 (KoGPT2, Polyglot, Kanana, DNA-R1)
- ✅ CPU/GPU 자동 감지 및 여러 GPU 지원 (NVIDIA, AMD)
- ✅ 대화형/비대화형/배치 처리 모드
- ✅ 예측 캐싱으로 빠른 응답
- ✅ Rich CLI로 보기 좋은 출력
- 🆕 REST API 지원 (FastAPI 기반)

## 목차

1. [설치 및 준비](#1-설치-및-준비)
2. [기본 사용법](#2-기본-사용법)
3. [모델 선택](#3-모델-선택)
4. [실행 모드 지정](#4-실행-모드-지정)
5. [고급 사용법](#5-고급-사용법)
6. [실전 활용 예시](#6-실전-활용-예시)
7. [팁과 트릭](#7-팁과-트릭)
8. [자주 묻는 질문](#8-자주-묻는-질문)
9. [REST API 사용법](#9-rest-api-사용법)

---

## 1. 설치 및 준비

### 1.1 필수 요구사항 확인

**Python 버전 확인**
```bash
python --version
# 또는 Mac/Linux
python3 --version
```
✅ Python 3.12 이상이어야 합니다.

**메모리 확인**
| 모델 | CPU RAM | GPU VRAM |
|------|---------|----------|
| KoGPT2 | ~2GB | ~2GB |
| Kanana (Base/Instruct) | ~8GB | ~5GB |
| Polyglot | ~4GB | ~3GB |
| DNA-R1 | 지원 안함 | ~28GB |

### 1.2 프로젝트 다운로드

프로젝트를 다운로드하거나 clone 후, 프로젝트 디렉토리로 이동합니다.

```bash
cd fine-tuning-from-db
```

### 1.3 가상환경 생성 (권장)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

가상환경이 활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다.

### 1.4 패키지 설치

```bash
pip install -r requirements.txt
```

이 과정에서 다음 패키지들이 설치됩니다:
- PyTorch (딥러닝 프레임워크)
- Transformers (Hugging Face 모델, >= 4.45.0)
- Rich (터미널 UI)
- Click (CLI 프레임워크)
- 기타 유틸리티

⏱️ 소요 시간: 3-10분 (네트워크 속도에 따라 다름)

### 1.5 설치 확인

```bash
python test_simple.py
```

이 테스트를 실행하면:
1. 모델이 자동으로 다운로드됩니다 (첫 실행 시만)
2. 간단한 예측 테스트를 수행합니다
3. 모델 정보를 출력합니다

✅ 모든 테스트가 통과하면 설치 완료!

---

## 2. 기본 사용법

### 2.1 대화형 모드 시작

가장 기본적인 사용 방법입니다.

```bash
python main.py
```

**화면 예시:**
```
╔══════════════════════════════════════════╗
║  한국어 다음 토큰 예측기 v1.0.0          ║
║  Korean Token Predictor                  ║
╚══════════════════════════════════════════╝

모델 초기화 중... (kogpt2, 모드: auto)
모델 로딩 중...
✓ 모델 'kogpt2' 로드 성공
  SKT KoGPT2 - 가볍고 빠른 한국어 GPT (125M)
✓ 캐시 서비스 활성화

╭─────────────────────────────────────────╮
│        한국어 다음 토큰 예측기          │
│ 텍스트를 입력하고 Enter를 누르면        │
│ 다음 단어를 예측합니다.                 │
│ 'quit', 'exit', 'q'를 입력하면 종료     │
╰─────────────────────────────────────────╯

텍스트 입력 >
```

### 2.2 텍스트 입력하고 예측하기

**예시 1: 간단한 문장**
```
텍스트 입력 > 오늘 날씨가

예측 중...

다음 단어 예측 (소요시간: 0.234초)
┏━━━┯━━━━━━━━┯━━━━━━┯━━━━━━━━┯━━━━┓
┃순위│예측 단어│신뢰도│확률 막대│타입┃
├───┼────────┼──────┼────────┼────┤
│ 1 │좋아요  │23.4% │████░░░░│일반│
│ 2 │맑습니다│18.7% │███░░░░░│일반│
│ 3 │좋네요  │15.2% │███░░░░░│일반│
│ 4 │어때요  │12.1% │██░░░░░░│일반│
│ 5 │괜찮네요│ 9.8% │██░░░░░░│일반│
└───┴────────┴──────┴────────┴────┘
발화 종료 확률: 0.0%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

**예시 2: 발화 종료 감지 (Kanana Base 모델 사용)**
```
텍스트 입력 > 대화를 종료하겠습니다.

예측 중...

다음 단어 예측 (소요시간: 0.187초)
┏━━━┯━━━━━━━━━┯━━━━━━┯━━━━━━━━┯━━━━━┓
┃순위│예측 단어│신뢰도│확률 막대│타입 ┃
├───┼─────────┼──────┼────────┼─────┤
│ 1 │<|end..> │14.2% │███░░░░░│종료 │
│ 2 │2        │ 1.2% │░░░░░░░░│일반 │
│ 3 │(다른..  │ 0.0% │░░░░░░░░│종료 │
└───┴─────────┴──────┴────────┴─────┘
발화 종료 확률: 14.2%  ████░░░░░░░░░░░░░░░░░░░░░░░░░░
```

### 2.3 대화형 모드 명령어

프로그램이 실행 중일 때 다음 명령어를 사용할 수 있습니다:

#### `/help` - 도움말 보기
```
텍스트 입력 > /help

╭──────────────────────────────────────╮
│              도움말                  │
│                                      │
│ /help              - 이 도움말 표시  │
│ /config            - 현재 설정 표시  │
│ /model info        - 모델 정보 표시  │
│ /model list        - 모델 목록 표시  │
│ /cache             - 캐시 통계 표시  │
│ /cache clear       - 캐시 비우기    │
│ /set top_k <숫자>  - 예측 개수 설정  │
│ /set temp <숫자>   - 온도 설정      │
│ quit, exit, q      - 프로그램 종료   │
╰──────────────────────────────────────╯
```

#### `/config` - 설정 확인
```
텍스트 입력 > /config

┏━━━━━━━━━━━━━━━━┯━━━━━━━┓
┃ 설정            │ 값    ┃
┡━━━━━━━━━━━━━━━━┿━━━━━━━┩
│ 예측 개수       │ 5     │
│ 온도            │ 0.8   │
│ 캐시 활성화     │ Yes   │
│ 최대 입력 길이  │ 128   │
└────────────────┴───────┘
```

#### `/model info` - 모델 정보
```
텍스트 입력 > /model info

┏━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━┓
┃ 항목        │ 값                  ┃
┡━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━┩
│ 모델명      │ skt/kogpt2-base-v2  │
│ 어휘 크기   │ 51,200              │
│ 최대 길이   │ 1024                │
│ 파라미터    │ 125.0M              │
│ 디바이스    │ cpu                 │
└─────────────┴─────────────────────┘
```

#### `/cache` - 캐시 통계
```
텍스트 입력 > /cache

┏━━━━━━━━━━━━┯━━━━━━━━━┓
┃ 항목        │ 값      ┃
┡━━━━━━━━━━━━┿━━━━━━━━━┩
│ 캐시 크기   │ 245.3KB │
│ 저장된 항목 │ 12      │
│ 캐시 히트   │ 8       │
│ 캐시 미스   │ 4       │
└─────────────┴─────────┘
```

#### `/set` - 설정 변경

**예측 개수 변경:**
```
텍스트 입력 > /set top_k 3
✓ 예측 개수를 3개로 설정했습니다.
```

**온도 변경 (창의성 조절):**
```
텍스트 입력 > /set temp 0.5
✓ 온도를 0.5로 설정했습니다.
```

- 낮은 온도 (0.1-0.5): 보수적, 일관된 예측
- 중간 온도 (0.6-1.0): 균형잡힌 예측
- 높은 온도 (1.1-2.0): 창의적, 다양한 예측

### 2.4 프로그램 종료

```
텍스트 입력 > quit
프로그램을 종료합니다.
프로그램을 종료합니다. 감사합니다!
```

또는 `exit`, `q`, `종료` 입력 또는 `Ctrl+C` 사용

---

## 3. 모델 선택

### 3.1 사용 가능한 모델

```bash
python main.py --list-models
```

**지원하는 모델 목록:**

| 모델명 | 파라미터 | 설명 | 메모리 |
|--------|---------|------|--------|
| `kogpt2` | 125M | SKT KoGPT2 - 가볍고 빠른 한국어 GPT | ~2GB |
| `polyglot` | 1.3B | EleutherAI Polyglot-Ko - 중형 한국어 모델 | ~4GB |
| `kanana-nano-2.1b-base` | 2.1B | Kakao Kanana Base - 순수 언어 모델 (대화 지속 경향 있음) | ~5GB (CPU: ~8GB) |
| `kanana-nano-2.1b-instruct` | 2.1B | Kakao Kanana Instruct - 대화 최적화 (대화 지속 경향 강함) | ~5GB (CPU: ~8GB) |
| `dna-r1` * | 14B | DNA-R1 - 추론 특화 한국어 모델 | ~28GB |

\* DNA-R1은 GPU 전용 모델입니다.

### 3.2 모델 지원 범위

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

### 3.3 모델 선택 방법

**기본 모델 (KoGPT2) 사용:**
```bash
python main.py
```

**다른 모델 선택:**
```bash
# Polyglot 모델 (더 큰 모델)
python main.py --model polyglot

# Kanana Base 모델 (대화 지속 경향 있음)
python main.py --model kanana-nano-2.1b-base

# Kanana Instruct 모델 (대화 지속 경향 있음)
python main.py --model kanana-nano-2.1b-instruct

# DNA-R1 모델
python main.py --model dna-r1

# Hugging Face 모델 직접 지정
python main.py --model "skt/kogpt2-base-v2"
```

### 3.4 모델별 특징

**KoGPT2 (기본값)**
- 가장 가볍고 빠름 (125M)
- 대부분의 한국어 작업에 적합
- CPU/GPU 모두 지원

**Polyglot-Ko**
- 중형 모델 (1.3B)
- 더 나은 이해도와 다양성

**Kanana Base**
- 순수 언어 모델 (2025년 최신)
- 발화 계속 경향 있음

**Kanana Instruct**
- 대화 최적화된 모델
- 발화 계속 경향 강함
- 대화형 AI에 적합

**DNA-R1**
- 14B 대형 추론 모델
- DeepSeek-R1 방식의 추론 특화
- GPU 필수 (NVIDIA)
- 매우 높은 성능

---

## 4. 실행 모드 지정

### 4.1 `--run-mode` 옵션

실행할 하드웨어를 명시적으로 지정합니다.

**옵션:**
- `auto` (기본값): 자동 감지 (GPU 있으면 GPU, 없으면 CPU)
- `cpu`: CPU 강제 사용
- `nvidia-gpu`: NVIDIA GPU (CUDA) 강제 사용
- `radeon-gpu`: AMD GPU (ROCm/DirectML) 강제 사용

### 4.2 사용 예시

**자동 감지 (기본값):**
```bash
python main.py
```

**CPU 사용:**
```bash
python main.py --run-mode cpu -t "테스트"
```

**NVIDIA GPU 사용:**
```bash
python main.py --run-mode nvidia-gpu --model kanana-nano-2.1b-base -t "테스트"
```

**AMD Radeon GPU 사용:**
```bash
python main.py --run-mode radeon-gpu --model kanana-nano-2.1b-base -t "테스트"
```

### 4.3 지원되지 않는 조합 시도

어떤 모델이 특정 실행 모드를 지원하지 않을 때:

**예: DNA-R1을 CPU 모드로 실행 시도**
```bash
python main.py --run-mode cpu --model dna-r1
```

**출력:**
```
모델 초기화 중... (dna-r1, 모드: cpu)
모델 로딩 중...

⚠️  경고: 모델 'dna-r1'은 cpu 모드를 지원하지 않습니다.
지원 모드: nvidia-gpu

이 조합은 공식적으로 지원되지 않으며, 오류가 발생할 수 있습니다.

강제로 시도하시겠습니까? (Y/N):
```

**Y(y) 입력**: 강제로 시도 (실패할 가능성 높음)
**N(n) 입력**: 취소하고 종료

**지원되지 않는 GPU 장치**
```bash
python main.py --run-mode radeon-gpu -t "테스트"
# AMD GPU가 설치되어 있지 않을 경우:
```

**출력:**
```
모델 초기화 중... (kogpt2, 모드: radeon-gpu)

⚠️  경고: AMD GPU를 사용할 수 없습니다.
ROCm 또는 DirectML 설치가 필요합니다.

요청한 장치를 사용할 수 없습니다. 강제로 시도하면 오류가 발생할 수 있습니다.

강제로 시도하시겠습니까? (Y/N):
```

**Y(y) 입력**: CPU 모드로 폴백하여 계속 진행
**N(n) 입력**: 취소하고 종료

---

## 5. 고급 사용법

### 5.1 비대화형 모드 (단일 예측)

터미널에서 바로 예측 결과를 받고 싶을 때:

```bash
python main.py -t "오늘 날씨가"
```

**옵션과 함께 사용:**
```bash
python main.py -t "오늘 날씨가" -k 3 --temperature 0.5
```
- `-k 3`: 상위 3개만 예측
- `--temperature 0.5`: 온도 0.5 사용

### 5.2 배치 모드 (파일 입력)

여러 문장을 한 번에 처리하고 싶을 때:

**1. 입력 파일 생성 (`my_input.txt`):**
```
오늘 날씨가
인공지능은 우리의
한국의 전통 음식은
프로그래밍을 배우면
서울에서 가장
```

**2. 배치 실행:**
```bash
python main.py -f my_input.txt
```

**3. 결과:**
```
5개 텍스트 처리 중...
예측 중... ━━━━━━━━━━━━━━━━━━━━━━━━ 100%

입력: 오늘 날씨가
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  다음 단어 예측               ┃
┡━━━━┯━━━━━━━━┯━━━━━━┯━━━━━━┩
│ 1  │ 좋아요  │ 23.4%│ ████ │
│ 2  │ 맑아요  │ 18.7%│ ███  │
...

입력: 인공지능은 우리의
...
```

### 5.3 캐시 비활성화

예측 결과를 캐싱하지 않으려면:

```bash
python main.py --no-cache
```

캐시를 비활성화하면:
- 장점: 항상 새로운 결과 생성
- 단점: 속도가 느려질 수 있음

### 5.4 로그 레벨 조정

디버깅이나 상세 정보가 필요할 때:

```bash
python main.py --log-level DEBUG
```

레벨:
- `DEBUG`: 모든 상세 정보
- `INFO`: 일반 정보 (기본값)
- `WARNING`: 경고만
- `ERROR`: 오류만

### 5.5 옵션 조합 사용

```bash
# Kanana Base 모델, CPU 모드, 예측 10개, 온도 1.2
python main.py --model polyglot --run-mode cpu -k 10 --temperature 1.2 -t "인공지능은"

# DNA-R1, GPU 모드, 배치 처리
python main.py --model dna-r1 --run-mode nvidia-gpu -f corpus.txt

# 캐시 비활성화, 높은 온도, 5개 예측
python main.py --no-cache --temperature 1.5 -k 5 -t "이것은"
```

---

## 6. 실전 활용 예시

### 6.1 텍스트 자동 완성 보조

**시나리오:** 이메일이나 문서 작성 중

```bash
python main.py -t "안녕하세요, 이번 프로젝트의 진행 상황을"
```

**예측 결과:**
```
1. 보고드립니다     (24.5%)
2. 공유드립니다     (21.3%)
3. 말씀드리겠습니다 (18.7%)
4. 알려드립니다     (15.2%)
5. 전달드립니다     (12.8%)
```

### 6.2 한국어 학습 보조

**시나리오:** 한국어를 배우는 외국인 학생

```bash
python main.py -t "저는 학교에"
```

**예측 결과:**
```
1. 갑니다    (32.1%)  ← 가장 자연스러운 표현
2. 가요      (25.4%)
3. 다녀요    (18.3%)
4. 있어요    (12.7%)
5. 다닙니다  (8.9%)
```

### 6.3 글쓰기 아이디어 얻기

**시나리오:** 블로그 포스트나 소설 작성

```
텍스트 입력 > 어두운 밤, 그녀는 홀로

예측 결과:
1. 걸어가고      (19.8%)
2. 있었다        (17.5%)
3. 집으로        (15.3%)
4. 생각했다      (13.7%)
5. 길을          (11.2%)
```

온도를 높여서 더 창의적인 결과:
```bash
python main.py --temperature 1.5 -t "어두운 밤, 그녀는 홀로"

예측 결과:
1. 숲속을        (16.2%)
2. 달빛을        (14.8%)
3. 노래했다      (13.5%)
4. 춤추었다      (12.9%)
5. 외쳤다        (11.7%)
```

### 6.4 화자 교대 시점 판단 (End-of-Turn)

**시나리오:** 대화형 AI 시스템에서 사용자가 말을 끝냈는지 판단

```bash
# Kanana Base 모델 (EOT 감지 최적화)
python main.py --model kanana-nano-2.1b-base -t "질문이 있으신가요?"
```

**예측 결과:**
```
다음 단어 예측 (소요시간: 0.187초)
┏━━━┯━━━━━━━━━━┯━━━━━━┯━━━━━━━━┯━━━━┓
┃순위│예측 단어 │신뢰도│확률 막대│타입┃
├───┼──────────┼──────┼────────┼────┤
│ 1 │<|end..>  │65.2% │██████  │종료│
│ 2 │있으면    │12.3% │██░░░░░ │일반│
│ 3 │아니면    │ 8.7% │█░░░░░░ │일반│
└───┴──────────┴──────┴────────┴────┘
발화 종료 확률: 65.2%  ██████░░░░░░░░░░░░░░░░░░░░░░░░
```

**활용 가이드:**
- **높은 EOT 확률 (>50%)**: AI가 응답을 시작해도 좋음
- **중간 EOT 확률 (20-50%)**: 짧은 대기 후 응답
- **낮은 EOT 확률 (<20%)**: 사용자가 계속 말할 것으로 예상

### 6.5 데이터 분석 및 연구

**배치 모드로 대량 처리:**

```bash
# 1000개 문장 처리
python main.py -f large_corpus.txt > results.txt
```

파이썬 스크립트로 통합:
```python
import subprocess
import json

def predict_next_word(text):
    result = subprocess.run(
        ['python', 'main.py', '-t', text],
        capture_output=True,
        text=True
    )
    return result.stdout

# 사용 예
prediction = predict_next_word("한국의 미래는")
print(prediction)
```

---

## 7. 팁과 트릭

### 7.1 성능 최적화

**첫 실행을 빠르게:**
```bash
# 미리 테스트로 모델 다운로드
python test_simple.py
```

**GPU 사용 확인:**
```
텍스트 입력 > /model info
# 디바이스가 'cuda'이면 GPU 사용 중
```

**캐시 활용:**
- 같은 텍스트는 캐시에서 즉시 반환
- 캐시는 5분간 유효
- 캐시 통계로 효율 확인: `/cache`

### 7.2 더 나은 결과 얻기

**적절한 온도 선택:**
- **공식 문서/기술 문서**: `temp 0.3-0.5`
  ```bash
  python main.py -t "시스템 구조는" --temperature 0.3
  ```

- **일상 대화**: `temp 0.7-0.9` (기본값)
  ```bash
  python main.py -t "오늘 뭐 먹을까" --temperature 0.8
  ```

- **창의적 글쓰기**: `temp 1.0-1.5`
  ```bash
  python main.py -t "우주의 끝에는" --temperature 1.3
  ```

**예측 개수 조절:**
- 빠른 제안: `-k 3`
- 다양한 옵션: `-k 10`

### 7.3 결과를 파일로 저장

**단일 예측 저장:**
```bash
python main.py -t "오늘 날씨가" > result.txt
```

**배치 처리 결과 저장:**
```bash
python main.py -f input.txt > output.txt 2>&1
```

### 7.4 대화형 모드 히스토리 활용

대화형 모드는 자동으로 입력 히스토리를 저장합니다.

- **위/아래 화살표**: 이전 입력 탐색
- **히스토리 파일**: `~/.korean_predictor_history`

### 7.5 모델 선택 가이드

**상황별 추천 모델:**

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| 빠른 응답 필요 | kogpt2 | 가장 가볍고 빠름 |
| 발화 종료 감지 | kanana-nano-2.1b-base | EOT 확률 높음 |
| 대화 지속 | kanana-nano-2.1b-instruct | 계속 말하도록 유도 |
| 높은 품질 필요 | polyglot 또는 dna-r1 | 더 큰 모델 |
| CPU 환경 | kogpt2 또는 kanana-* | 모두 지원 |

---

## 8. 자주 묻는 질문

### Q1. 첫 실행이 너무 느려요

**A:** 첫 실행 시 모델을 다운로드합니다. 예상 소요 시간:
- KoGPT2: ~500MB (3-5분)
- Kanana: ~5GB (10-30분)
- Polyglot: ~2.5GB (5-15분)

이후는 모델이 캐시되어 빠릅니다.

**진행 상황 확인:**
```bash
# 모델이 다운로드되는 위치
ls ~/.cache/korean_predictor/models/
```

### Q2. "Out of memory" 오류가 발생해요

**A:** 메모리 부족입니다. 해결 방법:

1. 다른 프로그램 종료
2. 더 작은 모델 사용
   ```bash
   python main.py --model kogpt2  # 가장 가벼운 모델
   ```
3. CPU 모드 사용 (GPU가 있는 경우)
   ```bash
   python main.py --run-mode cpu
   ```
4. 캐시 비우기 (대화형 모드에서): `/cache clear`

**메모리 확인:**
```bash
# Linux/Mac
free -h

# Windows (PowerShell)
Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory
```

### Q3. GPU를 사용하고 있는지 어떻게 확인하나요?

**A:** 대화형 모드에서:
```
텍스트 입력 > /model info
```
"디바이스" 항목을 확인:
- `cuda`: GPU 사용 중 ✅
- `cpu`: CPU 사용 중

**또는 시작 시 로그 확인:**
```
모델 초기화 중...
GPU 사용: NVIDIA GeForce RTX 3060  ← GPU 사용
또는
CPU 모드로 실행  ← CPU 사용
```

### Q4. 예측 결과가 이상해요

**A:** 다음을 시도해보세요:

1. **온도 조절**: 낮은 온도로 더 보수적인 예측
   ```bash
   python main.py -t "텍스트" --temperature 0.5
   ```

2. **다른 모델 시도**:
   ```bash
   python main.py --model kanana-nano-2.1b-base -t "텍스트"
   ```

3. **캐시 비우기** (대화형 모드):
   ```
   /cache clear
   ```

### Q5. 여러 문장을 한 번에 처리하려면?

**A:** 배치 모드 사용:

1. 텍스트 파일 생성 (`input.txt`)
2. 실행: `python main.py -f input.txt`

### Q6. 프로그램을 백그라운드에서 실행할 수 있나요?

**A:** 네, 비대화형 모드 사용:

**Linux/Mac:**
```bash
nohup python main.py -f large_file.txt > output.log 2>&1 &
```

**Windows:**
```cmd
start /B python main.py -f large_file.txt > output.log 2>&1
```

### Q7. 예측 속도를 더 빠르게 하려면?

**A:** 성능 향상 팁:

1. **GPU 사용** (가장 효과적)
2. **캐시 활성화** (기본값)
3. **예측 개수 줄이기**: `-k 3`
4. **온도 낮추기**: `--temperature 0.5`

### Q8. 지원되지 않는 모델-실행모드 조합이라고 나와요

**A:** 지원되지 않는 조합의 경우, Y/N 확인이 나옵니다.

**예시:**
```
⚠️  경고: 모델 'dna-r1'은 cpu 모드를 지원하지 않습니다.
지원 모드: nvidia-gpu

강제로 시도하시겠습니까? (Y/N):
```

- **Y 입력**: 강제 시도 (실패할 가능성 있음)
- **N 입력**: 취소하고 종료

권장: 지원되는 조합 사용
```bash
# DNA-R1은 GPU 전용
python main.py --run-mode nvidia-gpu --model dna-r1
```

### Q9. AMD GPU (Radeon)에서 지원되지 않는다고 나와요

**A:** AMD GPU 지원을 위해 다음을 설치하세요:

**Windows (DirectML):**
```bash
pip install torch-directml
```

**Linux (ROCm):**
```bash
pip install torch rocm
# 또는 공식 설치 가이드: https://rocmdocs.amd.com/
```

설치 후:
```bash
python main.py --run-mode radeon-gpu -t "테스트"
```

### Q10. Kanana Base와 Instruct 모델의 차이점은?

**A:** 두 모델의 EOT 확률이 다릅니다:

**Kanana Base:**
- 순수 언어 모델
- 종료 토큰(</d>)에 높은 확률 할당
- 자연스러운 발화 종료 감지
- 예: "대화를 종료하겠습니다." → 14.2% EOT 확률

**Kanana Instruct:**
- 대화 명령어 학습된 모델
- 대화 계속 경향 (낮은 EOT 확률)
- 대화형 AI에 적합하지만 EOT 감지에는 부적합
- 예: "대화를 종료하겠습니다." → 0.0% EOT 확률

**추천:**
- EOT 감지 필요: `kanana-nano-2.1b-base`
- 일반 대화: `kanana-nano-2.1b-instruct`

### Q11. DNA-R1 모델이란?

**A:** DNA-R1은:
- 14B 매개변수의 대형 한국어 추론 모델
- DeepSeek-R1 방식의 추론 특화
- **GPU 필수** (NVIDIA CUDA)
- ~28GB VRAM 필요
- 복잡한 추론이 필요한 경우 사용

```bash
# DNA-R1 사용 (NVIDIA GPU 필수)
python main.py --model dna-r1 --run-mode nvidia-gpu -t "복잡한 질문"
```

### Q12. 오프라인에서도 사용할 수 있나요?

**A:** 네! 모델이 한 번 다운로드되면 인터넷 없이 사용 가능합니다.

**오프라인 사용 준비:**
1. 인터넷 연결된 상태에서 각 모델 첫 실행
2. 모델 다운로드 완료 확인
3. 이후 오프라인 사용 가능

### Q13. transformers 업데이트가 필요하다고 나와요

**A:** Kanana 모델 사용 시 transformers >= 4.45.0이 필요합니다:

```bash
pip install --upgrade transformers
```

확인:
```bash
python -c "import transformers; print(transformers.__version__)"
```

---

## 9. REST API 사용법

이제 Korean Predictor를 REST API 서버로도 사용할 수 있습니다!

### 9.1 API 서버 시작

**기본 실행:**
```bash
python run_api.py
```

**옵션 지정:**
```bash
python run_api.py --model kogpt2 --run-mode cpu --port 8000
```

**사용 가능한 옵션:**
- `--model, -m`: 모델 선택 (kogpt2, kanana, polyglot-ko-5.8b, dna-r1)
- `--run-mode, -r`: 실행 모드 (auto, cpu, nvidia-gpu, radeon-gpu)
- `--host`: 호스트 주소 (기본값: 0.0.0.0)
- `--port, -p`: 포트 번호 (기본값: 8000)
- `--reload`: 개발 모드 (코드 변경 시 자동 재시작)
- `--workers, -w`: 워커 프로세스 수 (기본값: 1)

### 9.2 API 문서 확인

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 9.3 API 사용 예시

**Python 예시:**
```python
import requests

url = "http://localhost:8000/v1/predict"
headers = {
    "Authorization": "Bearer kp_test_development_key_12345",
    "Content-Type": "application/json"
}
payload = {
    "text": "안녕하세요",
    "model": "kogpt2",
    "top_k": 10,
    "temperature": 1.3
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()

if data["success"]:
    for pred in data["data"]["predictions"]:
        print(f"{pred['rank']}. {pred['token']}: {pred['probability']:.2%}")
```

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Authorization: Bearer kp_test_development_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요",
    "model": "kogpt2",
    "top_k": 10
  }'
```

### 9.4 주요 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/v1/predict` | POST | 다음 토큰 예측 |
| `/v1/predict/context` | POST | 컨텍스트 기반 예측 |
| `/v1/predict/batch` | POST | 배치 예측 |
| `/v1/models/current` | GET | 현재 모델 정보 |
| `/v1/models` | GET | 사용 가능한 모델 목록 |
| `/v1/cache/stats` | GET | 캐시 통계 |
| `/v1/cache` | DELETE | 캐시 삭제 |
| `/v1/health` | GET | 헬스체크 |

### 9.5 API 인증

개발 모드에서는 기본 테스트 API 키가 자동으로 설정됩니다:
```
Bearer kp_test_development_key_12345
```

프로덕션 환경에서는 환경 변수로 API 키를 설정하세요:
```bash
export API_KEYS="your_api_key_1,your_api_key_2"
python run_api.py
```

### 9.6 자세한 API 문서

전체 API 명세는 `REST-API.md` 파일을 참조하세요.

---

## 추가 도움말

**더 많은 정보:**
- `QUICKSTART.md`: 빠른 시작 가이드
- GitHub Issues: 버그 리포트 및 질문

**문제가 해결되지 않으면:**
1. 로그 레벨을 DEBUG로 설정: `--log-level DEBUG`
2. 에러 메시지 전체를 복사
3. GitHub Issues에 보고

**즐거운 사용 되세요! 🎉**