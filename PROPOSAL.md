# 경량 한국어 다음 토큰 예측기 서비스 계획서

## 1. 서비스 개요 및 목적

본 서비스는 연구용 "다음 토큰 예측기"로, 사용자가 텍스트를 입력할 때 어절 단위(스페이스 단위)로 실시간으로 다음에 올 후보 단어(또는 어절)를 추천합니다. 추천 결과는 confidence score와 함께 제공되어, 사용자 입력 보조 및 인터랙티브한 자연어 처리 실험에 활용될 수 있습니다.

## 2. 주요 기능

- 실시간 다음 어절 예측 (스페이스 입력마다)
- 예측 결과: 최대 N개의 후보 어절 + confidence score 표시
- CPU 또는 GPU 환경 모두에서 구동 가능 (모델 경량화)
- Web 기반 인터페이스 (프론트엔드 자유 구현)

## 3. 아키텍처 개요

```
[사용자 입력]
    ↓ (HTTP/WebSocket)
[프론트엔드]
    ↓
[FastAPI 백엔드 서버]
    ↓
[경량 한국어 언어모델 (KoGPT 등)] ← GPU or CPU
```

- 프론트엔드: 자유 (React 권장)
- 백엔드: FastAPI + transformers
- 모델: 경량 언어모델 상시 로딩 (GPU/CPU)
- 통신: REST 또는 WebSocket 기반

## 4. 후보 모델

| 모델명 | 파라미터 | 특징 |
|--------|----------|------|
| `skt/kogpt2-base-v2` | ~125M | 가볍고 한국어 특화 |
| `beomi/KcGPT-2` | ~124M | KoGPT2 기반, 경량 |
| `EleutherAI/polyglot-ko-125M` | 125M | 한국어 GPT 기반, HuggingFace 호환 |

## 5. 예측 방식과 모델 활용 흐름

### 입력 전처리
- 입력 텍스트를 토크나이저(SentencePiece 또는 GPT tokenizer)를 통해 토큰화
- 마지막 토큰 직전까지를 context로 설정

### 예측 흐름
1. 사용자가 입력한 문장을 토크나이즈하여 `input_ids`로 변환
2. `model.generate(...)` 또는 `model.forward(...)`로 다음 토큰 확률 분포 계산
3. softmax 후 top-k 후보 토큰 선택
4. 후보 토큰을 디코딩하여 문자열로 변환
5. 각 후보 토큰에 대해 확률값(=confidence score)을 함께 리턴

### 예시 코드 스니펫
```python
with torch.no_grad():
    outputs = model(input_ids)
    next_token_logits = outputs.logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=5)
    tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())
```

### confidence score 산출
- softmax(logits) → 확률값 (0~1)
- 상위 토큰들을 확률 내림차순으로 정렬하여 UI에 표시

## 6. UX 흐름

1. 사용자가 웹 입력창에 문장을 입력
2. 공백(Space) 입력 시 백엔드에 텍스트 전송
3. 모델이 다음 후보 토큰 5개 예측 + 확률 계산
4. 프론트엔드에 예측 리스트 렌더링 (드롭다운 등으로)
5. 사용자는 추천어를 클릭하거나 입력 지속

## 7. 성능 및 Latency 고려사항

| 항목 | 목표 |
|------|------|
| 전체 응답 시간 | 500 miliseconds 이내 |
| 모델 초기 로딩 | 앱 시작 시 1회만 |
| 요청당 추론 | input length < 32 tokens, top-5 예측 |
| CPU fallback | GPU 없을 경우에도 예측 가능 (모델 선택시 고려) |

### CPU 사용 예시 (KoGPT2 기준)
- KoGPT2가 PyTorch 기반으로 공개되어 있으며, from_pretrained() 로 모델을 로드한 뒤 model.to("cpu") 형태로 CPU 모드로 전환한 실사용 예시가 있습니다. [출처](https://ggaebap.tistory.com/121)

- 토크나이저 및 generate()/forward() 같은 메소드도 GPU 전용이 아닌 일반 CPU 환경에서 동작 가능하도록 설계되어 있습니다. 예컨대 CPU 환경으로 ctx = mx.cpu() 로 설정하는 MXNet 예시도 있습니다. [출처](https://github.com/thingsflow/KoGPT2)

### 최적화 방안
- `past_key_values` 캐시 사용 (session 단위 캐시)
- `generate()` 대신 `forward()` + logits softmax 사용
- FastAPI 서버에서 모델을 global 인스턴스로 재사용

## 8. 확장 고려사항

- 추천어 선택 로그 수집 → 향후 파인튜닝용
- 시간별/주제별 입력 통계 분석
- 도메인 특화(의료/법률 등) 데이터 파인튜닝
- 입력에 따라 prefix prompt 조정 기능

## 9. MVP 범위 및 구현 우선순위

### MVP 범위
- 실시간 추천어 예측 API 구현 (FastAPI)
- 경량 모델 로딩 및 추론 흐름 구현
- 프론트엔드에서 추천 리스트 UI 표시
- confidence score 기반 정렬

### 구현 우선순위
1. 모델 후보 결정 및 성능 테스트 (GPU/CPU)
2. API 기본 구조 설계 및 모델 서빙 구현
3. 프론트엔드 입력 이벤트 처리 및 API 연결
4. 추천어 리스트 UI 및 확률 표시 구현
5. 입력 로그 저장 기능 도입 (선택)
