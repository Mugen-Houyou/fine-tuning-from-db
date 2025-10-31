# EOT Prediction REST API 문서

## 1. 개요

### 1.1 목적
한국어 채팅 문맥에서 발화 종료(End-of-Turn) 확률을 예측하는 REST API 서비스를 제공합니다. 이 API는 챗봇, 대화 시스템, 채팅 분석 도구 등에서 활용될 수 있습니다.

### 1.2 주요 기능
- 단일 텍스트 EOT 확률 예측
- 배치 텍스트 EOT 확률 예측
- 컨텍스트 기반 EOT 확률 예측
- 다양한 한국어 종료 표현 인식
- 사용자 정의 EOT 토큰 지원
- 실시간 예측 결과 제공

### 1.3 기술 스택
- **프레임워크**: FastAPI (비동기 지원, 자동 문서화)
- **서버**: Uvicorn (ASGI 서버)
- **포트**: 8177 (기본값)
- **모델**: Polyglot-Ko (기본), KoGPT2, Kanana 지원

---

## 2. API 엔드포인트

### 2.1 Base URL
```
Production: https://eot-api.example.com
Development: http://localhost:8177
```

### 2.2 인증
모든 API 요청은 헤더에 API Key를 포함할 수 있음 (선택적):
```http
Authorization: Bearer YOUR_API_KEY
```

---

## 3. 엔드포인트 상세

### 3.1 EOT 예측 API

#### 3.1.1 단일 텍스트 EOT 예측
**POST** `/predict/eot`

입력 텍스트의 발화 종료 확률을 예측합니다.

**Request Body:**
```json
{
  "text": "그래 알았어",
  "model": "polyglot",
  "top_k": 10,
  "temperature": 0.5,
  "timeout": 60
}
```

**Request Parameters:**
| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| text | string | ✅ | - | 입력 텍스트 (1-1000자) |
| model | string | ❌ | polyglot | 사용할 모델 |
| top_k | integer | ❌ | 10 | 예측할 토큰 개수 (1-20) |
| temperature | float | ❌ | 0.5 | 샘플링 온도 (0.1-2.0) |
| timeout | integer | ❌ | 60 | 타임아웃 (0-300초) |

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "eot_probability": 0.9785,
    "predictions": [
      {
        "token": ".",
        "probability": 0.7941,
        "rank": 1,
        "is_eot": true,
        "type": "punctuation"
      },
      {
        "token": "\"",
        "probability": 0.0537,
        "rank": 2,
        "is_eot": true,
        "type": "punctuation"
      },
      {
        "token": "~",
        "probability": 0.0143,
        "rank": 3,
        "is_eot": true,
        "type": "user_defined"
      }
    ],
    "input_text": "그래 알았어",
    "model": "polyglot",
    "elapsed_time": 0.523,
    "eot_assessment": "high"
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_abc123def456"
  }
}
```

**EOT Assessment 기준:**
- `high`: EOT 확률 > 70% (발화 종료 가능성 높음)
- `medium`: EOT 확률 30-70% (상황에 따라 결정)
- `low`: EOT 확률 < 30% (발화 계속될 가능성 높음)

**토큰 타입 분류:**
- `eot_expression`: EOT-예측-첫-토큰.md의 종료 표현
- `user_defined`: 사용자 정의 EOT 토큰
- `punctuation`: 문장부호
- `abnormal`: 비정상적인 토큰 (10자 이상)
- `other_eot`: 기타 EOT 토큰
- `general`: 일반 토큰

---

#### 3.1.2 배치 EOT 예측
**POST** `/predict/batch`

여러 텍스트의 EOT 확률을 한 번에 예측합니다.

**Request Body:**
```json
{
  "texts": [
    "안녕하세요",
    "그래 알았어",
    "오늘 뭐 먹었어"
  ],
  "model": "polyglot",
  "top_k": 10,
  "temperature": 0.5,
  "timeout": 120
}
```

**Request Parameters:**
| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| texts | array | ✅ | - | 입력 텍스트 리스트 (1-100개) |
| model | string | ❌ | polyglot | 사용할 모델 |
| top_k | integer | ❌ | 10 | 예측할 토큰 개수 (1-20) |
| temperature | float | ❌ | 0.5 | 샘플링 온도 (0.1-2.0) |
| timeout | integer | ❌ | 120 | 타임아웃 (0-600초) |

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "안녕하세요",
        "eot_probability": 0.2341,
        "eot_assessment": "low",
        "top_prediction": "반갑습니다",
        "elapsed_time": 0.234,
        "success": true
      },
      {
        "text": "그래 알았어",
        "eot_probability": 0.9785,
        "eot_assessment": "high",
        "top_prediction": ".",
        "elapsed_time": 0.245,
        "success": true
      },
      {
        "text": "오늘 뭐 먹었어",
        "eot_probability": 0.8612,
        "eot_assessment": "high",
        "top_prediction": "?",
        "elapsed_time": 0.223,
        "success": true
      }
    ],
    "total_count": 3,
    "success_count": 3,
    "failure_count": 0,
    "total_elapsed_time": 0.702,
    "model": "polyglot"
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_ghi789jkl012"
  }
}
```

---

#### 3.1.3 컨텍스트 기반 EOT 예측
**POST** `/predict/context`

대화 컨텍스트를 고려하여 EOT 확률을 예측합니다.

**Request Body:**
```json
{
  "context": [
    "안녕하세요",
    "네 안녕하세요",
    "오늘 날씨 좋네요",
    "그렇네요 날씨가 정말 좋아요"
  ],
  "model": "polyglot",
  "top_k": 10,
  "temperature": 0.5,
  "timeout": 60
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "eot_probability": 0.8234,
    "predictions": [
      {
        "token": ".",
        "probability": 0.5678,
        "rank": 1,
        "is_eot": true
      },
      {
        "token": "네",
        "probability": 0.1234,
        "rank": 2,
        "is_eot": true
      }
    ],
    "context_length": 4,
    "combined_length": 47,
    "model": "polyglot",
    "elapsed_time": 0.345,
    "eot_assessment": "high",
    "recommendation": "대화를 종료해도 좋습니다"
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_mno345pqr678"
  }
}
```

---

### 3.2 헬스체크 API

#### 3.2.1 기본 헬스체크
**GET** `/health`

서비스 상태를 확인합니다.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-31T12:34:56Z",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

#### 3.2.2 상세 헬스체크
**GET** `/health/detailed`

서비스 구성요소별 상태를 확인합니다.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "components": {
    "predictor": {
      "status": "healthy",
      "loaded": true,
      "eot_tokens": 110,
      "user_eot_tokens": 34,
      "punctuation_count": 30
    },
    "cache": {
      "status": "healthy",
      "enabled": true,
      "size_kb": 1234.5,
      "items": 42
    }
  },
  "timestamp": "2025-10-31T12:34:56Z",
  "version": "1.0.0"
}
```

---

### 3.3 통계 및 정보 API

#### 3.3.1 API 통계
**GET** `/stats`

API 사용 통계를 반환합니다.

**Response (200 OK):**
```json
{
  "api_version": "1.0.0",
  "predictor": {
    "eot_tokens": 110,
    "user_eot_tokens": 34,
    "punctuation_marks": 30
  },
  "cache": {
    "size_kb": 1234.5,
    "items": 42,
    "hit_rate": 0.78
  },
  "rate_limits": {
    "standard": 100,
    "premium": 1000,
    "batch": 10
  },
  "timestamp": "2025-10-31T12:34:56Z"
}
```

---

#### 3.3.2 모델 정보
**GET** `/models`

사용 가능한 모델 목록을 반환합니다.

**Response (200 OK):**
```json
{
  "models": [
    {
      "id": "polyglot",
      "name": "EleutherAI Polyglot-Ko",
      "description": "한국어 특화 언어 모델",
      "params": "1.3B",
      "default": true
    },
    {
      "id": "kogpt2",
      "name": "SKT KoGPT2",
      "description": "가볍고 빠른 한국어 GPT",
      "params": "125M",
      "default": false
    },
    {
      "id": "kanana-nano-2.1b-base",
      "name": "Kanana Nano",
      "description": "경량 한국어 모델",
      "params": "2.1B",
      "default": false
    }
  ],
  "current": "polyglot"
}
```

---

## 4. 에러 코드

| 코드 | HTTP 상태 | 설명 |
|------|-----------|------|
| INVALID_INPUT | 400 | 잘못된 입력 형식 또는 값 |
| TEXT_TOO_LONG | 400 | 입력 텍스트 길이 초과 (1000자) |
| INVALID_PARAMETER | 400 | 파라미터 범위 벗어남 |
| RATE_LIMIT_EXCEEDED | 429 | 요청 제한 초과 |
| TIMEOUT | 408 | 요청 타임아웃 |
| INTERNAL_ERROR | 500 | 서버 내부 오류 |
| MODEL_NOT_LOADED | 503 | 모델 로드되지 않음 |

**에러 응답 예시:**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "요청 제한 초과",
    "details": {
      "limit": 100,
      "reset_at": "2025-10-31T12:35:00Z"
    }
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_stu901vwx234"
  }
}
```

---

## 5. Rate Limiting

### 5.1 요청 제한
- **기본 사용자**: 100 requests/minute
- **프리미엄 사용자**: 1000 requests/minute
- **배치 API**: 10 requests/minute

### 5.2 Rate Limit 헤더
응답 헤더에 포함되는 정보:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635782400
```

---

## 6. 사용 예시

### 6.1 Python
```python
import requests

url = "http://localhost:8177/predict/eot"
headers = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer YOUR_API_KEY"  # 선택적
}
payload = {
    "text": "그래 알았어",
    "model": "polyglot",
    "top_k": 10,
    "temperature": 0.5
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()

if data["success"]:
    eot_prob = data["data"]["eot_probability"]
    assessment = data["data"]["eot_assessment"]

    print(f"EOT 확률: {eot_prob:.1%}")
    print(f"평가: {assessment}")

    if assessment == "high":
        print("➜ 발화가 끝날 가능성이 높습니다")
    elif assessment == "medium":
        print("➜ 상황에 따라 결정하세요")
    else:
        print("➜ 발화가 계속될 가능성이 높습니다")
else:
    print(f"오류: {data['error']['message']}")
```

### 6.2 cURL
```bash
curl -X POST "http://localhost:8177/predict/eot" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "그래 알았어",
    "model": "polyglot",
    "top_k": 10,
    "temperature": 0.5
  }'
```

### 6.3 JavaScript (fetch)
```javascript
const url = "http://localhost:8177/predict/eot";
const headers = {
  "Content-Type": "application/json"
};
const payload = {
  text: "그래 알았어",
  model: "polyglot",
  top_k: 10,
  temperature: 0.5
};

fetch(url, {
  method: "POST",
  headers: headers,
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      const eotProb = data.data.eot_probability;
      const assessment = data.data.eot_assessment;

      console.log(`EOT 확률: ${(eotProb * 100).toFixed(1)}%`);
      console.log(`평가: ${assessment}`);

      // 예측 토큰 표시
      data.data.predictions.slice(0, 3).forEach(pred => {
        console.log(`${pred.rank}. ${pred.token}: ${(pred.probability * 100).toFixed(2)}% ${pred.is_eot ? '(EOT)' : ''}`);
      });
    } else {
      console.error(`오류: ${data.error.message}`);
    }
  })
  .catch(error => console.error("요청 실패:", error));
```

### 6.4 배치 처리 예시 (Python)
```python
import requests

url = "http://localhost:8177/predict/batch"
headers = {"Content-Type": "application/json"}

# 분석할 텍스트 리스트
texts = [
    "안녕하세요",
    "그래 알았어",
    "오늘 뭐 먹었어",
    "내일 봐",
    "어떻게 생각해"
]

payload = {
    "texts": texts,
    "model": "polyglot",
    "top_k": 10,
    "temperature": 0.5
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()

if data["success"]:
    print(f"처리된 텍스트: {data['data']['total_count']}개")
    print(f"총 소요 시간: {data['data']['total_elapsed_time']:.2f}초\n")

    for result in data["data"]["results"]:
        if result["success"]:
            text = result["text"]
            eot_prob = result["eot_probability"]
            assessment = result["eot_assessment"]

            # 이모지로 시각화
            emoji = "🔴" if assessment == "high" else "🟡" if assessment == "medium" else "🟢"

            print(f"{emoji} '{text}' - EOT: {eot_prob:.1%} ({assessment})")
        else:
            print(f"❌ '{result['text']}' - 오류: {result['error']}")
else:
    print(f"오류: {data['error']['message']}")
```

---

## 7. EOT 토큰 카테고리

### 7.1 종료 표현 (eot_expression)
`EOT-예측-첫-토큰.md` 파일의 한국어 종료 표현들:
- 네, 아니요, 그래, 맞아, 맞습니다
- 알겠습니다, 알았어요
- 감사합니다, 고마워요
- 미안합니다, 죄송합니다
- 기타 110개 이상의 한국어 종료 표현

### 7.2 사용자 정의 토큰 (user_defined)
`user-defined-eots.txt` 파일의 특수문자 및 비정상 문자:
- 키보드 특수문자: `-`, `_`, `=`, `+`, `|`, `\`, `/`, `*`, `&`, `%`, `$`, `#`, `@`, `~`, `` ` ``, `^`, `<`, `>`
- 비정상 문자: `�`, `�`, `□`, `■`, `▷`, `◇`, `◆`, `○`, `●`, `☆`, `★` 등

### 7.3 문장부호 (punctuation)
- 마침표: `.`, `。`
- 물음표: `?`, `？`
- 느낌표: `!`, `！`
- 쉼표: `,`, `、`
- 따옴표: `"`, `'`, `"`, `"`, `'`, `'`
- 괄호: `()`, `[]`, `{}`, `「」`, `『』`, `《》`, `〈〉`, `【】`

### 7.4 비정상 토큰 (abnormal)
- 10자 이상의 긴 토큰
- 알 수 없는 문자열

### 7.5 기타 EOT (other_eot)
- 공백만 있는 토큰
- 특수 토큰 패턴: `</d>`, `<|endoftext|>`, `[SEP]` 등

---

## 8. 성능 최적화

### 8.1 캐싱
- 동일한 입력에 대한 예측 결과를 캐싱
- TTL: 5분 (기본값)
- 캐시 키: `{model}:{text}:{top_k}:{temperature}`

### 8.2 모델 워밍업
- 서버 시작 시 자동으로 모델 워밍업 수행
- 첫 예측의 지연 시간 최소화

### 8.3 비동기 처리
- FastAPI의 async/await 활용
- 동시 요청 처리 최적화

---

## 9. 서버 실행

### 9.1 기본 실행
```bash
python eot_api.py
```

### 9.2 환경 변수 설정
```bash
export EOT_API_HOST=0.0.0.0
export EOT_API_PORT=8177
python eot_api.py
```

### 9.3 Docker 실행
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8177
CMD ["python", "eot_api.py"]
```

```bash
docker build -t eot-api .
docker run -p 8177:8177 eot-api
```

---

## 10. API 문서 자동 생성

FastAPI는 자동으로 OpenAPI 문서를 생성합니다:

- **Swagger UI**: http://localhost:8177/docs
- **ReDoc**: http://localhost:8177/redoc
- **OpenAPI JSON**: http://localhost:8177/openapi.json

---

## 11. 문의 및 지원

- **GitHub**: https://github.com/yourusername/eot-predictor
- **이메일**: support@eot-predictor.example.com
- **문서**: https://docs.eot-predictor.example.com

---

**버전**: 1.0.0
**최종 수정일**: 2025-10-31
**작성자**: EOT Predictor Team