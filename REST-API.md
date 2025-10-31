# Korean Predictor REST API 설계 문서

## 1. 개요

### 1.1 목적
한국어 다음 토큰/어절 예측 및 End-of-Turn 확률 분석 기능을 HTTP REST API로 제공하여 다양한 클라이언트 애플리케이션에서 활용 가능하도록 함.

### 1.2 주요 기능
- 다음 토큰/어절 예측
- End-of-Turn (EOT) 확률 계산
- 컨텍스트 기반 예측
- 다중 모델 지원 (KoGPT2, Kanana, Polyglot, DNA-R1)
- 설정 동적 조정 (temperature, top_k, timeout)
- 캐시 관리

### 1.3 기술 스택
- **프레임워크**: FastAPI (비동기 지원, 자동 문서화, 타입 검증)
- **서버**: Uvicorn (ASGI 서버)
- **인증**: API Key 기반
- **모니터링**: Prometheus + Grafana
- **로그**: Structured logging (JSON format)

---

## 2. API 엔드포인트

### 2.1 Base URL
```
Production: https://api.korean-predictor.example.com
Development: http://localhost:8000
```

### 2.2 인증
모든 API 요청은 헤더에 API Key를 포함해야 함:
```http
Authorization: Bearer YOUR_API_KEY
```

---

## 3. 엔드포인트 상세

### 3.1 예측 API

#### 3.1.1 다음 토큰 예측
**POST** `/predict`

입력 텍스트에 대한 다음 토큰/어절 예측 및 EOT 확률 반환

**Request Body:**
```json
{
  "text": "안녕하세요",
  "model": "kogpt2",
  "top_k": 10,
  "temperature": 1.3,
  "complete_word": true,
  "include_special_tokens": true,
  "timeout": 60
}
```

**Request Parameters:**
| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| text | string | ✅ | - | 입력 텍스트 (최대 128 토큰) |
| model | string | ❌ | kogpt2 | 사용할 모델 ID (kogpt2, kanana, polyglot-ko-5.8b, dna-r1) |
| top_k | integer | ❌ | 10 | 예측할 토큰 개수 (1-20) |
| temperature | float | ❌ | 1.3 | 샘플링 온도 (0.1-2.0, 추론 모델 제외) |
| complete_word | boolean | ❌ | true | 완전한 어절까지 생성 여부 |
| include_special_tokens | boolean | ❌ | true | 특수 토큰 포함 여부 |
| timeout | integer | ❌ | 60 | 타임아웃 (초, 0=무제한) |

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "token": "반갑습니다",
        "probability": 0.456,
        "rank": 1,
        "type": "normal"
      },
      {
        "token": ".",
        "probability": 0.234,
        "rank": 2,
        "type": "normal"
      },
      {
        "token": "</s>",
        "probability": 0.123,
        "rank": 3,
        "type": "eos"
      }
    ],
    "eot_probability": 0.123,
    "input_text": "안녕하세요",
    "model": "kogpt2",
    "elapsed_time": 0.234
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_abc123"
  }
}
```

**Response (400 Bad Request):**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text length exceeds maximum (128 tokens)",
    "details": {
      "field": "text",
      "current_length": 150,
      "max_length": 128
    }
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_abc123"
  }
}
```

**Response (408 Request Timeout):**
```json
{
  "success": false,
  "error": {
    "code": "TIMEOUT",
    "message": "Prediction timeout exceeded (60 seconds)",
    "details": {
      "timeout": 60,
      "elapsed": 61.5
    }
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_abc123"
  }
}
```

---

#### 3.1.2 컨텍스트 기반 예측
**POST** `/predict/context`

여러 턴의 대화 컨텍스트를 고려한 예측

**Request Body:**
```json
{
  "context": [
    "안녕하세요",
    "네 안녕하세요",
    "오늘 날씨가"
  ],
  "model": "kogpt2",
  "top_k": 10,
  "temperature": 1.3,
  "timeout": 60
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "token": "좋네요",
        "probability": 0.567,
        "rank": 1,
        "type": "normal"
      }
    ],
    "eot_probability": 0.234,
    "context_length": 3,
    "model": "kogpt2",
    "elapsed_time": 0.345
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_def456"
  }
}
```

---

#### 3.1.3 배치 예측
**POST** `/predict/batch`

여러 텍스트에 대한 일괄 예측

**Request Body:**
```json
{
  "texts": [
    "안녕하세요",
    "오늘 날씨는",
    "저는 학생"
  ],
  "model": "kogpt2",
  "top_k": 5,
  "temperature": 1.3,
  "timeout": 120
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "안녕하세요",
        "predictions": [...],
        "eot_probability": 0.123,
        "elapsed_time": 0.234
      },
      {
        "text": "오늘 날씨는",
        "predictions": [...],
        "eot_probability": 0.456,
        "elapsed_time": 0.245
      },
      {
        "text": "저는 학생",
        "predictions": [...],
        "eot_probability": 0.234,
        "elapsed_time": 0.223
      }
    ],
    "total_count": 3,
    "success_count": 3,
    "failure_count": 0,
    "total_elapsed_time": 0.702
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_ghi789"
  }
}
```

---

### 3.2 모델 관리 API

#### 3.2.1 모델 정보 조회
**GET** `/models/current`

현재 로드된 모델 정보 반환

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "model_name": "skt/kogpt2-base-v2",
    "model_id": "kogpt2",
    "vocab_size": 51200,
    "max_length": 1024,
    "parameters": 125.2,
    "device": "cuda:0",
    "supports_temperature": true,
    "loaded_at": "2025-10-31T12:00:00Z",
    "memory_usage_mb": 512
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_jkl012"
  }
}
```

---

#### 3.2.2 사용 가능한 모델 목록
**GET** `/models`

지원하는 모든 모델 목록 반환

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "kogpt2",
        "name": "SKT KoGPT2",
        "description": "가볍고 빠른 한국어 GPT",
        "params": "125M",
        "memory": "~2GB",
        "supports_temperature": true,
        "reasoning_model": false,
        "supported_modes": ["cpu", "nvidia-gpu", "radeon-gpu"]
      },
      {
        "id": "dna-r1",
        "name": "DNA-R1",
        "description": "추론 특화 한국어 모델 (DeepSeek-R1 방식)",
        "params": "14B",
        "memory": "~28GB",
        "supports_temperature": false,
        "reasoning_model": true,
        "supported_modes": ["nvidia-gpu"]
      }
    ]
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_mno345"
  }
}
```

---

#### 3.2.3 모델 전환 (관리자 전용)
**POST** `/models/switch`

다른 모델로 전환 (서버 재시작 필요)

**Request Body:**
```json
{
  "model_id": "dna-r1",
  "run_mode": "nvidia-gpu"
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "message": "Model switch scheduled",
    "current_model": "kogpt2",
    "target_model": "dna-r1",
    "estimated_time": "30-60 seconds"
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_pqr678"
  }
}
```

---

### 3.3 캐시 관리 API

#### 3.3.1 캐시 통계
**GET** `/cache/stats`

캐시 사용 통계 반환

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "enabled": true,
    "size_kb": 1234.5,
    "items": 567,
    "hits": 12345,
    "misses": 3456,
    "hit_rate": 0.781,
    "ttl_seconds": 300
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_stu901"
  }
}
```

---

#### 3.3.2 캐시 삭제 (관리자 전용)
**DELETE** `/cache`

캐시 전체 삭제

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "message": "Cache cleared successfully",
    "items_deleted": 567
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_vwx234"
  }
}
```

---

### 3.4 헬스체크 API

#### 3.4.1 기본 헬스체크
**GET** `/health`

서비스 상태 확인

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-31T12:34:56Z",
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "reason": "Model not loaded",
  "timestamp": "2025-10-31T12:34:56Z"
}
```

---

#### 3.4.2 상세 헬스체크
**GET** `/health/detailed`

서비스 상세 상태 확인

**Response (200 OK):**
```json
{
  "status": "healthy",
  "components": {
    "model": {
      "status": "healthy",
      "loaded": true,
      "model_name": "kogpt2"
    },
    "cache": {
      "status": "healthy",
      "enabled": true,
      "size_kb": 1234.5
    },
    "device": {
      "status": "healthy",
      "type": "cuda",
      "memory_used_mb": 512,
      "memory_total_mb": 8192
    }
  },
  "timestamp": "2025-10-31T12:34:56Z",
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

---

## 4. 에러 코드

| 코드 | HTTP 상태 | 설명 |
|------|-----------|------|
| INVALID_INPUT | 400 | 잘못된 입력 형식 또는 값 |
| MISSING_FIELD | 400 | 필수 필드 누락 |
| TEXT_TOO_LONG | 400 | 입력 텍스트 길이 초과 |
| INVALID_PARAMETER | 400 | 파라미터 범위 벗어남 |
| UNAUTHORIZED | 401 | 인증 실패 (API Key 없음 또는 무효) |
| FORBIDDEN | 403 | 권한 없음 (관리자 전용 API) |
| NOT_FOUND | 404 | 리소스 없음 |
| TIMEOUT | 408 | 요청 타임아웃 |
| RATE_LIMIT_EXCEEDED | 429 | 요청 제한 초과 |
| INTERNAL_ERROR | 500 | 서버 내부 오류 |
| MODEL_NOT_LOADED | 503 | 모델 로드되지 않음 |
| SERVICE_UNAVAILABLE | 503 | 서비스 일시 중단 |

---

## 5. Rate Limiting

### 5.1 요청 제한
- **기본 사용자**: 100 requests/minute
- **프리미엄 사용자**: 1000 requests/minute
- **배치 API**: 10 requests/minute

### 5.2 Rate Limit 헤더
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635782400
```

### 5.3 Rate Limit 초과 응답
**Response (429 Too Many Requests):**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "reset_at": "2025-10-31T12:35:00Z"
    }
  },
  "meta": {
    "api_version": "1.0.0",
    "timestamp": "2025-10-31T12:34:56Z",
    "request_id": "req_yz0123"
  }
}
```

---

## 6. 보안

### 6.1 API Key 관리
- **생성**: 관리자 대시보드에서 생성
- **형식**: `kp_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (프로덕션)
- **형식**: `kp_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (테스트)
- **저장**: 환경 변수 또는 보안 저장소에 저장
- **갱신**: 정기적 갱신 권장 (90일마다)

### 6.2 HTTPS
- 모든 API는 HTTPS로만 접근 가능
- HTTP 요청은 자동으로 HTTPS로 리다이렉트

### 6.3 CORS
- 허용된 origin만 접근 가능
- 설정 파일에서 origin 목록 관리

### 6.4 입력 검증
- 모든 입력은 타입 검증 및 범위 확인
- SQL Injection, XSS 방어
- 최대 입력 길이 제한

---

## 7. 성능 최적화

### 7.1 캐싱 전략
- **메모리 캐시**: 동일 요청에 대한 빠른 응답
- **TTL**: 기본 5분 (설정 가능)
- **캐시 키**: `{model}:{text}:{top_k}:{temperature}:{complete_word}:{include_special_tokens}`

### 7.2 비동기 처리
- FastAPI의 async/await 활용
- 배치 요청 병렬 처리
- 타임아웃 설정으로 무한 대기 방지

### 7.3 모델 최적화
- GPU 사용 시 Float16/BFloat16 사용
- 배치 추론 지원
- 모델 워밍업 (서버 시작 시)

---

## 8. 모니터링 및 로깅

### 8.1 메트릭 (Prometheus)
- 요청 수 (총/성공/실패)
- 응답 시간 (평균/P50/P95/P99)
- 에러율
- 캐시 히트율
- 모델 추론 시간
- GPU/CPU 사용률
- 메모리 사용량

### 8.2 로그
- **형식**: JSON (structured logging)
- **레벨**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **포함 정보**:
  - Request ID
  - Timestamp
  - API Key (마스킹)
  - Endpoint
  - Response Time
  - Status Code
  - Error details (if any)

**로그 예시:**
```json
{
  "timestamp": "2025-10-31T12:34:56.789Z",
  "level": "INFO",
  "request_id": "req_abc123",
  "api_key": "kp_live_xxxx...xxxx",
  "endpoint": "/v1/predict",
  "method": "POST",
  "status_code": 200,
  "response_time_ms": 234,
  "model": "kogpt2",
  "input_length": 15,
  "predictions_count": 10
}
```

---

## 9. 배포 계획

### 9.1 환경
- **개발**: 로컬 개발 서버
- **스테이징**: 테스트 및 검증
- **프로덕션**: 실제 서비스

### 9.2 인프라
- **서버**: Docker 컨테이너
- **오케스트레이션**: Kubernetes (선택)
- **로드 밸런서**: Nginx 또는 AWS ALB
- **데이터베이스**: PostgreSQL (API Key, 로그 저장)
- **캐시**: Redis (선택적 분산 캐시)

### 9.3 CI/CD
- **테스트**: Pytest 자동화 테스트
- **배포**: GitHub Actions
- **롤백**: 이전 버전으로 즉시 롤백 가능

---

## 10. API 사용 예시

### 10.1 Python
```python
import requests

url = "https://api.korean-predictor.example.com/predict"
headers = {
    "Authorization": "Bearer kp_live_your_api_key_here",
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
    print(f"EOT 확률: {data['data']['eot_probability']:.2%}")
else:
    print(f"Error: {data['error']['message']}")
```

### 10.2 cURL
```bash
curl -X POST "https://api.korean-predictor.example.com/predict" \
  -H "Authorization: Bearer kp_live_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요",
    "model": "kogpt2",
    "top_k": 10,
    "temperature": 1.3
  }'
```

### 10.3 JavaScript (fetch)
```javascript
const url = "https://api.korean-predictor.example.com/predict";
const headers = {
  "Authorization": "Bearer kp_live_your_api_key_here",
  "Content-Type": "application/json"
};
const payload = {
  text: "안녕하세요",
  model: "kogpt2",
  top_k: 10,
  temperature: 1.3
};

fetch(url, {
  method: "POST",
  headers: headers,
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      data.data.predictions.forEach(pred => {
        console.log(`${pred.rank}. ${pred.token}: ${(pred.probability * 100).toFixed(2)}%`);
      });
      console.log(`EOT 확률: ${(data.data.eot_probability * 100).toFixed(2)}%`);
    } else {
      console.error(`Error: ${data.error.message}`);
    }
  })
  .catch(error => console.error("Request failed:", error));
```

---

## 11. 향후 계획

### 11.1 Phase 2 기능
- WebSocket 지원 (실시간 스트리밍)
- 다국어 지원 (영어, 일본어 등)
- Fine-tuning API (사용자 커스텀 모델)
- 분석 대시보드 (사용 통계, 성능 지표)

### 11.2 Phase 3 기능
- GraphQL API 지원
- gRPC 지원 (고성능 클라이언트)
- 모델 앙상블 (여러 모델 결과 결합)
- A/B 테스팅 지원

---

## 12. 문의 및 지원

- **문서**: https://docs.korean-predictor.example.com
- **GitHub**: https://github.com/yourusername/korean-predictor
- **이메일**: support@korean-predictor.example.com
- **Discord**: https://discord.gg/korean-predictor

---

**버전**: 1.0.0
**최종 수정일**: 2025-10-31
**작성자**: Korean Predictor Team
