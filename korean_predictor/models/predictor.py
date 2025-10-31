"""
Prediction Service - 다음 토큰/어절 예측 서비스
"""
import torch
from typing import List, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)


class PredictionService:
    """한국어 다음 토큰/어절 예측 서비스"""

    def __init__(self, model_manager, cache_service=None):
        """
        Args:
            model_manager: ModelManager 인스턴스
            cache_service: 선택적 캐시 서비스
        """
        self.model_manager = model_manager
        self.cache = cache_service

        # 한국어 어절 경계 판단용 조사/어미
        self.korean_endings = [
            '은', '는', '이', '가', '을', '를', '에', '에서', '으로', '와', '과',
            '다', '요', '습니다', '했다', '한다', '하는', '하고', '해요', '해서',
            '에게', '한테', '께', '의', '로', '부터', '까지', '만', '도', '나', 
            '든', '거나', '든지', '던지', '하여', '기에', '으므로', '므로'
        ]

    def predict_next_tokens(
        self,
        text: str,
        top_k: int = 8,
        temperature: float = 1,
        complete_word: bool = True,
        max_length: int = 128,
        include_special_tokens: bool = True
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        다음 토큰/어절 예측 + End-of-Turn 확률

        Args:
            text: 입력 텍스트
            top_k: 상위 k개 예측
            temperature: 샘플링 온도 (낮을수록 보수적)
            complete_word: True면 완전한 어절까지 생성
            max_length: 최대 입력 길이
            include_special_tokens: True면 <eos> 등 특수 토큰 포함

        Returns:
            (end_of_turn_probability, [(예측 텍스트, 확률)]) 튜플
        """
        start_time = time.time()

        # 캐시 확인
        if self.cache:
            cache_key = f"{text}:{top_k}:{temperature}:{complete_word}:{include_special_tokens}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"캐시 히트 (키: {cache_key[:20]}...)")
                return cached_result

        # 입력 전처리
        inputs = self.model_manager.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length
        ).to(self.model_manager.device)

        # 예측 수행
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs)
            logits = outputs.logits[0, -1, :]  # 마지막 토큰의 logits

            # Temperature 적용
            if temperature != 1.0:
                logits = logits / temperature

            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)

            # EOS 토큰 확률 계산
            eot_prob = self._get_eos_probability(probs)

            # Top-k 선택 (더 많이 선택해서 나중에 필터링)
            k_expanded = min(top_k * 5, 100) if complete_word else top_k
            top_probs, top_indices = torch.topk(probs, k_expanded)

        # 예측 생성
        if complete_word:
            predictions = self._generate_complete_words(
                text, top_indices, top_probs, top_k, include_special_tokens
            )
        else:
            predictions = self._generate_tokens(
                top_indices[:top_k], top_probs[:top_k], include_special_tokens
            )

        # 캐시 저장
        result = (eot_prob, predictions)
        if self.cache and predictions:
            self.cache.set(cache_key, result, ttl=300)  # 5분 TTL

        elapsed = time.time() - start_time
        logger.info(f"예측 완료: {elapsed:.3f}초, {len(predictions)}개 결과, EOT: {eot_prob:.1%}")

        return result

    def _get_eos_probability(self, probs: torch.Tensor) -> float:
        """EOS 토큰 확률 추출"""
        tokenizer = self.model_manager.tokenizer

        # 모델별 EOS 토큰 찾기
        eos_token_id = None

        # KoGPT2의 경우 </d> 토큰 사용 (ID: 8)
        if 'kogpt2' in tokenizer.name_or_path.lower():
            eos_token_id = 8  # </d> 토큰
            logger.debug(f"Using </d> token for KoGPT2, ID: {eos_token_id}")
        # Kanana 모델의 경우 <|end_of_text|> 사용 (ID: 128001)
        elif 'kanana' in tokenizer.name_or_path.lower():
            eos_token_id = 128001  # <|end_of_text|> 토큰
            logger.debug(f"Using <|end_of_text|> token for Kanana, ID: {eos_token_id}")
        else:
            # 다른 모델의 경우 기본 eos_token_id 사용
            eos_token_id = tokenizer.eos_token_id

            # eos_token_id가 범위를 벗어나면 대체 토큰 찾기
            if eos_token_id is not None and eos_token_id >= len(probs):
                logger.debug(f"EOS token ID {eos_token_id} out of bounds (probs size: {len(probs)}), finding alternative")
                eos_token_id = None

        # EOS 토큰을 찾지 못했으면 여러 가능한 토큰들 시도
        if eos_token_id is None:
            possible_eos_tokens = ['</d>', '</s>', '<eos>', '<|endoftext|>', '<|end_of_text|>']
            for token in possible_eos_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != tokenizer.unk_token_id and token_id < len(probs):
                    eos_token_id = token_id
                    logger.debug(f"Found alternative EOS token: {token} with ID: {token_id}")
                    break

        # End-of-Turn 확률
        if eos_token_id is not None and eos_token_id < len(probs):
            eot_prob = float(probs[eos_token_id])
            logger.debug(f"EOT probability for token ID {eos_token_id}: {eot_prob:.1%}")
            return eot_prob

        logger.warning(f"Could not find valid EOS token ID (probs shape: {probs.shape})")
        return 0.0

    def predict_end_of_turn(
        self,
        text: str,
        temperature: float = 0.8,
        max_length: int = 128
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        발화 종료(end-of-turn) 확률 예측

        Args:
            text: 입력 텍스트
            temperature: 샘플링 온도
            max_length: 최대 입력 길이

        Returns:
            (end_of_turn_probability, [(다음 토큰, 확률)]) 튜플
        """
        # 입력 전처리
        inputs = self.model_manager.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length
        ).to(self.model_manager.device)

        # 예측 수행
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs)
            logits = outputs.logits[0, -1, :]  # 마지막 토큰의 logits

            # Temperature 적용
            if temperature != 1.0:
                logits = logits / temperature

            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)

        # EOS 토큰 ID 찾기
        tokenizer = self.model_manager.tokenizer
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            # </s> 토큰 직접 찾기
            eos_token_id = tokenizer.convert_tokens_to_ids('</s>')
            if eos_token_id == tokenizer.unk_token_id:
                eos_token_id = tokenizer.convert_tokens_to_ids('<eos>')

        # End-of-Turn 확률 계산
        eot_prob = 0.0
        if eos_token_id is not None and eos_token_id < len(probs):
            eot_prob = float(probs[eos_token_id])

        # 상위 5개 토큰 예측 (특수 토큰 포함)
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)

        predictions = []
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            token = tokenizer.decode([idx])
            predictions.append((token, float(prob)))

        # 확률 기준 내림차순 정렬
        predictions.sort(key=lambda x: x[1], reverse=True)

        return eot_prob, predictions

    def _generate_tokens(
        self,
        indices: torch.Tensor,
        probs: torch.Tensor,
        include_special_tokens: bool = True
    ) -> List[Tuple[str, float]]:
        """단순 토큰 예측"""
        predictions = []
        for idx, prob in zip(indices.tolist(), probs.tolist()):
            token = self.model_manager.tokenizer.decode([idx])

            # 특수 토큰 처리
            if not include_special_tokens:
                # 특수 토큰 제외
                if token.startswith('<') or token.startswith('['):
                    continue

            predictions.append((token.strip(), float(prob)))

        # 확률 기준 내림차순 정렬
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def _generate_complete_words(
        self,
        context: str,
        indices: torch.Tensor,
        probs: torch.Tensor,
        top_k: int,
        include_special_tokens: bool = True
    ) -> List[Tuple[str, float]]:
        """완전한 어절 생성"""
        predictions = []
        seen_words = set()

        for idx, prob in zip(indices.tolist(), probs.tolist()):
            if len(predictions) >= top_k:
                break

            # 시작 토큰으로 어절 생성
            word, word_prob = self._complete_single_word(
                context, idx, float(prob)
            )

            # 유효한 어절이고 중복이 아닌 경우만 추가
            if word and word not in seen_words and len(word.strip()) > 0:
                # 특수 토큰 처리
                if not include_special_tokens:
                    # 특수 토큰 제외
                    if word.startswith('<') or word.startswith('['):
                        continue

                seen_words.add(word)
                predictions.append((word, word_prob))

        # 확률 기준 내림차순 정렬
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def _complete_single_word(
        self,
        context: str,
        start_token_id: int,
        start_prob: float,
        max_word_length: int = 15
    ) -> Tuple[str, float]:
        """단일 토큰에서 시작해서 완전한 어절 생성"""
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        # 시작 토큰만 디코딩 (특수 토큰 포함 - <eos> 등이 중요함)
        start_word = tokenizer.decode([start_token_id])

        # 특수 토큰이면 그대로 반환 (화자 교대 판단에 중요)
        if start_word in ['</s>', '<eos>', '<pad>', '<unk>']:
            return start_word, start_prob

        if self._is_complete_word(start_word):
            return start_word.strip(), start_prob

        # 컨텍스트 인코딩
        context_ids = tokenizer.encode(context, return_tensors="pt").to(self.model_manager.device)
        context_length = context_ids.shape[1]

        # 컨텍스트 + 시작 토큰으로 입력 생성
        current_ids = torch.cat([
            context_ids,
            torch.tensor([[start_token_id]]).to(self.model_manager.device)
        ], dim=1)

        generated_ids = [start_token_id]
        accumulated_prob = start_prob

        # 어절 완성까지 토큰 생성
        for _ in range(max_word_length):
            with torch.no_grad():
                outputs = model(current_ids)
                next_logits = outputs.logits[0, -1, :]
                next_probs = torch.softmax(next_logits, dim=-1)

                # 가장 확률 높은 토큰 선택
                next_token_id = torch.argmax(next_probs).item()
                next_prob = next_probs[next_token_id].item()

            # 종료 토큰이나 공백 만나면 중단 (특수 토큰 유지)
            next_token = tokenizer.decode([next_token_id])

            # 특수 토큰은 유지하되 생성 중단
            if next_token in ['</s>', '<eos>', '<pad>', '<unk>']:
                break

            # 공백이나 구두점으로 어절 종료
            if (next_token.strip() == '' or
                next_token in ['.', ',', '!', '?'] or
                '▁' in next_token):  # sentencepiece의 공백 마커
                break

            generated_ids.append(next_token_id)
            accumulated_prob *= next_prob

            # 현재까지 생성된 어절 확인
            current_word = tokenizer.decode(generated_ids)
            if self._is_complete_word(current_word):
                break

            # 다음 반복을 위해 입력 업데이트
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token_id]]).to(self.model_manager.device)
            ], dim=1)

        # 전체 시퀀스를 디코딩한 후 context 부분 제거
        full_text = tokenizer.decode(current_ids[0])
        context_text = context.strip()

        # context 제거하여 새로 생성된 부분만 추출
        if full_text.startswith(context_text):
            word = full_text[len(context_text):].strip()
        else:
            # context와 정확히 매칭되지 않는 경우, generated_ids만 디코딩
            word = tokenizer.decode(generated_ids).strip()

        # 생성된 단어가 context에 이미 있는지 확인 (중복 방지)
        if word and word in context:
            # context의 마지막 부분과 겹치는 경우 빈 문자열 반환
            return "", 0.0

        return word, accumulated_prob

    def _is_complete_word(self, text: str) -> bool:
        """한국어 어절 완성 여부 판단"""
        text = text.strip()

        # 빈 문자열이면 미완성
        if not text:
            return False

        # 조사나 어미로 끝나면 완성된 어절
        for ending in self.korean_endings:
            if text.endswith(ending):
                return True

        # 마침표, 쉼표 등으로 끝나면 완성
        if text[-1] in '.!?,;':
            return True

        # 한글 글자수가 2개 이상이면 일단 완성으로 간주
        korean_chars = [c for c in text if '가' <= c <= '힣']
        if len(korean_chars) >= 2:
            return True

        return False

    def predict_with_context(
        self,
        context: str,
        prefix: str,
        top_k: int = 5,
        temperature: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        컨텍스트와 프리픽스를 구분해서 예측

        Args:
            context: 이전 문맥
            prefix: 현재 입력 중인 부분 단어
            top_k: 상위 k개 예측
            temperature: 샘플링 온도

        Returns:
            [(예측 텍스트, 확률)] 리스트
        """
        # 전체 텍스트 = 컨텍스트 + 프리픽스
        full_text = context + prefix

        # 예측 수행
        eot_prob, predictions = self.predict_next_tokens(
            full_text,
            top_k=top_k * 2,  # 더 많이 생성해서 필터링
            temperature=temperature,
            complete_word=True
        )

        # 프리픽스로 시작하는 예측만 필터링
        if prefix:
            filtered = []
            for word, prob in predictions:
                if word.startswith(prefix):
                    # 프리픽스 이후 부분만 반환
                    completion = word[len(prefix):]
                    if completion:
                        filtered.append((completion, prob))
            # 확률 기준 내림차순 정렬
            filtered.sort(key=lambda x: x[1], reverse=True)
            return filtered[:top_k]

        return predictions[:top_k]