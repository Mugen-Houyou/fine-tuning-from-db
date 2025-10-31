#!/usr/bin/env python3
"""
채팅 EOT (End-of-Turn) 판독기
한국어 채팅 문맥에서 발화가 끝났을 확률을 예측합니다.
"""

import re
import os
import sys
from pathlib import Path
import logging
import click
from typing import List, Tuple, Set
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import track

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from korean_predictor.models.model_manager import ModelManager
from korean_predictor.models.predictor import PredictionService
from korean_predictor.cache.cache import CacheService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class EOTPredictor:
    """EOT 예측기 클래스"""

    def __init__(self, model_name: str = "polyglot"):
        """초기화"""
        self.model_manager = ModelManager()
        self.model_manager.set_run_mode("auto")

        # polyglot 모델 로드
        success, message = self.model_manager.load_model(model_name)
        if not success:
            console.print(f"[red]모델 로드 실패: {message}[/red]")
            sys.exit(1)

        # 캐시 서비스 초기화 (기본값 사용)
        cache_dir = Path.home() / ".cache" / "korean_predictor" / "predictions"
        self.cache_service = CacheService(
            cache_dir=str(cache_dir),
            size_limit=100 * 1024 * 1024,  # 100MB
            ttl_seconds=300  # 5분
        )

        # 예측 서비스 초기화
        self.predictor = PredictionService(self.model_manager, self.cache_service)

        # EOT 토큰 목록 로드
        self.eot_tokens = self._load_eot_tokens()

        # 사용자 정의 EOT 토큰 로드
        self.user_eot_tokens = self._load_user_eot_tokens()

        # 문장부호 및 특수문자
        self.punctuation = set([
            '.', ',', '!', '?', ';', ':', '~', '…', '。', '、', '！', '？', '；', '：',
            '"', "'",  # ASCII 따옴표
            '\u201c', '\u201d', '\u2018', '\u2019',  # 유니코드 따옴표 (LEFT/RIGHT DOUBLE/SINGLE)
            '"', '"', ''', ''',  # 유니코드 따옴표 (예비)
            '「', '」', '『', '』', '（', '）', '(', ')',
            '[', ']', '{', '}', '《', '》', '〈', '〉', '【', '】'
        ])

        # 특수 토큰 패턴
        self.special_token_patterns = [
            r'^</[a-z]+>$',  # </d>, </s> 등
            r'^<\|.*\|>$',    # <|endoftext|>, <|unknown01|> 등
            r'^<[a-zA-Z_]+>$',  # <pad>, <eos>, <bos> 등
            r'^\[.*\]$',      # [SEP], [CLS] 등
        ]

        console.print(f"[green]EOT 예측기 초기화 완료 (모델: {model_name})[/green]")

    def _load_eot_tokens(self) -> Set[str]:
        """EOT 토큰 목록 로드"""
        eot_tokens = set()
        eot_file = project_root / "EOT-예측-첫-토큰.md"

        if not eot_file.exists():
            logger.warning(f"EOT 토큰 파일 없음: {eot_file}")
            return eot_tokens

        with open(eot_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 빈 줄이 아닌 모든 줄을 토큰으로 간주
                    eot_tokens.add(line)

        logger.info(f"EOT 토큰 {len(eot_tokens)}개 로드됨")
        return eot_tokens

    def _load_user_eot_tokens(self) -> Set[str]:
        """사용자 정의 EOT 토큰 로드"""
        user_eot_tokens = set()
        user_file = project_root / "user-defined-eots.txt"

        if not user_file.exists():
            logger.info("사용자 정의 EOT 파일 없음")
            return user_eot_tokens

        with open(user_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 빈 줄과 주석 제외
                    user_eot_tokens.add(line)

        # '#'은 파일에서 주석으로 처리되므로 여기서 직접 추가
        user_eot_tokens.add('#')

        logger.info(f"사용자 정의 EOT 토큰 {len(user_eot_tokens)}개 로드됨")
        return user_eot_tokens

    def is_eot_token(self, token: str) -> bool:
        """토큰이 EOT 토큰인지 확인"""
        # EOT 토큰 목록에 있는지 확인
        if token in self.eot_tokens:
            return True

        # 사용자 정의 EOT 토큰 확인
        if token in self.user_eot_tokens:
            return True

        # 문장부호인지 확인
        if any(p in token for p in self.punctuation):
            return True

        # 특수 토큰 패턴 확인
        for pattern in self.special_token_patterns:
            if re.match(pattern, token):
                return True

        # 10자 이상의 비정상적인 토큰
        if len(token) > 10:
            return True

        # 공백만 있는 토큰
        if token.strip() == '':
            return True

        return False

    def predict_eot(self, text: str, top_k: int = 10, temperature: float = 1.3) -> Tuple[float, List[Tuple[str, float, bool]]]:
        """
        EOT 확률 예측

        Args:
            text: 입력 텍스트
            top_k: 예측할 토큰 개수
            temperature: 샘플링 온도

        Returns:
            (EOT 확률, [(토큰, 확률, EOT여부), ...])
        """
        # 다음 토큰 예측
        _, predictions = self.predictor.predict_next_tokens(
            text=text,
            top_k=top_k,
            temperature=temperature,
            complete_word=False,  # 단일 토큰만 예측
            include_special_tokens=True
        )

        # EOT 토큰 확률 계산
        eot_probability = 0.0
        detailed_results = []

        for token, prob in predictions:
            is_eot = self.is_eot_token(token)
            if is_eot:
                eot_probability += prob
            detailed_results.append((token, prob, is_eot))

        return eot_probability, detailed_results

    def display_results(self, text: str, eot_prob: float, details: List[Tuple[str, float, bool]]):
        """결과 표시"""
        # console.print("\n" + "="*60)
        # console.print(f"[bold blue]입력 텍스트:[/bold blue] {text}")
        # console.print("="*60 + "\n")

        # EOT 확률 표시
        color = "red" if eot_prob > 0.7 else "yellow" if eot_prob > 0.3 else "green"
        bar_length = int(eot_prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)

        console.print(f"[bold]EOT 확률:[/bold] [{color}]{eot_prob*100:.1f}%[/{color}]  {bar}")

        if eot_prob > 0.7:
            console.print("[red]→ 발화가 끝날 가능성이 높습니다[/red]")
        elif eot_prob > 0.3:
            console.print("[yellow]→ 발화가 끝날 수도 있습니다[/yellow]")
        else:
            console.print("[green]→ 발화가 계속될 가능성이 높습니다[/green]")

        # 상세 예측 결과 표시
        console.print("\n[bold]예측 토큰 상세:[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("순위", style="dim", width=6)
        table.add_column("토큰", width=20)
        table.add_column("확률", justify="right", width=10)
        table.add_column("EOT", justify="center", width=8)
        table.add_column("타입", width=15)

        for i, (token, prob, is_eot) in enumerate(details, 1):
            # 토큰 타입 판별
            token_type = ""
            if is_eot:
                if token in self.eot_tokens:
                    token_type = "종료표현"
                elif token in self.user_eot_tokens:
                    token_type = "사용자정의"
                elif any(p in token for p in self.punctuation):
                    token_type = "문장부호"
                elif any(re.match(p, token) for p in self.special_token_patterns):
                    token_type = "특수토큰"
                elif len(token) > 10:
                    token_type = "비정상토큰"
                else:
                    token_type = "기타EOT"
            else:
                token_type = "일반"

            # 토큰 표시 (특수문자 이스케이프)
            display_token = repr(token)[1:-1] if len(token) > 20 or '\n' in token else token

            table.add_row(
                f"{i}",
                display_token[:20] + "..." if len(display_token) > 20 else display_token,
                f"{prob*100:.2f}%",
                "[red]●[/red]" if is_eot else "[green]○[/green]",
                token_type
            )

        console.print(table)
        console.print()

    def interactive_mode(self):
        """대화형 모드"""
        console.print("[bold cyan]채팅 EOT 판독기[/bold cyan]")
        console.print("텍스트를 입력하면 발화 종료 확률을 예측합니다.")
        console.print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
        console.print("명령어를 보려면 '/help'를 입력하세요.\n")

        # 기본 설정값
        self._top_k = 10
        self._temperature = 0.5

        while True:
            try:
                # 사용자 입력
                text = console.input("[bold yellow]텍스트 입력> [/bold yellow]").strip()

                # 종료 명령 확인
                if text.lower() in ['quit', 'exit', 'q']:
                    console.print("[dim]프로그램을 종료합니다.[/dim]")
                    break

                # 특수 명령 처리
                if text.startswith('/'):
                    self._handle_command(text)
                    continue

                # 빈 입력 처리
                if not text:
                    console.print("[dim]텍스트를 입력해주세요.[/dim]")
                    continue

                # EOT 예측
                with console.status("[bold green]예측 중...[/bold green]"):
                    eot_prob, details = self.predict_eot(text, self._top_k, self._temperature)

                # 결과 표시
                self.display_results(text, eot_prob, details)

            except KeyboardInterrupt:
                console.print("\n[dim]프로그램을 종료합니다.[/dim]")
                break
            except Exception as e:
                console.print(f"[red]오류 발생: {e}[/red]")
                logger.error(f"예측 중 오류: {e}", exc_info=True)

    def _handle_command(self, command: str):
        """특수 명령 처리"""
        cmd = command[1:].lower().split()

        if not cmd:
            return

        if cmd[0] == 'help':
            self._show_help()
        elif cmd[0] == 'config':
            self._show_config()
        elif cmd[0] == 'temperature' or cmd[0] == 'temp':
            if len(cmd) >= 2:
                self._set_temperature(cmd[1])
            else:
                console.print(f"[cyan]현재 Temperature: {self._temperature}[/cyan]")
        elif cmd[0] == 'topk' or cmd[0] == 'top_k':
            if len(cmd) >= 2:
                self._set_top_k(cmd[1])
            else:
                console.print(f"[cyan]현재 Top-K: {self._top_k}[/cyan]")
        elif cmd[0] == 'set':
            if len(cmd) >= 3:
                self._set_config(cmd[1], cmd[2])
            else:
                console.print("[yellow]사용법: /set <옵션> <값>[/yellow]")
        else:
            console.print(f"[red]알 수 없는 명령: {command}[/red]")
            console.print("[dim]/help를 입력하면 사용 가능한 명령을 볼 수 있습니다.[/dim]")

    def _show_help(self):
        """도움말 표시"""
        help_text = """
[bold]사용 가능한 명령:[/bold]

/help                  - 이 도움말 표시
/config                - 현재 설정 표시
/temperature <숫자>    - Temperature 조정 (0.1-2.0)
/temperature           - 현재 Temperature 확인
/topk <숫자>           - Top-K 조정 (1-20)
/topk                  - 현재 Top-K 확인
/set topk <숫자>       - Top-K 설정
/set temp <숫자>       - Temperature 설정
quit, exit, q          - 프로그램 종료

[bold]📊 EOT 예측 정보:[/bold]
다음 토큰 예측 결과를 분석하여 발화 종료 (End-Of-Turn; EOT) 확률을 계산합니다.
EOT 확률이 높으면 발화가 끝날 가능성이 높습니다.

[bold]🌡️  Temperature 설정:[/bold]
Temperature는 예측의 무작위성을 조절합니다.
- 낮은 값(0.1-0.5): 보수적, 일관된 예측
- 중간 값(0.5-1.5): 균형잡힌 예측
- 높은 값(1.5-2.0): 창의적, 다양한 예측

[bold]🔢 Top-K 설정:[/bold]
예측할 토큰의 개수를 지정합니다 (1-20).
더 많은 토큰을 분석할수록 EOT 확률이 정확해집니다.
        """
        console.print(help_text)

    def _show_config(self):
        """현재 설정 표시"""
        from rich.table import Table

        table = Table(title="현재 설정")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")

        table.add_row("Top-K", str(self._top_k))
        table.add_row("Temperature", str(self._temperature))
        table.add_row("EOT 토큰 개수", str(len(self.eot_tokens)))
        table.add_row("사용자 정의 EOT 토큰", str(len(self.user_eot_tokens)))

        console.print(table)

    def _set_temperature(self, value: str):
        """Temperature 설정"""
        try:
            new_value = float(value)
            if 0.1 <= new_value <= 2.0:
                self._temperature = new_value
                console.print(f"[green]Temperature를 {new_value}로 설정했습니다.[/green]")
            else:
                console.print("[red]Temperature는 0.1-2.0 사이여야 합니다.[/red]")
        except ValueError:
            console.print(f"[red]잘못된 값: {value}[/red]")

    def _set_top_k(self, value: str):
        """Top-K 설정"""
        try:
            new_value = int(value)
            if 1 <= new_value <= 20:
                self._top_k = new_value
                console.print(f"[green]Top-K를 {new_value}로 설정했습니다.[/green]")
            else:
                console.print("[red]Top-K는 1-20 사이여야 합니다.[/red]")
        except ValueError:
            console.print(f"[red]잘못된 값: {value}[/red]")

    def _set_config(self, option: str, value: str):
        """설정 변경"""
        if option in ['topk', 'top_k']:
            self._set_top_k(value)
        elif option in ['temp', 'temperature']:
            self._set_temperature(value)
        else:
            console.print(f"[red]알 수 없는 옵션: {option}[/red]")
            console.print("[dim]사용 가능한 옵션: topk, temp[/dim]")


@click.command()
@click.option('--text', '-t', help='예측할 텍스트')
@click.option('--model', '-m', default='polyglot',
              type=click.Choice(['polyglot', 'kogpt2', 'kanana-nano-2.1b-base'], case_sensitive=False),
              help='사용할 모델')
@click.option('--top-k', '-k', default=10, type=int, help='예측할 토큰 개수')
@click.option('--temperature', '--temp', default=0.5, type=float, help='샘플링 온도')
def main(text, model, top_k, temperature):
    """채팅 EOT (End-of-Turn) 판독기"""

    # EOT 예측기 초기화
    predictor = EOTPredictor(model_name=model)

    if text:
        # 단일 예측 모드
        # 한번 실행할 때마다 모델 load/unload하므로 비효율적일 수 있음
        eot_prob, details = predictor.predict_eot(text, top_k, temperature)
        predictor.display_results(text, eot_prob, details)
    else:
        # 대화형 모드
        predictor.interactive_mode()

    # 정리
    if hasattr(predictor, 'cache_service'):
        predictor.cache_service.close()


if __name__ == "__main__":
    main()