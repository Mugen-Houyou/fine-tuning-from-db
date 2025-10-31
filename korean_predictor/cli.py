"""
CLI Interface - 명령줄 인터페이스
"""
import sys
from typing import List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich import print as rprint
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from pathlib import Path
import time


class CLI:
    """Command Line Interface for Korean Token Predictor"""

    def __init__(self, predictor, config):
        """
        Args:
            predictor: PredictionService 인스턴스
            config: Config 클래스
        """
        self.predictor = predictor
        self.config = config
        self.console = Console()
        self.history_file = Path.home() / '.korean_predictor_history'

    def display_predictions(self, predictions: List[Tuple[str, float]], elapsed: float = 0, eot_prob: float = None):
        """예측 결과를 테이블로 표시"""
        if not predictions:
            self.console.print("[yellow]예측 결과가 없습니다.[/yellow]")
            return

        # 테이블 생성
        table = Table(title=f"다음 단어 예측 (소요시간: {elapsed:.3f}초)")
        table.add_column("순위", style="cyan", width=6)
        table.add_column("예측 단어", style="green", width=40, no_wrap=False)  # 너비 증가 + 줄바꿈 허용
        table.add_column("신뢰도", style="magenta", width=10)
        table.add_column("확률 막대", width=25)
        table.add_column("타입", width=12)

        # 예측 결과 추가
        for i, (word, confidence) in enumerate(predictions, 1):
            # 특수 토큰 여부 확인
            is_special = word in ['</s>', '<eos>', '<pad>', '<unk>', '</d>']

            # 긴 토큰 확인 (20자 이상)
            is_long = len(word) > 20

            # 단어 표시 처리
            if is_special:
                word_display = f"[bold red]{word}[/bold red]"
            elif is_long:
                # 긴 토큰은 잘라서 표시하되 전체 길이 표시
                word_display = f"{word[:35]}..." if len(word) > 35 else word
            else:
                word_display = word

            # 타입 표시
            if word in ['</s>', '<eos>', '</d>']:
                token_type = "종료"
                type_color = "red"
            elif is_special:
                token_type = "특수"
                type_color = "yellow"
            elif is_long:
                token_type = f"긴토큰({len(word)})"
                type_color = "cyan"
            else:
                token_type = "일반"
                type_color = "white"

            # 확률 막대 그래프
            bar_length = int(confidence * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            table.add_row(
                str(i),
                word_display,
                f"{confidence:.1%}",
                f"[blue]{bar}[/blue]",
                f"[{type_color}]{token_type}[/{type_color}]"
            )

        # self.console.print(predictions)
        self.console.print(table)

        # End-of-Turn 확률 표시 (한 줄로 간단하게)
        if eot_prob is not None:
            eot_color = "red" if eot_prob > 0.5 else "yellow" if eot_prob > 0.2 else "green"
            eot_bar_length = int(eot_prob * 30)
            eot_bar = "█" * eot_bar_length + "░" * (30 - eot_bar_length)
            self.console.print(f"발화 종료 확률: [{eot_color}]{eot_prob:.1%}[/{eot_color}]            {eot_bar}")

    def interactive_mode(self):
        """대화형 모드"""
        self.console.print(Panel.fit(
            "[bold green]한국어 다음 토큰 예측기[/bold green]\n"
            "텍스트를 입력하고 Enter를 누르면 다음 단어를 예측합니다.\n"
            "'quit', 'exit', 'q'를 입력하면 종료합니다.",
            title="환영합니다"
        ))

        # 모델 정보 표시
        self._display_model_info()

        # 입력 루프
        while True:
            try:
                # 입력 받기 (히스토리 지원)
                text = prompt(
                    "텍스트 입력 > ",
                    history=FileHistory(str(self.history_file)),
                    auto_suggest=AutoSuggestFromHistory()
                )

                # 종료 명령 확인
                if text.lower() in ['quit', 'exit', 'q', '종료']:
                    self.console.print("[yellow]프로그램을 종료합니다.[/yellow]")
                    break

                # 특수 명령 처리
                if text.startswith('/'):
                    self._handle_command(text)
                    continue

                # 빈 입력 무시
                if not text.strip():
                    continue

                # 예측 수행 (항상 EOT 확률 포함)
                self._predict_and_display(text)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Ctrl+C 감지. 종료하려면 'quit'를 입력하세요.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]오류 발생: {e}[/red]")

    def _predict_and_display(self, text: str):
        """예측 수행 및 결과 표시 (항상 EOT 확률 포함)"""
        with self.console.status("[bold green]예측 중...[/bold green]", spinner="dots") as status:
            start_time = time.time()

            try:
                # 다음 토큰 예측 + EOT 확률 (통합) + 타임아웃
                eot_prob, predictions = self.predictor.predict_next_tokens(
                    text,
                    top_k=self.config.DEFAULT_TOP_K,
                    temperature=self.config.DEFAULT_TEMPERATURE,
                    complete_word=self.config.COMPLETE_WORD,
                    include_special_tokens=self.config.INCLUDE_SPECIAL_TOKENS,
                    timeout=self.config.DEFAULT_TIMEOUT
                )

                elapsed = time.time() - start_time

                # 결과 표시 (항상 EOT 확률 포함)
                self.display_predictions(predictions, elapsed, eot_prob)

            except TimeoutError as e:
                elapsed = time.time() - start_time
                self.console.print(f"[red]타임아웃 오류: {str(e)}[/red]")
                self.console.print(f"[yellow]경과 시간: {elapsed:.1f}초 (제한: {self.config.DEFAULT_TIMEOUT}초)[/yellow]")
                self.console.print(f"[cyan]Tip: /timeout 명령으로 타임아웃 시간을 조정할 수 있습니다.[/cyan]")

    def _handle_command(self, command: str):
        """특수 명령 처리"""
        cmd = command[1:].lower().split()

        if not cmd:
            return

        if cmd[0] == 'help':
            self._show_help()
        elif cmd[0] == 'config':
            self._show_config()
        elif cmd[0] == 'cache':
            if len(cmd) > 1 and cmd[1] == 'clear':
                self._clear_cache()
            else:
                self._show_cache_stats()
        elif cmd[0] == 'model':
            if len(cmd) > 1:
                if cmd[1] == 'info':
                    self._display_model_info()
                elif cmd[1] == 'list':
                    self._list_available_models()
                else:
                    self.console.print("[yellow]사용법: /model info 또는 /model list[/yellow]")
            else:
                self.console.print("[yellow]사용법: /model info 또는 /model list[/yellow]")
        elif cmd[0] == 'temperature' or cmd[0] == 'temp':
            if len(cmd) >= 2:
                self._set_temperature(cmd[1])
            else:
                self.console.print(f"[cyan]현재 Temperature: {self.config.DEFAULT_TEMPERATURE}[/cyan]")
        elif cmd[0] == 'timeout':
            if len(cmd) >= 2:
                self._set_timeout(cmd[1])
            else:
                self.console.print(f"[cyan]현재 Timeout: {self.config.DEFAULT_TIMEOUT}초[/cyan]")
        elif cmd[0] == 'set':
            if len(cmd) >= 3:
                self._set_config(cmd[1], cmd[2])
            else:
                self.console.print("[yellow]사용법: /set <옵션> <값>[/yellow]")
        else:
            self.console.print(f"[red]알 수 없는 명령: {command}[/red]")

    def _show_help(self):
        """도움말 표시"""
        help_text = """
[bold]사용 가능한 명령:[/bold]

/help                  - 이 도움말 표시
/config                - 현재 설정 표시
/cache                 - 캐시 통계 표시
/cache clear           - 캐시 비우기
/model info            - 현재 로드된 모델 정보
/model list            - 사용 가능한 모델 목록
/set top_k <숫자>      - 예측 개수 설정 (1-10)
/temperature <숫자>    - Temperature 조정 (0.1-2.0, 지원 모델만)
/temperature           - 현재 Temperature 확인
/timeout <숫자>        - Timeout 설정 (초, 0=무제한)
/timeout               - 현재 Timeout 확인
quit, exit, q          - 프로그램 종료

[bold]📊 예측 정보:[/bold]
모든 예측에서 다음 토큰 목록과 발화 종료(EOT) 확률을 함께 표시합니다.
EOT 확률이 높으면 화자가 말을 끝낼 가능성이 높습니다.

[bold]🌡️  Temperature 설정:[/bold]
Temperature는 예측의 무작위성을 조절합니다. (높을수록 다양한 결과)
일부 추론 특화 모델(DNA-R1 등)은 temperature 조정이 불가능합니다.

[bold]⏱️  Timeout 설정:[/bold]
예측이 너무 오래 걸릴 경우 자동으로 중단합니다.
기본값: 60초, 0으로 설정하면 무제한으로 기다립니다.

[bold]💡 모델 변경:[/bold]
다른 모델을 사용하려면 프로그램 재시작이 필요합니다.
예: python main.py --model dna-r1
        """
        self.console.print(Panel(help_text, title="도움말"))

    def _show_config(self):
        """현재 설정 표시"""
        table = Table(title="현재 설정")
        table.add_column("설정", style="cyan")
        table.add_column("값", style="green")

        table.add_row("예측 개수 (top_k)", str(self.config.DEFAULT_TOP_K))
        table.add_row("온도 (temperature)", str(self.config.DEFAULT_TEMPERATURE))
        timeout_str = f"{self.config.DEFAULT_TIMEOUT}초" if self.config.DEFAULT_TIMEOUT else "무제한"
        table.add_row("타임아웃 (timeout)", timeout_str)
        table.add_row("완전한 어절 생성", "Yes" if self.config.COMPLETE_WORD else "No")
        table.add_row("특수 토큰 포함", "Yes" if self.config.INCLUDE_SPECIAL_TOKENS else "No")
        table.add_row("캐시 활성화", "Yes" if self.config.CACHE_ENABLED else "No")
        table.add_row("최대 입력 길이", str(self.config.MAX_INPUT_LENGTH))
        table.add_row("EOT 분석", "항상 활성화")

        self.console.print(table)

    def _show_cache_stats(self):
        """캐시 통계 표시"""
        if hasattr(self.predictor, 'cache') and self.predictor.cache:
            stats = self.predictor.cache.get_stats()

            table = Table(title="캐시 통계")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="green")

            table.add_row("캐시 크기", f"{stats['size'] / 1024:.1f} KB")
            table.add_row("저장된 항목", str(stats['items']))
            table.add_row("캐시 히트", str(stats.get('hits', 0)))
            table.add_row("캐시 미스", str(stats.get('misses', 0)))

            self.console.print(table)
        else:
            self.console.print("[yellow]캐시가 비활성화되어 있습니다.[/yellow]")

    def _clear_cache(self):
        """캐시 비우기"""
        if hasattr(self.predictor, 'cache') and self.predictor.cache:
            self.predictor.cache.clear()
            self.console.print("[green]캐시를 비웠습니다.[/green]")
        else:
            self.console.print("[yellow]캐시가 비활성화되어 있습니다.[/yellow]")

    def _display_model_info(self):
        """모델 정보 표시"""
        info = self.predictor.model_manager.get_model_info()

        if not info.get('loaded'):
            self.console.print("[red]모델이 로드되지 않았습니다.[/red]")
            return

        table = Table(title="현재 로드된 모델 정보")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")

        table.add_row("모델명", info['model_name'])
        table.add_row("어휘 크기", f"{info['vocab_size']:,}")
        table.add_row("최대 길이", str(info['max_length']))
        table.add_row("파라미터", f"{info['parameters']:.1f}M")
        table.add_row("디바이스", info['device'])

        # temperature 지원 시 현재 값 표시
        if info.get('supports_temperature', True):
            table.add_row("Temperature", f"{self.config.DEFAULT_TEMPERATURE}")

        self.console.print(table)

    def _list_available_models(self):
        """사용 가능한 모델 목록 표시"""
        from korean_predictor.models.model_manager import ModelManager

        table = Table(title="사용 가능한 모델 목록")
        table.add_column("모델명", style="cyan", width=15)
        table.add_column("파라미터", style="green", width=10)
        table.add_column("설명", style="white", width=45)
        table.add_column("메모리", style="yellow", width=15)

        models = ModelManager.list_models()
        for model_key, model_info in models.items():
            # 추론 모델은 별표 표시
            model_display = f"{model_key} *" if model_info.get('reasoning_model', False) else model_key
            table.add_row(
                model_display,
                model_info['params'],
                model_info['description'],
                model_info['memory']
            )

        self.console.print(table)
        self.console.print("\n[dim]* 추론 특화 모델 (DeepSeek-R1 방식)[/dim]")
        self.console.print("\n[bold]사용 방법:[/bold]")
        self.console.print("  프로그램 재시작: python main.py --model <모델명>")
        self.console.print("  예시: python main.py --model dna-r1")

    def _set_temperature(self, value: str):
        """Temperature 설정 (모델이 지원하는 경우에만)"""
        # temperature 지원 여부 확인
        info = self.predictor.model_manager.get_model_info()
        if not info.get('supports_temperature', True):
            self.console.print("[red]현재 모델은 temperature를 지원하지 않습니다.[/red]")
            self.console.print("[yellow]추론 특화 모델(DNA-R1 등)은 temperature 조정이 불가능합니다.[/yellow]")
            return

        try:
            new_value = float(value)
            if 0.1 <= new_value <= 2.0:
                self.config.DEFAULT_TEMPERATURE = new_value
                self.console.print(f"[green]Temperature를 {new_value}로 설정했습니다.[/green]")
            else:
                self.console.print("[red]Temperature는 0.1-2.0 사이여야 합니다.[/red]")
        except ValueError:
            self.console.print(f"[red]잘못된 값: {value}[/red]")

    def _set_timeout(self, value: str):
        """Timeout 설정"""
        try:
            new_value = int(value)
            if new_value > 0:
                self.config.DEFAULT_TIMEOUT = new_value
                self.console.print(f"[green]Timeout을 {new_value}초로 설정했습니다.[/green]")
            elif new_value == 0:
                self.config.DEFAULT_TIMEOUT = None
                self.console.print(f"[green]Timeout을 비활성화했습니다 (무제한).[/green]")
            else:
                self.console.print("[red]Timeout은 0 이상이어야 합니다 (0=무제한).[/red]")
        except ValueError:
            self.console.print(f"[red]잘못된 값: {value}[/red]")

    def _set_config(self, option: str, value: str):
        """설정 변경"""
        try:
            if option == 'top_k':
                new_value = int(value)
                if 1 <= new_value <= 10:
                    self.config.DEFAULT_TOP_K = new_value
                    self.console.print(f"[green]예측 개수를 {new_value}개로 설정했습니다.[/green]")
                else:
                    self.console.print("[red]예측 개수는 1-10 사이여야 합니다.[/red]")

            elif option == 'temp':
                # /set temp 명령은 _set_temperature로 리다이렉트
                self._set_temperature(value)

            else:
                self.console.print(f"[red]알 수 없는 옵션: {option}[/red]")

        except ValueError:
            self.console.print(f"[red]잘못된 값: {value}[/red]")

    def batch_mode(self, texts: List[str]):
        """배치 모드 - 여러 텍스트 한번에 처리"""
        self.console.print(f"[cyan]{len(texts)}개 텍스트 처리 중...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("예측 중...", total=len(texts))

            results = []
            for text in texts:
                predictions = self.predictor.predict_next_tokens(
                    text,
                    top_k=self.config.DEFAULT_TOP_K,
                    temperature=self.config.DEFAULT_TEMPERATURE,
                    complete_word=True
                )
                results.append((text, predictions))
                progress.update(task, advance=1)

        # 결과 표시
        for text, predictions in results:
            self.console.print(f"\n[bold]입력:[/bold] {text}")
            self.display_predictions(predictions)