#!/usr/bin/env python3
"""
Korean Token Predictor - Main Entry Point
한국어 다음 토큰 예측기
"""
import click
import sys
import logging
import asyncio
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 모듈 import
from korean_predictor.models.model_manager import ModelManager
from korean_predictor.models.predictor import PredictionService
from korean_predictor.cache.cache import CacheService
from korean_predictor.utils.config import Config
from korean_predictor.cli import CLI


console = Console()


def setup_logging(log_level: str):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # transformers 라이브러리 로그 레벨 조정
    logging.getLogger('transformers').setLevel(logging.WARNING)


@click.command()
@click.option(
    '--model',
    '-m',
    default='kogpt2',
    help='사용할 모델 (kogpt2, polyglot, kcgpt2, dna-r1 또는 HuggingFace 모델 경로)'
)
@click.option(
    '--run-mode',
    default='auto',
    type=click.Choice(['auto', 'cpu', 'nvidia-gpu', 'radeon-gpu']),
    help='실행 모드: auto(자동감지), cpu, nvidia-gpu, radeon-gpu'
)
@click.option(
    '--list-models',
    is_flag=True,
    help='사용 가능한 모델 목록 표시'
)
@click.option(
    '--text',
    '-t',
    help='예측할 텍스트 (비대화형 모드)'
)
@click.option(
    '--top-k',
    '-k',
    default=10,
    type=int,
    help='예측할 단어 개수'
)
@click.option(
    '--temperature',
    '-temp',
    default=0.8,
    type=float,
    help='샘플링 온도 (0.1-2.0)'
)
@click.option(
    '--no-cache',
    is_flag=True,
    help='캐시 비활성화'
)
@click.option(
    '--complete-word',
    is_flag=True,
    default=True,
    help='완전한 어절 예측 (기본값: True)'
)
@click.option(
    '--include-special-tokens',
    is_flag=True,
    default=True,
    help='<eos> 등 특수 토큰 포함 (기본값: True)'
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='로그 레벨'
)
@click.option(
    '--file',
    '-f',
    type=click.Path(exists=True),
    help='텍스트 파일에서 입력 읽기 (배치 모드)'
)
def main(model, run_mode, list_models, text, top_k, temperature, no_cache, complete_word, include_special_tokens, log_level, file):
    """
    한국어 다음 토큰 예측기

    대화형 모드: python main.py

    비대화형 모드: python main.py -t "오늘 날씨가"

    배치 모드: python main.py -f input.txt

    모델 목록: python main.py --list-models

    실행 모드 지정: python main.py --run-mode cpu
    """
    # 모델 목록 표시
    if list_models:
        console.print("\n[bold cyan]사용 가능한 모델 목록:[/bold cyan]\n")
        from rich.table import Table

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("모델명", style="cyan", width=15)
        table.add_column("파라미터", style="green", width=10)
        table.add_column("설명", style="white", width=50)
        table.add_column("메모리", style="yellow", width=20)

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

        console.print(table)
        console.print("\n[dim]* 추론 특화 모델 (DeepSeek-R1 방식)[/dim]")
        console.print("\n[bold]사용 예시:[/bold]")
        console.print("  python main.py --model kogpt2")
        console.print("  python main.py --model dna-r1  [dim](GPU 필수)[/dim]")
        return

    # 로깅 설정
    setup_logging(log_level)

    # 설정 초기화
    Config.setup_directories()
    Config.DEFAULT_TOP_K = top_k
    Config.DEFAULT_TEMPERATURE = temperature
    Config.COMPLETE_WORD = complete_word
    Config.INCLUDE_SPECIAL_TOKENS = include_special_tokens

    if no_cache:
        Config.CACHE_ENABLED = False

    # ASCII 아트 배너
    console.print("""
[bold cyan]╔══════════════════════════════════════════╗
║  한국어 다음 토큰 예측기 v1.0.0          ║
║  Korean Token Predictor                  ║
╚══════════════════════════════════════════╝[/bold cyan]
    """)

    try:
        # 모델 매니저 초기화 및 run_mode 설정
        console.print(f"[yellow]모델 초기화 중... ({model}, 모드: {run_mode})[/yellow]")
        model_manager = ModelManager()

        # run_mode 설정 시도 (장치 오류 가능)
        try:
            model_manager.set_run_mode(run_mode)
        except RuntimeError as e:
            error_msg = str(e)
            if "GPU를 사용할 수 없습니다" in error_msg or "GPU가 감지되지" in error_msg:
                console.print(f"\n[yellow]⚠️  경고: {error_msg}[/yellow]")
                console.print("\n[bold]요청한 장치를 사용할 수 없습니다. 강제로 시도하면 오류가 발생할 수 있습니다.[/bold]")

                try:
                    response = input("\n강제로 시도하시겠습니까? (Y/N): ").strip().lower()
                    if response in ['y', 'yes']:
                        console.print("[cyan]강제 시도 중... (CPU 모드로 폴백)[/cyan]")
                        # CPU로 폴백
                        model_manager.set_run_mode('cpu')
                    else:
                        console.print("[yellow]실행을 취소했습니다.[/yellow]")
                        sys.exit(1)
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]실행을 취소했습니다.[/yellow]")
                    sys.exit(1)
            else:
                raise

        # 모델 로드 (진행 상태 표시)
        force_load = False
        while True:
            with console.status("[bold green]모델 로딩 중...[/bold green]", spinner="dots") as status:
                success, message = model_manager.load_model(model, force=force_load)

            if not success:
                # 호환성 문제인지 확인
                if "지원하지 않습니다" in message or "지원 모드:" in message:
                    console.print(f"\n[yellow]⚠️  경고: {message}[/yellow]")
                    console.print("\n[bold]이 조합은 공식적으로 지원되지 않으며, 오류가 발생할 수 있습니다.[/bold]")

                    # 사용자 확인 받기
                    try:
                        response = input("\n강제로 시도하시겠습니까? (Y/N): ").strip().lower()
                        if response in ['y', 'yes']:
                            console.print("[cyan]강제 로드 시도 중...[/cyan]")
                            force_load = True
                            continue  # 다시 시도
                        else:
                            console.print("[yellow]모델 로드를 취소했습니다.[/yellow]")
                            sys.exit(1)
                    except (KeyboardInterrupt, EOFError):
                        console.print("\n[yellow]모델 로드를 취소했습니다.[/yellow]")
                        sys.exit(1)
                else:
                    # 다른 종류의 오류
                    console.print(f"[red]모델 로드 실패: {message}[/red]")
                    sys.exit(1)
            else:
                break  # 성공

        console.print(f"[green]✓ {message}[/green]")

        # 캐시 서비스 초기화
        cache_service = None
        if Config.CACHE_ENABLED:
            console.print("[yellow]캐시 서비스 초기화 중...[/yellow]")
            cache_service = CacheService(
                cache_dir=Config.CACHE_DIR,
                size_limit=Config.CACHE_SIZE_MB * 1024 * 1024
            )
            console.print("[green]✓ 캐시 서비스 활성화[/green]")

        # 예측 서비스 초기화
        predictor = PredictionService(model_manager, cache_service)

        # CLI 초기화
        cli = CLI(predictor, Config)

        # 실행 모드 결정
        if file:
            # 파일에서 읽기 (배치 모드)
            console.print(f"\n[cyan]파일 읽는 중: {file}[/cyan]")
            with open(file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            cli.batch_mode(texts)

        elif text:
            # 단일 텍스트 예측 (비대화형)
            console.print(f"\n[bold]입력 텍스트:[/bold] {text}")

            # 예측 (항상 EOT 확률 포함)
            with console.status("[bold green]예측 중...[/bold green]") as status:
                eot_prob, predictions = predictor.predict_next_tokens(
                    text,
                    top_k=top_k,
                    temperature=temperature,
                    complete_word=complete_word,
                    include_special_tokens=include_special_tokens
                )
            cli.display_predictions(predictions, eot_prob=eot_prob)

        else:
            # 대화형 모드
            console.print("\n[green]대화형 모드로 시작합니다...[/green]\n")
            cli.interactive_mode()

    except KeyboardInterrupt:
        console.print("\n[yellow]프로그램이 중단되었습니다.[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[red]오류 발생: {e}[/red]")
        logging.exception("치명적 오류")
        sys.exit(1)

    finally:
        # 정리 작업
        if 'cache_service' in locals() and cache_service:
            cache_service.close()
        if 'model_manager' in locals():
            model_manager.unload_model()
        console.print("\n[cyan]프로그램을 종료합니다.[/cyan]")


if __name__ == "__main__":
    main()