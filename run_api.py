#!/usr/bin/env python
"""
Korean Predictor API Server Runner

사용법:
    python run_api.py --model kogpt2 --run-mode cpu --port 8000

환경 변수:
    RUN_MODE: auto|cpu|nvidia-gpu|radeon-gpu
    MODEL_NAME: kogpt2|kanana|polyglot-ko-5.8b|dna-r1
    API_KEYS: comma-separated API keys
    ENVIRONMENT: development|production
"""
import os
import sys
import click
import uvicorn


@click.command()
@click.option('--model', '-m', default='kogpt2', help='모델 이름 (kogpt2, kanana, polyglot-ko-5.8b, dna-r1)')
@click.option('--run-mode', '-r', default='auto', help='실행 모드 (auto, cpu, nvidia-gpu, radeon-gpu)')
@click.option('--host', default='0.0.0.0', help='호스트 주소')
@click.option('--port', '-p', default=8000, type=int, help='포트 번호')
@click.option('--reload', is_flag=True, help='코드 변경 시 자동 재시작 (개발 모드)')
@click.option('--workers', '-w', default=1, type=int, help='워커 프로세스 수')
@click.option('--log-level', default='info', help='로그 레벨 (debug, info, warning, error)')
def main(model, run_mode, host, port, reload, workers, log_level):
    """Korean Predictor REST API 서버 실행"""
    # 환경 변수 설정
    os.environ['MODEL_NAME'] = model
    os.environ['RUN_MODE'] = run_mode

    # API 키 설정 (개발 모드)
    if 'API_KEYS' not in os.environ:
        os.environ['API_KEYS'] = 'kp_test_development_key_12345'
        click.echo('[개발 모드] 테스트 API 키 사용: kp_test_development_key_12345')

    # 환경 설정
    if 'ENVIRONMENT' not in os.environ:
        os.environ['ENVIRONMENT'] = 'development' if reload else 'production'

    click.echo('=' * 60)
    click.echo('Korean Predictor REST API Server')
    click.echo('=' * 60)
    click.echo(f'Model: {model}')
    click.echo(f'Run Mode: {run_mode}')
    click.echo(f'Host: {host}:{port}')
    click.echo(f'Workers: {workers}')
    click.echo(f'Reload: {reload}')
    click.echo(f'Environment: {os.environ.get("ENVIRONMENT")}')
    click.echo('=' * 60)
    click.echo()
    click.echo('서버를 시작합니다...')
    click.echo(f'API 문서: http://{host}:{port}/docs')
    click.echo(f'Health Check: http://{host}:{port}/v1/health')
    click.echo()

    # Uvicorn 서버 실행
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # reload 모드에서는 workers=1만 지원
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        click.echo('\n서버를 종료합니다.')
    except Exception as e:
        click.echo(f'\n오류 발생: {str(e)}', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
