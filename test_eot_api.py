#!/usr/bin/env python3
"""
EOT API 테스트 스크립트
EOT REST API 서버의 다양한 엔드포인트를 테스트합니다.
"""

import time
import requests
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# API 서버 설정
API_BASE_URL = "http://localhost:8177"

# 테스트할 텍스트 샘플
TEST_TEXTS = {
    "high_eot": [
        "그래 알았어",
        "내일 봐",
        "수고하셨습니다",
        "안녕히 계세요",
        "잘 자"
    ],
    "medium_eot": [
        "그런데 말이야",
        "어떻게 생각해",
        "혹시 가능할까",
        "그렇구나",
        "아 그래"
    ],
    "low_eot": [
        "안녕하세요",
        "오늘 날씨가",
        "저는 생각하기에",
        "그래서 제가",
        "왜냐하면"
    ]
}


def test_health_check():
    """헬스체크 테스트"""
    console.print("\n[bold blue]1. 헬스체크 테스트[/bold blue]")

    try:
        # 기본 헬스체크
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ 기본 헬스체크: {data['status']}")
        else:
            console.print(f"❌ 기본 헬스체크 실패: {response.status_code}")

        # 상세 헬스체크
        response = requests.get(f"{API_BASE_URL}/health/detailed")
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ 상세 헬스체크: {data['status']}")

            # 컴포넌트 상태 표시
            if 'components' in data:
                for comp_name, comp_data in data['components'].items():
                    status_emoji = "✅" if comp_data.get('status') == 'healthy' else "⚠️"
                    console.print(f"   {status_emoji} {comp_name}: {comp_data.get('status', 'unknown')}")
        else:
            console.print(f"❌ 상세 헬스체크 실패: {response.status_code}")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        return False

    return True


def test_single_prediction():
    """단일 텍스트 EOT 예측 테스트"""
    console.print("\n[bold blue]2. 단일 텍스트 EOT 예측 테스트[/bold blue]")

    test_cases = [
        ("그래 알았어", "high"),
        ("안녕하세요", "low"),
        ("어떻게 생각해", "medium")
    ]

    headers = {"Content-Type": "application/json"}

    for text, expected_level in test_cases:
        payload = {
            "text": text,
            "model": "polyglot",
            "top_k": 10,
            "temperature": 0.5
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/eot",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    eot_prob = data["data"]["eot_probability"]
                    assessment = data["data"]["eot_assessment"]

                    # 이모지로 시각화
                    emoji = "🔴" if assessment == "high" else "🟡" if assessment == "medium" else "🟢"

                    console.print(f"{emoji} '{text}' → EOT: {eot_prob:.1%} ({assessment})")

                    # 예상과 일치하는지 확인
                    if assessment == expected_level:
                        console.print(f"   ✅ 예상대로 {expected_level} 레벨")
                    else:
                        console.print(f"   ⚠️  예상: {expected_level}, 실제: {assessment}")

                    # 상위 3개 토큰 표시
                    if "predictions" in data["data"]:
                        console.print("   [dim]상위 예측 토큰:[/dim]")
                        for pred in data["data"]["predictions"][:3]:
                            token = pred["token"]
                            prob = pred["probability"]
                            is_eot = pred["is_eot"]
                            token_type = pred.get("type", "unknown")
                            eot_mark = "[red]●[/red]" if is_eot else "[green]○[/green]"
                            console.print(f"      {eot_mark} {repr(token)}: {prob:.2%} ({token_type})")
                else:
                    console.print(f"❌ 예측 실패: {data['error']['message']}")
            else:
                console.print(f"❌ HTTP 오류: {response.status_code}")

        except Exception as e:
            console.print(f"[red]오류: {e}[/red]")

        time.sleep(0.5)  # Rate limiting 방지


def test_batch_prediction():
    """배치 예측 테스트"""
    console.print("\n[bold blue]3. 배치 예측 테스트[/bold blue]")

    # 다양한 레벨의 텍스트 혼합
    batch_texts = (
        TEST_TEXTS["high_eot"][:2] +
        TEST_TEXTS["medium_eot"][:2] +
        TEST_TEXTS["low_eot"][:2]
    )

    payload = {
        "texts": batch_texts,
        "model": "polyglot",
        "top_k": 10,
        "temperature": 0.5
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                console.print(f"✅ 배치 처리 성공")
                console.print(f"   처리된 텍스트: {data['data']['total_count']}개")
                console.print(f"   성공: {data['data']['success_count']}개")
                console.print(f"   실패: {data['data']['failure_count']}개")
                console.print(f"   총 소요 시간: {data['data']['total_elapsed_time']:.2f}초")

                # 결과 테이블
                table = Table(title="배치 예측 결과")
                table.add_column("텍스트", style="cyan")
                table.add_column("EOT 확률", justify="right")
                table.add_column("평가", justify="center")
                table.add_column("상위 토큰")

                for result in data["data"]["results"]:
                    if result["success"]:
                        text = result["text"][:20] + "..." if len(result["text"]) > 20 else result["text"]
                        eot_prob = f"{result['eot_probability']:.1%}"
                        assessment = result["eot_assessment"]

                        # 평가에 따른 색상
                        if assessment == "high":
                            assessment_colored = "[red]HIGH[/red]"
                        elif assessment == "medium":
                            assessment_colored = "[yellow]MEDIUM[/yellow]"
                        else:
                            assessment_colored = "[green]LOW[/green]"

                        top_token = result.get("top_prediction", "N/A")

                        table.add_row(text, eot_prob, assessment_colored, top_token)

                console.print(table)
            else:
                console.print(f"❌ 배치 예측 실패: {data['error']['message']}")
        else:
            console.print(f"❌ HTTP 오류: {response.status_code}")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")


def test_context_prediction():
    """컨텍스트 기반 예측 테스트"""
    console.print("\n[bold blue]4. 컨텍스트 기반 예측 테스트[/bold blue]")

    contexts = [
        {
            "context": [
                "안녕하세요",
                "네 안녕하세요"
            ],
            "expected": "low"
        },
        {
            "context": [
                "오늘 날씨 좋네요",
                "네 정말 좋아요",
                "산책 가실래요",
                "좋아요 가시죠"
            ],
            "expected": "high"
        },
        {
            "context": [
                "내일 시간 있어?",
                "몇 시쯤?",
                "오후 3시는 어때"
            ],
            "expected": "medium"
        }
    ]

    headers = {"Content-Type": "application/json"}

    for test_case in contexts:
        payload = {
            "context": test_case["context"],
            "model": "polyglot",
            "top_k": 10,
            "temperature": 0.5
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/context",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    eot_prob = data["data"]["eot_probability"]
                    assessment = data["data"]["eot_assessment"]
                    recommendation = data["data"].get("recommendation", "")

                    console.print(f"\n[cyan]컨텍스트 (턴 수: {len(test_case['context'])})[/cyan]")
                    for i, turn in enumerate(test_case["context"], 1):
                        console.print(f"   {i}. {turn}")

                    # 이모지로 시각화
                    emoji = "🔴" if assessment == "high" else "🟡" if assessment == "medium" else "🟢"

                    console.print(f"{emoji} EOT: {eot_prob:.1%} ({assessment})")
                    console.print(f"   💬 {recommendation}")

                    # 예상과 비교
                    if assessment == test_case["expected"]:
                        console.print(f"   ✅ 예상대로 {test_case['expected']} 레벨")
                    else:
                        console.print(f"   ⚠️  예상: {test_case['expected']}, 실제: {assessment}")
                else:
                    console.print(f"❌ 컨텍스트 예측 실패: {data['error']['message']}")
            else:
                console.print(f"❌ HTTP 오류: {response.status_code}")

        except Exception as e:
            console.print(f"[red]오류: {e}[/red]")

        time.sleep(0.5)


def test_api_stats():
    """API 통계 테스트"""
    console.print("\n[bold blue]5. API 통계 테스트[/bold blue]")

    try:
        # 통계 조회
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            console.print("✅ API 통계:")
            console.print(f"   - API 버전: {data['api_version']}")
            console.print(f"   - EOT 토큰: {data['predictor']['eot_tokens']}개")
            console.print(f"   - 사용자 정의 토큰: {data['predictor']['user_eot_tokens']}개")
            console.print(f"   - 문장부호: {data['predictor']['punctuation_marks']}개")

            if 'cache' in data and data['cache']:
                console.print(f"   - 캐시 크기: {data['cache'].get('size_kb', 0):.1f} KB")
                console.print(f"   - 캐시 항목: {data['cache'].get('items', 0)}개")

            console.print("\n   Rate Limits:")
            for key, value in data['rate_limits'].items():
                console.print(f"   - {key}: {value} req/min")
        else:
            console.print(f"❌ 통계 조회 실패: {response.status_code}")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")


def test_model_info():
    """모델 정보 테스트"""
    console.print("\n[bold blue]6. 모델 정보 테스트[/bold blue]")

    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            console.print("✅ 사용 가능한 모델:")

            table = Table(title="모델 목록")
            table.add_column("ID", style="cyan")
            table.add_column("이름")
            table.add_column("설명")
            table.add_column("파라미터", justify="right")
            table.add_column("기본값", justify="center")

            for model in data["models"]:
                default_mark = "✓" if model.get("default", False) else ""
                table.add_row(
                    model["id"],
                    model["name"],
                    model["description"],
                    model["params"],
                    default_mark
                )

            console.print(table)
            console.print(f"\n   현재 사용 중: [bold]{data['current']}[/bold]")
        else:
            console.print(f"❌ 모델 정보 조회 실패: {response.status_code}")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")


def test_rate_limiting():
    """Rate Limiting 테스트"""
    console.print("\n[bold blue]7. Rate Limiting 테스트 (선택적)[/bold blue]")
    console.print("[dim]많은 요청을 보내 Rate Limiting을 테스트합니다...[/dim]")

    # 이 테스트는 선택적입니다 (실행 시간이 길 수 있음)
    return True


def main():
    """메인 테스트 실행"""
    console.print("[bold cyan]=" * 60)
    console.print("[bold cyan]EOT API 테스트 시작[/bold cyan]")
    console.print("[bold cyan]=" * 60)

    # API 서버 연결 확인
    console.print(f"\n[yellow]API 서버 연결 확인: {API_BASE_URL}[/yellow]")

    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ 서버 연결 성공: {data['name']} v{data['version']}")
        else:
            console.print(f"❌ 서버 응답 오류: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        console.print("[red]❌ 서버에 연결할 수 없습니다. API 서버가 실행 중인지 확인하세요.[/red]")
        console.print("[dim]서버 시작: python eot_api.py[/dim]")
        return
    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        return

    # 각 테스트 실행
    tests = [
        ("헬스체크", test_health_check),
        ("단일 예측", test_single_prediction),
        ("배치 예측", test_batch_prediction),
        ("컨텍스트 예측", test_context_prediction),
        ("API 통계", test_api_stats),
        ("모델 정보", test_model_info),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            console.print(f"[red]테스트 '{test_name}' 실행 중 오류: {e}[/red]")
            failed += 1

    # 결과 요약
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]테스트 결과 요약[/bold cyan]")
    console.print("[bold cyan]=" * 60)
    console.print(f"✅ 성공: {passed}개")
    console.print(f"❌ 실패: {failed}개")

    if failed == 0:
        console.print("\n[bold green]🎉 모든 테스트가 성공적으로 완료되었습니다![/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠️  {failed}개의 테스트가 실패했습니다.[/bold yellow]")


if __name__ == "__main__":
    main()