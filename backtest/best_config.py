"""
백테스트 결과 기반 통화·horizon별 최적 설정.

각 통화-horizon 조합마다 (V3 news-only vs V4 with-context) 중 더 잘 나온 쪽 선택.
"""

# (ticker, horizon) → {"use_context": bool, "accuracy": float, "n": int}
BEST_CONFIG = {
    # USDKRW: news only가 일관되게 우위
    ("USDKRW=X", 1): {"use_context": False, "accuracy": 30.8, "n": 13, "source": "V3"},
    ("USDKRW=X", 3): {"use_context": False, "accuracy": 38.5, "n": 13, "source": "V3"},
    ("USDKRW=X", 7): {"use_context": False, "accuracy": 38.5, "n": 13, "source": "V3"},

    # JPYKRW: context이 도움 (safe haven 신호와 상관)
    ("JPYKRW=X", 1): {"use_context": True,  "accuracy": 46.2, "n": 13, "source": "V4"},
    ("JPYKRW=X", 3): {"use_context": True,  "accuracy": 38.5, "n": 13, "source": "V4"},
    ("JPYKRW=X", 7): {"use_context": True,  "accuracy": 38.5, "n": 13, "source": "V4"},

    # GBPKRW: news only가 분명히 우위
    ("GBPKRW=X", 1): {"use_context": False, "accuracy": 53.8, "n": 13, "source": "V3"},
    ("GBPKRW=X", 3): {"use_context": False, "accuracy": 46.2, "n": 13, "source": "V3"},
    ("GBPKRW=X", 7): {"use_context": False, "accuracy": 38.5, "n": 13, "source": "V3"},

    # EURKRW: 둘 다 약함, 약간 context 우위
    ("EURKRW=X", 1): {"use_context": False, "accuracy":  7.7, "n": 13, "source": "tie"},
    ("EURKRW=X", 3): {"use_context": True,  "accuracy": 30.8, "n": 13, "source": "V4"},
    ("EURKRW=X", 7): {"use_context": True,  "accuracy": 23.1, "n": 13, "source": "V4"},

    # CNYKRW: 데이터 부족, 기본 news only
    ("CNYKRW=X", 1): {"use_context": False, "accuracy": None, "n": 0, "source": "unknown"},
    ("CNYKRW=X", 3): {"use_context": False, "accuracy": None, "n": 0, "source": "unknown"},
    ("CNYKRW=X", 7): {"use_context": False, "accuracy": None, "n": 0, "source": "unknown"},
}


def get_best(ticker: str, horizon: int) -> dict:
    """주어진 (ticker, horizon)의 최적 설정 반환"""
    return BEST_CONFIG.get((ticker, horizon), {"use_context": False, "accuracy": None, "n": 0})


def summary_table():
    """베스트 설정 요약 출력"""
    print("=" * 70)
    print("  통화별 최적 설정 (N=13 표본 기준, 신뢰구간 ±18%p)")
    print("=" * 70)
    print(f"  {'Ticker':<10} {'H':>3}  {'Context':<8} {'Acc':>7}  {'Source':<8}")
    print(f"  {'-'*10} {'-'*3}  {'-'*8} {'-'*7}  {'-'*8}")
    for (tk, h), cfg in BEST_CONFIG.items():
        ctx = "Yes" if cfg["use_context"] else "No"
        acc = f"{cfg['accuracy']:.1f}%" if cfg["accuracy"] else "—"
        print(f"  {tk:<10} T+{h:<2} {ctx:<8} {acc:>7}  {cfg['source']:<8}")
    # 베스트 단일 조합
    valid = [(k, v) for k, v in BEST_CONFIG.items() if v["accuracy"]]
    best = max(valid, key=lambda x: x[1]["accuracy"])
    print(f"\n  🏆 베스트 단일: {best[0][0]} T+{best[0][1]} → {best[1]['accuracy']}% (use_context={best[1]['use_context']})")


if __name__ == "__main__":
    summary_table()
