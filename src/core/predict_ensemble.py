"""
NEXUS 앙상블 예측 코어
==========================

백테스트에서 81.8% (전략A), 88.9% (전략B) 정확도를 달성한 두 앙상블 전략을
실시간 GBPKRW(또는 다른 통화) T+3 예측에 적용.

5개 "버추얼 실험"을 다른 조건으로 LLM 호출:
  V1: top-15 뉴스, no-ctx, Groq vote=3
  V2: top-30 뉴스, no-ctx, Groq vote=3 (more news breadth)
  V3: top-15 뉴스, WITH market context (DXY/VIX 등)
  V4: top-45 뉴스 (max breadth), no-ctx
  V5: top-10 뉴스 (가장 관련성 높음), Gemini single call

5개 결과의 pred_pct 를 모아 두 전략 적용:
  · 전략 A: 5개 평균 → |avg|≥0.4% 필터 → up/down (백테스트 N=11 시 81.8%)
  · 전략 B: K=3 콘센서스 + |pred|≥0.3% 필터 → up/down (N=9 시 88.9%)

사용:
  python predict_ensemble.py                 # GBPKRW T+3 예측 (기본)
  python predict_ensemble.py USDKRW=X 1      # USD/KRW T+1
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 단독 실행(예: backtest)에서도 동작하도록 프로젝트 루트의 .env 로드
# 파일 위치: src/core/predict_ensemble.py → parents[2] = 프로젝트 루트
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# src/ 를 sys.path 에 추가 (직접 실행 시 core/data 패키지 import 가능)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.analyzer import RateAnalyzer, SYSTEM_PROMPT
from core.market_context import fetch_context_at, format_for_prompt
from data.news_multisource import collect_all
from core.relevance_filter import filter_relevant


# ============================================================
# 통화/horizon 기본
# ============================================================
DEFAULT_TICKER  = "GBPKRW=X"   # 백테스트 최고 통화
DEFAULT_HORIZON = 3            # 백테스트 최고 horizon (T+3)

TICKER_LABELS = {
    "USDKRW=X": "USD/KRW (미국 달러)",
    "JPYKRW=X": "JPY/KRW (일본 엔)",
    "EURKRW=X": "EUR/KRW (유로)",
    "GBPKRW=X": "GBP/KRW (영국 파운드)",
    "CNYKRW=X": "CNY/KRW (중국 위안)",
}


# ============================================================
# 5개 버추얼 실험 구성
# ============================================================
EXPERIMENTS = [
    {"id": "V1", "label": "top-15, no-ctx, vote=3",       "top_k": 15, "use_ctx": False, "vote": 3, "method": "groq"},
    {"id": "V2", "label": "top-30, no-ctx, vote=3",       "top_k": 30, "use_ctx": False, "vote": 3, "method": "groq"},
    {"id": "V3", "label": "top-15, WITH market-ctx",      "top_k": 15, "use_ctx": True,  "vote": 3, "method": "groq"},
    {"id": "V4", "label": "top-45, max breadth, vote=3",  "top_k": 45, "use_ctx": False, "vote": 3, "method": "groq"},
    {"id": "V5", "label": "top-10 most-relevant, Gemini", "top_k": 10, "use_ctx": False, "vote": 1, "method": "gemini"},
]


# ============================================================
# Groq vote (백테스트 vote_predict 와 동일 로직, 응축판)
# ============================================================
def _groq_vote(news_list: List[Dict], market_ctx: Optional[Dict],
               vote: int = 3, throttle: float = 4.0) -> Optional[Dict]:
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
    except ImportError:
        return None
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    parts = []
    for i, n in enumerate(news_list):
        parts.append(f"[{i+1}|w={n.get('time_weight',1.0):.2f}] "
                     f"{n.get('title','')} {n.get('text','')}")
    combined = "\n".join(parts)
    if market_ctx:
        combined = format_for_prompt(market_ctx) + "\n\n=== NEWS ===\n" + combined

    results = []
    for i in range(vote):
        temp = 0.3 + i * 0.2
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Analyze this:\n\n{combined[:3000]}"},
                ],
                temperature=temp,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            print(f"      [Groq vote {i+1}/{vote} fail: {type(e).__name__}]")
        if i < vote - 1:
            time.sleep(throttle)

    if not results:
        return None

    tickers_all = list(TICKER_LABELS.keys())
    rate_preds: Dict[str, Dict] = {}
    for tk in tickers_all:
        preds = []
        for r in results:
            ri = r.get("rate_impacts", {}).get(tk, {})
            d  = ri.get("direction", "flat")
            m  = float(ri.get("magnitude", 0))
            pct = m if d == "up" else (-m if d == "down" else 0)
            preds.append(pct)
        avg = sum(preds) / len(preds) if preds else 0
        rate_preds[tk] = {
            "change_pct": avg,
            "confidence": 75,
            "direction": "up" if avg > 0.1 else ("down" if avg < -0.1 else "flat"),
        }
    return {"rate_preds": rate_preds, "method": f"Groq-8b vote={len(results)}"}


# ============================================================
# Gemini 단일 호출 (analyzer 재사용)
# ============================================================
_analyzer_singleton: Optional[RateAnalyzer] = None

def _get_analyzer() -> RateAnalyzer:
    global _analyzer_singleton
    if _analyzer_singleton is None:
        _analyzer_singleton = RateAnalyzer()
    return _analyzer_singleton


def _gemini_single(news_list: List[Dict], market_ctx: Optional[Dict]) -> Optional[Dict]:
    a = _get_analyzer()
    if not a._llm_ok:
        return None
    try:
        return a.analyze("", "", lang="en", news_list=news_list, market_context=market_ctx)
    except Exception as e:
        print(f"      [Gemini fail: {type(e).__name__}]")
        return None


# ============================================================
# 5개 실험 실행
# ============================================================
def run_experiments(articles: List[Dict], market_ctx: Optional[Dict],
                    verbose: bool = True) -> List[Dict]:
    """5개 실험 각각의 결과 (pred_pct per ticker) 반환"""
    results = []
    for cfg in EXPERIMENTS:
        if verbose:
            print(f"\n  [{cfg['id']}] {cfg['label']}")
        # 관련성 top-K
        filtered = filter_relevant(articles, top_k=cfg["top_k"])
        if verbose:
            print(f"      입력 뉴스: {len(filtered)}건")
        # 컨텍스트
        ctx = market_ctx if cfg["use_ctx"] else None
        # LLM 호출
        if cfg["method"] == "groq":
            res = _groq_vote(filtered, ctx, vote=cfg["vote"])
            if res is None:
                if verbose:
                    print(f"      Groq 실패 → Gemini 폴백")
                res = _gemini_single(filtered, ctx)
        else:
            res = _gemini_single(filtered, ctx)
        if res is None:
            if verbose:
                print(f"      [SKIP] LLM 응답 없음")
            continue
        method = res.get("method", "unknown")
        rate_preds = res.get("rate_preds", {})
        if verbose:
            print(f"      method: {method}")
            for tk in ["USDKRW=X", "JPYKRW=X", "EURKRW=X", "GBPKRW=X"]:
                p = rate_preds.get(tk, {}).get("change_pct", 0)
                arrow = "↑" if p > 0.1 else ("↓" if p < -0.1 else "→")
                print(f"        {tk}: {p:+.2f}% {arrow}")
        results.append({"id": cfg["id"], "label": cfg["label"],
                        "method": method, "rate_preds": rate_preds})
    return results


# ============================================================
# 두 앙상블 전략 적용
# ============================================================
def apply_ensemble(exp_results: List[Dict], ticker: str = DEFAULT_TICKER) -> Dict:
    """
    5개 실험의 pred_pct → 두 전략으로 최종 up/down 결정.
    """
    preds = []
    for r in exp_results:
        ri = r.get("rate_preds", {}).get(ticker, {})
        if "change_pct" not in ri:
            continue
        preds.append({"id": r["id"], "pred_pct": float(ri["change_pct"])})

    if not preds:
        return {"error": "no predictions", "ticker": ticker}

    # 전략 A: 평균
    avg = sum(p["pred_pct"] for p in preds) / len(preds)
    strat_a_dir = "up" if avg >= 0 else "down"
    strat_a_triggered = abs(avg) >= 0.4
    strat_a = {
        "name":      "전략 A — 5개 실험 평균 + |avg|≥0.4%",
        "avg_pred":  round(avg, 3),
        "direction": strat_a_dir,
        "triggered": strat_a_triggered,
        "confidence_label": "강함 (백테스트 81.8%)" if strat_a_triggered else "약함 (신호 부족 → 관망 권장)",
    }

    # 전략 B: 콘센서스 K=3 + |pred|≥0.3%
    strong = [p for p in preds if abs(p["pred_pct"]) >= 0.3]
    up_n   = sum(1 for p in strong if p["pred_pct"] > 0)
    down_n = sum(1 for p in strong if p["pred_pct"] < 0)
    if max(up_n, down_n) >= 3:
        strat_b_dir = "up" if up_n >= 3 else "down"
        strat_b_triggered = True
        conf_label = "강함 (백테스트 88.9%)"
    else:
        # 약한 신호: 단순 평균 방향 사용 (참고용)
        strat_b_dir = "up" if avg >= 0 else "down"
        strat_b_triggered = False
        conf_label = "약함 (콘센서스 K<3 → 관망 권장)"
    strat_b = {
        "name":      "전략 B — 콘센서스 K=3 + |pred|≥0.3%",
        "strong_signals": {"up": up_n, "down": down_n, "total": len(strong)},
        "direction": strat_b_dir,
        "triggered": strat_b_triggered,
        "confidence_label": conf_label,
    }

    # 합의: 두 전략이 모두 triggered이면서 같은 방향이면 "확정"
    consensus = None
    if strat_a_triggered and strat_b_triggered and strat_a_dir == strat_b_dir:
        consensus = {"direction": strat_a_dir, "label": "✅ 두 전략 모두 일치 (강한 신호)"}
    elif strat_a_dir == strat_b_dir:
        consensus = {"direction": strat_a_dir, "label": "⚠️ 방향은 일치하나 신호 약함"}
    else:
        consensus = {"direction": None, "label": "❌ 두 전략이 엇갈림 → 관망 권장"}

    return {
        "ticker":    ticker,
        "per_exp":   preds,
        "strategy_a": strat_a,
        "strategy_b": strat_b,
        "consensus": consensus,
    }


# ============================================================
# 메인 파이프라인
# ============================================================
def predict(ticker: str = DEFAULT_TICKER, horizon: int = DEFAULT_HORIZON,
            hours_back: int = 24, verbose: bool = True) -> Dict:
    """전체 예측 파이프라인"""
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print(f"  NEXUS 앙상블 예측 — {TICKER_LABELS.get(ticker, ticker)} T+{horizon}")
        print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    # 1) 뉴스 수집
    if verbose:
        print(f"\n[1] 뉴스 수집 (최근 {hours_back}시간, 멀티 소스)")
    news = collect_all(hours_back=hours_back, verbose=verbose)
    articles = news["articles"]
    if verbose:
        print(f"      총 수집: {news['total']}건 (소스 {len(news['by_source'])}개)")

    # 2) 시장 컨텍스트
    if verbose:
        print(f"\n[2] 시장 컨텍스트 (DXY, VIX 등)")
    try:
        market_ctx = fetch_context_at(datetime.now(timezone.utc), lookback_days=5)
        if verbose:
            print(f"      수집: {len(market_ctx)}/7 signals")
    except Exception as e:
        market_ctx = None
        if verbose:
            print(f"      [WARN] {e}")

    # 3) 5개 실험 실행
    if verbose:
        print(f"\n[3] 5개 버추얼 실험 실행")
    exp_results = run_experiments(articles, market_ctx, verbose=verbose)
    if verbose:
        print(f"\n      실험 성공: {len(exp_results)}/5")

    # 4) 앙상블 전략 적용
    if verbose:
        print(f"\n[4] 두 전략 앙상블 ({ticker})")
    ensemble = apply_ensemble(exp_results, ticker=ticker)

    dt = time.time() - t0
    result = {
        "ticker":       ticker,
        "horizon":      horizon,
        "ticker_label": TICKER_LABELS.get(ticker, ticker),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":  round(dt, 1),
        "news":         {"total": news["total"], "by_source": news["by_source"]},
        "experiments":  exp_results,
        "ensemble":     ensemble,
    }

    if verbose:
        print()
        print("=" * 70)
        print("  📊 최종 결과")
        print("=" * 70)
        print(f"  통화: {result['ticker_label']}  |  Horizon: T+{horizon}일")
        print()
        print(f"  {'실험':<5} {'결과 (pred_pct)':<15}")
        print(f"  {'-'*5} {'-'*15}")
        for p in ensemble["per_exp"]:
            arrow = "↑" if p["pred_pct"] > 0 else ("↓" if p["pred_pct"] < 0 else "→")
            print(f"  {p['id']:<5} {p['pred_pct']:+.3f}% {arrow}")
        print()
        sa = ensemble["strategy_a"]
        sb = ensemble["strategy_b"]
        print(f"  ⭐ {sa['name']}")
        print(f"      평균 예측: {sa['avg_pred']:+.3f}%  |  방향: {sa['direction'].upper()}")
        print(f"      신호: {sa['confidence_label']}")
        print()
        print(f"  ⭐ {sb['name']}")
        ss = sb["strong_signals"]
        print(f"      강신호: {ss['total']}개 (up={ss['up']}, down={ss['down']})  |  방향: {sb['direction'].upper()}")
        print(f"      신호: {sb['confidence_label']}")
        print()
        c = ensemble["consensus"]
        print(f"  🎯 최종 합의: {c['label']}")
        if c["direction"]:
            up_text = "원화 약세 (KRW↓, 외화↑)" if c["direction"] == "up" else "원화 강세 (KRW↑, 외화↓)"
            print(f"      → 예측 방향: {c['direction'].upper()}  ({up_text})")
        print()
        print(f"  ⏱  실행 시간: {dt:.1f}s")
        print("=" * 70)

    return result


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    h  = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HORIZON
    if tk not in TICKER_LABELS:
        print(f"[ERROR] 지원 통화: {list(TICKER_LABELS.keys())}")
        sys.exit(1)
    predict(ticker=tk, horizon=h)
