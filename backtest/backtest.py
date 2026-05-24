"""
NEXUS — 환율 예측 백테스트 프레임워크
=========================================

각 과거 시점 T에 대해:
  1. [T - news_window일, T] 구간의 뉴스를 GDELT에서 수집
  2. LLM(analyzer.RateAnalyzer)으로 5개 통화쌍 방향·강도 예측
  3. yfinance로 T~T+forward일 실제 환율 변동 가져옴
  4. 예측 vs 실제 비교 → 방향 정확도, MAE, confusion matrix

사용:
  python backtest.py --start 2026-01-01 --end 2026-05-01 --interval 7d \\
                     --news-window 3 --forward 1 --ticker USDKRW=X
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# 프로젝트 루트 경로 산출 후 src/ 를 sys.path 에 추가 (core/data 패키지 import)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
load_dotenv(os.path.join(_ROOT, ".env"))

from core.analyzer import RateAnalyzer, SYSTEM_PROMPT  # noqa: E402
from data.news_collector import collect_news_range  # noqa: E402
from core.market_context import fetch_context_at, format_for_prompt  # noqa: E402


def vote_predict(news_list, market_ctx, vote, throttle=2.5):
    """
    Self-consistency: Groq 8B를 N회 호출 (다른 temperature) 후
    각 ticker의 change_pct를 평균. analyzer 캐시 우회.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return None
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # 뉴스 + context 결합 (analyzer 로직 모방)
    parts = []
    for i, n in enumerate(news_list):
        parts.append(f"[{i+1}|w={n.get('weight',1.0):.2f}] {n.get('title','')} {n.get('text','')}")
    combined = "\n".join(parts)
    if market_ctx:
        combined = format_for_prompt(market_ctx) + "\n\n=== NEWS ===\n" + combined

    results = []
    for i in range(vote):
        temp = 0.3 + i * 0.2  # 0.3, 0.5, 0.7
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Analyze this:\n\n{combined[:3000]}"},
                ],
                temperature=temp,
                max_tokens=800,
                response_format={"type":"json_object"},
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            print(f"      vote {i+1}/{vote} fail: {type(e).__name__}")
        time.sleep(throttle)

    if not results:
        return None
    # Aggregate per ticker
    tickers_all = ["USDKRW=X","JPYKRW=X","CNYKRW=X","EURKRW=X","GBPKRW=X"]
    rate_preds = {}
    for tk in tickers_all:
        preds = []
        for r in results:
            ri = r.get("rate_impacts", {}).get(tk, {})
            d = ri.get("direction", "flat")
            m = float(ri.get("magnitude", 0))
            pct = m if d == "up" else (-m if d == "down" else 0)
            preds.append(pct)
        avg = sum(preds) / len(preds) if preds else 0
        rate_preds[tk] = {"change_pct": avg, "confidence": 75,
                          "direction": "up" if avg > 0.1 else ("down" if avg < -0.1 else "flat")}
    return {"rate_preds": rate_preds, "method": f"Groq-8b vote={vote}"}


TICKERS = {
    "USDKRW=X": "USD/KRW",
    "JPYKRW=X": "JPY/KRW",
    "CNYKRW=X": "CNY/KRW",
    "EURKRW=X": "EUR/KRW",
    "GBPKRW=X": "GBP/KRW",
}


def _yf():
    import yfinance as yf
    return yf


def fetch_history(ticker: str, start: datetime, end: datetime) -> Dict[str, float]:
    """yfinance에서 일별 종가 가져옴 → {date_str: close_price}"""
    yf = _yf()
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=2)).strftime("%Y-%m-%d"),
        interval="1d",
    )
    closes = {}
    for idx, row in df.iterrows():
        d = idx.strftime("%Y-%m-%d")
        closes[d] = float(row["Close"])
    return closes


def nearest_trading_date(closes: Dict[str, float], target: datetime) -> Optional[str]:
    """target 이후 가장 가까운 거래일 찾기 (휴장 대응)"""
    target_str = target.strftime("%Y-%m-%d")
    sorted_dates = sorted(closes.keys())
    for d in sorted_dates:
        if d >= target_str:
            return d
    return None


def direction_of(pct: float, threshold: float = 0.1) -> str:
    """% 변동 → up/down/flat (역치는 데이터 노이즈 흡수용)"""
    if pct > threshold:
        return "up"
    if pct < -threshold:
        return "down"
    return "flat"


def daterange(start: datetime, end: datetime, days: int):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=days)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n  python backtest.py --start 2026-01-01 --end 2026-05-01 \\\n    --interval 7d --news-window 3 --forward 1")
    ap.add_argument("--start", required=True, help="시작 날짜 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="끝 날짜 YYYY-MM-DD")
    ap.add_argument("--interval", default="7d",
                    help="테스트 날짜 간격. 예: 7d, 14d (기본 7일)")
    ap.add_argument("--news-window", type=int, default=3,
                    help="각 테스트 날짜의 N일 전부터 뉴스 수집 (기본 3)")
    ap.add_argument("--forward", type=int, default=1,
                    help="(legacy) 단일 horizon. --forwards가 우선")
    ap.add_argument("--forwards", default="1,3,7",
                    help="comma-sep horizons (예: 1,3,7). 다중 horizon F1 측정")
    ap.add_argument("--ticker", default="ALL",
                    choices=list(TICKERS.keys()) + ["ALL"],
                    help="대상 통화쌍 (기본 ALL = 5종 전부)")
    ap.add_argument("--keyword",
                    default='"South Korea" (won OR KRW OR currency OR economy OR exchange)',
                    help='GDELT query. AND가 기본이므로 OR로 명시. 예: \'"South Korea" (won OR KRW)\'')
    ap.add_argument("--max-news", type=int, default=20,
                    help="각 시점 최대 뉴스 수 (LLM 비용 영향, 기본 20)")
    ap.add_argument("--throttle", type=float, default=60.0,
                    help="시점 간 대기 시간(초) — GDELT 안정성 위해 60+ 권장")
    ap.add_argument("--out", default="backtest_results.csv")
    ap.add_argument("--resume", action="store_true",
                    help="기존 CSV에 있는 시점은 스킵 (이어 돌리기)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="CSV에서 direction_match가 N/A인 시점(GDELT 실패 등)만 재시도")
    ap.add_argument("--no-context", action="store_true",
                    help="cross-asset market context 사용 안 함 (news only baseline)")
    ap.add_argument("--vote", type=int, default=1,
                    help="self-consistency: N>=2면 각 시점 LLM을 N회 호출 후 결과 평균")
    ap.add_argument("--relevance-filter", type=int, default=0,
                    help="뉴스 관련성 pre-filter: Groq로 상위 K개만 선별 (0=비활성)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start >= end:
        print("[ERROR] start < end 이어야 함"); return

    # interval 파싱 (Nd 형식)
    if not args.interval.endswith("d"):
        print("[ERROR] interval은 Nd 형식 (예: 7d)"); return
    interval_days = int(args.interval[:-1])

    # forwards 파싱 (1,3,7 형태)
    try:
        forwards = sorted(set(int(x) for x in args.forwards.split(",") if x.strip()))
    except ValueError:
        print("[ERROR] --forwards는 콤마구분 정수 (예: 1,3,7)"); return
    max_fwd = max(forwards)

    print("=" * 70)
    print("  NEXUS — 환율 예측 백테스트 (multi-horizon × multi-ticker)")
    print("=" * 70)
    print(f"  기간       : {args.start} ~ {args.end}")
    print(f"  테스트 간격: {interval_days}일")
    print(f"  뉴스 윈도우: T-{args.news_window}일 ~ T")
    print(f"  예측 horizons: T+{forwards}")
    print(f"  대상 통화쌍 : {args.ticker}")

    # 분석기 로드
    a = RateAnalyzer()
    if not a._llm_ok:
        print("[ERROR] LLM 활성 안 됨"); return

    # 환율 데이터 미리 받기 (yfinance 한 번)
    print("\n[1] 과거 환율 데이터 수집 중...")
    if args.ticker == "ALL":
        tickers = list(TICKERS.keys())
    else:
        tickers = [args.ticker]
    rate_history = {}
    for tk in tickers:
        rate_history[tk] = fetch_history(tk,
            start - timedelta(days=2),
            end + timedelta(days=max_fwd + 5))
        print(f"   {tk}: {len(rate_history[tk])}거래일")

    # 테스트 날짜 생성
    test_dates = list(daterange(start, end, interval_days))

    # 기존 CSV에서 이미 처리된 (date, ticker) 추적
    skip_pairs = set()           # resume 시 스킵할 (date, ticker)
    failed_dates = set()         # retry-failed 시 재시도할 date
    existing_rows = []           # 그대로 유지할 rows
    if (args.resume or args.retry_failed) and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
                d = r["test_date"]; tk = r["ticker"]
                if r.get("direction_match") in ("0","1"):
                    if args.resume:
                        skip_pairs.add((d, tk))
                elif args.retry_failed:
                    failed_dates.add(d)
        print(f"\n[resume] 기존 {len(existing_rows)}개 row 발견, "
              f"성공 {len(skip_pairs)}개 (스킵), 실패 {len(failed_dates)}개 (재시도)")

    if args.retry_failed:
        # 실패 날짜 + 한 번도 안 한 날짜만 진행
        done_dates = {r["test_date"] for r in existing_rows
                      if r.get("direction_match") in ("0","1")}
        test_dates = [d for d in test_dates
                      if d.strftime("%Y-%m-%d") in failed_dates
                      or d.strftime("%Y-%m-%d") not in done_dates]

    print(f"\n[2] 테스트 시점 {len(test_dates)}개 — 백테스트 시작\n")

    # 새 long-format CSV — 한 row = (date, ticker, horizon)
    # 스키마 변경되어 기존 CSV는 사용 못 함. resume 시 (date, ticker) 단위 스킵
    CSV_COLS = ["test_date","ticker","horizon","news_count",
                "rate_at_T","rate_at_T_plus","actual_pct","actual_dir",
                "pred_pct","pred_dir","pred_conf","direction_match",
                "llm_method"]
    if not os.path.exists(args.out) or not (args.resume or args.retry_failed):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLS)

    results = []
    for i, td in enumerate(test_dates):
        td_str = td.strftime("%Y-%m-%d")
        # 모든 ticker가 이미 처리된 시점은 스킵
        if args.resume and all((td_str, tk) in skip_pairs for tk in tickers):
            print(f"  [{i+1}/{len(test_dates)}] {td_str}  [resume: 모든 ticker 완료, 스킵]")
            continue
        print(f"  [{i+1}/{len(test_dates)}] {td_str} 시점")

        # 뉴스 수집
        news_start = td - timedelta(days=args.news_window)
        news_data = collect_news_range(news_start, td, keyword=args.keyword,
                                        max_total=args.max_news)
        n_articles = news_data["total"]
        if n_articles == 0:
            print(f"      뉴스 0건 → 스킵")
            time.sleep(args.throttle)
            continue
        print(f"      뉴스 {n_articles}건 수집")

        # 시장 context (cross-asset 신호) 수집
        market_ctx = None
        if not args.no_context:
            try:
                market_ctx = fetch_context_at(td, lookback_days=5)
                print(f"      market context: {len(market_ctx)} signals")
            except Exception as e:
                print(f"      [WARN] context 수집 실패: {e}")

        # LLM 분석
        news_list = [
            {"title": x["title"], "text": x["text"], "weight": x.get("normalized_weight", 1.0)}
            for x in news_data["articles"]
        ]

        # 관련성 pre-filter (선택적)
        if args.relevance_filter > 0:
            try:
                from core.relevance_filter import filter_relevant
                before = len(news_list)
                news_list = filter_relevant(news_list, top_k=args.relevance_filter)
                print(f"      relevance filter: {before} → {len(news_list)}")
            except Exception as e:
                print(f"      [WARN] relevance filter 실패: {e}")

        t0 = time.time()
        try:
            if args.vote > 1:
                # Self-consistency 모드: Groq 8b를 N회, 평균
                # inter-vote throttle: max(2.5, args.throttle/vote) → TPM 예산 분산
                vote_throttle = max(2.5, args.throttle / max(args.vote, 1))
                result = vote_predict(news_list, market_ctx, vote=args.vote, throttle=vote_throttle)
                if result is None:
                    print(f"      vote 모드 실패 → 단일 호출로 폴백")
                    result = a.analyze("", "", lang="en", news_list=news_list,
                                      market_context=market_ctx)
            else:
                result = a.analyze("", "", lang="en", news_list=news_list,
                                  market_context=market_ctx)
            llm_method = result.get("method", "")
        except Exception as e:
            print(f"      LLM 실패: {e}")
            time.sleep(args.throttle)
            continue
        dt_llm = time.time() - t0
        print(f"      LLM 분석 완료 ({dt_llm:.1f}s, method={llm_method[:40]})")

        # 각 ticker × 각 horizon 조합
        for tk in tickers:
            preds = result.get("rate_preds", {}).get(tk, {})
            pred_pct  = preds.get("change_pct", 0.0)
            pred_dir  = direction_of(pred_pct)
            pred_conf = preds.get("confidence", 0)

            closes = rate_history.get(tk, {})
            t_date = nearest_trading_date(closes, td)
            if t_date is None:
                continue
            rate_t  = closes[t_date]
            t_date_dt = datetime.strptime(t_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            for fwd in forwards:
                # horizon별 threshold 자동 조정: T+N일에는 누적 변동 더 크므로 threshold도 조정
                thr = 0.1 if fwd == 1 else (0.25 if fwd <= 3 else 0.5)
                p_dir = direction_of(pred_pct, threshold=0.1)  # 예측은 1일 기준 그대로

                t_plus = nearest_trading_date(closes, t_date_dt + timedelta(days=fwd))
                if t_plus is None or t_plus == t_date:
                    actual_pct = None; actual_dir = "N/A"; match = "N/A"; rate_tp = None
                else:
                    rate_tp = closes[t_plus]
                    actual_pct = (rate_tp - rate_t) / rate_t * 100
                    actual_dir = direction_of(actual_pct, threshold=thr)
                    match = int(p_dir == actual_dir)

                # CSV append
                with open(args.out, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        td.strftime("%Y-%m-%d"), tk, fwd, n_articles,
                        f"{rate_t:.2f}",
                        f"{rate_tp:.2f}" if rate_tp else "",
                        f"{actual_pct:.3f}" if actual_pct is not None else "",
                        actual_dir,
                        f"{pred_pct:.3f}",
                        p_dir, pred_conf,
                        match,
                        llm_method,
                    ])
                results.append({
                    "date": td.strftime("%Y-%m-%d"),
                    "ticker": tk,
                    "horizon": fwd,
                    "actual_pct": actual_pct,
                    "actual_dir": actual_dir,
                    "pred_pct":   pred_pct,
                    "pred_dir":   p_dir,
                    "match":      match,
                })

            # 시점 요약 (한 줄에 모든 horizon)
            results_for_this = [r for r in results
                                if r["date"]==td.strftime("%Y-%m-%d") and r["ticker"]==tk]
            line = f"        {tk}  pred={pred_pct:+.2f}%({pred_dir})"
            for r in results_for_this:
                mark = "✓" if r["match"]==1 else ("✗" if r["match"]==0 else "?")
                ap = f"{r['actual_pct']:+.2f}" if r['actual_pct'] is not None else "N/A"
                line += f"  T+{r['horizon']}:{ap}({r['actual_dir'][:1]}){mark}"
            print(line)

        time.sleep(args.throttle)

    # ── 집계: (ticker, horizon) 조합별 정확도 ──
    print("\n" + "=" * 70)
    print("  📊 백테스트 결과 — (ticker × horizon)별 정확도")
    print("=" * 70)

    from collections import Counter
    by_combo: Dict[tuple, List[dict]] = {}
    for r in results:
        by_combo.setdefault((r["ticker"], r["horizon"]), []).append(r)

    print(f"\n  {'Ticker':<10} {'H':>3}  {'N':>3}  {'Acc':>7}  {'MAE':>6}")
    print(f"  {'-'*10} {'-'*3}  {'-'*3}  {'-'*7}  {'-'*6}")
    summary_rows = []
    for (tk, h), rows in sorted(by_combo.items()):
        valid = [r for r in rows if r["match"] in (0,1)]
        if not valid:
            continue
        n = len(valid)
        correct = sum(r["match"] for r in valid)
        acc = correct / n * 100
        mae = sum(abs(r["pred_pct"] - r["actual_pct"]) for r in valid) / n
        summary_rows.append({"ticker": tk, "horizon": h, "n": n, "acc": acc, "mae": mae})
        print(f"  {tk:<10} T+{h:<2} {n:>3}  {acc:>6.1f}%  {mae:>5.2f}%")

    # 최고 조합 강조
    if summary_rows:
        best = max(summary_rows, key=lambda x: x["acc"])
        print(f"\n  🏆 최고 조합: {best['ticker']} T+{best['horizon']} → {best['acc']:.1f}%  ({best['n']}개 시점, MAE {best['mae']:.2f}%)")

    # ── Ensemble: 5개 통화 다수결 ──
    print(f"\n  📊 통화 ensemble (5개 통화 다수결, horizon별)")
    print(f"  {'Horizon':<8} {'N':>3}  {'Acc':>7}")
    print(f"  {'-'*8} {'-'*3}  {'-'*7}")
    ensemble_rows = []
    by_date_horizon: Dict[tuple, List[dict]] = {}
    for r in results:
        if r["match"] in (0,1):
            by_date_horizon.setdefault((r["date"], r["horizon"]), []).append(r)
    for h in forwards:
        valid = [v for k, v in by_date_horizon.items() if k[1] == h]
        if not valid:
            continue
        n = 0; correct = 0
        for vs in valid:
            preds = [r["pred_dir"] for r in vs]
            actuals = [r["actual_dir"] for r in vs]
            if not preds:
                continue
            # 다수결 (5종 합의 방향)
            pred_winner = Counter(preds).most_common(1)[0][0]
            actual_winner = Counter(actuals).most_common(1)[0][0]
            n += 1
            if pred_winner == actual_winner:
                correct += 1
        if n:
            acc = correct / n * 100
            ensemble_rows.append({"horizon": h, "n": n, "acc": acc})
            print(f"  T+{h:<6} {n:>3}  {acc:>6.1f}%")

    # ── Ensemble: 강한 신호만 (|pred_pct| > 1%) ──
    print(f"\n  📊 강한 신호만 (|pred_pct| > 1.0%, horizon별)")
    print(f"  {'Horizon':<8} {'N':>3}  {'Acc':>7}")
    print(f"  {'-'*8} {'-'*3}  {'-'*7}")
    for h in forwards:
        valid = [r for r in results
                 if r["horizon"]==h and r["match"] in (0,1) and abs(r["pred_pct"]) > 1.0]
        if not valid:
            continue
        n = len(valid)
        correct = sum(r["match"] for r in valid)
        acc = correct / n * 100
        print(f"  T+{h:<6} {n:>3}  {acc:>6.1f}%")

    # 기존 confusion 출력 (각 조합 너무 많으니 USDKRW + T+1만 자세히)
    target_combo = ("USDKRW=X", forwards[0])
    if target_combo in by_combo:
        rows = [r for r in by_combo[target_combo] if r["match"] in (0,1)]
        confusion = Counter((r["actual_dir"], r["pred_dir"]) for r in rows)
        print(f"\n  USDKRW=X T+{forwards[0]} Confusion (실제→예측):")
        for actual in ["up","flat","down"]:
            row_str = "      " + f"실제 {actual:<5} → "
            for pred in ["up","flat","down"]:
                c = confusion.get((actual,pred),0)
                row_str += f"{pred}:{c:>2}  "
            print(row_str)

    print(f"\n  ✅ CSV 저장: {args.out}")
    return summary_rows


if __name__ == "__main__":
    main()
