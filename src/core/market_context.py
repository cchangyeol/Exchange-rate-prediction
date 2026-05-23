"""
환율에 영향을 미치는 cross-asset 신호 수집 (yfinance 무료 사용).

학술적으로 검증된 FX 예측 보조 신호:
  · DXY    — 달러 지수 (USD/KRW과 강한 양의 상관)
  · VIX    — 변동성 지수 (위험 회피 proxy: ↑ → KRW 약세)
  · TNX    — 미 10Y 국채 금리 (금리차 시그널)
  · BZ     — Brent 원유 (한국 수입원가 ↑ → KRW 약세)
  · ^GSPC  — S&P 500 (글로벌 위험 sentiment)
  · GC=F   — 금 선물 (safe haven)
  · KS11   — 코스피 (한국 위험자산 proxy)
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


SIGNALS = {
    "DXY":  ("DX-Y.NYB",  "Dollar Index"),
    "VIX":  ("^VIX",      "Volatility (Fear) Index"),
    "TNX":  ("^TNX",      "US 10Y Treasury Yield (%)"),
    "BRENT":("BZ=F",      "Brent Crude Oil (USD/bbl)"),
    "SP500":("^GSPC",     "S&P 500"),
    "GOLD": ("GC=F",      "Gold Futures (USD/oz)"),
    "KOSPI":("^KS11",     "KOSPI"),
}


def _yf():
    import yfinance as yf
    return yf


def fetch_context_at(target_date: datetime, lookback_days: int = 5) -> Dict:
    """
    target_date 시점 기준 시장 context.
    반환 형식:
      {
        "DXY": {"level": 104.2, "change_3d_pct": +0.5, "z_score": +1.2},
        ...
      }
    """
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=timezone.utc)
    end = target_date + timedelta(days=1)
    start = target_date - timedelta(days=lookback_days * 3)   # 휴장 마진
    yf = _yf()

    out = {}
    for key, (yf_sym, name) in SIGNALS.items():
        try:
            df = yf.Ticker(yf_sym).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
            )
            if df.empty:
                continue
            closes = df["Close"].dropna()
            if len(closes) < 2:
                continue
            current = float(closes.iloc[-1])
            # 3거래일 전과 비교 (없으면 가능한 첫 값)
            prev_idx = max(0, len(closes) - 1 - lookback_days)
            prev = float(closes.iloc[prev_idx])
            change_pct = (current - prev) / prev * 100 if prev else 0.0
            # 간단한 z-score (mean과 std 기반)
            mean = float(closes.mean())
            std  = float(closes.std()) or 1.0
            z = (current - mean) / std
            out[key] = {
                "name":          name,
                "level":         round(current, 2),
                "change_pct":    round(change_pct, 2),
                "z_score":       round(z, 2),
                "trend":         "up" if change_pct > 0.3 else ("down" if change_pct < -0.3 else "flat"),
            }
        except Exception as e:
            # print(f"  [context SKIP] {key}: {e}")
            continue
    return out


def format_for_prompt(context: Dict) -> str:
    """LLM 프롬프트용 텍스트 포맷"""
    if not context:
        return ""
    lines = ["Market context (cross-asset signals at this date):"]
    for key, info in context.items():
        arrow = "↑" if info["trend"] == "up" else ("↓" if info["trend"] == "down" else "→")
        lines.append(
            f"  · {key} ({info['name']}): {info['level']} "
            f"{arrow} {info['change_pct']:+.2f}% over ~3 trading days"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # CLI 테스트
    import sys
    target = datetime.now(timezone.utc) - timedelta(days=7)
    if len(sys.argv) > 1:
        target = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"=== Market context @ {target.date()} ===")
    ctx = fetch_context_at(target)
    print(f"  수집: {len(ctx)}/{len(SIGNALS)}\n")
    print(format_for_prompt(ctx))
