"""
backtest_results.csv의 raw 데이터에서 (LLM 재호출 없이) 여러 전략 비교.
"""
import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import product

LABELS = ["BEARISH", "NEUTRAL", "BULLISH"]

# 기본 CSV 위치: results/csv/ 우선, 없으면 현재 디렉토리
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "backtest" else _HERE
_DEFAULT_CSV = os.path.join(_ROOT, "results", "csv", "backtest_results.csv")
if not os.path.exists(_DEFAULT_CSV):
    _DEFAULT_CSV = "backtest_results.csv"


def direction_of(pct, threshold):
    if pct > threshold:  return "up"
    if pct < -threshold: return "down"
    return "flat"


def load_csv(path=_DEFAULT_CSV):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append({
                "date": r["test_date"],
                "ticker": r["ticker"],
                "horizon": int(r["horizon"]),
                "pred_pct": float(r["pred_pct"]),
                "actual_pct": float(r["actual_pct"]) if r["actual_pct"] else None,
            })
        except ValueError:
            continue
    return [r for r in rows if r["actual_pct"] is not None]


def accuracy(rows, pred_thr, actual_thr):
    """주어진 threshold로 정확도 계산"""
    if not rows:
        return 0.0, 0, 0
    correct = 0
    for r in rows:
        p = direction_of(r["pred_pct"], pred_thr)
        a = direction_of(r["actual_pct"], actual_thr)
        if p == a:
            correct += 1
    return correct / len(rows), correct, len(rows)


def main():
    rows = load_csv()
    if not rows:
        print("[ERROR] CSV 데이터 없음"); return
    print(f"전체 row: {len(rows)}\n")

    # === 1) 단일 변수 영향: ticker별 평균 정확도 ===
    print("="*70)
    print("  [1] Ticker × Horizon 매트릭스 (현재 threshold 0.1%/0.25%/0.5%)")
    print("="*70)
    by_th = {1: 0.1, 3: 0.25, 7: 0.5}
    grid = defaultdict(list)
    for r in rows:
        grid[(r["ticker"], r["horizon"])].append(r)
    print(f"  {'Ticker':<10} {'T+1':>7} {'T+3':>7} {'T+7':>7}")
    ticker_avg = {}
    for tk in sorted({r["ticker"] for r in rows}):
        line = f"  {tk:<10}"
        accs = []
        for h in [1,3,7]:
            sub = grid.get((tk, h), [])
            acc, c, n = accuracy(sub, 0.1, by_th[h])
            line += f"  {acc*100:>5.1f}%"
            accs.append(acc)
        line += f"   평균 {sum(accs)/len(accs)*100:>5.1f}%"
        ticker_avg[tk] = sum(accs)/len(accs)
        print(line)

    # === 2) Threshold sweep ===
    print("\n"+"="*70)
    print("  [2] Prediction threshold sweep (모든 ticker, 모든 horizon 합산)")
    print("="*70)
    print(f"  {'pred_thr':<10} {'N (filtered)':<14} {'Acc':>7}")
    for pred_thr in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]:
        # 강한 신호 필터 — |pred|>=pred_thr인 row만 사용
        filtered = [r for r in rows if abs(r["pred_pct"]) >= pred_thr]
        if not filtered:
            continue
        # actual_thr는 horizon에 맞춰 자동
        correct = 0
        for r in filtered:
            p_dir = direction_of(r["pred_pct"], 0.1)
            a_dir = direction_of(r["actual_pct"], by_th[r["horizon"]])
            if p_dir == a_dir: correct += 1
        acc = correct/len(filtered)
        print(f"  {pred_thr:<10.1f} {len(filtered):<14} {acc*100:>6.1f}%")

    # === 3) Single ticker + threshold sweep ===
    print("\n"+"="*70)
    print("  [3] 최고 ticker만 + strong signal sweep")
    print("="*70)
    best_tk = max(ticker_avg, key=ticker_avg.get)
    print(f"  최고 ticker: {best_tk} (평균 {ticker_avg[best_tk]*100:.1f}%)")
    print(f"\n  {'horizon':<8} {'pred_thr':<10} {'N':<5} {'Acc':>7}")
    for h in [1,3,7]:
        for pred_thr in [0.1, 0.5, 1.0, 1.5]:
            filtered = [r for r in rows
                        if r["ticker"]==best_tk and r["horizon"]==h
                        and abs(r["pred_pct"]) >= pred_thr]
            if not filtered: continue
            acc, c, n = accuracy(filtered, 0.1, by_th[h])
            print(f"  T+{h:<6} {pred_thr:<10.1f} {n:<5} {acc*100:>6.1f}%")

    # === 4) Multi-ticker consensus ===
    print("\n"+"="*70)
    print("  [4] Multi-ticker consensus (5종 통화 중 K개 이상 같은 방향이면 trigger)")
    print("="*70)
    by_date_h = defaultdict(list)
    for r in rows:
        by_date_h[(r["date"], r["horizon"])].append(r)
    print(f"  {'horizon':<8} {'K':<3} {'N (trigger)':<14} {'Acc':>7}")
    for h in [1,3,7]:
        for K in [3, 4, 5]:
            n = 0; correct = 0
            for (d, hh), rs in by_date_h.items():
                if hh != h or len(rs) < 3:
                    continue
                preds = Counter(direction_of(r["pred_pct"], 0.1) for r in rs)
                actuals = Counter(direction_of(r["actual_pct"], by_th[h]) for r in rs)
                top_pred, top_count = preds.most_common(1)[0]
                if top_count < K:
                    continue  # consensus 부족 → skip
                top_actual = actuals.most_common(1)[0][0]
                n += 1
                if top_pred == top_actual:
                    correct += 1
            if n:
                print(f"  T+{h:<6} {K:<3} {n:<14} {correct/n*100:>6.1f}%")

    # === 5) GBPKRW + T+3 + strong signal ===
    print("\n"+"="*70)
    print("  [5] 통합 전략 — GBPKRW=X × T+3 × |pred|>1.0%")
    print("="*70)
    filtered = [r for r in rows
                if r["ticker"]=="GBPKRW=X" and r["horizon"]==3
                and abs(r["pred_pct"]) > 1.0]
    if filtered:
        correct = sum(1 for r in filtered
                      if direction_of(r["pred_pct"], 0.1) == direction_of(r["actual_pct"], 0.25))
        print(f"  N: {len(filtered)}, Acc: {correct/len(filtered)*100:.1f}%")
        for r in filtered:
            p = direction_of(r["pred_pct"], 0.1)
            a = direction_of(r["actual_pct"], 0.25)
            m = "✓" if p==a else "✗"
            print(f"    {r['date']}  pred={r['pred_pct']:+.2f}({p})  actual={r['actual_pct']:+.2f}({a}) {m}")

    # === 6) 최종 권장 전략 찾기 (grid search) ===
    print("\n"+"="*70)
    print("  [6] 🏆 자동 grid search — N>=10인 조합 중 최고 정확도")
    print("="*70)
    best = []
    for tk in sorted({r["ticker"] for r in rows}):
        for h in [1,3,7]:
            for pred_thr in [0.1, 0.5, 1.0, 1.5, 2.0]:
                filtered = [r for r in rows
                            if r["ticker"]==tk and r["horizon"]==h
                            and abs(r["pred_pct"]) >= pred_thr]
                if len(filtered) < 10: continue
                acc, c, n = accuracy(filtered, 0.1, by_th[h])
                best.append((acc, n, tk, h, pred_thr))
    best.sort(key=lambda x: (-x[0], -x[1]))
    print(f"  {'Acc':>7} {'N':>4}  {'Ticker':<10} {'Horizon':<8} {'pred_thr':<10}")
    for acc, n, tk, h, thr in best[:10]:
        print(f"  {acc*100:>6.1f}% {n:>4}  {tk:<10} T+{h:<6} {thr:<10.1f}")


if __name__ == "__main__":
    main()
