"""
LLM 실측 F1 벤치마크 (v2: CSV 저장 + 조기 종료)
================================================

NOSIBLE/Twitter Financial 데이터에서 균형 샘플링 후
analyzer.RateAnalyzer를 통해 분류 → 실제 라벨과 비교 → F1 계산

사용:
  python benchmark_llm.py                 # 클래스당 30건 (총 90건)
  python benchmark_llm.py --per-class 50  # 클래스당 50건
  python benchmark_llm.py --provider groq # 특정 provider만 측정

특징:
  · 매 샘플 결과를 즉시 CSV 저장 (중단되어도 데이터 안 잃음)
  · 연속 N건 폴백 발생 시 조기 종료 (모든 LLM quota 소진 의미)
  · 최종 F1은 LLM 호출 성공 샘플만으로 별도 계산
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

# 프로젝트 루트를 sys.path에 추가 (backtest/ 서브 폴더에서도 동작)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
# archive/train.py 도 import 가능하게 (legacy → archive 로 이동됨)
sys.path.insert(0, os.path.join(_ROOT, "archive"))
load_dotenv(os.path.join(_ROOT, ".env"))

from core.analyzer import RateAnalyzer  # noqa: E402
from train import build_dataset, META_PATH  # noqa: E402

LABELS = ["BEARISH", "NEUTRAL", "BULLISH"]

# 공정 비교용 프롬프트 — 짧고 명확 (70B의 TPM 절감 + 명확한 룰)
SENTIMENT_PROMPT = """Classify financial news into BULLISH / BEARISH / NEUTRAL.

BULLISH = positive direction (growth, beat, profit, gain, surge, contract, expansion, recovery)
BEARISH = negative direction (decline, miss, loss, drop, crisis, layoffs, downgrade, concern)
NEUTRAL = factual without direction (scheduled meetings, neutral announcements, pure facts)

Even subtle cues count: "9% growth expected" → BULLISH; "cost pressure" → BEARISH.

Output only JSON: {"label": "BULLISH" | "BEARISH" | "NEUTRAL"}"""


def is_llm(method: str) -> bool:
    m = (method or "").lower()
    if "error" in m:
        return False
    return any(s in m for s in ("gemini", "groq", "gpt", "openai", "sentiment"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--provider", default=None,
                    help="특정 provider만 사용 (gemini/groq/openai). 기본은 전체 chain")
    ap.add_argument("--task", choices=["fx", "sentiment"], default="fx",
                    help="fx: 운영용 FX 트레이더 프롬프트 (분석기 그대로) / "
                         "sentiment: 공정 비교용 일반 금융 감성 프롬프트")
    ap.add_argument("--max-fallback-streak", type=int, default=10)
    ap.add_argument("--throttle", type=float, default=2.1)
    ap.add_argument("--vote", type=int, default=1,
                    help="self-consistency: N>=2면 각 샘플을 N회 호출 후 다수결 (temp=0.5)")
    ap.add_argument("--out", default="bench_results.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  NEXUS — LLM 실측 F1 벤치마크 v2")
    print("=" * 60)

    # 데이터
    df_full = build_dataset(extra_csv=None)
    if df_full.empty:
        print("[ERROR] 데이터셋 비어있음"); return

    parts = []
    for lbl in LABELS:
        sub = df_full[df_full["label"] == lbl]
        parts.append(sub.sample(min(len(sub), args.per_class), random_state=args.seed))
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"  샘플: {len(df):,}건 (클래스당 {args.per_class:,})")

    # 분석기
    a = RateAnalyzer()
    if not a._llm_ok:
        print("[ERROR] LLM 활성 안 됨"); return
    print(f"  체인: {[p[0] for p in a._llm_chain]}")

    # 특정 provider만 사용하려면 chain을 그것만 남기기
    if args.provider:
        a._llm_chain = [(n, fn) for n, fn in a._llm_chain if n == args.provider]
        if not a._llm_chain:
            print(f"[ERROR] provider '{args.provider}' 활성 아님"); return
        print(f"  → {args.provider}만 사용")

    # CSV 헤더
    csv_path = os.path.join(os.path.dirname(__file__) or ".", args.out)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label", "pred_label", "match", "method", "is_llm",
                    "duration_s", "text_preview"])

    # 감성 분류 모드: provider client를 직접 호출 + Rate limit 대응
    _last_call_time = [0.0]
    MIN_INTERVAL = args.throttle

    def _throttle():
        now = time.time()
        wait = MIN_INTERVAL - (now - _last_call_time[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()

    def _one_classify_call(text: str, temperature: float) -> tuple:
        provider = args.provider or a._llm_chain[0][0]
        prompt = f"Classify this financial text:\n\n{text[:1500]}"
        delays = [2.0, 5.0, 12.0]  # 429 발생 시 backoff
        for attempt, delay in enumerate(delays + [0], 1):
            _throttle()
            try:
                if provider == "groq":
                    resp = a._groq.chat.completions.create(
                        model=a.GROQ_MODELS[0],
                        messages=[{"role":"system","content":SENTIMENT_PROMPT},
                                  {"role":"user","content":prompt}],
                        temperature=temperature, max_tokens=50,
                        response_format={"type":"json_object"},
                    )
                    return json.loads(resp.choices[0].message.content).get("label","NEUTRAL"), f"Groq-sentiment-{a.GROQ_MODELS[0]}"
                if provider == "gemini":
                    from google.genai import types as gt
                    resp = a._gemini.models.generate_content(
                        model=a.GEMINI_MODELS[0],
                        contents=prompt,
                        config=gt.GenerateContentConfig(
                            system_instruction=SENTIMENT_PROMPT,
                            response_mime_type="application/json",
                            temperature=temperature, max_output_tokens=50,
                        ),
                    )
                    return json.loads(resp.text or '{"label":"NEUTRAL"}').get("label","NEUTRAL"), f"Gemini-sentiment-{a.GEMINI_MODELS[0]}"
                if provider == "openai":
                    resp = a._openai.chat.completions.create(
                        model=a.OPENAI_MODEL,
                        messages=[{"role":"system","content":SENTIMENT_PROMPT},
                                  {"role":"user","content":prompt}],
                        temperature=temperature, max_tokens=50,
                        response_format={"type":"json_object"},
                    )
                    return json.loads(resp.choices[0].message.content).get("label","NEUTRAL"), f"OpenAI-sentiment-{a.OPENAI_MODEL}"
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
                if is_rate_limit and attempt <= len(delays):
                    print(f"    [retry {attempt}/{len(delays)}] rate-limit → {delay}s wait")
                    time.sleep(delay)
                    continue
                return "NEUTRAL", f"error: {type(e).__name__}"
        return "NEUTRAL", "error: max_retries"

    def classify_sentiment(text: str) -> tuple:
        if args.vote <= 1:
            return _one_classify_call(text, temperature=0.1)
        # Self-consistency: 여러 번 호출 후 다수결
        from collections import Counter
        labels, methods = [], []
        for _ in range(args.vote):
            lbl, m = _one_classify_call(text, temperature=0.5)
            labels.append(lbl); methods.append(m)
        counts = Counter(labels)
        winner, _ = counts.most_common(1)[0]
        return winner, f"{methods[0]} [vote {args.vote}: {dict(counts)}]"

    # 추론 루프
    rows, fallback_streak = [], 0
    mode_label = "sentiment 직접 호출" if args.task == "sentiment" else "FX 프롬프트 (analyzer)"
    print(f"\n  진행 ({len(df)}건), 모드: {mode_label}")
    for i, row in df.iterrows():
        true_lbl = row["label"]
        text = str(row["text"])[:1000]
        t0 = time.time()
        try:
            if args.task == "sentiment":
                pred, method = classify_sentiment(text)
            else:
                r = a.analyze("", text, lang=args.lang)
                pred = r.get("label", "NEUTRAL")
                method = r.get("method", "")
        except Exception as e:
            pred, method = "NEUTRAL", f"error: {type(e).__name__}"
        dt = time.time() - t0
        match = pred == true_lbl
        llm = is_llm(method)
        rows.append({
            "idx": i, "true": true_lbl, "pred": pred, "match": match,
            "method": method, "is_llm": llm, "duration": dt,
        })

        # 즉시 CSV append
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                i, true_lbl, pred, int(match), method, int(llm),
                round(dt, 2), text[:120].replace("\n", " ")
            ])

        # 조기 종료 카운터
        fallback_streak = 0 if llm else fallback_streak + 1

        if (i + 1) % 5 == 0 or i == len(df) - 1:
            llm_n = sum(1 for r in rows if r["is_llm"])
            avg_dt = sum(r["duration"] for r in rows) / len(rows)
            chk = "✓" if match else "✗"
            src = "LLM" if llm else "FALLBACK"
            print(f"    [{i+1:>3}/{len(df)}] {chk} {true_lbl:<8}→{pred:<8}  "
                  f"({dt:>5.1f}s, avg {avg_dt:.1f}s, LLM {llm_n}/{len(rows)} = {llm_n/len(rows)*100:.0f}%, {src})")

        if fallback_streak >= args.max_fallback_streak:
            print(f"\n  ⚠️ 연속 {fallback_streak}건 폴백 — 모든 LLM quota 소진 추정 → 조기 종료")
            break

    # ── 평가 ──
    print("\n" + "=" * 60)
    print("  📊 평가 결과")
    print("=" * 60)
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]

    # 전체 (LLM + 폴백 섞임)
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        print(f"\n  [전체 {len(rows)}건, LLM+폴백 혼합]")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  F1 (weighted) : {f1_w*100:.2f}%")

    # LLM만
    llm_rows = [r for r in rows if r["is_llm"]]
    if llm_rows:
        y_true_l = [r["true"] for r in llm_rows]
        y_pred_l = [r["pred"] for r in llm_rows]
        acc_l = accuracy_score(y_true_l, y_pred_l)
        f1_w_l = f1_score(y_true_l, y_pred_l, average="weighted", labels=LABELS, zero_division=0)
        f1_m_l = f1_score(y_true_l, y_pred_l, average="macro", labels=LABELS, zero_division=0)
        f1_each = f1_score(y_true_l, y_pred_l, average=None, labels=LABELS, zero_division=0)

        print(f"\n  [⭐ LLM만 {len(llm_rows)}건]")
        print(f"  Accuracy            : {acc_l*100:.2f}%")
        print(f"  F1-score (weighted) : {f1_w_l*100:.2f}%")
        print(f"  F1-score (macro)    : {f1_m_l*100:.2f}%")
        for lbl, v in zip(LABELS, f1_each):
            bar = "█" * int(v * 25)
            print(f"  F1 {lbl:<10}   : {v*100:5.2f}%  {bar}")
        print()
        print(classification_report(y_true_l, y_pred_l, labels=LABELS,
                                     target_names=LABELS, digits=4, zero_division=0))
        cm = confusion_matrix(y_true_l, y_pred_l, labels=LABELS)
        print("  Confusion Matrix (행=실제, 열=예측)")
        print("                " + "  ".join(f"{l:>9}" for l in LABELS))
        for i, rrow in enumerate(cm):
            print(f"  {LABELS[i]:>10}    " + "  ".join(f"{v:>9}" for v in rrow))

        # 메타 저장
        meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
        meta["llm_f1"] = round(f1_w_l * 100, 2)
        meta["llm_benchmark"] = {
            "samples_total":     len(rows),
            "samples_llm":       len(llm_rows),
            "per_class":         args.per_class,
            "accuracy":          round(acc_l * 100, 2),
            "f1_weighted":       round(f1_w_l * 100, 2),
            "f1_macro":          round(f1_m_l * 100, 2),
            "f1_per_class":      {lbl: round(float(v) * 100, 2) for lbl, v in zip(LABELS, f1_each)},
            "avg_seconds":       round(sum(r["duration"] for r in llm_rows) / len(llm_rows), 2),
            "csv":               csv_path,
            "provider_filter":   args.provider,
            "benchmarked_at":    datetime.now().isoformat(),
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n  ✅ 메타 업데이트: llm_f1={meta['llm_f1']}%")
        print(f"  ✅ CSV 저장: {csv_path}")
    else:
        print("\n  ⚠️ LLM 호출 0건 — F1 계산 불가")


if __name__ == "__main__":
    main()
