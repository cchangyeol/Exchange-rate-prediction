"""
NEXUS — 환율 변동 예측 분석 엔진

LLM 우선순위 (순서대로 시도, 실패/쿼타 시 다음으로 cascade):
  1. Gemini    (GEMINI_API_KEY)  — 무료, 일일 한도 적음 (모델별 20~수천/일)
  2. Groq      (GROQ_API_KEY)    — 무료, 14,400/일 (Llama 3.3 70B)
  3. OpenAI    (OPENAI_API_KEY)  — 유료, 거의 무제한

LLM 모두 실패 시 폴백:
  4. KR-FinBERT (models/kr_finbert/)
  5. sklearn    (models/sentiment_model.pkl)
  6. 키워드 기반
"""

import os, json, random, hashlib, time
from datetime import datetime
from typing import Dict, List, Optional
from collections import OrderedDict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "models", "model_meta.json")
BERT_DIR   = os.path.join(os.path.dirname(__file__), "models", "kr_finbert")

BULL_THRESHOLD =  0.30
BEAR_THRESHOLD = -0.30

# ── 키워드 사전 ───────────────────────────────────────────────
BULLISH_KW = [
    "급등","상승","돌파","신고가","호재","성장","회복","반등","강세","흑자",
    "승인","협력","실적개선","확대","surge","rally","record","bullish","gain","growth"
]
BEARISH_KW = [
    "급락","하락","폭락","위기","손실","적자","감소","약세","제재","전쟁",
    "침공","봉쇄","파산","금리인상","긴축","침체","경기둔화",
    "drop","fall","crash","crisis","bearish","decline","recession","war","sanction"
]
RISK_KW = [
    "전쟁","핵","침공","군사","테러","봉쇄","제재","팬데믹","파산","디폴트",
    "war","nuclear","invasion","military","terror","pandemic","default","blockade"
]
CAT_MAP = {
    "지정학적 위기": ["전쟁","침공","분쟁","군사","제재","지정학","핵","테러","war","invasion","military"],
    "금융 정책":    ["Fed","금리","연준","인플레이션","긴축","통화정책","기준금리","FOMC","interest rate"],
    "무역·경제":    ["무역","수출","수입","관세","GDP","경기","성장률","trade","tariff","GDP","growth"],
    "기업·실적":    ["실적","매출","영업이익","반도체","AI","earnings","revenue","profit"],
    "암호화폐":     ["비트코인","코인","암호화폐","BTC","bitcoin","crypto"],
}

# 환율별 카테고리 편향 (방향: + = 원화 약세 = 환율 상승)
RATE_BIAS = {
    "지정학적 위기": {"USDKRW=X": +0.6, "JPYKRW=X": +0.4, "EURKRW=X": -0.1, "CNYKRW=X": +0.2, "GBPKRW=X": -0.2},
    "금융 정책":    {"USDKRW=X": +0.8, "JPYKRW=X": -0.2, "EURKRW=X": -0.3, "CNYKRW=X": +0.1, "GBPKRW=X": -0.1},
    "무역·경제":    {"USDKRW=X": +0.5, "CNYKRW=X": +0.6, "JPYKRW=X": +0.2, "EURKRW=X": +0.1, "GBPKRW=X": +0.1},
}

# GPT 시스템 프롬프트 — V3 hybrid: balanced direction + assertive magnitude
SYSTEM_PROMPT = """You are an FX analyst specializing in Korean Won (KRW) currency pairs.
Analyze the news and predict KRW exchange rate direction for the NEXT 7 TRADING DAYS.

KEY RULES:
1. BIDIRECTIONAL: Don't default to "up". Look for BOTH bullish (KRW strengthens, rate DOWN) AND
   bearish (KRW weakens, rate UP) signals. Make a clear directional call when evidence is present.
   · UP signals: Fed hawkish, geopolitical risk, Korean trade deficit, capital outflow, USD strength
   · DOWN signals: Korean exports surge, BOK hike, foreign inflows, Fed dovish, semiconductor boom
2. ASSERTIVE: Reserve NEUTRAL only for truly directionless news. If there's a meaningful catalyst,
   commit to a direction even if mild.
3. MAGNITUDE: Realistic. 7-day KRW moves are typically 0.5-1.5%. Major events (FOMC, war, crisis)
   may justify 1.5-3.0. Default magnitude 0.8-1.5 when news has clear directional signal.
4. USE MARKET CONTEXT (if provided): Cross-asset signals are STRONG predictors:
   · DXY up → USD strong → USD/KRW UP
   · VIX up → risk-off → KRW weakens (UP)
   · TNX (US 10Y yield) up → USD demand → USD/KRW UP
   · BRENT/oil up → Korea import cost ↑ → KRW weakens (UP)
   · SP500/KOSPI up → risk-on → KRW strengthens (DOWN)
   · GOLD up → safe haven demand → KRW weakens (UP)
   Weight news vs context: short-term moves often track context; news provides catalysts.

Respond ONLY in valid JSON with this exact structure:
{
  "label": "BULLISH" | "BEARISH" | "NEUTRAL",
  "raw_score": <float -1.0 to 1.0>,
  "confidence": <integer 40-97>,
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "category": "지정학적 위기" | "금융 정책" | "무역·경제" | "기업·실적" | "암호화폐" | "기타",
  "reasoning_ko": "<2-3 sentences in Korean explaining the KRW impact>",
  "reasoning_en": "<2-3 sentences in English explaining the KRW impact>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "rate_impacts": {
    "USDKRW=X": {"direction": "up"|"down"|"flat", "magnitude": <0.0-3.0>, "confidence": <40-95>, "reason": "<brief reason>"},
    "JPYKRW=X": {"direction": "up"|"down"|"flat", "magnitude": <0.0-3.0>, "confidence": <40-95>, "reason": "<brief reason>"},
    "CNYKRW=X": {"direction": "up"|"down"|"flat", "magnitude": <0.0-3.0>, "confidence": <40-95>, "reason": "<brief reason>"},
    "EURKRW=X": {"direction": "up"|"down"|"flat", "magnitude": <0.0-3.0>, "confidence": <40-95>, "reason": "<brief reason>"},
    "GBPKRW=X": {"direction": "up"|"down"|"flat", "magnitude": <0.0-3.0>, "confidence": <40-95>, "reason": "<brief reason>"}
  }
}

Direction semantics:
- "up" = KRW weakens (USD/KRW rises from 1300 to 1320)
- "down" = KRW strengthens (USD/KRW falls)
- raw_score: positive = risk-on/KRW strong, negative = risk-off/KRW weak

Self-check:
- Did I consider BOTH up AND down? If only one direction, what's my justification?
- Are 5 currency pairs evaluated independently? (e.g. JPY safe-haven often opposite to USD)"""


class RateAnalyzer:
    GPT_CACHE_SIZE = 256
    GEMINI_MODELS  = [
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]
    GROQ_MODELS    = [
        os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    OPENAI_MODEL   = "gpt-4o-mini"

    @property
    def GEMINI_MODEL(self) -> str:
        return self._gemini_active_model or self.GEMINI_MODELS[0]

    @property
    def GROQ_MODEL(self) -> str:
        return self._groq_active_model or self.GROQ_MODELS[0]

    def __init__(self):
        self._gemini      = None
        self._gemini_active_model = None
        self._groq        = None
        self._groq_active_model = None
        self._openai      = None
        self._bert        = None
        self._sklearn     = None
        self._meta        = {}
        self._llm_chain: List = []  # 활성 LLM provider 목록 (순서대로 시도)
        self._gpt_cache: "OrderedDict[str, Dict]" = OrderedDict()

        # 가능한 LLM 모두 로드 (독립적, 어느 것이 실패해도 다음 시도)
        self._load_gemini()
        self._load_groq()
        self._load_openai()
        if not self._llm_chain:
            self._load_bert()
        if not self._llm_chain and not self._bert:
            self._load_sklearn()
        if not self._llm_chain and not self._bert and not self._sklearn:
            print("[INFO] 키워드 기반 분석 사용 (폴백)")

        # 학습 메타 표시용 로드
        if not self._meta and os.path.exists(META_PATH):
            try:
                with open(META_PATH, encoding="utf-8") as f:
                    self._meta = json.load(f)
            except Exception:
                pass

        if self._llm_chain:
            print(f"[INFO] LLM 체인: {' → '.join(p[0] for p in self._llm_chain)} → (sklearn/keyword 폴백)")

    @property
    def _llm_ok(self) -> bool:
        return bool(self._llm_chain)

    @property
    def _llm_provider(self) -> Optional[str]:
        return self._llm_chain[0][0] if self._llm_chain else None

    # ── 모델 로드 ─────────────────────────────────────────────
    def _load_gemini(self):
        key = os.environ.get("GEMINI_API_KEY", "").strip() \
              or os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            print("[INFO] GEMINI_API_KEY 없음 → Gemini 비활성")
            return
        try:
            from google import genai
            self._gemini = genai.Client(api_key=key)
            list(self._gemini.models.list())
            self._gemini_active_model = self.GEMINI_MODELS[0]
            self._llm_chain.append(("gemini", self._try_gemini))
            chain = " → ".join(dict.fromkeys(self.GEMINI_MODELS))
            print(f"[OK] Gemini 연동 완료 — 모델 체인: {chain}")
        except Exception as e:
            print(f"[WARN] Gemini 연결 실패: {e}")
            self._gemini = None

    def _load_groq(self):
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            print("[INFO] GROQ_API_KEY 없음 → Groq 비활성")
            return
        if not key.startswith("gsk_"):
            print(f"[WARN] GROQ_API_KEY 형식 이상 (gsk_로 시작해야 함, 길이={len(key)}) → Groq 비활성")
            return
        try:
            from groq import Groq
            self._groq = Groq(api_key=key)
            # 간단 핑: 모델 목록
            self._groq.models.list()
            self._groq_active_model = self.GROQ_MODELS[0]
            self._llm_chain.append(("groq", self._try_groq))
            chain = " → ".join(dict.fromkeys(self.GROQ_MODELS))
            print(f"[OK] Groq 연동 완료 — 모델 체인: {chain}")
        except Exception as e:
            print(f"[WARN] Groq 연결 실패: {e}")
            self._groq = None

    def _load_openai(self):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            print("[INFO] OPENAI_API_KEY 없음 → GPT 비활성")
            return
        if not key.startswith("sk-"):
            print(f"[WARN] OPENAI_API_KEY 형식 이상 (sk-로 시작해야 함, 길이={len(key)}) → GPT 비활성")
            return
        try:
            from openai import OpenAI
            self._openai = OpenAI(api_key=key)
            self._openai.models.list()
            self._llm_chain.append(("openai", self._try_openai))
            print(f"[OK] OpenAI ({self.OPENAI_MODEL}) 연동 완료")
        except Exception as e:
            print(f"[WARN] OpenAI 연결 실패: {e}")

    def _load_bert(self):
        if not os.path.exists(BERT_DIR):
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            print("[INFO] KR-FinBERT 로드 중...")
            tok   = AutoTokenizer.from_pretrained(BERT_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
            self._bert = pipeline(
                "text-classification", model=model, tokenizer=tok,
                device=-1, truncation=True, max_length=128
            )
            if os.path.exists(META_PATH):
                with open(META_PATH, encoding="utf-8") as f:
                    self._meta = json.load(f)
            print(f"[OK] KR-FinBERT 로드 완료 — F1: {self._meta.get('f1','?')}%")
        except Exception as e:
            print(f"[WARN] KR-FinBERT 로드 실패: {e}")

    def _load_sklearn(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            import pickle
            with open(MODEL_PATH, "rb") as f:
                self._sklearn = pickle.load(f)
            if os.path.exists(META_PATH):
                with open(META_PATH, encoding="utf-8") as f:
                    self._meta = json.load(f)
            print(f"[OK] sklearn 로드 완료 — F1: {self._meta.get('f1','?')}%")
        except Exception as e:
            print(f"[WARN] sklearn 로드 실패: {e}")

    # ── 공개 메서드 ───────────────────────────────────────────
    def analyze(self, title: str, text: str, lang: str = "ko",
                news_list: Optional[List[Dict]] = None,
                market_context: Optional[Dict] = None) -> Dict:
        if news_list and len(news_list) > 1:
            combined      = self._merge_news(news_list)
            method_prefix = f"[다중 뉴스 {len(news_list)}건] "
        else:
            combined      = (title + " " + text).strip()
            method_prefix = ""

        # market_context 있으면 입력에 포함 (LLM에 cross-asset 신호 제공)
        if market_context:
            try:
                from core.market_context import format_for_prompt
                ctx_text = format_for_prompt(market_context)
                if ctx_text:
                    combined = ctx_text + "\n\n=== NEWS ===\n" + combined
                    method_prefix += "[+context] "
            except ImportError:
                pass

        # LLM 체인을 순회: 첫 번째 성공이 답, 모두 실패 시 sklearn/키워드
        for provider_name, try_fn in self._llm_chain:
            result = try_fn(combined, lang, method_prefix)
            if result is not None:
                return result
        return self._analyze_fallback(combined, lang, method_prefix)

    # ── LLM 분석 ──────────────────────────────────────────────
    def _cache_key(self, text: str, lang: str, provider: str = "") -> str:
        return hashlib.sha256(f"{lang}::{text[:3000]}::{provider}".encode("utf-8")).hexdigest()

    def _cache_get(self, ck: str, prefix: str, label: str) -> Optional[Dict]:
        if ck in self._gpt_cache:
            self._gpt_cache.move_to_end(ck)
            cached = dict(self._gpt_cache[ck])
            cached["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cached["method"] = prefix + label + " (cached)"
            return cached
        return None

    def _cache_put(self, ck: str, result: Dict):
        self._gpt_cache[ck] = result
        if len(self._gpt_cache) > self.GPT_CACHE_SIZE:
            self._gpt_cache.popitem(last=False)

    def _try_gemini(self, text: str, lang: str, prefix: str) -> Optional[Dict]:
        """Gemini 호출. 성공하면 결과 dict, 모든 모델 실패하면 None (→ 다음 provider)."""
        ck = self._cache_key(text, lang, "gemini")
        hit = self._cache_get(ck, prefix, f"Gemini-{self.GEMINI_MODEL}")
        if hit: return hit
        from google.genai import types as genai_types
        cfg = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=4000,
        )
        last_err = None
        models_to_try = list(dict.fromkeys(self.GEMINI_MODELS))
        for model_name in models_to_try:
            delays = [1.0, 3.0, 7.0]
            for attempt, delay in enumerate(delays, 1):
                try:
                    resp = self._gemini.models.generate_content(
                        model=model_name,
                        contents=f"Analyze this:\n\n{text[:3000]}",
                        config=cfg,
                    )
                    raw = resp.text or ""
                    if not raw.strip():
                        raise ValueError("Gemini 빈 응답")
                    gpt = json.loads(raw)
                    result = self._build_from_gpt(gpt, lang, prefix + f"Gemini-{model_name}")
                    self._cache_put(ck, result)
                    self._gemini_active_model = model_name
                    return result
                except Exception as e:
                    msg = str(e)
                    last_err = e
                    is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                    is_transient = any(s in msg for s in
                        ("503", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")) or \
                        isinstance(e, (json.JSONDecodeError, ValueError))
                    if is_quota:
                        print(f"[INFO] Gemini {model_name} quota 초과 → 다음 모델")
                        break
                    if is_transient and attempt < len(delays):
                        print(f"[INFO] Gemini {model_name} 일시 오류 (시도 {attempt}/{len(delays)}: {type(e).__name__}) → {delay}s 후 재시도")
                        time.sleep(delay)
                        continue
                    break
        print(f"[WARN] Gemini 전체 실패: {type(last_err).__name__} → 다음 LLM provider 시도")
        return None

    def _try_groq(self, text: str, lang: str, prefix: str) -> Optional[Dict]:
        """Groq 호출. 성공하면 결과 dict, 모든 모델 실패하면 None."""
        ck = self._cache_key(text, lang, "groq")
        hit = self._cache_get(ck, prefix, f"Groq-{self.GROQ_MODEL}")
        if hit: return hit
        last_err = None
        models_to_try = list(dict.fromkeys(self.GROQ_MODELS))
        for model_name in models_to_try:
            delays = [1.0, 3.0, 7.0]
            for attempt, delay in enumerate(delays, 1):
                try:
                    resp = self._groq.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": f"Analyze this:\n\n{text[:3000]}"},
                        ],
                        temperature=0.2,
                        max_tokens=4000,
                        response_format={"type": "json_object"},
                    )
                    raw = resp.choices[0].message.content or ""
                    if not raw.strip():
                        raise ValueError("Groq 빈 응답")
                    gpt = json.loads(raw)
                    result = self._build_from_gpt(gpt, lang, prefix + f"Groq-{model_name}")
                    self._cache_put(ck, result)
                    self._groq_active_model = model_name
                    return result
                except Exception as e:
                    msg = str(e)
                    last_err = e
                    is_quota = "429" in msg or "rate_limit" in msg.lower() or "quota" in msg.lower()
                    is_transient = any(s in msg for s in ("503", "502", "504", "timeout", "Timeout")) or \
                        isinstance(e, (json.JSONDecodeError, ValueError))
                    if is_quota:
                        print(f"[INFO] Groq {model_name} 한도 초과 → 다음 모델")
                        break
                    if is_transient and attempt < len(delays):
                        print(f"[INFO] Groq {model_name} 일시 오류 (시도 {attempt}/{len(delays)}: {type(e).__name__}) → {delay}s 후 재시도")
                        time.sleep(delay)
                        continue
                    break
        print(f"[WARN] Groq 전체 실패: {type(last_err).__name__} → 다음 LLM provider 시도")
        return None

    def _try_openai(self, text: str, lang: str, prefix: str) -> Optional[Dict]:
        """OpenAI 호출. 성공하면 결과, 실패하면 None."""
        ck = self._cache_key(text, lang, "openai")
        hit = self._cache_get(ck, prefix, self.OPENAI_MODEL)
        if hit: return hit
        try:
            resp = self._openai.chat.completions.create(
                model=self.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Analyze this:\n\n{text[:3000]}"}
                ],
                temperature=0.2, max_tokens=1500,
                response_format={"type": "json_object"}
            )
            gpt = json.loads(resp.choices[0].message.content)
            result = self._build_from_gpt(gpt, lang, prefix + self.OPENAI_MODEL)
            self._cache_put(ck, result)
            return result
        except Exception as e:
            print(f"[WARN] OpenAI 실패: {e} → 다음 LLM provider 시도")
            return None

    def _build_from_gpt(self, gpt: Dict, lang: str, method: str) -> Dict:
        label      = gpt.get("label", "NEUTRAL")
        raw_score  = float(gpt.get("raw_score", 0.0))
        confidence = int(gpt.get("confidence", 60))
        risk_level = gpt.get("risk_level", "LOW")
        category   = gpt.get("category", "기타")
        reasoning  = gpt.get("reasoning_ko" if lang == "ko" else "reasoning_en", "")
        key_factors= gpt.get("key_factors", [])

        rate_preds = {}
        gpt_rates  = gpt.get("rate_impacts", {})
        for r in self._rate_meta():
            tk   = r["ticker"]
            info = gpt_rates.get(tk, {})
            d    = info.get("direction", "flat")
            mag  = float(info.get("magnitude", abs(raw_score) * 1.2))
            conf = int(info.get("confidence", confidence))
            reason = info.get("reason", "")
            pct  = round(mag if d == "up" else (-mag if d == "down" else 0.0), 2)
            rate_preds[tk] = {
                **r,
                "change_pct":   pct,
                "direction":    "원화 약세" if pct > 0 else ("원화 강세" if pct < 0 else "보합"),
                "direction_en": "KRW Weaken" if pct > 0 else ("KRW Strengthen" if pct < 0 else "Flat"),
                "confidence":   conf,
                "reason":       reason,
            }

        causes = [{"keyword": f, "snippet": f, "weight": round(0.95 - i * 0.1, 2)}
                  for i, f in enumerate(key_factors[:5])]

        return {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label":       label,
            "raw_score":   round(raw_score, 4),
            "confidence":  confidence,
            "risk_level":  risk_level,
            "category":    category,
            "method":      method,
            "reasoning":   reasoning,
            "key_factors": key_factors,
            "causes":      causes,
            "threshold": {
                "raw_score":      round(raw_score, 4),
                "bull_threshold": BULL_THRESHOLD,
                "bear_threshold": BEAR_THRESHOLD,
                "explanation_ko": f"GPT 분석 → 감성 점수 {round(raw_score,3)} → {label} 판정",
                "explanation_en": f"GPT analysis → score {round(raw_score,3)} → {label}",
            },
            "verdict":     self._verdict(label, risk_level),
            "rate_preds":  rate_preds,
            "lang":        lang,
            "llm_used":    True,
        }

    # ── 폴백 분석 ─────────────────────────────────────────────
    def _analyze_fallback(self, text: str, lang: str, prefix: str) -> Dict:
        raw_score, method = self._score(text)
        label = ("BULLISH" if raw_score > BULL_THRESHOLD
                 else "BEARISH" if raw_score < BEAR_THRESHOLD
                 else "NEUTRAL")
        confidence = round(min(97, abs(raw_score) * 100 + 45), 1)
        risk_count = sum(1 for k in RISK_KW if k.lower() in text.lower())
        risk_level = "HIGH" if risk_count >= 2 else ("MEDIUM" if risk_count == 1 else "LOW")
        risk_mult  = {"HIGH": 1.8, "MEDIUM": 1.2, "LOW": 0.9}[risk_level]
        category   = self._detect_category(text)
        causes     = self._extract_causes(text, label)

        rate_preds = self._predict_rates(raw_score, risk_mult, category, label)

        return {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label":       label,
            "raw_score":   round(raw_score, 4),
            "confidence":  confidence,
            "risk_level":  risk_level,
            "category":    category,
            "method":      prefix + method,
            "reasoning":   "",
            "key_factors": [],
            "causes":      causes,
            "threshold": {
                "raw_score":      round(raw_score, 4),
                "bull_threshold": BULL_THRESHOLD,
                "bear_threshold": BEAR_THRESHOLD,
                "explanation_ko": f"감성 점수 {round(raw_score,3)} → 기준값 ±{BULL_THRESHOLD} 비교 → {label} 판정",
                "explanation_en": f"Sentiment score {round(raw_score,3)} vs ±{BULL_THRESHOLD} → {label}",
            },
            "verdict":     self._verdict(label, risk_level),
            "rate_preds":  rate_preds,
            "lang":        lang,
            "llm_used":    False,
        }

    # ── 내부 유틸 ─────────────────────────────────────────────
    def _merge_news(self, news_list: List[Dict]) -> str:
        parts = []
        for i, n in enumerate(news_list):
            w = n.get("weight", 1.0)
            parts.append(f"[뉴스{i+1}|가중치:{w:.2f}] {n.get('title','')} {n.get('text','')}")
        return "\n".join(parts)

    def _score(self, text: str):
        if self._bert:
            return self._score_bert(text), "KR-FinBERT (파인튜닝)"
        if self._sklearn:
            return self._score_sklearn(text), "sklearn"
        return self._score_keyword(text), "keyword-based"

    def _score_bert(self, text: str):
        try:
            r = self._bert(text[:512], top_k=None)[0]
            s = {x["label"]: x["score"] for x in r}
            return float(s.get("BULLISH", s.get("LABEL_2", 0)) - s.get("BEARISH", s.get("LABEL_0", 0)))
        except Exception:
            return self._score_keyword(text)

    def _score_sklearn(self, text: str):
        try:
            proba   = self._sklearn.predict_proba([text])[0]
            classes = list(self._sklearn.classes_)
            bull = proba[classes.index("BULLISH")] if "BULLISH" in classes else 0
            bear = proba[classes.index("BEARISH")] if "BEARISH" in classes else 0
            return float(bull - bear)
        except Exception:
            return self._score_keyword(text)

    def _score_keyword(self, text: str):
        tl   = text.lower()
        bull = sum(1 for k in BULLISH_KW if k.lower() in tl)
        bear = sum(1 for k in BEARISH_KW if k.lower() in tl)
        return (bull - bear) / (bull + bear or 1)

    def _detect_category(self, text: str) -> str:
        tl = text.lower()
        for cat, kws in CAT_MAP.items():
            if any(k.lower() in tl for k in kws):
                return cat
        return "기타"

    def _extract_causes(self, text: str, label: str) -> List[Dict]:
        tl      = text.lower()
        kw_list = BEARISH_KW if label == "BEARISH" else BULLISH_KW
        causes  = []
        for kw in kw_list:
            if kw.lower() in tl:
                idx     = tl.find(kw.lower())
                snippet = text[max(0, idx - 20): idx + len(kw) + 20].strip()
                causes.append({"keyword": kw, "snippet": snippet,
                               "weight": round(random.uniform(0.6, 1.0), 2)})
        return causes[:5]

    def _rate_meta(self):
        return [
            {"ticker": "USDKRW=X", "pair": "USD/KRW", "flag": "🇺🇸", "name": "미국 달러"},
            {"ticker": "JPYKRW=X", "pair": "JPY/KRW", "flag": "🇯🇵", "name": "일본 엔"},
            {"ticker": "CNYKRW=X", "pair": "CNY/KRW", "flag": "🇨🇳", "name": "중국 위안"},
            {"ticker": "EURKRW=X", "pair": "EUR/KRW", "flag": "🇪🇺", "name": "유로"},
            {"ticker": "GBPKRW=X", "pair": "GBP/KRW", "flag": "🇬🇧", "name": "파운드"},
        ]

    def _predict_rates(self, score: float, risk_mult: float,
                       category: str, label: str) -> Dict:
        # 위기 시 달러·엔 강세(KRW 약세 = 환율 상승)
        direction = 1 if label == "BEARISH" else -1
        result = {}
        for r in self._rate_meta():
            tk    = r["ticker"]
            base  = direction * abs(score) * risk_mult * 1.8
            bias  = RATE_BIAS.get(category, {}).get(tk, 0) * abs(score)
            noise = (random.random() - 0.5) * 0.3
            pct   = round(max(-4, min(4, base + bias + noise)), 2)
            result[tk] = {
                **r,
                "change_pct":   pct,
                "direction":    "원화 약세" if pct > 0 else ("원화 강세" if pct < 0 else "보합"),
                "direction_en": "KRW Weaken" if pct > 0 else ("KRW Strengthen" if pct < 0 else "Flat"),
                "confidence":   round(min(90, abs(pct) * 15 + 40), 1),
                "reason":       "",
            }
        return result

    def _verdict(self, label: str, risk: str) -> str:
        if label == "BEARISH" and risk == "HIGH": return "DANGER"
        if label == "BEARISH":                    return "CAUTION"
        if label == "BULLISH":                    return "BULLISH"
        return "NEUTRAL"

    def get_status(self) -> Dict:
        p = self._llm_provider  # 첫 번째 활성 provider
        if p == "gemini":
            mode = f"Gemini-{self._gemini_active_model or self.GEMINI_MODELS[0]}"
        elif p == "groq":
            mode = f"Groq-{self._groq_active_model or self.GROQ_MODELS[0]}"
        elif p == "openai":
            mode = self.OPENAI_MODEL
        elif self._bert:
            mode = "KR-FinBERT"
        elif self._sklearn:
            mode = "sklearn"
        else:
            mode = "keyword"

        # F1은 어느 모델 것인지 명확히 분리
        # - active_f1: 현재 분석에 쓰이는 모델 점수 (LLM은 zero-shot이라 별도 벤치마크 필요)
        # - fallback_f1: LLM 실패 시 받아주는 sklearn 모델 점수
        active_f1 = self._meta.get("llm_f1") if self._llm_ok else self._meta.get("f1", "미학습")
        fallback_f1 = self._meta.get("f1", "미학습")

        return {
            "mode":         mode,
            "llm_available":self._llm_ok,
            "llm_provider": p or "none",
            "llm_chain":    [name for name, _ in self._llm_chain],
            "active_f1":    active_f1 if active_f1 is not None else "미측정",
            "fallback_model": "sklearn" if self._sklearn or os.path.exists(MODEL_PATH) else None,
            "fallback_f1":  fallback_f1,
            "f1_score":     active_f1 if active_f1 not in (None, "미측정") else fallback_f1,
            "train_size":   self._meta.get("train_size", 0),
            "trained_at":   self._meta.get("trained_at", "—"),
            "llm_benchmark": self._meta.get("llm_benchmark"),
        }

    def model_info(self) -> str:
        s = self.get_status()
        if s["llm_available"]:
            llm_f1 = s["active_f1"]
            fb_f1  = s["fallback_f1"]
            return f"{s['mode']} (F1={llm_f1}) | fallback={s.get('fallback_model','—')} F1={fb_f1}"
        return f"{s['mode']} | F1={s['fallback_f1']}"
