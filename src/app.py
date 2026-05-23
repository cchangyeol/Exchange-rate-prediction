"""
NEXUS — 환율 변동 예측 시스템
Flask 백엔드

실행:
  pip install -r requirements.txt
  cp .env.example .env  # GROQ_API_KEY 등 입력
  python src/app.py  →  http://127.0.0.1:5050
"""

import os
import sys
import random
from datetime import datetime

# ── 경로 상수 ─────────────────────────────────────────────────
# src/app.py 기준:  BASE_DIR = .../src,  PROJECT_ROOT = .../
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEB_DIR      = os.path.join(PROJECT_ROOT, "web")
DELIV_DIR    = os.path.join(PROJECT_ROOT, "deliverables")

# core/ 및 data/ 패키지 import 를 위해 src/ 를 모듈 경로에 추가
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# .env 자동 로드 (프로젝트 루트의 .env)
try:
    from dotenv import load_dotenv
    if load_dotenv(os.path.join(PROJECT_ROOT, ".env")):
        print("[OK] .env 로드 완료")
except ImportError:
    pass  # python-dotenv 없어도 export로 설정한 경우 동작

# 정적/템플릿 폴더는 프로젝트 루트의 web/ 디렉토리
app = Flask(__name__, template_folder=WEB_DIR, static_folder=WEB_DIR)
CORS(app)

try:
    import yfinance as yf
    YFINANCE_OK = True
    print("[OK] yfinance 실시간 모드")
except ImportError:
    YFINANCE_OK = False
    print("[WARN] yfinance 없음 → 데모 모드")

from core.analyzer import RateAnalyzer
analyzer = RateAnalyzer()

# ── 환율 정의 ─────────────────────────────────────────────────
EXCHANGE_RATES = [
    {"ticker": "USDKRW=X", "pair": "USD/KRW", "base": "USD", "flag": "🇺🇸", "name": "미국 달러"},
    {"ticker": "JPYKRW=X", "pair": "JPY/KRW", "base": "JPY", "flag": "🇯🇵", "name": "일본 엔"},
    {"ticker": "CNYKRW=X", "pair": "CNY/KRW", "base": "CNY", "flag": "🇨🇳", "name": "중국 위안"},
    {"ticker": "EURKRW=X", "pair": "EUR/KRW", "base": "EUR", "flag": "🇪🇺", "name": "유로"},
    {"ticker": "GBPKRW=X", "pair": "GBP/KRW", "base": "GBP", "flag": "🇬🇧", "name": "영국 파운드"},
]

# ── 데모 시세 ─────────────────────────────────────────────────
_BASE = {
    "USDKRW=X": 1340.5, "JPYKRW=X": 9.12,
    "CNYKRW=X": 185.3,  "EURKRW=X": 1450.2, "GBPKRW=X": 1698.7,
}

def _demo(ticker):
    base = _BASE.get(ticker, 100)
    pct  = round(random.uniform(-1.5, 1.5), 2)
    p    = round(base * (1 + pct / 100), 2)
    return {
        "price": p, "change": round(p - base, 2),
        "change_pct": pct, "high": round(p * 1.008, 2),
        "low": round(p * 0.992, 2), "demo": True
    }

def _real(ticker):
    try:
        info  = yf.Ticker(ticker).fast_info
        price = round(float(info.last_price), 2)
        prev  = round(float(info.previous_close), 2)
        chg   = round(price - prev, 2)
        pct   = round(chg / prev * 100, 2) if prev else 0
        high  = float(info.day_high) if info.day_high is not None else price
        low   = float(info.day_low)  if info.day_low  is not None else price
        return {
            "price": price, "change": chg, "change_pct": pct,
            "high": round(high, 2), "low":  round(low, 2), "demo": False
        }
    except Exception as e:
        print(f"[WARN] {ticker} 시세 조회 실패: {e} → 데모값")
        return _demo(ticker)

def quote(ticker):
    return _real(ticker) if YFINANCE_OK else _demo(ticker)


# ── 라우트 ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

# CBD 표준 산출물 문서 (정적 HTML)
# 라우트는 호환을 위해 /docs/* 그대로 두고, 실 파일은 deliverables/ 에서 서빙
@app.route("/docs")
def docs_root():
    from flask import redirect
    return redirect("/docs/cbd/index.html")

@app.route("/docs/<path:filename>")
def docs_static(filename):
    from flask import send_from_directory
    return send_from_directory(DELIV_DIR, filename)

@app.route("/api/rates")
def api_rates():
    data = [{**r, **quote(r["ticker"])} for r in EXCHANGE_RATES]
    return jsonify({
        "data":    data,
        "updated": datetime.now().strftime("%H:%M:%S"),
        "demo":    not YFINANCE_OK
    })

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json()
    if not d:
        return jsonify({"error": "요청 본문이 없습니다"}), 400

    news_list = d.get("news_list")
    if news_list and len(news_list) > 1:
        result = analyzer.analyze("", "", lang=d.get("lang", "ko"), news_list=news_list)
    else:
        text = d.get("news_text", "").strip()
        if not text:
            return jsonify({"error": "뉴스 텍스트를 입력하세요"}), 400
        result = analyzer.analyze(
            d.get("news_title", ""), text, lang=d.get("lang", "ko")
        )
    return jsonify(result)

@app.route("/api/news/collect")
def api_news_collect():
    from data.news_collector import collect_news, demo_news, FEEDPARSER_OK, parse_period
    # period 우선, 없으면 hours
    period = request.args.get("period")
    if period:
        hours = parse_period(period)
    else:
        try:
            hours = int(request.args.get("hours", 3))
        except (TypeError, ValueError):
            hours = 3
    # 안전 상한 1년
    hours = max(1, min(24 * 365, hours))
    try:
        max_total = int(request.args.get("max_total", 20))
    except (TypeError, ValueError):
        max_total = 20
    max_total = max(1, min(500, max_total))
    keyword = request.args.get("keyword", "")
    if not FEEDPARSER_OK:
        data = demo_news(hours=hours)
        data["warning"] = "feedparser 미설치 — 데모 데이터. pip install feedparser 후 재시작하세요."
    else:
        data = collect_news(hours=hours, max_total=max_total, keyword=keyword, period=period)
    return jsonify(data)

@app.route("/api/news/analyze", methods=["POST"])
def api_news_analyze():
    d = request.get_json()
    if not d or not d.get("articles"):
        return jsonify({"error": "articles 필드가 필요합니다"}), 400
    news_list = [
        {"title": a.get("title", ""), "text": a.get("text", ""),
         "weight": a.get("normalized_weight", a.get("time_weight", 1.0))}
        for a in d["articles"]
    ]
    result = analyzer.analyze("", "", lang=d.get("lang", "ko"), news_list=news_list)
    return jsonify(result)

@app.route("/api/model/status")
def api_model_status():
    return jsonify(analyzer.get_status())


# ── 앙상블 예측 (predict 페이지의 🎯 앙상블 탭에서 사용) ─────
@app.route("/api/ensemble/predict", methods=["POST"])
def api_ensemble_predict():
    """
    백테스트 81.8% / 88.9% 두 전략 앙상블 예측.
    POST { "ticker": "GBPKRW=X", "horizon": 3, "hours_back": 24 }
    """
    d = request.get_json() or {}
    ticker     = d.get("ticker",     "GBPKRW=X")
    horizon    = int(d.get("horizon", 3))
    hours_back = int(d.get("hours_back", 24))

    try:
        from core.predict_ensemble import predict as ensemble_predict, TICKER_LABELS
        if ticker not in TICKER_LABELS:
            return jsonify({"error": f"지원 통화: {list(TICKER_LABELS.keys())}"}), 400
        result = ensemble_predict(
            ticker=ticker, horizon=horizon,
            hours_back=hours_back, verbose=False,
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# 메모리에 최근 수집된 뉴스 캐시 (모달 본문 표시용)
_LAST_NEWS_CACHE = {"hours": None, "data": None, "fetched_at": None}


@app.route("/api/ensemble/news_by_source")
def api_ensemble_news_by_source():
    """특정 source(빈 값이면 전체)의 기사를 모달용으로 반환 (캐시 우선, 없으면 새로 수집)"""
    source = request.args.get("source", "").strip()
    hours_back = int(request.args.get("hours_back", 24))
    try:
        max_total = int(request.args.get("max", 50))
    except (TypeError, ValueError):
        max_total = 50
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cache = _LAST_NEWS_CACHE
        fresh = False
        if cache["data"] and cache["hours"] == hours_back and cache["fetched_at"]:
            age = _dt.now(_tz.utc) - cache["fetched_at"]
            fresh = age < _td(minutes=10)
        if not fresh:
            from data.news_multisource import collect_all
            data = collect_all(hours_back=hours_back, verbose=False)
            _LAST_NEWS_CACHE.update({
                "hours": hours_back, "data": data,
                "fetched_at": _dt.now(_tz.utc),
            })
        else:
            data = cache["data"]

        if not source:
            # 전체 기사 (정밀 모드 결과 영역에서 사용)
            articles = data["articles"][:max_total]
        else:
            # 특정 source 매칭
            articles = [
                a for a in data["articles"]
                if a.get("source") == source or a.get("source", "").startswith(source.split("/")[0])
            ][:max_total]
        return jsonify({"source": source or "전체", "articles": articles, "total": len(articles)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    # macOS AirPlay Receiver가 기본으로 5000을 점유하므로 환경변수로 변경 가능
    port = int(os.environ.get("NEXUS_PORT", "5050"))
    print("=" * 50)
    print(f"  NEXUS  →  http://127.0.0.1:{port}")
    print(f"  yfinance : {'실시간' if YFINANCE_OK else '데모'}")
    print(f"  분석 모드 : {analyzer.model_info()}")
    print("=" * 50)
    app.run(debug=True, port=port, use_reloader=False)
