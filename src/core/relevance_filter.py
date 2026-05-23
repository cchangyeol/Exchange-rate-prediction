"""
뉴스 관련성 pre-filter: Groq 8B로 각 기사가 환율 예측에 유용한지 0~10 점수 매김.
상위 K개만 LLM 분석에 넘겨 노이즈를 줄임.

사용:
  from core.relevance_filter import filter_relevant
  filtered = filter_relevant(news_list, top_k=10)
"""
import os
import time
import json
from typing import List, Dict

RELEVANCE_PROMPT = """You are a financial analyst specializing in Korean Won (KRW) exchange rate prediction.

Score each news article for its relevance to predicting KRW exchange rate movements (0-10 scale):
- 10: Directly mentions KRW rate, Bank of Korea policy, Korea trade data, major geopolitical event affecting Korea
- 7-9: Indirect but strong signal: US Fed decisions, DXY moves, major Asia-Pacific economic data
- 4-6: Moderate relevance: global trade news, commodity prices, general risk sentiment
- 1-3: Weak relevance: general business news, minor political news
- 0: Completely irrelevant to KRW

Return JSON only:
{"scores": [{"idx": 0, "score": 8}, {"idx": 1, "score": 3}, ...]}
"""


HIGH_RELEVANCE = [
    # 영문 — 직접 환율 키워드
    "exchange rate", "fx rate", "krw", "won", "forex", "fx ",
    "bank of korea", "bok", "federal reserve", "fed",
    "interest rate", "rate hike", "rate cut", "rate decision", "monetary policy",
    "trade balance", "current account", "inflation", "cpi", "ppi",
    "dollar", "usd", "tariff", "trade war",
    "yuan", "yen", "pound", "euro",
    # 한글 — 직접 환율 키워드
    "환율", "원달러", "원·달러", "원/달러", "외환", "외화", "통화",
    "기준금리", "금리 인상", "금리 인하", "한국은행", "한은", "총재",
    "연준", "fomc", "연방준비",
    "무역수지", "수출", "수입", "관세",
    "달러", "엔화", "위안", "파운드", "유로",
]
MEDIUM_RELEVANCE = [
    # 영문
    "korea", "korean", "economy", "economic", "kospi", "kosdaq",
    "bond", "yield", "treasury", "oil", "energy", "commodity",
    "china", "japan", "europe", "global", "samsung", "hyundai",
    "geopolit", "sanction", "recession", "growth",
    # 한글
    "경제", "금융", "시장", "증시", "주식", "코스피", "코스닥",
    "수익률", "국채", "원유", "에너지",
    "중국", "일본", "미국", "유럽", "삼성", "현대",
    "지정학", "제재", "경기침체", "성장",
]
# 환율과 무관한 기사 패널티 (정치, 사건/사고, 연예 등)
NEGATIVE_KEYWORDS = [
    "유상증자", "제3자배정", "최종합격", "선발시험",
    "헤드라인", "보도자료",
    "5·18", "선거", "범죄", "사건",
    "야구", "축구", "스포츠", "연예", "방송",
]


def _keyword_score(title: str, text: str = "") -> float:
    """키워드 기반 관련성 점수 (-10~10, 음수면 환율 무관)"""
    combined = (title + " " + text).lower()
    score = 0.0
    for kw in HIGH_RELEVANCE:
        if kw.lower() in combined:
            score += 2.0
    for kw in MEDIUM_RELEVANCE:
        if kw.lower() in combined:
            score += 0.5
    for kw in NEGATIVE_KEYWORDS:
        if kw.lower() in combined:
            score -= 3.0
    return max(-10.0, min(score, 10.0))


def filter_relevant(news_list: List[Dict], top_k: int = 10,
                    groq_api_key: str = None, throttle: float = 2.0) -> List[Dict]:
    """
    뉴스 목록에서 상위 top_k개 관련 기사만 반환.
    키워드 기반 scoring으로 API 호출 없이 동작.
    """
    if not news_list:
        return news_list

    scored = [
        (_keyword_score(a.get("title", ""), a.get("text", "")), i, a)
        for i, a in enumerate(news_list)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [a for _, _, a in scored[:top_k]]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dotenv import load_dotenv
    from data.news_collector import collect_news_range
    from datetime import datetime, timedelta, timezone

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_PROJECT_ROOT / ".env")

    td = datetime(2026, 4, 12, tzinfo=timezone.utc)
    news_data = collect_news_range(td - timedelta(days=3), td, max_total=45)
    articles = news_data["articles"]
    print(f"원본: {len(articles)} 기사")

    filtered = filter_relevant(articles, top_k=10)
    print(f"필터 후: {len(filtered)} 기사 (top-10)\n")
    for i, a in enumerate(filtered):
        print(f"  {i+1}. {a.get('title','')[:80]}")
