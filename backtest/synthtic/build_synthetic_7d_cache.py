"""
기존 3d-window 캐시 두 개를 합쳐 7d-window 캐시 엔트리를 생성.

7d 백테스트 (--interval 7d --news-window 7):
  T = 2026-02-08 → 필요 범위: [2026-02-01, 2026-02-08]
  이미 캐시에 있는 것:
    [2026-01-29, 2026-02-01] (T_prev=2026-02-01 의 3d 윈도우)
    [2026-02-05, 2026-02-08] (T=2026-02-08 의 3d 윈도우)
  합치면: 두 3d 창에서 모은 기사 → synthetic [2026-02-01, 2026-02-08] 엔트리 생성

결과: GDELT API 호출 없이 7d 백테스트 실행 가능.
"""
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta, timezone

# DB 는 프로젝트 루트의 data/ 디렉토리 (backtest/synthetic/ → ../../data/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
CACHE_PATH = os.path.join(_ROOT, "data", "gdelt_cache.db")
KEYWORD = '"South Korea" (won OR KRW OR currency OR economy OR exchange)'

TEST_DATES = [
    "2026-02-01", "2026-02-08", "2026-02-15", "2026-02-22",
    "2026-03-01", "2026-03-08", "2026-03-15", "2026-03-22",
    "2026-03-29", "2026-04-05", "2026-04-12", "2026-04-19", "2026-04-26",
]


def _make_key(keyword, start_dt, end_dt):
    s = f"{keyword}::{start_dt.strftime('%Y%m%d%H%M')}::{end_dt.strftime('%Y%m%d%H%M')}"
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def get_articles_for_window(conn, keyword, start_dt, end_dt):
    key = _make_key(keyword, start_dt, end_dt)
    cur = conn.execute("SELECT id, status, article_count FROM queries WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return None, None
    qid, status, cnt = row
    if status != "ok":
        return None, None
    cur = conn.execute(
        "SELECT title, url, source, published_at, time_weight FROM articles WHERE query_id=?",
        (qid,)
    )
    arts = []
    for title, url, source, pub, tw in cur.fetchall():
        arts.append({
            "title": title, "url": url or "", "source": source or "",
            "published_at": pub, "time_weight": tw or 0.0,
        })
    return arts, cnt


def put_synthetic(conn, keyword, start_dt, end_dt, articles):
    key = _make_key(keyword, start_dt, end_dt)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO queries(key,keyword,start_dt,end_dt,fetched_at,status,article_count,error)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(key) DO UPDATE SET
            fetched_at=excluded.fetched_at,
            status=excluded.status,
            article_count=excluded.article_count,
            error=excluded.error
    """, (key, keyword + " [synthetic-7d]",
          start_dt.isoformat(), end_dt.isoformat(),
          now, "ok", len(articles), ""))
    cur = conn.execute("SELECT id FROM queries WHERE key=?", (key,))
    qid = cur.fetchone()[0]
    conn.execute("DELETE FROM articles WHERE query_id=?", (qid,))
    for a in articles:
        conn.execute("""
            INSERT INTO articles(query_id,title,url,source,published_at,time_weight)
            VALUES (?,?,?,?,?,?)
        """, (qid, a["title"], a["url"], a["source"], a["published_at"], a["time_weight"]))
    conn.commit()
    return qid


def main():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    created = 0
    skipped = 0
    missing = 0

    print(f"{'T':12} {'3d-prev':8} {'3d-curr':8} {'7d-synth':9} status")
    print("-" * 55)

    for i, td_str in enumerate(TEST_DATES):
        td = datetime.strptime(td_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # 7d 윈도우: [T-7, T]
        win7_start = td - timedelta(days=7)
        win7_end   = td

        # 이미 있으면 스킵
        key7 = _make_key(KEYWORD, win7_start, win7_end)
        row = conn.execute("SELECT status FROM queries WHERE key=?", (key7,)).fetchone()
        if row and row[0] == "ok":
            arts7, _ = get_articles_for_window(conn, KEYWORD, win7_start, win7_end)
            print(f"{td_str:12} {'—':8} {'—':8} {len(arts7 or []):9}  [already exists]")
            skipped += 1
            continue

        # T의 3d 윈도우 ([T-3, T]) 로드
        win3_curr_start = td - timedelta(days=3)
        arts_curr, _ = get_articles_for_window(conn, KEYWORD, win3_curr_start, td)

        # T_prev (= T-7) 의 3d 윈도우 ([T-10, T-7]) 로드
        if i > 0:
            t_prev = datetime.strptime(TEST_DATES[i-1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            win3_prev_start = t_prev - timedelta(days=3)
            arts_prev, _ = get_articles_for_window(conn, KEYWORD, win3_prev_start, t_prev)
        else:
            arts_prev = None

        n_curr = len(arts_curr) if arts_curr else 0
        n_prev = len(arts_prev) if arts_prev else 0

        if n_curr == 0 and n_prev == 0:
            print(f"{td_str:12} {n_prev:8} {n_curr:8} {'—':9}  [no source data]")
            missing += 1
            continue

        # 합치기 (URL로 중복 제거)
        combined = {}
        for a in (arts_prev or []):
            key = a.get("url") or a["title"]
            combined[key] = a
        for a in (arts_curr or []):
            key = a.get("url") or a["title"]
            combined[key] = a
        merged = list(combined.values())

        put_synthetic(conn, KEYWORD, win7_start, win7_end, merged)
        print(f"{td_str:12} {n_prev:8} {n_curr:8} {len(merged):9}  ✓ created")
        created += 1

    conn.close()
    print(f"\n결과: created={created}, skipped={skipped}, missing={missing}")
    print("이제 --news-window 7 백테스트를 GDELT API 없이 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
