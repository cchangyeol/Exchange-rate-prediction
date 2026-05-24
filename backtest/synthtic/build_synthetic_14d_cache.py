"""
3d 캐시 3~4개를 합쳐 14d-window 캐시 엔트리 생성.

test interval=7d, 각 T에 대해 [T-14, T] 범위:
  - T의 3d 윈도우: [T-3, T]
  - T_prev (T-7) 의 3d 윈도우: [T-10, T-7]
  - T_prev2 (T-14) 의 3d 윈도우: [T-17, T-14]
  → 세 3d 창 합집합 ≈ 14d 대리 (중간 갭 있음)
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
    cur = conn.execute("SELECT id, status FROM queries WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return None
    qid, status = row
    if status != "ok":
        return None
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
    return arts


def put_synthetic(conn, keyword, start_dt, end_dt, articles, label="14d"):
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
    """, (key, keyword + f" [synthetic-{label}]",
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


def main():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    created = 0
    skipped = 0

    print(f"{'T':12} {'3d×3':8} {'14d-synth':9} status")
    print("-" * 40)

    for i, td_str in enumerate(TEST_DATES):
        td = datetime.strptime(td_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        win14_start = td - timedelta(days=14)
        win14_end   = td

        key14 = _make_key(KEYWORD, win14_start, win14_end)
        row = conn.execute("SELECT status FROM queries WHERE key=?", (key14,)).fetchone()
        if row and row[0] == "ok":
            arts, _ = get_articles_for_window(conn, KEYWORD, win14_start, win14_end), None
            existing = arts[0] if arts else []
            print(f"{td_str:12} {'—':8} {'—':9}  [already exists]")
            skipped += 1
            continue

        combined = {}

        # 3d windows for T, T-7, T-14
        for offset in [0, 7, 14]:
            t_ref = td - timedelta(days=offset)
            win3_start = t_ref - timedelta(days=3)
            arts = get_articles_for_window(conn, KEYWORD, win3_start, t_ref)
            if arts:
                for a in arts:
                    k = a.get("url") or a["title"]
                    combined[k] = a

        merged = list(combined.values())

        if not merged:
            print(f"{td_str:12} {'0+0+0':8} {'—':9}  [no source data]")
            continue

        put_synthetic(conn, KEYWORD, win14_start, win14_end, merged, label="14d")
        print(f"{td_str:12} {len(merged):8} {len(merged):9}  ✓ created")
        created += 1

    conn.close()
    print(f"\n결과: created={created}, skipped={skipped}")
    print("이제 --news-window 14 백테스트를 GDELT API 없이 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
