"""
NEXUS 자동 예측 스케줄러
==========================

백그라운드 daemon 스레드가 지정된 간격마다:
  1) 미검증 예측의 outcome 검증 (T+horizon 도래한 것)
  2) 모든 활성 (ticker, horizon) 조합에 대해 새 앙상블 예측 실행
  3) 결과를 prediction_history DB에 저장

Flask app.py에서 import 시 자동 시작 (NEXUS_AUTO_REFRESH=1 일 때).
"""
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import List, Tuple


# 자동 갱신할 (ticker, horizon) 조합
ACTIVE_TARGETS: List[Tuple[str, int]] = [
    ("GBPKRW=X", 3),  # 검증된 최고 조합 (88.9%)
    ("GBPKRW=X", 1),
    ("USDKRW=X", 3),
    ("JPYKRW=X", 3),
    ("EURKRW=X", 3),
]

# 기본 갱신 간격 (초)
DEFAULT_INTERVAL_SEC = 3600  # 1시간


class PredictionScheduler:
    def __init__(self, interval_sec: int = DEFAULT_INTERVAL_SEC,
                 targets: List[Tuple[str, int]] = None,
                 hours_back: int = 24):
        self.interval_sec = interval_sec
        self.targets      = targets or ACTIVE_TARGETS
        self.hours_back   = hours_back
        self._stop_event  = threading.Event()
        self._thread:     threading.Thread = None
        self.last_run     = None
        self.last_results = []
        self.running      = False

    def start(self):
        if self._thread and self._thread.is_alive():
            print("[scheduler] 이미 실행 중")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NexusScheduler")
        self._thread.start()
        self.running = True
        print(f"[scheduler] 시작 — {self.interval_sec}s 간격, {len(self.targets)}개 타겟")

    def stop(self):
        self._stop_event.set()
        self.running = False
        print("[scheduler] 정지 신호 전송")

    def _loop(self):
        # 처음 시작 시 30초 후 첫 실행 (서버 안정화 대기)
        if self._stop_event.wait(30):
            return
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception as e:
                print(f"[scheduler ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()
            # 다음 cycle 대기 (interrupt 가능)
            if self._stop_event.wait(self.interval_sec):
                break

    def _run_once(self):
        from core.predict_ensemble import predict as ensemble_predict
        from data.prediction_history import save_prediction, verify_pending

        run_at = datetime.now(timezone.utc).isoformat()
        print(f"\n[scheduler] === cycle 시작 {run_at} ===")

        # 1) 검증 가능한 예측 검증
        try:
            n_verified = verify_pending(verbose=False)
            if n_verified > 0:
                print(f"[scheduler] 검증 완료: {n_verified}건")
        except Exception as e:
            print(f"[scheduler] 검증 실패: {e}")

        # 2) 활성 타겟에 대해 예측 실행
        results_summary = []
        for ticker, horizon in self.targets:
            if self._stop_event.is_set():
                break
            try:
                t0 = time.time()
                print(f"[scheduler] 예측: {ticker} T+{horizon} ...")
                result = ensemble_predict(
                    ticker=ticker, horizon=horizon,
                    hours_back=self.hours_back, verbose=False,
                )
                # yfinance 현재가
                rate_now = None
                try:
                    import yfinance as yf
                    rate_now = float(yf.Ticker(ticker).fast_info.last_price)
                except Exception:
                    pass
                pid = save_prediction(result, rate_at_prediction=rate_now)
                dt = time.time() - t0
                cons = result["ensemble"]["consensus"]
                print(f"  ✓ {ticker} T+{horizon}: cons={cons.get('direction')} ({dt:.1f}s, id={pid})")
                results_summary.append({
                    "ticker":   ticker, "horizon": horizon,
                    "consensus": cons.get("direction"),
                    "duration":  dt, "history_id": pid,
                })
            except Exception as e:
                print(f"  ✗ {ticker} T+{horizon} 실패: {type(e).__name__}: {e}")
                results_summary.append({
                    "ticker": ticker, "horizon": horizon, "error": str(e),
                })

        self.last_run     = run_at
        self.last_results = results_summary
        next_run = time.time() + self.interval_sec
        print(f"[scheduler] === cycle 완료 — 다음: {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')} ===\n")

    def status(self):
        return {
            "running":      self.running,
            "interval_sec": self.interval_sec,
            "targets":      [{"ticker": t, "horizon": h} for t, h in self.targets],
            "hours_back":   self.hours_back,
            "last_run":     self.last_run,
            "last_results": self.last_results,
        }


# ────────────────────────────────────────────
# 전역 싱글톤
# ────────────────────────────────────────────
_scheduler: PredictionScheduler = None


def get_scheduler() -> PredictionScheduler:
    global _scheduler
    if _scheduler is None:
        interval = int(os.environ.get("NEXUS_SCHEDULER_INTERVAL", DEFAULT_INTERVAL_SEC))
        _scheduler = PredictionScheduler(interval_sec=interval)
    return _scheduler


def auto_start_if_enabled():
    """환경변수 NEXUS_AUTO_REFRESH=1 이면 자동 시작"""
    if os.environ.get("NEXUS_AUTO_REFRESH", "").strip() in ("1", "true", "True", "TRUE"):
        sched = get_scheduler()
        sched.start()
        return sched
    return None


if __name__ == "__main__":
    # CLI: 한 번 실행만 (테스트용)
    from dotenv import load_dotenv
    load_dotenv(".env")
    sched = PredictionScheduler(interval_sec=999999)  # 1회만
    sched._run_once()
