"""Drift monitor. Runs as its own process so heavy statistics never slow down inference.

Reference window : finished runs whose audio has dataset_tag == DRIFT_REFERENCE_TAG ("baseline")
Current window   : runs finished in the last DRIFT_WINDOW_MINUTES (any tag)
Test             : two-sample Kolmogorov-Smirnov per signal; drift = significant AND big enough effect
Exports gauges on :9102/metrics for Prometheus.

  python -m monitor.drift"""
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from prometheus_client import Gauge, start_http_server
from scipy.stats import ks_2samp
from sqlalchemy import select

from app.db.models import Audio, Run, RunStatus
from app.db.session import SessionLocal

log = logging.getLogger("asr.drift")

FEATURES = ["avg_logprob", "no_speech_prob", "rtf", "input_rms", "silence_ratio"]
REF_TAG = os.environ.get("DRIFT_REFERENCE_TAG", "baseline")
WINDOW_MIN = int(os.environ.get("DRIFT_WINDOW_MINUTES", 15))
ALPHA = float(os.environ.get("DRIFT_ALPHA", 0.01))
MIN_EFFECT = float(os.environ.get("DRIFT_MIN_KS", 0.3))
MIN_SAMPLES = int(os.environ.get("DRIFT_MIN_SAMPLES", 30))
INTERVAL_S = int(os.environ.get("DRIFT_INTERVAL_SECONDS", 30))


@dataclass
class FeatureDrift:
    ks: float
    p_value: float
    drifted: bool


def compute_drift(ref: dict[str, list[float]], cur: dict[str, list[float]], alpha=ALPHA,
                  min_effect=MIN_EFFECT, min_samples=MIN_SAMPLES) -> dict[str, FeatureDrift]:
    """Pure function (easy to unit-test). Features with too few samples are skipped."""
    out = {}
    for f in FEATURES:
        a, b = ref.get(f, []), cur.get(f, [])
        if len(a) < min_samples or len(b) < min_samples:
            continue
        res = ks_2samp(a, b)
        out[f] = FeatureDrift(float(res.statistic), float(res.pvalue),
                              bool(res.pvalue < alpha and res.statistic >= min_effect))
    return out


def _collect(runs: list[Run]) -> dict[str, list[float]]:
    return {f: [getattr(r, f) for r in runs if getattr(r, f) is not None] for f in FEATURES}


def load_windows(db):
    base = select(Run).join(Audio, Audio.id == Run.audio_id).where(Run.status == RunStatus.done)
    ref = list(db.scalars(base.where(Audio.dataset_tag == REF_TAG)
                          .order_by(Run.finished_at.desc()).limit(500)))
    since = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MIN)
    cur = list(db.scalars(base.where(Run.finished_at >= since)))
    return ref, cur


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    g_ks = Gauge("asr_drift_ks", "KS statistic vs reference window", ["feature"])
    g_p = Gauge("asr_drift_pvalue", "KS p-value vs reference window", ["feature"])
    g_flag = Gauge("asr_drift_detected", "1 if any signal drifted")
    g_n = Gauge("asr_drift_samples", "Runs in each window", ["window"])
    start_http_server(9102)
    log.info("drift monitor on :9102 (reference tag=%s, window=%d min)", REF_TAG, WINDOW_MIN)

    while True:
        try:
            with SessionLocal() as db:
                ref_runs, cur_runs = load_windows(db)
            g_n.labels("reference").set(len(ref_runs))
            g_n.labels("current").set(len(cur_runs))
            report = compute_drift(_collect(ref_runs), _collect(cur_runs))
            for f, d in report.items():
                g_ks.labels(f).set(d.ks)
                g_p.labels(f).set(d.p_value)
            g_flag.set(1 if any(d.drifted for d in report.values()) else 0)
            log.info("ref=%d cur=%d drifted=%s", len(ref_runs), len(cur_runs),
                     [f for f, d in report.items() if d.drifted])
        except Exception:
            log.exception("drift check failed")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
