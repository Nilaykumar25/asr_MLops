"""After a load test: wait for the queue to drain, then prove nothing was lost.
  python -m scripts.verify_drain --timeout 1800
Prints PASS only if every run reached done/failed (none stuck queued/processing)."""
import argparse
import time
from collections import Counter

from sqlalchemy import select

from app.db import repositories as repo
from app.db.models import Run, RunStatus
from app.db.session import SessionLocal


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    t0 = time.time()
    while True:
        with SessionLocal() as db:
            counts = repo.status_counts(db)
        active = counts.get("queued", 0) + counts.get("processing", 0)
        print(f"[{time.time() - t0:6.0f}s] {counts}")
        if active == 0 or time.time() - t0 > args.timeout:
            break
        time.sleep(10)

    with SessionLocal() as db:
        runs = list(db.scalars(select(Run)))
    done = [r for r in runs if r.status == RunStatus.done]
    qw = [(r.started_at - r.queued_at).total_seconds() for r in done if r.started_at]
    pr = [(r.finished_at - r.started_at).total_seconds() for r in done if r.finished_at and r.started_at]
    tot = [(r.finished_at - r.queued_at).total_seconds() for r in done if r.finished_at]
    failed = [r for r in runs if r.status == RunStatus.failed]

    print("\n=== RESULT ===")
    print(f"total runs        : {len(runs)}")
    print(f"done / failed     : {len(done)} / {len(failed)}")
    print(f"retried (attempts>1): {sum(1 for r in runs if r.attempts > 1)}")
    for name, v in (("queue wait", qw), ("processing", pr), ("total latency", tot)):
        print(f"{name:14s}: p50={pct(v, .5):.2f}s  p95={pct(v, .95):.2f}s  max={max(v) if v else float('nan'):.2f}s")
    if failed:
        print("failure reasons:", Counter((r.error or "")[:60] for r in failed).most_common(3))
    stuck = len(runs) - len(done) - len(failed)
    if not runs:
        print("NO RUNS FOUND: the load test did not submit anything (check --host and API keys)")
    elif stuck == 0:
        print("PASS: every run reached a terminal state")
    else:
        print(f"FAIL: {stuck} run(s) still active")


if __name__ == "__main__":
    main()
