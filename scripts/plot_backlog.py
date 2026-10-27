"""Backlog over time, computed from the runs table. Run it BEFORE truncating the table.
  python -m scripts.plot_backlog --out figures/stageC_backlog.png
Top panel: arrival rate vs completion rate (runs per second). Bottom panel: runs waiting or in progress."""
import argparse
import bisect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db.models import Run  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402


def rate_series(times: list[float], end: int, window: int = 10) -> list[float]:
    """Events per second, averaged over a sliding window."""
    out = []
    for x in range(end + 1):
        lo = max(0, x - window)
        n = bisect.bisect_right(times, x) - bisect.bisect_right(times, lo)
        out.append(n / max(1, x - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/backlog.png")
    args = ap.parse_args()

    with SessionLocal() as db:
        rows = db.execute(select(Run.queued_at, Run.finished_at)).all()
    if not rows:
        raise SystemExit("no runs in the database (did you truncate it already?)")

    t0 = min(r.queued_at for r in rows)
    arrivals = sorted((r.queued_at - t0).total_seconds() for r in rows)
    finished = sorted((r.finished_at - t0).total_seconds() for r in rows if r.finished_at)
    end = int(max(arrivals[-1], finished[-1])) + 1
    xs = list(range(end + 1))
    backlog = [bisect.bisect_right(arrivals, x) - bisect.bisect_right(finished, x) for x in xs]
    arr_rate, fin_rate = rate_series(arrivals, end), rate_series(finished, end)

    load_end = arrivals[-1]
    done_during_load = bisect.bisect_right(finished, load_end)
    drain_time = max(1e-9, finished[-1] - load_end)
    print(f"runs                         : {len(arrivals)}")
    print(f"load phase                   : {load_end:.0f} s")
    print(f"mean arrival rate            : {len(arrivals) / max(load_end, 1e-9):.1f} runs/s")
    print(f"mean completion rate (load)  : {done_during_load / max(load_end, 1e-9):.1f} runs/s")
    print(f"drain rate after load        : {(len(finished) - done_during_load) / drain_time:.1f} runs/s")
    print(f"peak backlog                 : {max(backlog)} runs at t = {backlog.index(max(backlog))} s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(xs, arr_rate, label="arrivals")
    a1.plot(xs, fin_rate, label="completions")
    a1.set_ylabel("runs per second")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(xs, backlog, color="tab:red")
    a2.set_ylabel("backlog (queued + processing)")
    a2.set_xlabel("seconds since first request")
    a2.grid(alpha=0.3)
    fig.suptitle("Queue behaviour under load")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()