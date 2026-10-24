import random

import pytest

pytest.importorskip("scipy")
pytest.importorskip("jiwer")

from eval.metrics import compute_wer, normalize  # noqa: E402
from monitor.drift import compute_drift  # noqa: E402


def sample(mu, n=200, seed=0):
    r = random.Random(seed)
    return [r.gauss(mu, 1.0) for _ in range(n)]


def test_no_drift_when_distributions_match():
    rep = compute_drift({"avg_logprob": sample(0, seed=1)}, {"avg_logprob": sample(0, seed=2)})
    assert not rep["avg_logprob"].drifted


def test_drift_detected_when_distribution_shifts():
    rep = compute_drift({"avg_logprob": sample(0, seed=1)}, {"avg_logprob": sample(2.0, seed=2)})
    assert rep["avg_logprob"].drifted and rep["avg_logprob"].ks > 0.5


def test_too_few_samples_is_skipped():
    assert compute_drift({"rtf": sample(0, n=5)}, {"rtf": sample(3, n=5)}) == {}


def test_normalize_and_wer():
    assert normalize("Hello,  World!  It's-fine.") == "hello world it's fine"
    assert compute_wer(["the cat sat"], ["the cat sat"]) == 0.0
    assert compute_wer(["the cat sat"], ["the bat sat"]) == pytest.approx(1 / 3)


def test_plot_results_makes_charts(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import pandas as pd
    from eval.plot_results import plot
    df = pd.DataFrame([
        {"model_size": m, "sample_rate_hz": sr, "wer": w, "rtf_mean": r}
        for m, r, base in (("tiny", 0.1, 0.08), ("small", 0.4, 0.04))
        for sr, w in ((16000, base), (8000, base * 1.4), (4000, base * 2.5))
    ])
    files = plot(df, tmp_path)
    assert all(f.exists() and f.stat().st_size > 1000 for f in files)
