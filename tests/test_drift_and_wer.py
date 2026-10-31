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


def test_mix_at_snr_hits_the_target():
    np = pytest.importorskip("numpy")
    from eval.make_noisy import mix_at_snr, noise_segment
    rng = np.random.default_rng(0)
    speech = (0.3 * np.sin(np.linspace(0, 400, 16000 * 3))).astype("float32")
    noise = noise_segment(rng.normal(0, 0.2, 8000).astype("float32"), len(speech), rng)  # short noise is looped
    for target in (20, 10, 5, 0):
        noisy = mix_at_snr(speech, noise, target)
        # measure SNR in a way that ignores overall loudness (the mix is scaled down if it would clip):
        # split noisy into (a * speech) + residual, then SNR = a^2 * P_speech / P_residual
        a = float(np.dot(noisy, speech) / np.dot(speech, speech))
        residual = noisy - a * speech
        measured = 10 * np.log10((a ** 2) * np.mean(speech ** 2) / np.mean(residual ** 2))
        assert abs(measured - target) < 0.3


def test_plot_results_with_noise_conditions(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import pandas as pd
    from eval.plot_results import plot
    rows = []
    for m, base in (("base", 0.06), ("small", 0.04)):
        rows.append({"model_size": m, "sample_rate_hz": 16000, "wer": base, "rtf_mean": 0.2,
                     "condition": "clean", "noise": "none", "snr_db": "clean"})
        for snr, mult in ((20, 1.1), (10, 1.6), (5, 2.5), (0, 5.0)):
            rows.append({"model_size": m, "sample_rate_hz": 16000, "wer": base * mult, "rtf_mean": 0.2,
                         "condition": f"cafeteria_{snr}db", "noise": "cafeteria", "snr_db": str(float(snr))})
    files = plot(pd.DataFrame(rows), tmp_path)
    assert any(f.name == "wer_vs_snr.png" and f.stat().st_size > 1000 for f in files)


def test_analyze_proxy_finds_a_negative_correlation(tmp_path):
    pytest.importorskip("pandas")
    import numpy as np
    import pandas as pd
    from eval.analyze_proxy import correlations, load, scatter
    rng = np.random.default_rng(0)
    for name in ("clean_base_16000hz", "cafeteria_5db_base_16000hz"):
        lp = rng.uniform(-1.2, -0.1, 60)
        wer = np.clip(-lp * 0.3 + rng.normal(0, 0.03, 60), 0, 1)          # lower confidence -> more errors
        pd.DataFrame({"path": range(60), "wer": wer, "avg_logprob": lp, "no_speech_prob": rng.uniform(0, 0.1, 60),
                      "rtf": 0.1, "silence_ratio": rng.uniform(0, 0.3, 60), "input_rms": rng.uniform(0.02, 0.1, 60)}
                     ).to_csv(tmp_path / f"{name}.csv", index=False)
    df = load(str(tmp_path))
    assert set(df["condition"]) == {"clean", "cafeteria_5db"}            # underscores in the condition survive
    table = correlations(df)
    rho = table[(table["scope"] == "ALL (pooled)") & (table["signal"] == "avg_logprob")]["spearman_rho"].iloc[0]
    assert rho < -0.8
    scatter(df, tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000
