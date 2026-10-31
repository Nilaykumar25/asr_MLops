"""THE research question: do confidence / audio signals predict transcription errors?
Pools every per-file CSV that eval.run_eval wrote (clean + noisy conditions) and reports rank correlations.

  python -m eval.analyze_proxy --results results --out figures

Spearman rho between a signal and per-file WER:
  avg_logprob should be NEGATIVE (lower confidence goes with more errors),
  no_speech_prob and silence_ratio should be POSITIVE. |rho| near 0 means the signal is not a useful proxy."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SIGNALS = ["avg_logprob", "no_speech_prob", "silence_ratio", "input_rms"]


def load(results_dir: str) -> pd.DataFrame:
    """File names are <condition>_<model>_<rate>hz.csv; condition may itself contain underscores."""
    frames = []
    for f in sorted(Path(results_dir).glob("*.csv")):
        parts = f.stem.rsplit("_", 2)
        if len(parts) != 3 or not parts[2].endswith("hz") or not parts[2][:-2].isdigit():
            continue
        df = pd.read_csv(f)
        df["condition"], df["model_size"], df["sample_rate_hz"] = parts[0], parts[1], int(parts[2][:-2])
        frames.append(df)
    if not frames:
        raise SystemExit(f"no per-file result CSVs found in {results_dir} (run eval.run_eval first)")
    return pd.concat(frames, ignore_index=True)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    scopes = [("ALL (pooled)", df)] + [(c, g) for c, g in df.groupby("condition")]
    rows = []
    for scope, g in scopes:
        for sig in SIGNALS:
            d = g[[sig, "wer"]].dropna()
            if len(d) >= 5 and d[sig].nunique() > 1 and d["wer"].nunique() > 1:
                res = spearmanr(d[sig], d["wer"])
                rows.append({"scope": scope, "signal": sig, "n": len(d),
                             "spearman_rho": round(float(res.statistic), 3), "p_value": float(res.pvalue)})
    return pd.DataFrame(rows)


def scatter(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cond, g in df.groupby("condition"):
        ax.scatter(g["avg_logprob"], g["wer"] * 100, s=14, alpha=0.6, label=cond)
    ax.set_xlabel("avg_logprob (closer to 0 = more confident)")
    ax.set_ylabel("per-file WER (%)")
    ax.set_title("Does decoder confidence predict errors?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, title="condition")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    df = load(a.results)
    table = correlations(df)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(a.results) / "proxy_correlations.csv", index=False)
    print(f"{len(df)} per-file results from {df['condition'].nunique()} condition(s)\n")
    if table.empty:
        print("No correlations computed: the signals or the WER are constant (this is normal with the stub "
              "model; it means real Whisper results are needed).")
    else:
        print(table.to_string(index=False))
    scatter(df, Path(a.out) / "proxy_scatter.png")
    print(f"\nwrote {Path(a.out) / 'proxy_scatter.png'} and {Path(a.results) / 'proxy_correlations.csv'}")


if __name__ == "__main__":
    main()
