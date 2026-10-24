"""Turn MLflow results into report-ready charts.
  python -m eval.plot_results --experiment asr-whisper-grid --out figures
Produces: wer_vs_sample_rate.png, wer_vs_latency.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_from_mlflow(experiment: str) -> pd.DataFrame:
    import mlflow
    df = mlflow.search_runs(experiment_names=[experiment])
    df = df.rename(columns=lambda c: c.split(".", 1)[1] if c.startswith(("params.", "metrics.")) else c)
    df["sample_rate_hz"] = df["sample_rate_hz"].astype(int)
    return df


def plot(df: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for size, g in df.groupby("model_size"):
        g = g.sort_values("sample_rate_hz")
        ax.plot(g["sample_rate_hz"] / 1000, g["wer"] * 100, marker="o", label=size)
    ax.set_xlabel("Simulated source sampling rate (kHz)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Word error rate vs sampling rate")
    ax.legend(title="Whisper model")
    ax.grid(alpha=0.3)
    f = out / "wer_vs_sample_rate.png"
    fig.tight_layout()
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f)

    clean = df[df["sample_rate_hz"] == df["sample_rate_hz"].max()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(clean["rtf_mean"], clean["wer"] * 100)
    for _, r in clean.iterrows():
        ax.annotate(r["model_size"], (r["rtf_mean"], r["wer"] * 100), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Real-time factor (lower = faster)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Accuracy vs speed trade-off (16 kHz)")
    ax.grid(alpha=0.3)
    f = out / "wer_vs_latency.png"
    fig.tight_layout()
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="asr-whisper-grid")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    for f in plot(load_from_mlflow(a.experiment), Path(a.out)):
        print("wrote", f)


if __name__ == "__main__":
    main()
