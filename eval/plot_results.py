"""Turn MLflow results into report-ready charts.
  python -m eval.plot_results --experiment asr-whisper-grid --out figures
Produces: wer_vs_sample_rate.png, wer_vs_latency.png (clean test set only) and, when noisy
conditions are present in the experiment, wer_vs_snr.png.
Tip: log the clean and noisy runs to the SAME experiment (the default) so they appear together."""
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
    if "condition" not in df:
        df = df.assign(condition="clean")
    clean_df = df[df["condition"] == "clean"]
    if clean_df.empty:
        clean_df = df

    fig, ax = plt.subplots(figsize=(6, 4))
    for size, g in clean_df.groupby("model_size"):
        g = g.sort_values("sample_rate_hz")
        ax.plot(g["sample_rate_hz"] / 1000, g["wer"] * 100, marker="o", label=size)
    ax.set_xlabel("Simulated source sampling rate (kHz)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Word error rate vs sampling rate (clean speech)")
    ax.legend(title="Whisper model")
    ax.grid(alpha=0.3)
    f = out / "wer_vs_sample_rate.png"
    fig.tight_layout()
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f)

    best = clean_df[clean_df["sample_rate_hz"] == clean_df["sample_rate_hz"].max()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(best["rtf_mean"], best["wer"] * 100)
    for _, r in best.iterrows():
        ax.annotate(r["model_size"], (r["rtf_mean"], r["wer"] * 100), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Real-time factor (lower = faster)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Accuracy vs speed trade-off (16 kHz, clean)")
    ax.grid(alpha=0.3)
    f = out / "wer_vs_latency.png"
    fig.tight_layout()
    fig.savefig(f, dpi=200)
    plt.close(fig)
    files.append(f)

    # ---- WER against noise level (only when noisy conditions were evaluated) ----
    if "snr_db" in df and df["condition"].nunique() > 1:
        d = df[df["sample_rate_hz"] == 16000].copy()
        d["snr_num"] = pd.to_numeric(d["snr_db"], errors="coerce")       # "clean" -> NaN
        levels = sorted(d["snr_num"].dropna().unique(), reverse=True)    # e.g. 20, 10, 5, 0
        if levels:
            labels = ["clean"] + [f"{l:g} dB" for l in levels]
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            noisy = d[d["snr_num"].notna()]
            for (size, noise), g in noisy.groupby(["model_size", "noise"]):
                clean_wer = d[(d["snr_num"].isna()) & (d["model_size"] == size)]["wer"].mean()
                ys = [clean_wer] + [g[g["snr_num"] == l]["wer"].mean() for l in levels]
                ax.plot(range(len(ys)), [y * 100 for y in ys], marker="o", label=f"{size} / {noise}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlabel("Signal-to-noise ratio (lower = noisier)")
            ax.set_ylabel("WER (%)")
            ax.set_title("Word error rate vs noise level (16 kHz)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            f = out / "wer_vs_snr.png"
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
