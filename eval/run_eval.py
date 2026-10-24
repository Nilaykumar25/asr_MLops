"""Experiment grid: model size x simulated sample rate -> WER, latency, RTF, confidence. Logged to MLflow.
Also answers the research question: do the confidence signals predict per-file errors?

  python -m eval.run_eval --key YOUR_KEY --manifest eval/data/manifest.csv \
         --model-sizes tiny,base,small --sample-rates 16000,8000,4000

The worker must run with MODEL_BACKEND=whisper. Manifest CSV columns: path,reference"""
import argparse
import csv
import statistics
from pathlib import Path

from eval.client import AsrClient
from eval.metrics import compute_wer


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else float("nan")


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else float("nan")


def spearman(xs, ys) -> float:
    """Rank correlation between a confidence signal and per-file WER. Negative for avg_logprob
    means: lower confidence goes with higher error, i.e. the signal is a useful proxy."""
    from scipy.stats import spearmanr
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5:
        return float("nan")
    return float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--experiment", default="asr-whisper-grid")
    ap.add_argument("--model-sizes", default="small")
    ap.add_argument("--sample-rates", default="16000")
    ap.add_argument("--dataset-tag", default="librispeech-test-clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    client = AsrClient(a.api, a.key)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    print(f"uploading {len(rows)} files ...")
    audio_ids = [client.upload(r["path"], a.dataset_tag, r["reference"])["id"] for r in rows]
    refs = [r["reference"] for r in rows]
    Path(a.out).mkdir(parents=True, exist_ok=True)

    mlflow = None
    if not a.no_mlflow:
        import mlflow
        mlflow.set_experiment(a.experiment)

    for size in a.model_sizes.split(","):
        for sr in (int(x) for x in a.sample_rates.split(",")):
            config = {"model_size": size, "simulate_sample_rate": None if sr == 16000 else sr}
            print(f"\n== model={size} sample_rate={sr} ==")
            run_ids = [client.start(aid, config) for aid in audio_ids]
            results = client.wait_all(run_ids)
            ok = [results[r] for r in run_ids if r in results and results[r]["status"] == "done"]
            hyps = [(results[r]["transcript"] or "") if r in results else "" for r in run_ids]

            # per-file table (also saved as CSV for the report)
            per_file, file_wer, logprobs, nsp = [], [], [], []
            for r, ref, hyp, rid in zip(rows, refs, hyps, run_ids):
                res = results.get(rid)
                sig = (res or {}).get("signals", {})
                w = compute_wer([ref], [hyp]) if ref.strip() else float("nan")
                file_wer.append(w)
                logprobs.append(sig.get("avg_logprob"))
                nsp.append(sig.get("no_speech_prob"))
                per_file.append({"path": r["path"], "wer": w, "avg_logprob": sig.get("avg_logprob"),
                                 "no_speech_prob": sig.get("no_speech_prob"), "rtf": sig.get("rtf"),
                                 "silence_ratio": sig.get("silence_ratio"), "input_rms": sig.get("input_rms")})
            csv_path = Path(a.out) / f"{size}_{sr}hz.csv"
            with open(csv_path, "w", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=list(per_file[0]))
                w_.writeheader()
                w_.writerows(per_file)

            metrics = {
                "wer": compute_wer(refs, hyps),
                "rtf_mean": mean([r["signals"]["rtf"] for r in ok]),
                "latency_p50_s": pct([r["timings"]["total_s"] for r in ok], 0.50),
                "latency_p95_s": pct([r["timings"]["total_s"] for r in ok], 0.95),
                "avg_logprob_mean": mean([r["signals"]["avg_logprob"] for r in ok]),
                "no_speech_prob_mean": mean([r["signals"]["no_speech_prob"] for r in ok]),
                "spearman_logprob_vs_wer": spearman(logprobs, file_wer),
                "spearman_nospeech_vs_wer": spearman(nsp, file_wer),
                "failed_runs": len(run_ids) - len(ok),
            }
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
            if mlflow:
                with mlflow.start_run(run_name=f"{size}-{sr}hz"):
                    mlflow.log_params({"model_size": size, "sample_rate_hz": sr, "n_files": len(rows),
                                       "dataset": a.dataset_tag})
                    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # skip NaN
                    mlflow.log_artifact(str(csv_path))


if __name__ == "__main__":
    main()
