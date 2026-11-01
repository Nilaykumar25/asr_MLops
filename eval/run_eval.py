"""Experiment grid: model size x simulated sample rate -> WER, latency, RTF, confidence. Logged to MLflow.
Also answers the research question: do the confidence signals predict per-file errors?

  python -m eval.run_eval --key YOUR_KEY --manifest eval/data/manifest.csv --language en \
         --model-sizes tiny,base,small --sample-rates 16000,8000,4000

The worker must run with MODEL_BACKEND=whisper. Manifest CSV columns: path,reference.
Manifests made by eval.make_noisy also carry noise and snr_db columns; they are picked up automatically
and the experiment is labelled e.g. "cafeteria_5db" (clean manifests are labelled "clean")."""
import argparse
import csv
import statistics
from pathlib import Path

from eval.client import AsrClient
from eval.metrics import bootstrap_wer_ci, compute_wer, word_errors


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
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and y == y]
    if len(pairs) < 5:
        return float("nan")
    return float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)


def detect_condition(rows: list[dict], override: str | None):
    """Label for this test set: from the manifest's noise/snr_db columns, or 'clean'."""
    if override:
        return override, "custom", None
    if rows and "noise" in rows[0] and "snr_db" in rows[0] and rows[0]["snr_db"] != "":
        snr = float(rows[0]["snr_db"])
        return f"{rows[0]['noise']}_{int(snr)}db", rows[0]["noise"], snr
    return "clean", "none", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--experiment", default="asr-whisper-grid")
    ap.add_argument("--model-sizes", default="small")
    ap.add_argument("--sample-rates", default="16000")
    ap.add_argument("--language", help="force a language (en, hi, ...). Strongly recommended for single-language "
                                       "data: auto-detection on noisy chunks can pick the wrong language")
    ap.add_argument("--condition", help="override the automatic label of this test set")
    ap.add_argument("--dataset-tag", default="eval")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    client = AsrClient(a.api, a.key)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    condition, noise, snr = detect_condition(rows, a.condition)
    print(f"test set: {len(rows)} files, condition = {condition}")
    print("uploading ...")
    uploads = [client.upload(r["path"], f"{a.dataset_tag}-{condition}", r["reference"]) for r in rows]
    audio_ids = [u["id"] for u in uploads]
    durations = [u["duration_seconds"] for u in uploads]
    refs = [r["reference"] for r in rows]
    Path(a.out).mkdir(parents=True, exist_ok=True)

    mlflow = None
    if not a.no_mlflow:
        import mlflow
        mlflow.set_experiment(a.experiment)

    for size in a.model_sizes.split(","):
        for sr in (int(x) for x in a.sample_rates.split(",")):
            config = {"model_size": size, "simulate_sample_rate": None if sr == 16000 else sr}
            if a.language:
                config["language"] = a.language
            print(f"\n== condition={condition} model={size} sample_rate={sr} ==")
            run_ids = [client.start(aid, config) for aid in audio_ids]
            results = client.wait_all(run_ids)
            ok = [results[r] for r in run_ids if r in results and results[r]["status"] == "done"]
            hyps = [(results[r]["transcript"] or "") if r in results else "" for r in run_ids]

            per_file, file_wer, logprobs, nsp = [], [], [], []
            for r, ref, hyp, rid in zip(rows, refs, hyps, run_ids):
                res = results.get(rid)
                sig = (res or {}).get("signals", {})
                w = compute_wer([ref], [hyp]) if ref.strip() else float("nan")
                n_err, n_words = word_errors(ref, hyp) if ref.strip() else (0, 0)
                file_wer.append(w)
                logprobs.append(sig.get("avg_logprob"))
                nsp.append(sig.get("no_speech_prob"))
                per_file.append({"path": r["path"], "wer": w, "errors": n_err, "ref_words": n_words,
                                 "avg_logprob": sig.get("avg_logprob"),
                                 "no_speech_prob": sig.get("no_speech_prob"), "rtf": sig.get("rtf"),
                                 "silence_ratio": sig.get("silence_ratio"), "input_rms": sig.get("input_rms")})
            # file name carries the experiment identity: <condition>_<model>_<rate>hz.csv
            csv_path = Path(a.out) / f"{condition}_{size}_{sr}hz.csv"
            with open(csv_path, "w", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=list(per_file[0]))
                w_.writeheader()
                w_.writerows(per_file)

            # corpus RTF = total compute time / total audio time (the honest throughput figure; the mean of
            # per-file RTFs is inflated by fixed per-file overhead on short clips)
            done_pairs = [(results[r]["signals"]["rtf"] * d, d) for r, d in zip(run_ids, durations)
                          if r in results and results[r]["status"] == "done" and results[r]["signals"]["rtf"] is not None]
            rtf_corpus = (sum(c for c, _ in done_pairs) / sum(d for _, d in done_pairs)) if done_pairs else float("nan")
            scored = [(f["errors"], f["ref_words"]) for f in per_file if f["ref_words"] > 0]
            ci_low, ci_high = bootstrap_wer_ci([e for e, _ in scored], [n for _, n in scored])
            metrics = {
                "wer": compute_wer(refs, hyps),
                "wer_ci_low": ci_low,
                "wer_ci_high": ci_high,
                "ref_words": float(sum(n for _, n in scored)),
                "rtf_corpus": rtf_corpus,
                "rtf_mean": mean([r["signals"]["rtf"] for r in ok]),
                "processing_p50_s": pct([r["timings"]["processing_s"] for r in ok], 0.50),
                "processing_p95_s": pct([r["timings"]["processing_s"] for r in ok], 0.95),
                # total latency includes QUEUE WAIT: all files are submitted at once, so this is dominated by the
                # backlog and says little about speed. Use rtf_corpus / processing_* for speed.
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
                with mlflow.start_run(run_name=f"{condition}-{size}-{sr}hz"):
                    mlflow.log_params({"model_size": size, "sample_rate_hz": sr, "n_files": len(rows),
                                       "condition": condition, "noise": noise,
                                       "snr_db": "clean" if snr is None else snr,
                                       "language": a.language or "auto"})
                    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # skip NaN
                    mlflow.log_artifact(str(csv_path))


if __name__ == "__main__":
    main()
