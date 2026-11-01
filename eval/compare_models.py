"""Is model B really better than model A? A paired bootstrap over files.

  python -m eval.compare_models results/clean_small_16000hz.csv results/clean_medium_16000hz.csv

Both files must come from the same test set (same audio files). Needs the 'errors' and 'ref_words'
columns written by the current eval.run_eval. Resampling the SAME files for both models removes the
difficulty of individual files from the comparison, so small real differences can show up."""
import argparse

import pandas as pd

from eval.metrics import bootstrap_wer_ci, paired_bootstrap_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="per-file CSV of model A (the baseline)")
    ap.add_argument("b", help="per-file CSV of model B")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    A, B = pd.read_csv(args.a), pd.read_csv(args.b)
    for name, df in (("A", A), ("B", B)):
        if not {"errors", "ref_words"} <= set(df.columns):
            raise SystemExit(f"{name} has no errors/ref_words columns: re-run eval.run_eval with the updated script")
    m = A.merge(B, on="path", suffixes=("_a", "_b"))
    m = m[m["ref_words_a"] > 0]
    if m.empty:
        raise SystemExit("the two result files share no audio files: compare results from the same test set")

    words = m["ref_words_a"].to_numpy()
    ea, eb = m["errors_a"].to_numpy(), m["errors_b"].to_numpy()
    for label, path, e in (("A", args.a, ea), ("B", args.b, eb)):
        lo, hi = bootstrap_wer_ci(e, words)
        print(f"{label}: {path}\n   WER {e.sum() / words.sum() * 100:.2f}%   (95% CI {lo * 100:.2f} to {hi * 100:.2f})")

    diff, lo, hi, p_better = paired_bootstrap_diff(ea, eb, words, n_boot=args.n_boot)
    print(f"\n{len(m)} files, {int(words.sum())} reference words")
    print(f"B minus A: {diff * 100:+.2f} percentage points   (95% CI {lo * 100:+.2f} to {hi * 100:+.2f})")
    print(f"B was better in {p_better * 100:.0f}% of resamples")
    if hi < 0:
        print("-> B is reliably better than A on this test set.")
    elif lo > 0:
        print("-> B is reliably WORSE than A on this test set.")
    else:
        print("-> The interval includes 0: this data cannot tell A and B apart. Use more files before claiming a difference.")


if __name__ == "__main__":
    main()
