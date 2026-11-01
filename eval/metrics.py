import re


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace: the usual ASR scoring normalisation."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\u0900-\u097f' ]+", " ", text)   # keeps Devanagari for Hindi data
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]          # jiwer rejects empty references
    return float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]))


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word errors, reference word count) for one file: minimum substitutions + deletions + insertions.
    Summing errors / summing words over files gives exactly the corpus WER."""
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def bootstrap_wer_ci(errors, words, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """95% confidence interval for the corpus WER, by resampling FILES with replacement."""
    import numpy as np
    e, w = np.asarray(errors, dtype=float), np.asarray(words, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    wers = e[idx].sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.quantile(wers, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(errors_a, errors_b, words, n_boot: int = 5000, seed: int = 0, level: float = 0.95):
    """Is B different from A on the SAME files? Returns (WER_B - WER_A, ci_low, ci_high, share of resamples
    in which B is better). Resampling the same files for both models removes file difficulty from the noise."""
    import numpy as np
    a, b, w = (np.asarray(x, dtype=float) for x in (errors_a, errors_b, words))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    diffs = (b[idx].sum(axis=1) - a[idx].sum(axis=1)) / w[idx].sum(axis=1)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = (b.sum() - a.sum()) / w.sum()
    return float(point), float(lo), float(hi), float((diffs < 0).mean())
