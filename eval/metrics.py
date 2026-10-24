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
