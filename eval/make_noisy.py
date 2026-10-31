"""Create a noisy copy of a manifest (to test accuracy and drift detection under noise).

  White noise:      python -m eval.make_noisy --manifest eval/data/manifest.csv --snr-db 5 --out eval/data/white_5db
  Real noise files: python -m eval.make_noisy --manifest eval/data/manifest.csv --snr-db 5 \
                           --noise-dir eval/data/noise/cafeteria --out eval/data/cafeteria_5db

SNR = speech power / noise power in dB. 20 dB mild, 10 dB noticeable, 5 dB bad, 0 dB noise as loud as speech.
With --noise-dir each utterance gets a random noise file and a random start point inside it (reproducible
through --seed). The noise TYPE label is the name of the noise folder (cafeteria, traffic, ...), or --noise-name;
the output manifest records that label, the exact noise file and the SNR. Keep one noise type per folder."""
import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

NOISE_SUFFIXES = {".wav", ".flac", ".ogg"}


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def noise_segment(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A piece of noise exactly n samples long: loop it if too short, random start if longer."""
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    start = int(rng.integers(0, len(noise) - n + 1))
    return noise[start:start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale the noise so that 10*log10(P_speech / P_noise) equals snr_db, then add it.
    If the sum would exceed full scale, the whole mix is scaled down (this keeps the SNR; it
    changes only the overall loudness)."""
    p_speech = float(np.mean(speech ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = speech + gain * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def load_noise_files(noise_dir: Path) -> list[Path]:
    files = sorted(p for p in noise_dir.rglob("*") if p.suffix.lower() in NOISE_SUFFIXES)
    if not files:
        raise SystemExit(f"no wav/flac/ogg noise files found under {noise_dir}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--snr-db", type=float, default=5.0)
    ap.add_argument("--noise-dir", help="folder with real noise recordings; omit for white noise")
    ap.add_argument("--noise-name", help="label for this noise type (default: the folder name, or 'white')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    noise_files = load_noise_files(Path(a.noise_dir)) if a.noise_dir else []
    label = a.noise_name or (Path(a.noise_dir).resolve().name if a.noise_dir else "white")
    noise_cache: dict[Path, tuple[np.ndarray, int]] = {}

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    new_rows = []
    for r in rows:
        speech, sr = sf.read(r["path"], dtype="float32")
        speech = to_mono(speech)
        if noise_files:
            nf = noise_files[int(rng.integers(0, len(noise_files)))]
            if nf not in noise_cache:
                raw, nsr = sf.read(nf, dtype="float32")
                noise_cache[nf] = (to_mono(raw), nsr)
            raw, nsr = noise_cache[nf]
            noise = noise_segment(resample(raw, nsr, sr), len(speech), rng)
            noise_file = nf.stem
        else:
            noise = rng.normal(0.0, 1.0, size=speech.shape).astype(np.float32)
            noise_file = "white"
        dest = out / f"{Path(r['path']).stem}_{label}_snr{int(a.snr_db)}.wav"
        sf.write(dest, mix_at_snr(speech, noise, a.snr_db), sr, subtype="PCM_16")
        new_rows.append({"path": str(dest.resolve()), "reference": r["reference"],
                         "noise": label, "noise_file": noise_file, "snr_db": a.snr_db})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "reference", "noise", "noise_file", "snr_db"])
        w.writeheader()
        w.writerows(new_rows)
    print(f"wrote {len(new_rows)} noisy files + {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
