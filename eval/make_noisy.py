"""Create a noisy copy of a manifest to simulate input drift.
  python -m eval.make_noisy --manifest eval/data/manifest.csv --snr-db 5 --out eval/data/noisy_5db
Lower SNR = noisier. 20 dB is mild, 5 dB is bad, 0 dB is very bad."""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(x ** 2)) + 1e-12
    noise_power = power / (10 ** (snr_db / 10))
    noisy = x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--snr-db", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    new_rows = []
    for r in rows:
        x, sr = sf.read(r["path"], dtype="float32")
        dest = out / (Path(r["path"]).stem + f"_snr{int(a.snr_db)}.wav")
        sf.write(dest, add_noise(x, a.snr_db, rng), sr)
        new_rows.append({"path": str(dest.resolve()), "reference": r["reference"]})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "reference"])
        w.writeheader()
        w.writerows(new_rows)
    print(f"wrote {len(new_rows)} noisy files + {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
