"""Download LibriSpeech test-clean (~350 MB) and build a manifest of N utterances.
  python -m eval.prepare_librispeech --n 200
Creates eval/data/manifest.csv with columns: path,reference"""
import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="eval/data")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "test-clean.tar.gz"
    if not archive.exists():
        print("downloading ...")
        urllib.request.urlretrieve(URL, archive)
    root = out / "LibriSpeech"
    if not root.exists():
        print("extracting ...")
        with tarfile.open(archive) as t:
            t.extractall(out)

    rows = []
    for trans in sorted(root.glob("test-clean/*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists():
                rows.append((str(flac.resolve()), text.lower()))
    rows = rows[:: max(1, len(rows) // a.n)][: a.n]       # spread across many speakers
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reference"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
