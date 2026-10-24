"""Send a steady stream of real requests (used to feed the monitoring + drift demo).
  python -m eval.send_traffic --key KEY --manifest eval/data/manifest.csv --tag baseline --rate 1 --count 150
  python -m eval.send_traffic --key KEY --manifest eval/data/noisy_5db/manifest.csv --tag noisy --rate 1 --count 150"""
import argparse
import csv
import itertools
import time

from eval.client import AsrClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True, help="dataset_tag stored with the audio (e.g. baseline / noisy)")
    ap.add_argument("--rate", type=float, default=1.0, help="requests per second")
    ap.add_argument("--count", type=int, default=100)
    a = ap.parse_args()

    client = AsrClient(a.api, a.key)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    ids = {}                                              # upload each file once, reuse the audio id
    for r in rows:
        ids[r["path"]] = client.upload(r["path"], a.tag, r["reference"])["id"]
    for i, r in zip(range(a.count), itertools.cycle(rows)):
        client.start(ids[r["path"]], {})
        if (i + 1) % 10 == 0:
            print(f"sent {i + 1}/{a.count}")
        time.sleep(1.0 / a.rate)


if __name__ == "__main__":
    main()
