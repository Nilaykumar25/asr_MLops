"""Send a steady stream of real requests (used to feed the monitoring + drift demo).

  python -m eval.send_traffic --key KEY --manifest eval/data/manifest.csv --tag baseline --rate 0.5 --count 120 --language en
  python -m eval.send_traffic --key KEY --manifest eval/data/white_5db/manifest.csv --tag noisy --rate 0.5 --count 120 --language en

IMPORTANT: use a user that has NOT uploaded these files before (for example a dedicated 'traffic' user).
Uploads are de-duplicated per user, so a repeated file keeps its OLD dataset_tag and would never count as 'baseline'.
"""
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
    ap.add_argument("--language", help="force a language such as en (recommended for English data)")
    ap.add_argument("--model-size", help="model size for these runs (default: the worker's WHISPER_MODEL_SIZE)")
    a = ap.parse_args()

    config = {}
    if a.language:
        config["language"] = a.language
    if a.model_size:
        config["model_size"] = a.model_size

    client = AsrClient(a.api, a.key)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    ids = {}                                              # upload each file once, reuse the audio id
    for r in rows:
        up = client.upload(r["path"], a.tag, r["reference"])
        ids[r["path"]] = up["id"]
        if up.get("duplicate"):
            print(f"WARNING: {r['path']} was uploaded before by this user; its old dataset_tag is kept, "
                  f"so it will NOT count as '{a.tag}'. Use a fresh user.")
            break
    for i, r in zip(range(a.count), itertools.cycle(rows)):
        client.start(ids[r["path"]], config)
        if (i + 1) % 10 == 0:
            print(f"sent {i + 1}/{a.count}")
        time.sleep(1.0 / a.rate)


if __name__ == "__main__":
    main()
