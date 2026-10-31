"""Transcribe ONE file through the API and print everything that matters (no copy-pasting of IDs).

  python -m eval.transcribe_one --key $KEY eval/data/long.wav
  python -m eval.transcribe_one --key $KEY clip.flac --reference "the correct text" --language en
  python -m eval.transcribe_one --key $KEY clip.flac --config '{"model_size": "small", "simulate_sample_rate": 8000}'"""
import argparse
import json
import sys

from eval.client import AsrClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--config", default="{}", help='JSON, e.g. \'{"model_size": "small"}\'')
    ap.add_argument("--language", help="force a language code such as en or hi (default: auto-detect)")
    ap.add_argument("--reference", help="the correct text; if given, the WER is printed")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()

    config = json.loads(a.config)
    if a.language:
        config["language"] = a.language

    client = AsrClient(a.api, a.key)
    audio = client.upload(a.path, reference=a.reference)
    print(f"uploaded   : {audio['id']}  ({audio['duration_seconds']:.1f} s, duplicate={audio['duplicate']})")
    run_id = client.start(audio["id"], config)
    run = client.wait_all([run_id], timeout=a.timeout, poll=1.0).get(run_id)
    if run is None:
        sys.exit(f"timed out waiting for run {run_id}")

    print(f"status     : {run['status']}   attempts: {run['attempts']}   model: {run['model_version']}")
    if run["status"] != "done":
        sys.exit(f"error      : {run['error']}")
    print(f"transcript : {run['transcript']}")
    if a.reference:
        from eval.metrics import compute_wer
        print(f"reference  : {a.reference}")
        print(f"WER        : {compute_wer([a.reference], [run['transcript'] or '']) * 100:.1f} %")
    t, s = run["timings"], run["signals"]
    print(f"timings    : queue wait {t['queue_wait_s']} s, processing {t['processing_s']} s")
    print("signals    : " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
