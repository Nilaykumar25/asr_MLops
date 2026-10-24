"""Small HTTP client for the ASR backend, shared by the evaluation and traffic scripts."""
import time
from pathlib import Path

import requests


class AsrClient:
    def __init__(self, base_url: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}

    def upload(self, path: str, dataset_tag: str | None = None, reference: str | None = None) -> dict:
        data = {k: v for k, v in (("dataset_tag", dataset_tag), ("reference_text", reference)) if v}
        with open(path, "rb") as f:
            r = requests.post(f"{self.base}/audio", headers=self.headers, data=data,
                              files={"file": (Path(path).name, f)})
        r.raise_for_status()
        return r.json()

    def start(self, audio_id: str, config: dict | None = None) -> str:
        while True:
            r = requests.post(f"{self.base}/audio/{audio_id}/transcribe", headers=self.headers,
                              json={"config": config or {}})
            if r.status_code == 429:      # backpressure: wait and retry
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()["run_id"]

    def get_run(self, run_id: str) -> dict:
        r = requests.get(f"{self.base}/runs/{run_id}", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def wait_all(self, run_ids: list[str], timeout: float = 3600, poll: float = 2.0) -> dict[str, dict]:
        results: dict[str, dict] = {}
        t0 = time.time()
        while len(results) < len(run_ids) and time.time() - t0 < timeout:
            for rid in run_ids:
                if rid not in results:
                    run = self.get_run(rid)
                    if run["status"] in ("done", "failed"):
                        results[rid] = run
            time.sleep(poll)
        return results
