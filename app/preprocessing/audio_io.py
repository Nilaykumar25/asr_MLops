"""Thin wrappers around ffprobe / ffmpeg. Requires ffmpeg installed on the machine."""
import json
import subprocess
from pathlib import Path


class AudioError(Exception):
    pass


def probe(path: Path) -> dict:
    """Return duration (s), sample_rate, channels. Raises AudioError if undecodable."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=sample_rate,channels:format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        raise AudioError("ffprobe timed out") from e
    if out.returncode != 0:
        raise AudioError(out.stderr.strip() or "ffprobe failed")
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise AudioError("no audio stream found")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError) as e:
        raise AudioError("could not determine duration") from e
    return {
        "duration": duration,
        "sample_rate": int(streams[0].get("sample_rate") or 0) or None,
        "channels": int(streams[0].get("channels") or 0) or None,
    }


def resample_to_wav(src: Path, dst: Path, sample_rate: int = 16000, via_sr: int | None = None) -> None:
    """Decode -> mono -> sample_rate PCM wav.
    via_sr simulates a low-bandwidth source: audio is squeezed to via_sr first
    (e.g. 8000 = telephone quality) and then brought back up to sample_rate."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1"]
    if via_sr:
        cmd += ["-af", f"aresample={via_sr},aresample={sample_rate}"]
    cmd += ["-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise AudioError(out.stderr.strip() or "ffmpeg failed")


def load_wav(path: Path):
    """Load a wav as mono float32 numpy array (imported lazily: numpy/soundfile are Phase 4 deps)."""
    import soundfile as sf
    data, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data
