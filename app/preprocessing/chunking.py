"""Turns VAD speech segments into transcription chunks. Pure Python: easy to unit-test."""


def build_chunks(segments: list[tuple[float, float]], max_len: float = 30.0) -> list[tuple[float, float]]:
    """1) Any speech segment longer than max_len is force-split into max_len pieces.
    2) Neighbouring segments are packed together (pauses included) while the chunk stays <= max_len.
    Result: few, long chunks that start and end at pauses wherever possible."""
    pieces: list[tuple[float, float]] = []
    for s, e in segments:
        while e - s > max_len:
            pieces.append((s, s + max_len))
            s += max_len
        if e - s > 0:
            pieces.append((s, e))

    chunks: list[tuple[float, float]] = []
    cur: list[float] | None = None
    for s, e in pieces:
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_len:
            cur[1] = e
        else:
            chunks.append((cur[0], cur[1]))
            cur = [s, e]
    if cur is not None:
        chunks.append((cur[0], cur[1]))
    return chunks
