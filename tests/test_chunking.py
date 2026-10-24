from app.preprocessing.chunking import build_chunks


def test_packs_neighbouring_segments():
    # 0-20 and 25-40 would exceed 30 s together, so they stay separate
    assert build_chunks([(0, 10), (11, 20), (25, 40)], 30) == [(0, 20), (25, 40)]


def test_force_splits_long_continuous_speech():
    chunks = build_chunks([(0, 75)], 30)
    assert chunks == [(0, 30), (30, 60), (60, 75)]


def test_no_speech_gives_no_chunks():
    assert build_chunks([], 30) == []


def test_no_chunk_exceeds_limit():
    segs = [(i * 7.0, i * 7.0 + 5.0) for i in range(40)]
    for s, e in build_chunks(segs, 30):
        assert e - s <= 30
