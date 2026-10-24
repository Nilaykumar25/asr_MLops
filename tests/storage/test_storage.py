"""One test suite, run against every storage backend. Add a backend = add it to the fixture."""
import hashlib
import io
import os

import pytest

from app.storage.base import InvalidKey
from app.storage.local import LocalStorage


@pytest.fixture(params=["local", "s3"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorage(str(tmp_path))
    endpoint = os.environ.get("S3_TEST_ENDPOINT")        # e.g. http://localhost:9000 (MinIO)
    if not endpoint:
        pytest.skip("set S3_TEST_ENDPOINT to test the S3/MinIO backend")
    from app.storage.s3 import S3Storage
    return S3Storage("asr-test", endpoint, "minioadmin", "minioadmin")


def test_roundtrip_and_checksum(storage):
    data = os.urandom(3 * 1024 * 1024 + 17)
    size, sha = storage.save("audio/2026-01-01/a.wav", io.BytesIO(data))
    assert size == len(data)
    assert sha == hashlib.sha256(data).hexdigest()
    with storage.get("audio/2026-01-01/a.wav") as f:
        assert f.read() == data
    storage.delete("audio/2026-01-01/a.wav")


def test_exists_list_delete(storage):
    storage.save("audio/x/1.wav", io.BytesIO(b"1"))
    storage.save("audio/x/2.wav", io.BytesIO(b"2"))
    assert storage.exists("audio/x/1.wav")
    assert set(storage.list("audio/x/")) >= {"audio/x/1.wav", "audio/x/2.wav"}
    storage.delete("audio/x/1.wav")
    assert not storage.exists("audio/x/1.wav")
    storage.delete("audio/x/2.wav")


def test_objects_are_immutable(storage):
    storage.save("audio/y/1.wav", io.BytesIO(b"first"))
    with pytest.raises(FileExistsError):
        storage.save("audio/y/1.wav", io.BytesIO(b"second"))
    with storage.get("audio/y/1.wav") as f:
        assert f.read() == b"first"
    storage.delete("audio/y/1.wav")


def test_missing_object_raises(storage):
    with pytest.raises(FileNotFoundError):
        storage.get("audio/nope.wav")


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs/path", "a/../../b", "", "a\\b"])
def test_rejects_path_traversal(storage, bad):
    with pytest.raises(InvalidKey):
        storage.save(bad, io.BytesIO(b"x"))
