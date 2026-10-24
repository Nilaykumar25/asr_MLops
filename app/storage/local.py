import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from app.storage.base import CHUNK, InvalidKey, StorageBackend, validate_key


class LocalStorage(StorageBackend):
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise InvalidKey(f"key escapes storage root: {key!r}")
        return p

    def save(self, key: str, stream: BinaryIO) -> tuple[int, str]:
        dest = self._path(key)
        if dest.exists():
            raise FileExistsError(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha, size = hashlib.sha256(), 0
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    buf = stream.read(CHUNK)
                    if not buf:
                        break
                    sha.update(buf)
                    size += len(buf)
                    out.write(buf)
            os.replace(tmp, dest)  # atomic: readers never see a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return size, sha.hexdigest()

    def get(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        keys = []
        for p in self.root.rglob("*"):
            if p.is_file() and not p.name.endswith(".part"):
                k = p.relative_to(self.root).as_posix()
                if k.startswith(prefix):
                    keys.append(k)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def local_path(self, key: str) -> Path | None:
        p = self._path(key)
        return p if p.is_file() else None
