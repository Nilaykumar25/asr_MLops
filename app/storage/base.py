"""Storage interface. Local disk and S3/MinIO both implement this."""
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

CHUNK = 1024 * 1024  # 1 MiB streaming unit


class InvalidKey(ValueError):
    pass


def validate_key(key: str) -> None:
    if not key or key.startswith("/") or "\\" in key or ".." in key.split("/"):
        raise InvalidKey(f"invalid storage key: {key!r}")


class StorageBackend(ABC):
    @abstractmethod
    def save(self, key: str, stream: BinaryIO) -> tuple[int, str]:
        """Stream `stream` to `key`. Returns (size_bytes, sha256_hex).
        Raises FileExistsError if the key already exists (objects are immutable)."""

    @abstractmethod
    def get(self, key: str) -> BinaryIO:
        """Open the object for streaming reads. Raises FileNotFoundError."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def local_path(self, key: str) -> Path | None:
        """Return a real filesystem path if the backend has one (local disk), else None."""
        return None

    def fetch_to_path(self, key: str, dest: Path) -> Path:
        """Copy an object to a local file (used by ffprobe / ffmpeg)."""
        src = self.get(key)
        try:
            with open(dest, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
        finally:
            src.close()
        return dest
