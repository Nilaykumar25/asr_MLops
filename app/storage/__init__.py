from functools import lru_cache

from app.config import get_settings
from app.storage.base import StorageBackend


@lru_cache
def get_storage() -> StorageBackend:
    s = get_settings()
    if s.storage_backend == "local":
        from app.storage.local import LocalStorage
        return LocalStorage(s.data_root)
    if s.storage_backend == "s3":
        from app.storage.s3 import S3Storage
        return S3Storage(s.s3_bucket, s.s3_endpoint_url, s.s3_access_key, s.s3_secret_key, s.s3_region)
    raise ValueError(f"unknown STORAGE_BACKEND: {s.storage_backend}")
