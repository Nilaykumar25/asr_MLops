"""S3-compatible backend. Works with MinIO locally and AWS S3 in the cloud."""
import hashlib
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from app.storage.base import StorageBackend, validate_key


class _HashingReader:
    """Wraps a stream so size + sha256 are computed while boto3 uploads it."""

    def __init__(self, stream: BinaryIO):
        self._stream = stream
        self.sha = hashlib.sha256()
        self.size = 0

    def read(self, n: int = -1) -> bytes:
        buf = self._stream.read(n)
        self.sha.update(buf)
        self.size += len(buf)
        return buf


class S3Storage(StorageBackend):
    def __init__(self, bucket, endpoint_url=None, access_key=None, secret_key=None, region="us-east-1"):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key or None,      # None -> boto3 uses the IAM role / env credentials
            aws_secret_access_key=secret_key or None,
            region_name=region,
        )
        if endpoint_url:  # MinIO: auto-create. On AWS create the bucket yourself in the console.
            try:
                self.client.head_bucket(Bucket=bucket)
            except ClientError:
                self.client.create_bucket(Bucket=bucket)

    def save(self, key: str, stream: BinaryIO) -> tuple[int, str]:
        validate_key(key)
        if self.exists(key):
            raise FileExistsError(key)
        reader = _HashingReader(stream)
        self.client.upload_fileobj(reader, self.bucket, key)
        return reader.size, reader.sha.hexdigest()

    def get(self, key: str) -> BinaryIO:
        validate_key(key)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(key) from e
            raise

    def exists(self, key: str) -> bool:
        validate_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        return sorted(keys)

    def delete(self, key: str) -> None:
        validate_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)
