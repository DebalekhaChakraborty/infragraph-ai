#!/usr/bin/env python3
"""
Upload infragraph-ai/model_artifacts to S3 using boto3.

Target:
  s3://my-hackathons/infragraph-ai/model_artifacts

Usage:
  python scripts/upload_model_artifacts_to_s3.py

Optional:
  python scripts/upload_model_artifacts_to_s3.py --dry-run
  python scripts/upload_model_artifacts_to_s3.py --local-dir model_artifacts
  python scripts/upload_model_artifacts_to_s3.py --bucket my-hackathons --prefix infragraph-ai/model_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from boto3.s3.transfer import TransferConfig


DEFAULT_BUCKET = "my-hackathons"
DEFAULT_PREFIX = "infragraph-ai/model_artifacts"


def find_local_model_artifacts() -> Path:
    candidates = [
        Path.cwd() / "model_artifacts",
        Path.cwd() / "infragraph-ai" / "model_artifacts",
        Path(__file__).resolve().parents[1] / "model_artifacts",
    ]

    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()

    raise FileNotFoundError(
        "model_artifacts folder not found. Expected one of:\n"
        "  ./model_artifacts\n"
        "  ./infragraph-ai/model_artifacts\n"
        "  <repo>/model_artifacts"
    )


def guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_skip_upload(s3_client, bucket: str, key: str, local_path: Path) -> bool:
    """
    Skip if S3 object exists and size matches.
    ETag is not reliable for multipart uploads, so size check is safer/simple.
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        remote_size = int(response.get("ContentLength", -1))
        local_size = local_path.stat().st_size
        return remote_size == local_size
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_directory(
    local_dir: Path,
    bucket: str,
    prefix: str,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    s3_client = boto3.client("s3")

    prefix = prefix.strip("/")

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        print(f"[WARN] No files found under: {local_dir}")
        return

    total_bytes = sum(p.stat().st_size for p in files)

    print("====================================================")
    print(" InfraGraph AI — Upload model_artifacts to S3")
    print("====================================================")
    print(f"Local folder : {local_dir}")
    print(f"S3 target    : s3://{bucket}/{prefix}")
    print(f"Files        : {len(files)}")
    print(f"Total size   : {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Dry run      : {dry_run}")
    print(f"Force upload : {force}")
    print()

    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    uploaded = 0
    skipped = 0

    for idx, path in enumerate(files, start=1):
        relative_path = path.relative_to(local_dir).as_posix()
        s3_key = f"{prefix}/{relative_path}"

        size_mb = path.stat().st_size / (1024 * 1024)

        print(f"[{idx}/{len(files)}] {relative_path} ({size_mb:.2f} MB)")

        if dry_run:
            print(f"  DRY RUN -> s3://{bucket}/{s3_key}")
            continue

        if not force and should_skip_upload(s3_client, bucket, s3_key, path):
            print("  skipped: same size already exists")
            skipped += 1
            continue

        extra_args = {
            "ContentType": guess_content_type(path),
            "Metadata": {
                "source": "infragraph-ai",
                "artifact_type": "model_artifacts",
                "md5": file_md5(path),
            },
        }

        s3_client.upload_file(
            Filename=str(path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs=extra_args,
            Config=transfer_config,
        )

        print(f"  uploaded -> s3://{bucket}/{s3_key}")
        uploaded += 1

    print()
    print("====================================================")
    print(" Upload complete")
    print("====================================================")
    print(f"Uploaded : {uploaded}")
    print(f"Skipped  : {skipped}")
    print(f"Target   : s3://{bucket}/{prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload InfraGraph model_artifacts folder to S3 using boto3."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Upload even if same-size object exists")
    args = parser.parse_args()

    try:
        local_dir = Path(args.local_dir).resolve() if args.local_dir else find_local_model_artifacts()

        if not local_dir.exists() or not local_dir.is_dir():
            print(f"[ERROR] Local directory does not exist: {local_dir}")
            sys.exit(1)

        upload_directory(
            local_dir=local_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            dry_run=args.dry_run,
            force=args.force,
        )

    except NoCredentialsError:
        print("[ERROR] AWS credentials not found.")
        print("Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or use an IAM role.")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Upload failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
