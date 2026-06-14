#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    import boto3
except ImportError:
    print("ERROR: boto3 is not installed.")
    print("Install with: pip install boto3")
    sys.exit(1)

DEFAULT_S3_PREFIX = "s3://my-hackathon/infragraph-ai/model_artifacts/rfdetr_v3"
DEFAULT_LOCAL_DIR = "/workspace/shared/infragraph-ai/model_artifacts/rfdetr_v3/model"

FILES = [
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_total.pth",
]

def parse_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    return bucket, prefix

def main():
    s3_prefix = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RFDETR_S3_PREFIX", DEFAULT_S3_PREFIX)
    local_dir = Path(os.environ.get("RFDETR_LOCAL_DIR", DEFAULT_LOCAL_DIR))

    bucket, prefix = parse_s3_uri(s3_prefix)

    print("S3 bucket :", bucket)
    print("S3 prefix :", prefix)
    print("Local dir :", local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    session_kwargs = {}
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    for filename in FILES:
        key = f"{prefix}/{filename}"
        dest = local_dir / filename

        print(f"\nDownloading s3://{bucket}/{key}")
        print(f"        -> {dest}")

        try:
            s3.download_file(bucket, key, str(dest))
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(2)

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK: {filename} ({size_mb:.1f} MB)")

    recommended = local_dir / "checkpoint_best_total.pth"

    print("\nDone.")
    print("\nRecommended checkpoint:")
    print(recommended)

    print("\nExport this before starting RF-DETR/Streamlit:")
    print(f'export INFRAGRAPH_RFDETR_CHECKPOINT="{recommended}"')

if __name__ == "__main__":
    main()
