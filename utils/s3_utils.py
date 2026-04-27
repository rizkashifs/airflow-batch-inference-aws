"""S3 helpers for reading batch transform output and staging results.

All functions are stateless and rely solely on boto3 + pandas.
"""

from __future__ import annotations

import io
import logging

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_OUTPUT_SUFFIX = ".csv.out"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_input_files(bucket: str, prefix: str, suffix: str = ".csv") -> list[str]:
    """Return S3 keys of input CSV files under *prefix*."""
    return _list_keys(bucket, prefix, suffix)


def list_output_files(bucket: str, prefix: str) -> list[str]:
    """Return S3 keys of SageMaker output files (.csv.out) under *prefix*."""
    return _list_keys(bucket, prefix, _OUTPUT_SUFFIX)


def _list_keys(bucket: str, prefix: str, suffix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith(suffix):
                    keys.append(key)
    except ClientError as exc:
        logger.error("s3 list failed s3://%s/%s: %s", bucket, prefix, exc)
        raise
    logger.info("Found %d '%s' file(s) at s3://%s/%s", len(keys), suffix, bucket, prefix)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Validation of input data presence
# ---------------------------------------------------------------------------

def validate_input_data(bucket: str, prefix: str) -> dict:
    """Verify that input CSV files exist and are non-empty.

    Returns a summary dict with 'file_count' and 'estimated_row_count'.
    Raises RuntimeError if no files are found.
    """
    keys = list_input_files(bucket, prefix)
    if not keys:
        raise RuntimeError(
            f"No input CSV files found at s3://{bucket}/{prefix}. "
            "Ensure the upstream pipeline has placed data there before the DAG runs."
        )

    s3 = boto3.client("s3")
    total_lines = 0
    for key in keys:
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            size = head.get("ContentLength", 0)
            if size == 0:
                raise RuntimeError(f"Input file s3://{bucket}/{key} is empty.")
            # rough estimate: average 100 bytes per CSV row
            total_lines += max(1, size // 100)
        except ClientError as exc:
            logger.error("head_object failed for s3://%s/%s: %s", bucket, key, exc)
            raise

    summary = {"file_count": len(keys), "estimated_row_count": total_lines}
    logger.info("Input data validated: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Collecting inference results
# ---------------------------------------------------------------------------

def collect_inference_results(bucket: str, output_prefix: str) -> pd.DataFrame:
    """Read all .csv.out files and combine them into a single DataFrame.

    Output schema:
        source_file  – S3 key of the .csv.out file
        row_index    – 0-based row position (global, across all files ordered by key)
        raw_output   – raw text line from the SageMaker output
        prediction   – first CSV field cast to float (None if unparseable)
    """
    keys = list_output_files(bucket, output_prefix)
    if not keys:
        raise RuntimeError(
            f"No .csv.out files found at s3://{bucket}/{output_prefix}. "
            "Ensure the SageMaker Batch Transform job finished successfully."
        )

    s3 = boto3.client("s3")
    records: list[dict] = []
    global_idx = 0

    for key in keys:
        lines = _read_lines(s3, bucket, key)
        for line in lines:
            records.append(
                {
                    "source_file": key,
                    "row_index": global_idx,
                    "raw_output": line,
                    "prediction": _parse_first_field(line),
                }
            )
            global_idx += 1

    df = pd.DataFrame(records)
    logger.info(
        "Collected %d inference record(s) from %d file(s)", len(df), len(keys)
    )
    return df


def _read_lines(s3_client, bucket: str, key: str) -> list[str]:
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        content = resp["Body"].read().decode("utf-8")
    except ClientError as exc:
        logger.error("Failed to read s3://%s/%s: %s", bucket, key, exc)
        raise
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def _parse_first_field(raw: str) -> float | None:
    """Cast the first comma-separated field to float; return None on failure."""
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Staging processed results to S3
# ---------------------------------------------------------------------------

def write_dataframe_to_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    """Serialise *df* to CSV and upload to s3://bucket/key.

    Returns the full s3:// URI.
    """
    s3 = boto3.client("s3")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    body = buf.getvalue().encode("utf-8")

    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")
    except ClientError as exc:
        logger.error("Failed to write s3://%s/%s: %s", bucket, key, exc)
        raise

    uri = f"s3://{bucket}/{key}"
    logger.info("Staged %d rows → %s", len(df), uri)
    return uri


def read_dataframe_from_s3(s3_uri: str) -> pd.DataFrame:
    """Read a CSV from an s3:// URI into a DataFrame."""
    bucket, key = _split_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        content = resp["Body"].read().decode("utf-8")
    except ClientError as exc:
        logger.error("Failed to read %s: %s", s3_uri, exc)
        raise
    df = pd.read_csv(io.StringIO(content))
    logger.info("Read %d rows from %s", len(df), s3_uri)
    return df


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key/path' → ('bucket', 'key/path')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid s3:// URI: {uri!r}")
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key
