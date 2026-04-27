"""S3 helpers for reading batch transform output and staging results.

All functions are stateless; they create a boto3 client internally so callers
don't need to manage AWS sessions.
"""

from __future__ import annotations

import io
import logging

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_OUTPUT_SUFFIX = ".csv.out"
_INPUT_SUFFIX  = ".csv"


# ---------------------------------------------------------------------------
# Key listing
# ---------------------------------------------------------------------------

def list_input_files(bucket: str, prefix: str) -> list[str]:
    """Return sorted S3 keys of CSV input files under *prefix*."""
    return _list_keys(bucket, prefix, _INPUT_SUFFIX)


def list_output_files(bucket: str, prefix: str) -> list[str]:
    """Return sorted S3 keys of SageMaker output files (.csv.out) under *prefix*."""
    return _list_keys(bucket, prefix, _OUTPUT_SUFFIX)


def _list_keys(bucket: str, prefix: str, suffix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(suffix):
                    keys.append(obj["Key"])
    except ClientError as exc:
        logger.error("S3 list failed — s3://%s/%s: %s", bucket, prefix, exc)
        raise
    logger.info("Found %d '%s' file(s) at s3://%s/%s", len(keys), suffix, bucket, prefix)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Input data validation
# ---------------------------------------------------------------------------

def validate_input_data(bucket: str, prefix: str) -> dict:
    """Confirm that at least one non-empty CSV exists at *prefix*.

    Uses a single paginated list call so the ``Size`` field from
    ``list_objects_v2`` avoids an extra ``head_object`` per file.
    Raises RuntimeError if no files are found or any file is empty.

    Returns ``{"file_count": N}`` for use in log_status.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    found: list[str] = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(_INPUT_SUFFIX):
                    continue
                if obj["Size"] == 0:
                    # Empty input file would silently produce zero output rows —
                    # fail loudly here so the error is obvious.
                    raise RuntimeError(
                        f"Input file s3://{bucket}/{obj['Key']} has 0 bytes. "
                        "Upload valid CSV data before triggering the DAG."
                    )
                found.append(obj["Key"])
    except ClientError as exc:
        logger.error("S3 list failed during input validation: %s", exc)
        raise

    if not found:
        raise RuntimeError(
            f"No CSV files found at s3://{bucket}/{prefix}. "
            "Ensure the upstream pipeline has deposited input data before running."
        )

    logger.info("Input validated — %d file(s) at s3://%s/%s", len(found), bucket, prefix)
    return {"file_count": len(found)}


# ---------------------------------------------------------------------------
# Collecting inference results
# ---------------------------------------------------------------------------

def collect_inference_results(bucket: str, output_prefix: str) -> pd.DataFrame:
    """Read every .csv.out file under *output_prefix* and combine into one DataFrame.

    SageMaker writes one output line per input line (SplitType=Line /
    AssembleWith=Line), so row order across files is deterministic.

    Output columns:
        source_file  – S3 key of the originating .csv.out file
        row_index    – 0-based position across all files (sorted by key name)
        raw_output   – raw text line exactly as SageMaker wrote it
        prediction   – first CSV field cast to float (None when unparseable)
    """
    keys = list_output_files(bucket, output_prefix)
    if not keys:
        raise RuntimeError(
            f"No .csv.out files found at s3://{bucket}/{output_prefix}. "
            "The SageMaker Batch Transform job may not have completed successfully."
        )

    s3 = boto3.client("s3")
    records: list[dict] = []
    global_idx = 0

    for key in keys:
        for line in _read_lines(s3, bucket, key):
            records.append({
                "source_file": key,
                "row_index":   global_idx,
                "raw_output":  line,
                "prediction":  _parse_first_field(line),
            })
            global_idx += 1

    df = pd.DataFrame(records)
    logger.info("Collected %d record(s) from %d file(s)", len(df), len(keys))
    return df


def _read_lines(s3_client, bucket: str, key: str) -> list[str]:
    """Download an S3 object and return its non-empty lines."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        content = resp["Body"].read().decode("utf-8")
    except ClientError as exc:
        logger.error("Failed to read s3://%s/%s: %s", bucket, key, exc)
        raise
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def _parse_first_field(raw: str) -> float | None:
    """Extract the prediction score from a raw output line.

    SageMaker batch transform writes one prediction per line.  For CSV output
    the score is the first comma-separated field.  Returns None when the line
    cannot be parsed as a number (e.g. a header row or malformed output).
    """
    first = raw.split(",")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Staging processed results to S3
# ---------------------------------------------------------------------------

def write_dataframe_to_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    """Serialize *df* to CSV and upload to ``s3://bucket/key``.

    Returns the full ``s3://`` URI so the caller can push it to XCom for
    downstream tasks without re-computing the path.
    """
    s3 = boto3.client("s3")
    body = df.to_csv(index=False).encode("utf-8")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/csv")
    except ClientError as exc:
        logger.error("Failed to write s3://%s/%s: %s", bucket, key, exc)
        raise

    uri = f"s3://{bucket}/{key}"
    logger.info("Staged %d rows → %s", len(df), uri)
    return uri


def read_dataframe_from_s3(s3_uri: str) -> pd.DataFrame:
    """Read a CSV from an ``s3://`` URI into a DataFrame."""
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
    """Parse ``'s3://bucket/key/path'`` → ``('bucket', 'key/path')``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key
