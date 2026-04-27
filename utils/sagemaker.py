"""SageMaker Batch Transform job configuration builder.

Produces the flat config dict that the DAG pushes to XCom key-by-key so that
the SageMakerTransformOperator (and S3KeySensor) can reference individual
values through Jinja templates at render time.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# SageMaker enforces a 63-character job name limit.
_MAX_JOB_NAME_LEN = 63
_INVALID_CHARS = re.compile(r"[^a-z0-9\-]")
_MULTI_HYPHEN = re.compile(r"-{2,}")


def build_transform_config(
    model_info: dict,
    s3_bucket: str,
    execution_date: str,
    dag_id: str,
    ts_nodash: str,
) -> dict:
    """Return a flat config dict for the current DAG run.

    Every key in the returned dict is pushed to XCom by the
    ``build_transform_config`` task.  Downstream Jinja templates reference them
    with ``{{ ti.xcom_pull(task_ids='build_transform_config', key='<key>') }}``.

    Args:
        model_info:     Output of ``registry.get_latest_approved_model()``.
        s3_bucket:      S3 bucket name (no ``s3://`` prefix).
        execution_date: Airflow ``ds`` macro value (YYYY-MM-DD).
        dag_id:         DAG identifier — baked into the job name.
        ts_nodash:      Airflow ``ts_nodash`` macro (YYYYMMDDTHHmmss) — ensures
                        the job name is unique per run even when re-triggered on
                        the same date.
    """
    job_name = _make_job_name(dag_id, ts_nodash)

    return {
        # --- model identity (sourced from Model Registry) ---
        "model_name":        model_info["model_name"],
        "model_version":     model_info["model_version"],
        "model_package_arn": model_info["model_package_arn"],
        # --- job identity ---
        "job_name":          job_name,
        # --- S3 paths (date-partitioned by execution_date) ---
        "input_s3":          f"s3://{s3_bucket}/batch/input/{execution_date}/",
        "output_s3":         f"s3://{s3_bucket}/batch/output/{execution_date}/",
        "output_bucket":     s3_bucket,
        "output_prefix":     f"batch/output/{execution_date}/",
        # staged combined CSV written after postprocessing
        "processed_s3_key":  f"batch/processed/{execution_date}/combined.csv",
        # --- compute ---
        "instance_type":     model_info["instance_type"],
    }


def _make_job_name(dag_id: str, ts_nodash: str) -> str:
    """Produce a SageMaker-compliant job name.

    Format: ``{dag_id}-transform-{ts_nodash}`` (lowercase, hyphens only).
    SageMaker allows [a-z0-9-], max 63 chars, no leading/trailing hyphen.
    
    GUIDE: WHY THIS LOGIC?
    1. SageMaker allows [a-z0-9-] only. No underscores or uppercase.
    2. Max length is 63 chars. Exceeding this causes API failure immediately.
    3. We add 'ts_nodash' to ensure the name is unique even if the DAG is 
       re-triggered multiple times on the same calendar day.
    """
    # 1. Replace underscores (common in DAG IDs) with hyphens
    raw = f"{dag_id.replace('_', '-')}-transform-{ts_nodash}".lower()
    
    # 2. Sanitize: Drop characters that aren't a-z, 0-9, or hyphen
    sanitized = _INVALID_CHARS.sub("-", raw)
    
    # 3. Clean up: Avoid double hyphens '--' and trailing hyphens
    sanitized = _MULTI_HYPHEN.sub("-", sanitized).strip("-")
    
    # 4. Enforce: Trim to SageMaker's hard limit of 63 characters
    return sanitized[:_MAX_JOB_NAME_LEN]
