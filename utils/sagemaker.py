"""SageMaker Batch Transform job configuration builder.

Produces the config dict consumed by SageMakerTransformOperator and the
S3 path metadata used by downstream tasks via XCom.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MAX_JOB_NAME_LEN = 63
_INVALID_CHARS = re.compile(r"[^a-z0-9\-]")
_MULTI_HYPHEN = re.compile(r"-{2,}")


def build_transform_config(
    model_info: dict,
    s3_bucket: str,
    execution_date: str,
    dag_id: str,
    ts_nodash: str,
    instance_count: int = 1,
    max_concurrent_transforms: int = 1,
    max_payload_mb: int = 6,
) -> dict:
    """Build the full Batch Transform job configuration.

    Args:
        model_info:              Output of registry.get_latest_approved_model().
        s3_bucket:               S3 bucket name (no s3:// prefix).
        execution_date:          Airflow ``ds`` macro value  (YYYY-MM-DD).
        dag_id:                  DAG identifier used in the job name.
        ts_nodash:               Airflow ``ts_nodash`` macro (YYYYMMDDTHHmmss).
        instance_count:          Number of ML instances.
        max_concurrent_transforms: Max concurrent requests per instance.
        max_payload_mb:          Payload cap in MB.

    Returns:
        Config dict.  Every key is pushed to XCom individually by the
        build_transform_config task so that downstream Jinja templates can
        reference them via  ti.xcom_pull(task_ids='build_transform_config', key=…).
    """
    job_name = _make_job_name(dag_id, ts_nodash)
    input_s3 = f"s3://{s3_bucket}/batch/input/{execution_date}/"
    output_s3 = f"s3://{s3_bucket}/batch/output/{execution_date}/"
    output_prefix = f"batch/output/{execution_date}/"
    processed_key = f"batch/processed/{execution_date}/combined.csv"

    config = {
        # model identity
        "model_name": model_info["model_name"],
        "model_version": model_info["model_version"],
        "model_package_arn": model_info["model_package_arn"],
        # job identity
        "job_name": job_name,
        # s3 paths
        "input_s3": input_s3,
        "output_s3": output_s3,
        "output_bucket": s3_bucket,
        "output_prefix": output_prefix,
        "processed_s3_key": processed_key,
        # compute
        "instance_type": model_info["instance_type"],
        "instance_count": instance_count,
        "max_concurrent_transforms": max_concurrent_transforms,
        "max_payload_mb": max_payload_mb,
        "strategy": "SingleRecord",
    }

    logger.info(
        "Transform config built — job: %s  model: %s  in: %s  out: %s",
        job_name,
        model_info["model_name"],
        input_s3,
        output_s3,
    )
    return config


def make_operator_config(transform_config: dict) -> dict:
    """Convert the internal config dict into the SageMakerTransformOperator shape.

    This is called by the DAG when NOT using Jinja (i.e., when the operator
    config is assembled at task runtime inside a PythonOperator-launched
    SageMakerHook call, or when the template approach isn't available).

    Returns the dict that maps 1-to-1 with boto3 create_transform_job kwargs.
    """
    return {
        "TransformJobName": transform_config["job_name"],
        "ModelName": transform_config["model_name"],
        "BatchStrategy": transform_config["strategy"],
        "MaxConcurrentTransforms": transform_config["max_concurrent_transforms"],
        "MaxPayloadInMB": transform_config["max_payload_mb"],
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": transform_config["input_s3"],
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        "TransformOutput": {
            "S3OutputPath": transform_config["output_s3"],
            "AssembleWith": "Line",
        },
        "TransformResources": {
            "InstanceType": transform_config["instance_type"],
            "InstanceCount": transform_config["instance_count"],
        },
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_job_name(dag_id: str, ts_nodash: str) -> str:
    """Produce a SageMaker-compliant job name from dag_id and ts_nodash.

    Format  :  {dag_id (underscores→hyphens)}-transform-{ts_nodash}  (lowercase)
    Constraints: [a-z0-9\\-], max 63 chars, no leading/trailing hyphen.
    """
    raw = f"{dag_id.replace('_', '-')}-transform-{ts_nodash}".lower()
    sanitized = _INVALID_CHARS.sub("-", raw)
    sanitized = _MULTI_HYPHEN.sub("-", sanitized).strip("-")
    return sanitized[:_MAX_JOB_NAME_LEN]
