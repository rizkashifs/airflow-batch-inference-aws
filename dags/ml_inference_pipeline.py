"""Unified ML Inference Pipeline DAG.

Merges three previously separate DAGs into a single end-to-end pipeline:
  Batch Transform  +  Model Registry resolution  +  Output → RDS loading

Flow:
  validate_input
    → preprocessing_data
    → load_model_from_registry
    → build_transform_config
    → run_batch_transform          (SageMakerTransformOperator)
    → wait_for_inference_output    (S3KeySensor)
    → postprocess_output
    → validate_output
    → load_to_rds
    → log_status                   (trigger_rule=ALL_DONE)

Configuration (Airflow Variables — set in MWAA UI or via API):
    ml_pipeline_aws_region              default: ca-central-1
    ml_pipeline_model_package_group     REQUIRED
    ml_pipeline_s3_bucket               REQUIRED
    ml_pipeline_db_secret_arn           REQUIRED
    ml_pipeline_db_table                default: batch_inference_results
    ml_pipeline_default_instance_type   default: ml.m5.xlarge

MWAA deployment note:
    Upload the entire utils/ directory inside the dags/ S3 prefix so that
    the DAG processor finds it on sys.path:
        s3://<mwaa-bucket>/dags/utils/
        s3://<mwaa-bucket>/dags/ml_inference_pipeline.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTransformOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.utils.trigger_rule import TriggerRule

from utils import registry, s3_utils, sagemaker as sm_utils, validation, db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — fetch Variables at task runtime to avoid parse-time failures
# ---------------------------------------------------------------------------

def _cfg(key: str, default: str | None = None) -> str:
    """Wrapper around Variable.get() that fails clearly when a required var is absent."""
    val = Variable.get(key, default_var=default)
    if val is None:
        raise RuntimeError(
            f"Airflow Variable '{key}' is not set. "
            "Configure it in the MWAA UI under Admin → Variables before running."
        )
    return val


# ---------------------------------------------------------------------------
# SageMakerTransformOperator config — all dynamic fields resolved via XCom
# Jinja rendering happens before execute(); string values with {{ }} are safe.
# ---------------------------------------------------------------------------
_TRANSFORM_OP_CONFIG = {
    "TransformJobName": (
        "{{ ti.xcom_pull(task_ids='build_transform_config', key='job_name') }}"
    ),
    "ModelName": (
        "{{ ti.xcom_pull(task_ids='build_transform_config', key='model_name') }}"
    ),
    "BatchStrategy": "SingleRecord",
    "MaxConcurrentTransforms": 1,
    "MaxPayloadInMB": 6,
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": (
                    "{{ ti.xcom_pull(task_ids='build_transform_config', key='input_s3') }}"
                ),
            }
        },
        "ContentType": "text/csv",
        "SplitType": "Line",
    },
    "TransformOutput": {
        "S3OutputPath": (
            "{{ ti.xcom_pull(task_ids='build_transform_config', key='output_s3') }}"
        ),
        "AssembleWith": "Line",
    },
    "TransformResources": {
        "InstanceType": (
            "{{ ti.xcom_pull(task_ids='build_transform_config', key='instance_type') }}"
        ),
        "InstanceCount": 1,
    },
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _validate_input(**context) -> dict:
    """Validate dag_run.conf and push normalised overrides to XCom."""
    conf: dict = context.get("dag_run").conf or {}
    validated = validation.validate_dag_conf(conf)
    context["ti"].xcom_push(key="validated_conf", value=validated)
    return validated


def _preprocessing_data(**context) -> dict:
    """Verify input data is present in S3 and return a summary.

    In a more complex pipeline this task would also perform feature engineering
    or format normalisation before inference.
    """
    ti = context["ti"]
    validated_conf: dict = ti.xcom_pull(task_ids="validate_input", key="validated_conf") or {}

    execution_date: str = (
        validated_conf.get("execution_date_override")
        or context["ds"]
    )
    bucket = _cfg("ml_pipeline_s3_bucket")
    prefix = f"batch/input/{execution_date}/"

    summary = s3_utils.validate_input_data(bucket, prefix)
    logger.info(
        "Preprocessing complete — execution_date: %s  input summary: %s",
        execution_date,
        summary,
    )
    ti.xcom_push(key="execution_date", value=execution_date)
    return summary


def _load_model_from_registry(**context) -> dict:
    """Query SageMaker Model Registry for the latest APPROVED model package."""
    ti = context["ti"]
    validated_conf: dict = (
        ti.xcom_pull(task_ids="validate_input", key="validated_conf") or {}
    )

    region = _cfg("ml_pipeline_aws_region", default="ca-central-1")
    group = validated_conf.get("model_package_group_override") or _cfg(
        "ml_pipeline_model_package_group"
    )
    default_instance = validated_conf.get("instance_type_override") or _cfg(
        "ml_pipeline_default_instance_type", default="ml.m5.xlarge"
    )

    model_info = registry.get_latest_approved_model(
        model_package_group_name=group,
        region=region,
        default_instance_type=default_instance,
    )

    # Apply instance_type_override from dag_run.conf if provided
    if "instance_type_override" in validated_conf:
        model_info["instance_type"] = validated_conf["instance_type_override"]
        logger.info(
            "Instance type overridden to: %s", model_info["instance_type"]
        )

    ti.xcom_push(key="model_info", value=model_info)
    return model_info


def _build_transform_config(**context) -> dict:
    """Assemble the full transform job config and push each key to XCom."""
    ti = context["ti"]

    model_info: dict = ti.xcom_pull(task_ids="load_model_from_registry", key="model_info")
    if not model_info:
        raise RuntimeError("model_info XCom is empty — load_model_from_registry may have failed.")

    execution_date: str = (
        ti.xcom_pull(task_ids="preprocessing_data", key="execution_date")
        or context["ds"]
    )

    bucket = _cfg("ml_pipeline_s3_bucket")
    dag_id: str = context["dag"].dag_id
    ts_nodash: str = context["ts_nodash"]

    config = sm_utils.build_transform_config(
        model_info=model_info,
        s3_bucket=bucket,
        execution_date=execution_date,
        dag_id=dag_id,
        ts_nodash=ts_nodash,
    )

    # Push each key individually so Jinja templates in the SageMaker operator
    # and S3KeySensor can reference them as:
    #   {{ ti.xcom_pull(task_ids='build_transform_config', key='<key>') }}
    for k, v in config.items():
        ti.xcom_push(key=k, value=v)

    logger.info("Transform config pushed to XCom: %s", config)
    return config


def _postprocess_output(**context) -> str:
    """Read .csv.out files, combine them, and stage the result to S3.

    Returns the s3:// URI of the staged combined CSV.
    """
    ti = context["ti"]
    bucket: str = ti.xcom_pull(task_ids="build_transform_config", key="output_bucket")
    output_prefix: str = ti.xcom_pull(task_ids="build_transform_config", key="output_prefix")
    processed_key: str = ti.xcom_pull(task_ids="build_transform_config", key="processed_s3_key")

    df = s3_utils.collect_inference_results(bucket=bucket, output_prefix=output_prefix)

    processed_uri = s3_utils.write_dataframe_to_s3(
        df=df, bucket=bucket, key=processed_key
    )
    ti.xcom_push(key="processed_s3_uri", value=processed_uri)
    logger.info("Postprocessing complete — %d rows staged at %s", len(df), processed_uri)
    return processed_uri


def _validate_output(**context) -> None:
    """Read the staged CSV and run structural/quality checks."""
    ti = context["ti"]
    processed_uri: str = ti.xcom_pull(
        task_ids="postprocess_output", key="processed_s3_uri"
    )
    if not processed_uri:
        raise RuntimeError("processed_s3_uri XCom is empty — postprocess_output may have failed.")

    df = s3_utils.read_dataframe_from_s3(processed_uri)
    validation.validate_output_dataframe(df)
    logger.info("Output validation passed for %s", processed_uri)


def _load_to_rds(**context) -> int:
    """Bulk-insert validated inference results into PostgreSQL RDS."""
    ti = context["ti"]

    processed_uri: str = ti.xcom_pull(
        task_ids="postprocess_output", key="processed_s3_uri"
    )
    job_name: str = ti.xcom_pull(task_ids="build_transform_config", key="job_name")
    model_name: str = ti.xcom_pull(task_ids="build_transform_config", key="model_name")
    model_version: str = ti.xcom_pull(task_ids="build_transform_config", key="model_version")
    execution_date: str = (
        ti.xcom_pull(task_ids="preprocessing_data", key="execution_date") or context["ds"]
    )

    region = _cfg("ml_pipeline_aws_region", default="ca-central-1")
    secret_arn = _cfg("ml_pipeline_db_secret_arn")
    table = _cfg("ml_pipeline_db_table", default="batch_inference_results")

    df = s3_utils.read_dataframe_from_s3(processed_uri)

    inserted = db.load_to_rds(
        df=df,
        table=table,
        secret_arn=secret_arn,
        region=region,
        job_name=job_name,
        model_name=model_name,
        model_version=model_version,
        execution_date=execution_date,
    )

    ti.xcom_push(key="inserted_rows", value=inserted)
    logger.info("Loaded %d rows to RDS table '%s'.", inserted, table)
    return inserted


def _log_status(**context) -> None:
    """Log the final pipeline execution summary.

    Uses trigger_rule=ALL_DONE so this task always runs, even after upstream
    failures, providing an audit trail for every DAG run.
    """
    ti = context["ti"]

    def _pull(task_id: str, key: str, default: str = "N/A") -> str:
        val = ti.xcom_pull(task_ids=task_id, key=key)
        return str(val) if val is not None else default

    summary = {
        "dag_id": context["dag"].dag_id,
        "run_id": context["run_id"],
        "execution_date": context["ds"],
        "model_name": _pull("build_transform_config", "model_name"),
        "model_version": _pull("build_transform_config", "model_version"),
        "model_package_arn": _pull("build_transform_config", "model_package_arn"),
        "job_name": _pull("build_transform_config", "job_name"),
        "input_s3": _pull("build_transform_config", "input_s3"),
        "output_s3": _pull("build_transform_config", "output_s3"),
        "processed_s3_uri": _pull("postprocess_output", "processed_s3_uri"),
        "inserted_rows": _pull("load_to_rds", "inserted_rows", default="0"),
        "status": (
            "SUCCESS"
            if context["dag_run"].get_task_instance("load_to_rds").state == "success"
            else "FAILED"
        ),
    }

    border = "=" * 72
    lines = [border, "  ML INFERENCE PIPELINE — EXECUTION SUMMARY", border]
    for k, v in summary.items():
        lines.append(f"  {k:<22}: {v}")
    lines.append(border)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

toronto_tz = pendulum.timezone("America/Toronto")

_DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="ml_inference_pipeline",
    description="Unified end-to-end ML batch inference: registry → transform → RDS",
    schedule_interval="0 23 * * *",
    start_date=datetime(2024, 1, 1, tzinfo=toronto_tz),
    catchup=False,
    max_active_runs=1,
    default_args=_DEFAULT_ARGS,
    tags=["ml", "batch-inference", "sagemaker", "model-registry"],
    doc_md=__doc__,
) as dag:

    # ── 1 ──────────────────────────────────────────────────────────────────
    validate_input = PythonOperator(
        task_id="validate_input",
        python_callable=_validate_input,
        retries=0,
    )

    # ── 2 ──────────────────────────────────────────────────────────────────
    preprocessing_data = PythonOperator(
        task_id="preprocessing_data",
        python_callable=_preprocessing_data,
        retries=2,
    )

    # ── 3 ──────────────────────────────────────────────────────────────────
    load_model_from_registry = PythonOperator(
        task_id="load_model_from_registry",
        python_callable=_load_model_from_registry,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )

    # ── 4 ──────────────────────────────────────────────────────────────────
    build_transform_config = PythonOperator(
        task_id="build_transform_config",
        python_callable=_build_transform_config,
        retries=0,
    )

    # ── 5 ──────────────────────────────────────────────────────────────────
    # SageMakerTransformOperator.template_fields = ("config",)
    # Airflow renders all string values in the config dict before execute(),
    # so XCom pulls resolve correctly even inside nested dicts.
    run_batch_transform = SageMakerTransformOperator(
        task_id="run_batch_transform",
        config=_TRANSFORM_OP_CONFIG,
        aws_conn_id="aws_default",
        wait_for_completion=True,
        check_interval=30,
        max_ingestion_time=6 * 3600,   # 6-hour timeout
        check_if_job_exists=True,
        action_if_job_exists="timestamp",   # auto-suffix to allow DAG re-runs
        retries=1,
        retry_delay=timedelta(minutes=10),
    )

    # ── 6 ──────────────────────────────────────────────────────────────────
    # Secondary safety gate: confirms .csv.out files are visible in S3 before
    # proceeding.  mode=reschedule releases the worker slot while waiting.
    wait_for_inference_output = S3KeySensor(
        task_id="wait_for_inference_output",
        bucket_name=(
            "{{ ti.xcom_pull(task_ids='build_transform_config', key='output_bucket') }}"
        ),
        bucket_key=(
            "{{ ti.xcom_pull(task_ids='build_transform_config', key='output_prefix') }}"
            "*.csv.out"
        ),
        wildcard_match=True,
        aws_conn_id="aws_default",
        timeout=30 * 60,        # 30-minute window after transform job succeeds
        poke_interval=30,
        mode="reschedule",
        soft_fail=False,
    )

    # ── 7 ──────────────────────────────────────────────────────────────────
    postprocess_output = PythonOperator(
        task_id="postprocess_output",
        python_callable=_postprocess_output,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )

    # ── 8 ──────────────────────────────────────────────────────────────────
    validate_output = PythonOperator(
        task_id="validate_output",
        python_callable=_validate_output,
        retries=0,
    )

    # ── 9 ──────────────────────────────────────────────────────────────────
    load_to_rds = PythonOperator(
        task_id="load_to_rds",
        python_callable=_load_to_rds,
        retries=3,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )

    # ── 10 ─────────────────────────────────────────────────────────────────
    log_status = PythonOperator(
        task_id="log_status",
        python_callable=_log_status,
        trigger_rule=TriggerRule.ALL_DONE,   # always runs, even after upstream failure
        retries=0,
    )

    # ── Dependency chain ───────────────────────────────────────────────────
    (
        validate_input
        >> preprocessing_data
        >> load_model_from_registry
        >> build_transform_config
        >> run_batch_transform
        >> wait_for_inference_output
        >> postprocess_output
        >> validate_output
        >> load_to_rds
        >> log_status
    )
