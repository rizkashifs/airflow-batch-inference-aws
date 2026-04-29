"""Unified ML Inference Pipeline DAG.

Merges three previously separate DAGs into one end-to-end pipeline:
  Model Registry resolution  →  SageMaker Batch Transform  →  Output → RDS

Task flow:
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

Required Airflow Variables (set in MWAA UI → Admin → Variables):
    ml_pipeline_s3_bucket              S3 bucket holding input/output data
    ml_pipeline_model_package_group    SageMaker Model Package Group name
    ml_pipeline_db_secret_arn          Secrets Manager ARN for RDS credentials

Optional Airflow Variables (have sensible defaults):
    ml_pipeline_aws_region             default: ca-central-1
    ml_pipeline_db_table               default: batch_inference_results
    ml_pipeline_default_instance_type  default: ml.m5.xlarge

MWAA deployment:
    Place utils/ inside the dags/ S3 prefix so the DAG processor adds it to
    sys.path automatically:
        s3://<mwaa-bucket>/dags/ml_inference_pipeline.py
        s3://<mwaa-bucket>/dags/utils/
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

from utils import db, registry, s3_utils
from utils import sagemaker as sm_utils
from utils import validation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _cfg(key: str, default: str | None = None) -> str:
    """Fetch an Airflow Variable at task runtime.

    Fetching inside the task callable (not at module level) prevents parse-time
    errors when the scheduler imports the DAG before Variables are set.
    Raises RuntimeError with a clear message when a required variable is absent.
    """
    val = Variable.get(key, default_var=default)
    if val is None:
        raise RuntimeError(
            f"Airflow Variable '{key}' is not set. "
            "Add it under Admin → Variables in the MWAA UI before running."
        )
    return val


# ---------------------------------------------------------------------------
# SageMakerTransformOperator config
#
# SageMakerTransformOperator has template_fields = ("config",), which means
# Airflow renders every string value in this dict through Jinja before the
# operator's execute() is called.  That lets us pull dynamic values — job name,
# model name, S3 paths — directly from XCom without extra glue code.
# Integer values (InstanceCount, MaxConcurrentTransforms, MaxPayloadInMB) are
# not templated and stay constant.
# ---------------------------------------------------------------------------
_TRANSFORM_OP_CONFIG = {
    "TransformJobName": "{{ ti.xcom_pull(task_ids='build_transform_config', key='job_name') }}",
    "ModelName":        "{{ ti.xcom_pull(task_ids='build_transform_config', key='model_name') }}",
    "BatchStrategy":    "SingleRecord",
    "MaxConcurrentTransforms": 1,
    "MaxPayloadInMB": 6,
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "{{ ti.xcom_pull(task_ids='build_transform_config', key='input_s3') }}",
            }
        },
        "ContentType": "text/csv",
        "SplitType":   "Line",
    },
    "TransformOutput": {
        "S3OutputPath": "{{ ti.xcom_pull(task_ids='build_transform_config', key='output_s3') }}",
        "AssembleWith": "Line",
    },
    "TransformResources": {
        "InstanceType":  "{{ ti.xcom_pull(task_ids='build_transform_config', key='instance_type') }}",
        "InstanceCount": 1,
    },
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _validate_input(**context) -> dict:
    """Parse and validate dag_run.conf.

    Accepted optional keys: execution_date_override, model_package_group_override,
    instance_type_override.  All are format-checked before being stored in XCom
    so downstream tasks can trust their values.
    """
    conf = context["dag_run"].conf or {}
    validated = validation.validate_dag_conf(conf)
    context["ti"].xcom_push(key="validated_conf", value=validated)
    return validated


def _preprocessing_data(**context) -> dict:
    """Confirm that input CSV files exist in S3 for the target execution date.

    Also resolves the effective execution_date (which may be overridden via
    dag_run.conf) and stores it in XCom so all downstream tasks use the same
    date consistently.
    """
    ti = context["ti"]
    conf = ti.xcom_pull(task_ids="validate_input", key="validated_conf") or {}

    # Use the override date when provided; otherwise fall back to the Airflow ds macro.
    execution_date: str = conf.get("execution_date_override") or context["ds"]
    bucket = _cfg("ml_pipeline_s3_bucket")

    summary = s3_utils.validate_input_data(bucket, f"batch/input/{execution_date}/")
    ti.xcom_push(key="execution_date", value=execution_date)
    logger.info("Input OK for %s — %s", execution_date, summary)
    return summary


def _load_model_from_registry(**context) -> dict:
    """Query SageMaker Model Registry for the latest APPROVED model package.

    Resolves the model name, instance type, and version without creating any
    SageMaker resources.  dag_run.conf overrides take precedence over the
    Airflow Variable defaults.
    """
    ti = context["ti"]
    conf = ti.xcom_pull(task_ids="validate_input", key="validated_conf") or {}

    region   = _cfg("ml_pipeline_aws_region", default="ca-central-1")
    group    = conf.get("model_package_group_override") or _cfg("ml_pipeline_model_package_group")
    default_instance = conf.get("instance_type_override") or _cfg(
        "ml_pipeline_default_instance_type", default="ml.m5.xlarge"
    )

    model_info = registry.get_latest_approved_model(
        model_package_group_name=group,
        region=region,
        default_instance_type=default_instance,
    )

    # dag_run.conf instance override wins over whatever the registry returned.
    if "instance_type_override" in conf:
        model_info["instance_type"] = conf["instance_type_override"]
        logger.info("Instance type overridden to: %s", model_info["instance_type"])

    ti.xcom_push(key="model_info", value=model_info)
    return model_info


def _build_transform_config(**context) -> dict:
    """Assemble the full transform job config and push each field to XCom.

    Pushes individual keys (not just the whole dict) so the Jinja templates
    inside _TRANSFORM_OP_CONFIG and the S3KeySensor can reference them directly
    with  ti.xcom_pull(task_ids='build_transform_config', key='<field>').
    """
    ti = context["ti"]

    model_info = ti.xcom_pull(task_ids="load_model_from_registry", key="model_info")
    if not model_info:
        raise RuntimeError("model_info XCom is empty — load_model_from_registry failed.")

    execution_date = (
        ti.xcom_pull(task_ids="preprocessing_data", key="execution_date") or context["ds"]
    )

    config = sm_utils.build_transform_config(
        model_info=model_info,
        s3_bucket=_cfg("ml_pipeline_s3_bucket"),
        execution_date=execution_date,
        dag_id=context["dag"].dag_id,
        ts_nodash=context["ts_nodash"],
    )

    for k, v in config.items():
        ti.xcom_push(key=k, value=v)

    logger.info("Transform config: %s", config)
    return config


def _postprocess_output(**context) -> str:
    """Combine all .csv.out files into one DataFrame and stage it to S3.

    Staging to S3 (rather than storing in XCom) avoids the XCom size limit for
    large result sets.  The S3 URI is stored in XCom so validate_output and
    load_to_rds can read the same file independently.
    """
    ti = context["ti"]
    bucket         = ti.xcom_pull(task_ids="build_transform_config", key="output_bucket")
    output_prefix  = ti.xcom_pull(task_ids="build_transform_config", key="output_prefix")
    processed_key  = ti.xcom_pull(task_ids="build_transform_config", key="processed_s3_key")

    df = s3_utils.collect_inference_results(bucket=bucket, output_prefix=output_prefix)
    uri = s3_utils.write_dataframe_to_s3(df=df, bucket=bucket, key=processed_key)

    ti.xcom_push(key="processed_s3_uri", value=uri)
    logger.info("Postprocessing complete — %d rows staged at %s", len(df), uri)
    return uri


def _validate_output(**context) -> None:
    """Read the staged CSV and run structural / quality checks.

    Intentionally kept separate from postprocess_output so that validation
    failures show up as a distinct task failure in the Airflow UI.
    """
    ti = context["ti"]
    uri = ti.xcom_pull(task_ids="postprocess_output", key="processed_s3_uri")
    if not uri:
        raise RuntimeError("processed_s3_uri XCom is empty — postprocess_output failed.")

    df = s3_utils.read_dataframe_from_s3(uri)
    validation.validate_output_dataframe(df)


def _load_to_rds(**context) -> int:
    """Bulk-insert validated inference results into PostgreSQL RDS."""
    ti = context["ti"]

    uri           = ti.xcom_pull(task_ids="postprocess_output", key="processed_s3_uri")
    job_name      = ti.xcom_pull(task_ids="build_transform_config", key="job_name")
    model_name    = ti.xcom_pull(task_ids="build_transform_config", key="model_name")
    model_version = ti.xcom_pull(task_ids="build_transform_config", key="model_version")
    execution_date = (
        ti.xcom_pull(task_ids="preprocessing_data", key="execution_date") or context["ds"]
    )

    inserted = db.load_to_rds(
        df=s3_utils.read_dataframe_from_s3(uri),
        table=_cfg("ml_pipeline_db_table", default="batch_inference_results"),
        secret_arn=_cfg("ml_pipeline_db_secret_arn"),
        region=_cfg("ml_pipeline_aws_region", default="ca-central-1"),
        job_name=job_name,
        model_name=model_name,
        model_version=model_version,
        execution_date=execution_date,
    )

    ti.xcom_push(key="inserted_rows", value=inserted)
    return inserted


def _log_status(**context) -> None:
    """Log a structured summary of the entire pipeline run.

    Runs with trigger_rule=ALL_DONE so it executes even when an upstream task
    fails, providing a consistent audit trail for every DAG run.
    """
    ti = context["ti"]

    def _pull(task_id: str, key: str, default: str = "N/A") -> str:
        val = ti.xcom_pull(task_ids=task_id, key=key)
        return str(val) if val is not None else default

    # Determine run status by checking whether load_to_rds produced a result.
    # Using XCom presence is more reliable than reading task state from the DB.
    inserted = ti.xcom_pull(task_ids="load_to_rds", key="inserted_rows")
    status = "SUCCESS" if inserted is not None else "FAILED"

    summary = {
        "dag_id":            context["dag"].dag_id,
        "run_id":            context["run_id"],
        "execution_date":    context["ds"],
        "status":            status,
        "model_name":        _pull("build_transform_config", "model_name"),
        "model_version":     _pull("build_transform_config", "model_version"),
        "model_package_arn": _pull("build_transform_config", "model_package_arn"),
        "job_name":          _pull("build_transform_config", "job_name"),
        "input_s3":          _pull("build_transform_config", "input_s3"),
        "output_s3":         _pull("build_transform_config", "output_s3"),
        "processed_uri":     _pull("postprocess_output",     "processed_s3_uri"),
        "rows_inserted":     _pull("load_to_rds",            "inserted_rows", default="0"),
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

_DEFAULT_ARGS = {
    "owner":                    "ml-platform",
    "depends_on_past":          False,
    "email_on_failure":         False,
    "email_on_retry":           False,
    "retries":                  1,
    "retry_delay":              timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="ml_inference_pipeline",
    description="End-to-end ML batch inference: Model Registry → SageMaker Transform → RDS",
    schedule="0 23 * * *",             # daily at 23:00 Toronto time
    start_date=datetime(2024, 1, 1, tzinfo=pendulum.timezone("America/Toronto")),
    catchup=False,
    max_active_runs=1,
    default_args=_DEFAULT_ARGS,
    tags=["ml", "batch-inference", "sagemaker", "model-registry"],
    doc_md=__doc__,
) as dag:

    validate_input = PythonOperator(
        task_id="validate_input",
        python_callable=_validate_input,
        retries=0,
    )

    preprocessing_data = PythonOperator(
        task_id="preprocessing_data",
        python_callable=_preprocessing_data,
        retries=2,
    )

    load_model_from_registry = PythonOperator(
        task_id="load_model_from_registry",
        python_callable=_load_model_from_registry,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )

    build_transform_config = PythonOperator(
        task_id="build_transform_config",
        python_callable=_build_transform_config,
        retries=0,
    )

    # SageMakerTransformOperator polls every 30 s and times out after 6 hours.
    # action_if_job_exists="timestamp" appends a suffix on re-runs so the same
    # execution date never causes a job name collision in SageMaker.
    run_batch_transform = SageMakerTransformOperator(
        task_id="run_batch_transform",
        config=_TRANSFORM_OP_CONFIG,
        aws_conn_id="aws_default",
        wait_for_completion=True,
        check_interval=30,
        max_ingestion_time=6 * 3600,
        check_if_job_exists=True,
        action_if_job_exists="timestamp",
        retries=1,
        retry_delay=timedelta(minutes=10),
    )

    # S3KeySensor runs in mode=reschedule — it releases the worker slot between
    # pokes so long waits don't tie up MWAA worker capacity.
    wait_for_inference_output = S3KeySensor(
        task_id="wait_for_inference_output",
        bucket_name="{{ ti.xcom_pull(task_ids='build_transform_config', key='output_bucket') }}",
        bucket_key=(
            "{{ ti.xcom_pull(task_ids='build_transform_config', key='output_prefix') }}"
            "*.csv.out"
        ),
        wildcard_match=True,
        aws_conn_id="aws_default",
        timeout=30 * 60,
        poke_interval=30,
        mode="reschedule",
        soft_fail=False,
    )

    postprocess_output = PythonOperator(
        task_id="postprocess_output",
        python_callable=_postprocess_output,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )

    validate_output = PythonOperator(
        task_id="validate_output",
        python_callable=_validate_output,
        retries=0,
    )

    load_to_rds = PythonOperator(
        task_id="load_to_rds",
        python_callable=_load_to_rds,
        retries=3,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )

    log_status = PythonOperator(
        task_id="log_status",
        python_callable=_log_status,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

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
