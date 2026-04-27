"""Input and output validation utilities.

All validators raise ValueError on failure so Airflow marks the task as FAILED
with an actionable message rather than silently continuing with bad data.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Every combined inference DataFrame must contain these columns.
# They are produced by s3_utils.collect_inference_results().
REQUIRED_OUTPUT_COLUMNS: frozenset[str] = frozenset(
    {"source_file", "row_index", "raw_output", "prediction"}
)


# ---------------------------------------------------------------------------
# DAG-run conf validation
# ---------------------------------------------------------------------------

def validate_dag_conf(conf: dict | None) -> dict:
    """Validate and normalise ``dag_run.conf``.

    All fields are optional.  When present, each is format-checked before being
    forwarded to downstream tasks.  Supplying an invalid value fails the run
    immediately rather than letting a bad value propagate silently.

    Accepted keys:
        execution_date_override      – YYYY-MM-DD; overrides the Airflow ``ds`` macro
        model_package_group_override – use a different Model Registry group for this run
        instance_type_override       – override the ML instance type (e.g. ml.m5.2xlarge)

    Returns the validated subset of *conf*.
    """
    if not conf:
        logger.info("No dag_run.conf supplied — using Airflow Variable defaults.")
        return {}

    validated: dict = {}
    errors: list[str] = []

    exec_date = conf.get("execution_date_override")
    if exec_date is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(exec_date)):
            errors.append(f"execution_date_override must be YYYY-MM-DD, got: {exec_date!r}")
        else:
            validated["execution_date_override"] = str(exec_date)

    group_override = conf.get("model_package_group_override")
    if group_override is not None:
        if not str(group_override).strip():
            errors.append("model_package_group_override must be a non-empty string.")
        else:
            validated["model_package_group_override"] = str(group_override).strip()

    instance_override = conf.get("instance_type_override")
    if instance_override is not None:
        if not re.fullmatch(r"ml\.[a-z0-9]+\.[a-z0-9]+", str(instance_override)):
            errors.append(
                f"instance_type_override must match 'ml.<family>.<size>', "
                f"got: {instance_override!r}"
            )
        else:
            validated["instance_type_override"] = str(instance_override)

    if errors:
        msg = "dag_run.conf validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("dag_run.conf validated — effective overrides: %s", validated)
    return validated


# ---------------------------------------------------------------------------
# Inference output validation
# ---------------------------------------------------------------------------

def validate_output_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: frozenset | None = None,
    max_null_prediction_ratio: float = 0.05,
) -> None:
    """Validate the combined inference output DataFrame.

    Runs all checks before raising so the error message lists every problem at
    once rather than forcing repeated fix-run-fix cycles.

    Checks:
        1. Row count is at least *min_rows*.
        2. All *required_columns* are present.
        3. Null ``prediction`` ratio does not exceed *max_null_prediction_ratio*.
    """
    if required_columns is None:
        required_columns = REQUIRED_OUTPUT_COLUMNS

    errors: list[str] = []

    if len(df) < min_rows:
        errors.append(f"Row count {len(df)} is below minimum {min_rows}.")

    missing = required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}.")

    if "prediction" in df.columns:
        null_count = int(df["prediction"].isna().sum())
        null_ratio = null_count / max(len(df), 1)
        if null_ratio > max_null_prediction_ratio:
            errors.append(
                f"Null prediction ratio {null_ratio:.1%} ({null_count}/{len(df)} rows) "
                f"exceeds threshold of {max_null_prediction_ratio:.1%}."
            )

    if errors:
        msg = "Output validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Output validation passed — %d rows, columns: %s",
        len(df),
        sorted(df.columns.tolist()),
    )
