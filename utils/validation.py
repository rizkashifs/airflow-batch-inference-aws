"""Input and output validation utilities.

All validators raise ValueError (or a subclass) on failure so that the
Airflow task is marked as FAILED with an actionable error message.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in the combined inference output DataFrame.
REQUIRED_OUTPUT_COLUMNS: frozenset[str] = frozenset(
    {"source_file", "row_index", "raw_output", "prediction"}
)


# ---------------------------------------------------------------------------
# DAG-run conf validation
# ---------------------------------------------------------------------------

def validate_dag_conf(conf: Optional[dict]) -> dict:
    """Validate and normalise the dag_run.conf payload.

    All fields are optional; their presence triggers format checks.

    Accepted keys:
        execution_date_override      – YYYY-MM-DD override (default: use ds)
        model_package_group_override – override Model Package Group name
        instance_type_override       – override ML instance type for the job

    Returns the validated subset of conf (keys that were supplied and valid).
    Raises ValueError on any format violation.
    """
    if not conf:
        logger.info("No dag_run.conf supplied; using Airflow Variable defaults.")
        return {}

    validated: dict = {}
    errors: list[str] = []

    exec_date = conf.get("execution_date_override")
    if exec_date is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(exec_date)):
            errors.append(
                f"execution_date_override must be YYYY-MM-DD, got: {exec_date!r}"
            )
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

    logger.info("dag_run.conf validated. Effective overrides: %s", validated)
    return validated


# ---------------------------------------------------------------------------
# Output DataFrame validation
# ---------------------------------------------------------------------------

def validate_output_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: Optional[frozenset] = None,
    max_null_prediction_ratio: float = 0.05,
) -> None:
    """Validate the combined inference output DataFrame.

    Checks:
      1. At least *min_rows* rows exist.
      2. All *required_columns* are present.
      3. Null prediction ratio does not exceed *max_null_prediction_ratio*.

    Raises ValueError listing all failures (not just the first).
    """
    if required_columns is None:
        required_columns = REQUIRED_OUTPUT_COLUMNS

    errors: list[str] = []

    # 1 – row count
    if len(df) < min_rows:
        errors.append(
            f"Row count {len(df)} is below the minimum of {min_rows}."
        )

    # 2 – required columns
    missing = required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}.")

    # 3 – null predictions
    if "prediction" in df.columns:
        null_count: int = int(df["prediction"].isna().sum())
        null_ratio: float = null_count / max(len(df), 1)
        if null_ratio > max_null_prediction_ratio:
            errors.append(
                f"Null prediction ratio {null_ratio:.1%} ({null_count}/{len(df)} rows) "
                f"exceeds the allowed threshold of {max_null_prediction_ratio:.1%}."
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
