"""PostgreSQL (RDS) loader for batch inference results.

Uses AWS Secrets Manager for credentials and psycopg2 ``execute_values`` for
efficient bulk inserts.  Inserts are idempotent: re-running the DAG for the
same job_name skips rows that already exist (ON CONFLICT DO NOTHING).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import boto3
import pandas as pd
import psycopg2
import psycopg2.extras
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# For production MWAA, create this table via IaC (Terraform / CloudFormation)
# and grant only INSERT + SELECT to the pipeline user.  The runtime CREATE IF
# NOT EXISTS below is a convenience for first-run bootstrapping.
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id              BIGSERIAL        PRIMARY KEY,
    job_name        VARCHAR(63)      NOT NULL,
    model_name      VARCHAR(255)     NOT NULL,
    model_version   VARCHAR(63),
    execution_date  DATE             NOT NULL,
    row_index       INTEGER          NOT NULL,
    source_file     TEXT,
    raw_output      TEXT,
    prediction      DOUBLE PRECISION,
    loaded_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_{safe_table}_job_row UNIQUE (job_name, row_index)
);
CREATE INDEX IF NOT EXISTS idx_{safe_table}_job       ON {table} (job_name);
CREATE INDEX IF NOT EXISTS idx_{safe_table}_exec_date ON {table} (execution_date);
"""

_UPSERT_SQL = """
INSERT INTO {table}
    (job_name, model_name, model_version, execution_date,
     row_index, source_file, raw_output, prediction)
VALUES %s
ON CONFLICT (job_name, row_index) DO NOTHING
RETURNING 1
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_db_secret(secret_arn: str, region: str) -> dict:
    """Retrieve RDS connection credentials from AWS Secrets Manager.

    Expected secret format (JSON):
        { "host": "…", "port": 5432, "dbname": "…",
          "username": "…", "password": "…" }

    Returns a dict ready to be unpacked as ``**kwargs`` into ``psycopg2.connect()``.
    """
    sm = boto3.client("secretsmanager", region_name=region)
    try:
        resp = sm.get_secret_value(SecretId=secret_arn)
    except ClientError as exc:
        logger.error(
            "Secrets Manager error (%s) for '%s': %s",
            exc.response["Error"]["Code"], secret_arn, exc,
        )
        raise

    raw = resp.get("SecretString")
    if not raw:
        raise RuntimeError(
            f"Secret '{secret_arn}' contains no SecretString. "
            "Binary secrets are not supported."
        )

    try:
        secret = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Secret '{secret_arn}' is not valid JSON.") from exc

    missing = {"host", "username", "password", "dbname"} - set(secret)
    if missing:
        raise KeyError(
            f"Secret '{secret_arn}' is missing key(s): {missing}. "
            f"Found: {set(secret)}."
        )

    return {
        "host":     secret["host"],
        "port":     int(secret.get("port", 5432)),
        "dbname":   secret["dbname"],
        "user":     secret["username"],
        "password": secret["password"],
    }


def load_to_rds(
    df: pd.DataFrame,
    table: str,
    secret_arn: str,
    region: str,
    job_name: str,
    model_name: str,
    model_version: str,
    execution_date: str,
) -> int:
    """Bulk-insert inference results into PostgreSQL.

    Args:
        df:             Combined inference DataFrame (from s3_utils.collect_inference_results).
        table:          Target table name.
        secret_arn:     Secrets Manager ARN for the RDS credentials.
        region:         AWS region (used for both Secrets Manager and the RDS endpoint).
        job_name:       SageMaker transform job name — stored as metadata, also used
                        as part of the uniqueness key for idempotent inserts.
        model_name:     Model name sourced from the Model Registry.
        model_version:  Model version string.
        execution_date: Airflow ds macro value (YYYY-MM-DD).

    Returns:
        Number of rows actually inserted (0 when all rows already exist).
    """
    conn_params = fetch_db_secret(secret_arn, region)
    rows = _build_rows(df, job_name, model_name, model_version, execution_date)

    with _connection(conn_params) as conn:
        with conn.cursor() as cur:
            _ensure_table(cur, table)
            inserted = _bulk_upsert(cur, table, rows)
        conn.commit()

    logger.info(
        "RDS load complete — table: %s  job: %s  inserted: %d / %d",
        table, job_name, inserted, len(rows),
    )
    return inserted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

@contextmanager
def _connection(params: dict):
    """Open a psycopg2 connection, roll back on error, always close."""
    conn = psycopg2.connect(**params)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_table(cursor, table: str) -> None:
    """Create the target table if it does not already exist.

    Wraps the DDL in a savepoint because psycopg2 does not support partial
    transaction rollbacks without one.  If CREATE TABLE fails (e.g. the pipeline
    user lacks DDL privileges), we verify the table actually exists before
    continuing — raising a clear RuntimeError if it does not.

    GUIDE: WHY THE SAVEPOINT?
    Postgres treats any failed command (like a CREATE TABLE that errors on
    permissions) as a "dead" transaction that must be rolled back entirely.
    """
    safe = table.replace(".", "_").replace("-", "_")
    try:
        cursor.execute("SAVEPOINT ensure_table_sp")
        cursor.execute(_CREATE_TABLE_SQL.format(table=table, safe_table=safe))
        cursor.execute("RELEASE SAVEPOINT ensure_table_sp")
    except psycopg2.Error as exc:
        cursor.execute("ROLLBACK TO SAVEPOINT ensure_table_sp")
        cursor.execute("RELEASE SAVEPOINT ensure_table_sp")
        # Confirm the table exists before continuing; to_regclass returns NULL when absent.
        cursor.execute("SELECT to_regclass(%s)", (table,))
        if cursor.fetchone()[0] is None:
            raise RuntimeError(
                f"Table '{table}' does not exist and could not be created: {exc}. "
                "Pre-create it via IaC or grant CREATE TABLE to the pipeline user."
            ) from exc
        logger.info("Table '%s' already exists — DDL skipped.", table)


def _bulk_upsert(cursor, table: str, rows: list[tuple]) -> int:
    if not rows:
        logger.warning("No rows to insert into '%s'.", table)
        return 0
    # fetch=True + RETURNING 1: each actually-inserted row returns one record.
    # Rows skipped by ON CONFLICT DO NOTHING are absent from the result, giving
    # an accurate count even when execute_values pages across multiple batches.
    result = psycopg2.extras.execute_values(
        cursor, _UPSERT_SQL.format(table=table), rows, page_size=500, fetch=True,
    )
    return len(result)


def _build_rows(
    df: pd.DataFrame,
    job_name: str,
    model_name: str,
    model_version: str,
    execution_date: str,
) -> list[tuple]:
    """Convert the inference DataFrame into a list of tuples for execute_values.

    Uses itertuples (roughly 10x faster than iterrows for large DataFrames)
    because we are reading every row sequentially to build the insert batch.

    GUIDE: WHY ITERTUPLES?
    We use itertuples() because it is significantly faster (roughly 10x) than
    iterrows(). For bulk loads of 100k+ rows, iterrows() can stall the Airflow
    worker for minutes.
    """
    return [
        (
            job_name,
            model_name,
            model_version,
            execution_date,
            int(row.row_index),
            str(row.source_file),
            str(row.raw_output),
            float(row.prediction) if pd.notna(row.prediction) else None,
        )
        for row in df.itertuples(index=False)
    ]
