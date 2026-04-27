"""PostgreSQL (RDS) loader for batch inference results.

Uses AWS Secrets Manager for credentials, psycopg2 for the connection, and
execute_values for efficient bulk inserts with ON CONFLICT DO NOTHING
idempotency (safe to re-run for the same job_name).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Generator

import boto3
import pandas as pd
import psycopg2
import psycopg2.extras
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — created at runtime only if the table is absent.
# For production MWAA, prefer creating this table via IaC (Terraform /
# CloudFormation) and granting INSERT/SELECT to the pipeline user only.
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
CREATE INDEX IF NOT EXISTS idx_{safe_table}_job ON {table} (job_name);
CREATE INDEX IF NOT EXISTS idx_{safe_table}_exec_date ON {table} (execution_date);
"""

_UPSERT_SQL = """
INSERT INTO {table}
    (job_name, model_name, model_version, execution_date,
     row_index, source_file, raw_output, prediction)
VALUES %s
ON CONFLICT (job_name, row_index) DO NOTHING;
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_db_secret(secret_arn: str, region: str) -> dict:
    """Retrieve RDS credentials from AWS Secrets Manager.

    Expected secret format (JSON string):
        { "host": "…", "port": 5432, "dbname": "…",
          "username": "…", "password": "…" }

    Returns a dict suitable as **kwargs for psycopg2.connect().
    """
    sm = boto3.client("secretsmanager", region_name=region)
    try:
        resp = sm.get_secret_value(SecretId=secret_arn)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        logger.error("Secrets Manager error (%s) fetching '%s': %s", code, secret_arn, exc)
        raise

    raw = resp.get("SecretString")
    if not raw:
        raise RuntimeError(f"Secret '{secret_arn}' has no SecretString (binary secrets not supported).")

    try:
        secret = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Secret '{secret_arn}' is not valid JSON.") from exc

    required = {"host", "username", "password", "dbname"}
    missing = required - set(secret.keys())
    if missing:
        raise KeyError(
            f"Secret '{secret_arn}' is missing required key(s): {missing}. "
            f"Present keys: {set(secret.keys())}."
        )

    return {
        "host": secret["host"],
        "port": int(secret.get("port", 5432)),
        "dbname": secret["dbname"],
        "user": secret["username"],
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

    Idempotent: rows with an existing (job_name, row_index) pair are skipped
    (ON CONFLICT DO NOTHING), so re-running the DAG for the same date is safe.

    Args:
        df:             Combined inference DataFrame from s3_utils.collect_inference_results.
        table:          Target table name (e.g. 'batch_inference_results').
        secret_arn:     Secrets Manager ARN holding the RDS credentials.
        region:         AWS region for both Secrets Manager and the RDS endpoint.
        job_name:       SageMaker transform job name (stored as metadata).
        model_name:     SageMaker model name.
        model_version:  Model version string.
        execution_date: Airflow ds macro value YYYY-MM-DD.

    Returns:
        Number of rows actually inserted (0 if all were duplicates).
    """
    conn_params = fetch_db_secret(secret_arn, region)
    rows = _to_row_tuples(df, job_name, model_name, model_version, execution_date)

    with _connection(conn_params) as conn:
        with conn.cursor() as cur:
            _ensure_table(cur, table)
            inserted = _bulk_upsert(cur, table, rows)
        conn.commit()

    logger.info(
        "RDS load complete — table: %s  job: %s  inserted: %d / %d",
        table,
        job_name,
        inserted,
        len(rows),
    )
    return inserted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

@contextmanager
def _connection(params: dict) -> Generator:
    conn = psycopg2.connect(**params)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_table(cursor, table: str) -> None:
    """Idempotently create the target table and its indexes.

    Uses a savepoint so that a failure (e.g. insufficient CREATE privilege)
    does not abort the surrounding transaction.  If creation fails we log a
    warning and assume the table already exists.
    """
    safe = table.replace(".", "_").replace("-", "_")
    ddl = _CREATE_TABLE_SQL.format(table=table, safe_table=safe)
    try:
        cursor.execute("SAVEPOINT ensure_table_sp")
        cursor.execute(ddl)
        cursor.execute("RELEASE SAVEPOINT ensure_table_sp")
        logger.debug("Table '%s' ensured.", table)
    except psycopg2.Error as exc:
        cursor.execute("ROLLBACK TO SAVEPOINT ensure_table_sp")
        cursor.execute("RELEASE SAVEPOINT ensure_table_sp")
        logger.warning(
            "Could not create table '%s' (assuming it exists): %s", table, exc
        )


def _bulk_upsert(cursor, table: str, rows: list[tuple]) -> int:
    if not rows:
        logger.warning("No rows to insert into '%s'.", table)
        return 0

    psycopg2.extras.execute_values(
        cursor,
        _UPSERT_SQL.format(table=table),
        rows,
        page_size=500,
    )
    # rowcount after execute_values reflects actual rows inserted
    return cursor.rowcount if cursor.rowcount != -1 else len(rows)


def _to_row_tuples(
    df: pd.DataFrame,
    job_name: str,
    model_name: str,
    model_version: str,
    execution_date: str,
) -> list[tuple]:
    tuples: list[tuple] = []
    for _, row in df.iterrows():
        prediction = row.get("prediction")
        tuples.append(
            (
                job_name,
                model_name,
                model_version,
                execution_date,
                int(row.get("row_index", 0)),
                str(row.get("source_file", "")),
                str(row.get("raw_output", "")),
                float(prediction) if pd.notna(prediction) else None,
            )
        )
    return tuples
