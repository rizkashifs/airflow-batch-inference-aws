# airflow-batch-inference-aws

An AWS MWAA Airflow pipeline that runs daily ML batch inference end-to-end:
resolves the latest approved model from SageMaker Model Registry, runs a
SageMaker Batch Transform job, validates the output, and loads results into
PostgreSQL RDS.

---

## What it does (pipeline overview)

```
validate_input
  → preprocessing_data
  → load_model_from_registry
  → build_transform_config
  → run_batch_transform          ← SageMakerTransformOperator
  → wait_for_inference_output    ← S3KeySensor
  → postprocess_output
  → validate_output
  → load_to_rds
  → log_status
```

| Step | What happens |
|------|-------------|
| **validate_input** | Checks `dag_run.conf` for optional overrides (date, model group, instance type) and rejects malformed values immediately |
| **preprocessing_data** | Confirms input CSVs exist and are non-empty in `s3://<bucket>/batch/input/<date>/`; resolves the effective execution date |
| **load_model_from_registry** | Calls `list_model_packages` + `describe_model_package` to find the latest `Approved` model; extracts model name, version, and instance type — **never creates a model** |
| **build_transform_config** | Builds the job config (name, S3 paths, compute) and pushes every field to XCom so downstream tasks can reference them |
| **run_batch_transform** | Submits a SageMaker Batch Transform job, polls every 30 s, waits up to 6 hours |
| **wait_for_inference_output** | S3KeySensor confirms `.csv.out` files are visible in S3 before proceeding (safety gate after the transform) |
| **postprocess_output** | Reads all `.csv.out` files, combines them into a single DataFrame, and stages it to `s3://<bucket>/batch/processed/<date>/combined.csv` |
| **validate_output** | Checks row count > 0, required columns exist, and null-prediction ratio ≤ 5% |
| **load_to_rds** | Bulk-inserts results into PostgreSQL using `ON CONFLICT DO NOTHING` — safe to re-run |
| **log_status** | Always runs (even after failures) and logs a structured summary: model, job, S3 paths, rows inserted |

---

## Repository structure

```
.
├── dags/
│   └── ml_inference_pipeline.py   # Airflow DAG — thin orchestrator only
│                                  # All business logic lives in utils/
└── utils/
    ├── __init__.py
    ├── registry.py      # Queries SageMaker Model Registry
    ├── sagemaker.py     # Builds the transform job config dict
    ├── s3_utils.py      # Lists, reads, and writes S3 objects
    ├── validation.py    # Validates dag_run.conf and output DataFrames
    └── db.py            # Loads results into PostgreSQL via psycopg2
```

### File descriptions

**`dags/ml_inference_pipeline.py`**
The only file Airflow directly parses.  Defines the DAG, all task operators,
and thin callable functions that delegate to `utils/`.  Does not contain any
AWS or database logic.

**`utils/registry.py`**
`get_latest_approved_model(group, region)` — pages through
`list_model_packages` (sorted by CreationTime DESC, filtered to `Approved`),
calls `describe_model_package` on the newest one, and returns a plain dict
with `model_name`, `model_version`, `instance_type`, and `model_package_arn`.
The model name is read from `CustomerMetadataProperties` (key
`sagemaker_model_name` or `model_name`), falling back to the group name.

**`utils/sagemaker.py`**
`build_transform_config(model_info, bucket, execution_date, dag_id, ts_nodash)`
— assembles the flat config dict for a single run.  All S3 paths are
date-partitioned (`batch/input/<date>/`, `batch/output/<date>/`).  The job name
is built as `<dag_id>-transform-<ts_nodash>` (lowercase, hyphens only, max 63
chars) so it is unique per trigger even when re-running the same date.

**`utils/s3_utils.py`**
Four public functions:
- `validate_input_data(bucket, prefix)` — verifies CSV files exist and are non-empty
- `collect_inference_results(bucket, prefix)` — reads all `.csv.out` files and
  returns a combined DataFrame (`source_file`, `row_index`, `raw_output`, `prediction`)
- `write_dataframe_to_s3(df, bucket, key)` — uploads a DataFrame as CSV; returns the S3 URI
- `read_dataframe_from_s3(uri)` — reads a CSV from an `s3://` URI into a DataFrame

**`utils/validation.py`**
- `validate_dag_conf(conf)` — format-checks optional overrides in `dag_run.conf`
- `validate_output_dataframe(df)` — checks row count, required columns, and null-prediction ratio;
  collects all errors before raising so every problem is reported at once

**`utils/db.py`**
- `fetch_db_secret(arn, region)` — retrieves credentials from AWS Secrets Manager
- `load_to_rds(df, table, ...)` — bulk-inserts via `execute_values` with
  `ON CONFLICT (job_name, row_index) DO NOTHING` for idempotency;
  creates the target table if it does not exist (using a savepoint so a
  privilege failure does not abort the transaction)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS MWAA 2.8.x | Airflow 2.8, Python 3.11 |
| SageMaker Model Registry | At least one `Approved` model package in the configured group |
| S3 bucket | Input CSVs placed at `batch/input/<YYYY-MM-DD>/` before the DAG runs |
| RDS PostgreSQL | Accessible from MWAA VPC; credentials stored in Secrets Manager |
| MWAA execution role | Needs permissions for SageMaker, S3, Secrets Manager (see below) |

### Minimum IAM permissions for the MWAA execution role

```json
{
  "Effect": "Allow",
  "Action": [
    "sagemaker:ListModelPackages",
    "sagemaker:DescribeModelPackage",
    "sagemaker:CreateTransformJob",
    "sagemaker:DescribeTransformJob",
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket",
    "secretsmanager:GetSecretValue"
  ],
  "Resource": "*"
}
```

---

## Setup steps

### 1. Install the extra dependency

Upload `requirements.txt` to the MWAA S3 bucket and point the environment at it.
The only non-bundled package is `psycopg2-binary`.

```
s3://<mwaa-bucket>/requirements.txt
```

### 2. Deploy the DAG and utils

MWAA adds everything inside `dags/` to `sys.path`, so `utils/` must live
**inside** the `dags/` S3 prefix:

```
s3://<mwaa-bucket>/dags/ml_inference_pipeline.py
s3://<mwaa-bucket>/dags/utils/__init__.py
s3://<mwaa-bucket>/dags/utils/registry.py
s3://<mwaa-bucket>/dags/utils/sagemaker.py
s3://<mwaa-bucket>/dags/utils/s3_utils.py
s3://<mwaa-bucket>/dags/utils/validation.py
s3://<mwaa-bucket>/dags/utils/db.py
```

Sync command:

```bash
aws s3 sync dags/         s3://<mwaa-bucket>/dags/
aws s3 sync utils/        s3://<mwaa-bucket>/dags/utils/
aws s3 cp requirements.txt s3://<mwaa-bucket>/requirements.txt
```

### 3. Set Airflow Variables

In the MWAA UI go to **Admin → Variables** and add:

| Key | Example value | Required |
|-----|---------------|----------|
| `ml_pipeline_s3_bucket` | `my-mlops-bucket` | Yes |
| `ml_pipeline_model_package_group` | `my-model-group` | Yes |
| `ml_pipeline_db_secret_arn` | `arn:aws:secretsmanager:ca-central-1:123:secret:rds-creds` | Yes |
| `ml_pipeline_aws_region` | `ca-central-1` | No (default: `ca-central-1`) |
| `ml_pipeline_db_table` | `batch_inference_results` | No (default: `batch_inference_results`) |
| `ml_pipeline_default_instance_type` | `ml.m5.xlarge` | No (default: `ml.m5.xlarge`) |

### 4. Confirm the Secrets Manager secret format

The secret pointed to by `ml_pipeline_db_secret_arn` must be a JSON string:

```json
{
  "host":     "my-rds-instance.xxxx.ca-central-1.rds.amazonaws.com",
  "port":     5432,
  "dbname":   "mlops",
  "username": "pipeline_user",
  "password": "…"
}
```

### 5. Place input data

Upload your input CSV files to S3 before the scheduled run time:

```
s3://<bucket>/batch/input/2024-01-15/features.csv
```

The DAG reads whatever files are under that prefix, so multiple files are fine.

---

## Running the DAG

### Scheduled run
The DAG runs daily at **23:00 Toronto time** (`0 23 * * *`).

### Manual trigger (standard)
In the MWAA UI, click **Trigger DAG** with no config.

### Manual trigger with overrides
Trigger with a JSON conf body to override defaults for a single run:

```json
{
  "execution_date_override":      "2024-01-10",
  "model_package_group_override": "my-other-group",
  "instance_type_override":       "ml.m5.2xlarge"
}
```

All three keys are optional and independently validated.

---

## Database schema

The table is created automatically on first run if it does not exist.

```sql
CREATE TABLE batch_inference_results (
    id              BIGSERIAL        PRIMARY KEY,
    job_name        VARCHAR(63)      NOT NULL,   -- SageMaker transform job name
    model_name      VARCHAR(255)     NOT NULL,
    model_version   VARCHAR(63),
    execution_date  DATE             NOT NULL,
    row_index       INTEGER          NOT NULL,   -- 0-based, unique per job
    source_file     TEXT,                        -- S3 key of the .csv.out file
    raw_output      TEXT,                        -- raw prediction line
    prediction      DOUBLE PRECISION,            -- first CSV field cast to float
    loaded_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    UNIQUE (job_name, row_index)
);
```

Re-running the DAG for the same date produces the same `job_name` only if using
the same trigger timestamp.  Because `action_if_job_exists="timestamp"` appends
a suffix to duplicate job names in SageMaker, a re-run always creates a new job
and inserts its output as new rows.

---

## Local development

```bash
# 1 – install dependencies
pip install apache-airflow apache-airflow-providers-amazon psycopg2-binary pandas boto3

# 2 – set PYTHONPATH so utils/ is importable from dags/
export PYTHONPATH=$(pwd):$PYTHONPATH
export AIRFLOW_HOME=$(pwd)

# 3 – initialise the Airflow metadata DB (SQLite, for local use only)
airflow db init

# 4 – parse the DAG (no task execution)
python dags/ml_inference_pipeline.py

# 5 – run a single task locally (requires real AWS credentials)
airflow tasks test ml_inference_pipeline validate_input 2024-01-15
```
