# airflow-batch-inference-aws

An AWS MWAA Airflow pipeline that runs daily ML batch inference end-to-end:
resolves the latest approved model from SageMaker Model Registry, runs a
SageMaker Batch Transform job, validates the output, and loads results into
PostgreSQL RDS.

> **Deployment target:** AWS MWAA 2.8.x (Airflow 2.8, Python 3.11).
> The DAG imports Airflow and Amazon provider classes that are only available
> inside an MWAA environment; it cannot be executed locally.
> See [Local development](#local-development) for running the unit tests locally.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Infrastructure requirements](#infrastructure-requirements)
4. [Deployment](#deployment)
5. [Airflow Variables](#airflow-variables)
6. [First-run checklist](#first-run-checklist)
7. [Running the DAG](#running-the-dag)
8. [Database schema](#database-schema)
9. [Local development](#local-development)

---

## Pipeline overview

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
| **wait_for_inference_output** | S3KeySensor confirms `.csv.out` files are visible in S3 before proceeding |
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
└── utils/
    ├── __init__.py
    ├── registry.py      # Queries SageMaker Model Registry
    ├── sagemaker.py     # Builds the transform job config dict
    ├── s3_utils.py      # Lists, reads, and writes S3 objects
    ├── validation.py    # Validates dag_run.conf and output DataFrames
    └── db.py            # Loads results into PostgreSQL via psycopg2
```

`utils/` contains all business logic; `dags/ml_inference_pipeline.py` is a thin
orchestrator that wires the tasks together and delegates to `utils/`.

---

## Infrastructure requirements

All of the following must exist before the DAG can run.

### AWS services

| Service | What is needed |
|---------|---------------|
| **AWS MWAA 2.8.x** | The execution environment. No other Airflow deployment is supported. |
| **S3 bucket** | One bucket for both input data and output. The pipeline user needs `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket` on the bucket. |
| **SageMaker Model Registry** | A Model Package Group containing at least one `Approved` versioned model package. The approved package must store the deployed SageMaker Model name in `CustomerMetadataProperties` under the key `sagemaker_model_name` or `model_name` (see [Model name convention](#model-name-convention) below). |
| **SageMaker Model resource** | A pre-existing SageMaker Model (not created by this pipeline) whose name matches what is stored in the registry metadata above. |
| **AWS Secrets Manager** | A secret containing the RDS credentials in the JSON format described in [Secrets Manager secret format](#secrets-manager-secret-format). |
| **RDS PostgreSQL** | A PostgreSQL instance reachable from the MWAA VPC. The pipeline user needs `INSERT` and `SELECT` on the target table. If the user also has `CREATE TABLE`, the table is bootstrapped automatically on first run; otherwise pre-create it with the DDL in [Database schema](#database-schema). |

### IAM permissions for the MWAA execution role

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

Scope `Resource` to specific ARNs in production.

### Model name convention

The pipeline resolves the SageMaker Model name from the model package's
`CustomerMetadataProperties`. Set one of these keys when registering a model
package version:

```python
customer_metadata_properties = {
    "sagemaker_model_name": "my-deployed-model-name"
    # or: "model_name": "my-deployed-model-name"
}
```

If neither key is present, the Model Package Group name is used as the model
name (works when a single Model resource is shared across all package versions).

### Secrets Manager secret format

The secret pointed to by `ml_pipeline_db_secret_arn` must be a JSON string with
these exact keys:

```json
{
  "host":     "my-rds-instance.xxxx.ca-central-1.rds.amazonaws.com",
  "port":     5432,
  "dbname":   "mlops",
  "username": "pipeline_user",
  "password": "s3cr3t"
}
```

`port` is optional and defaults to `5432`.

---

## Deployment

### 1. Upload `requirements.txt`

Upload `requirements.txt` to the MWAA S3 bucket and configure the MWAA
environment to use it. The only package it installs is `psycopg2-binary`
(all other dependencies are pre-installed by MWAA 2.8.x).

```bash
aws s3 cp requirements.txt s3://<mwaa-bucket>/requirements.txt
```

Then update the MWAA environment to point at the new requirements file and
wait for the environment to finish updating before deploying the DAG.

### 2. Deploy `dags/` and `utils/`

MWAA adds everything under the `dags/` S3 prefix to `sys.path` automatically,
so `utils/` **must be placed inside `dags/`** — not at the bucket root.

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
aws s3 sync dags/  s3://<mwaa-bucket>/dags/
aws s3 sync utils/ s3://<mwaa-bucket>/dags/utils/
```

### 3. Set Airflow Variables

See the full table in [Airflow Variables](#airflow-variables).

### 4. Place input data

Upload input CSV files to S3 before the scheduled run or manual trigger:

```
s3://<bucket>/batch/input/2024-01-15/features.csv
```

Multiple CSV files under the same prefix are supported.

---

## Airflow Variables

Set these in the MWAA UI under **Admin → Variables**.

### Required

| Variable | Example value | Description |
|----------|---------------|-------------|
| `ml_pipeline_s3_bucket` | `my-mlops-bucket` | S3 bucket name — no `s3://` prefix |
| `ml_pipeline_model_package_group` | `my-model-group` | SageMaker Model Package Group name |
| `ml_pipeline_db_secret_arn` | `arn:aws:secretsmanager:ca-central-1:123:secret:rds-creds-abc123` | Full ARN of the Secrets Manager secret |

### Optional (have sensible defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `ml_pipeline_aws_region` | `ca-central-1` | AWS region for SageMaker, S3, and Secrets Manager |
| `ml_pipeline_db_table` | `batch_inference_results` | Target PostgreSQL table name |
| `ml_pipeline_default_instance_type` | `ml.m5.xlarge` | Fallback instance type when the registry provides no preference |

---

## First-run checklist

Work through this list top-to-bottom before triggering the DAG for the first
time.

```
Infrastructure
  [ ] MWAA environment is on version 2.8.x
  [ ] MWAA execution role has the IAM permissions listed above
  [ ] S3 bucket exists and is accessible from MWAA
  [ ] SageMaker Model Package Group has at least one Approved package
  [ ] Approved package has sagemaker_model_name / model_name in CustomerMetadataProperties
  [ ] A SageMaker Model resource exists with that name
  [ ] RDS PostgreSQL is reachable from the MWAA VPC
  [ ] Secrets Manager secret is in the required JSON format
  [ ] Pipeline DB user has INSERT + SELECT on the target table
      (or CREATE TABLE if relying on the bootstrap path)

Deployment
  [ ] requirements.txt uploaded and MWAA environment updated (wait for "Available")
  [ ] utils/ synced to s3://<mwaa-bucket>/dags/utils/
  [ ] ml_inference_pipeline.py synced to s3://<mwaa-bucket>/dags/
  [ ] DAG appears in the MWAA UI without import errors

Configuration
  [ ] ml_pipeline_s3_bucket Variable set
  [ ] ml_pipeline_model_package_group Variable set
  [ ] ml_pipeline_db_secret_arn Variable set

Data
  [ ] Input CSV files uploaded to s3://<bucket>/batch/input/<YYYY-MM-DD>/
```

---

## Running the DAG

### Scheduled run

The DAG runs daily at **23:00 Toronto time** (`0 23 * * *`).

### Manual trigger (no overrides)

In the MWAA UI click **Trigger DAG** with no config. The pipeline uses the
current Airflow execution date and the default Variables.

### Manual trigger with overrides

Provide a JSON body when triggering to override defaults for a single run:

```json
{
  "execution_date_override":      "2024-01-10",
  "model_package_group_override": "my-other-group",
  "instance_type_override":       "ml.m5.2xlarge"
}
```

All three keys are optional and independently validated. An invalid value
(wrong date format, blank group name, unrecognised instance type pattern)
fails the `validate_input` task immediately with an actionable error message.

---

## Database schema

The table is created automatically on the first successful run if the pipeline
user has `CREATE TABLE` privileges. For production deployments, pre-create it
via IaC and grant only `INSERT` and `SELECT` to the pipeline user.

```sql
CREATE TABLE batch_inference_results (
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
    CONSTRAINT uq_batch_inference_results_job_row UNIQUE (job_name, row_index)
);
CREATE INDEX IF NOT EXISTS idx_batch_inference_results_job       ON batch_inference_results (job_name);
CREATE INDEX IF NOT EXISTS idx_batch_inference_results_exec_date ON batch_inference_results (execution_date);
```

Inserts are idempotent: re-triggering a run for the same `job_name` skips rows
that already exist (`ON CONFLICT DO NOTHING`). Each trigger of the DAG produces
a new `job_name` (based on the trigger timestamp), so re-runs on the same date
insert new rows rather than updating existing ones.

---

## Local development

The DAG file itself cannot run locally — it imports Airflow and Amazon provider
classes only available inside MWAA. However, the utility modules (`utils/`) have
no Airflow dependency and can be developed and tested locally.

### Running the unit tests

```bash
# Install test dependencies (does not require Airflow or AWS credentials)
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=utils --cov-report=term-missing
```

The tests use `unittest.mock` to stub all AWS calls — no real AWS credentials
or infrastructure are needed.

### Checking DAG syntax without MWAA

To verify the DAG file parses without syntax errors (imports will fail without
Airflow installed, but the AST check is sufficient):

```bash
python -c "import ast; ast.parse(open('dags/ml_inference_pipeline.py').read()); print('OK')"
```

### Testing utils against real AWS (optional)

If you have valid AWS credentials and want to exercise the utils against a real
account, set the relevant environment variables:

```bash
export AWS_DEFAULT_REGION=ca-central-1
export AWS_PROFILE=your-profile   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

python -c "
from utils.s3_utils import validate_input_data
print(validate_input_data('your-bucket', 'batch/input/2024-01-15/'))
"
```
