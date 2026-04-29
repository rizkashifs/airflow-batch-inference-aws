import re
import pytest
from utils.sagemaker import build_transform_config, _make_job_name

_VALID_JOB_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")

MODEL_INFO = {
    "model_name":        "my-model",
    "model_version":     "3",
    "model_package_arn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/3",
    "instance_type":     "ml.m5.xlarge",
}


# ---------------------------------------------------------------------------
# _make_job_name
# ---------------------------------------------------------------------------

class TestMakeJobName:

    def test_basic_format(self):
        assert _make_job_name("my-dag", "20240115t120000") == "my-dag-transform-20240115t120000"

    def test_underscores_replaced_with_hyphens(self):
        name = _make_job_name("my_dag_id", "20240115t120000")
        assert "_" not in name
        assert name.startswith("my-dag-id-")

    def test_result_is_lowercase(self):
        name = _make_job_name("MY-DAG", "20240115T120000")
        assert name == name.lower()

    def test_ts_nodash_uppercase_t_lowercased(self):
        # Airflow's ts_nodash macro includes an uppercase 'T' (e.g. 20240115T120000)
        name = _make_job_name("my-dag", "20240115T120000")
        assert "T" not in name
        assert "20240115t120000" in name

    def test_special_chars_sanitized_to_hyphens(self):
        name = _make_job_name("dag.id@v2!", "20240115t120000")
        assert re.match(r"^[a-z0-9-]+$", name), f"Invalid chars in: {name!r}"

    def test_consecutive_hyphens_collapsed(self):
        # Double underscores → double hyphens → collapsed to single
        name = _make_job_name("dag__id", "20240115t120000")
        assert "--" not in name

    def test_max_length_63_chars(self):
        name = _make_job_name("a" * 100, "20240115t120000")
        assert len(name) <= 63

    def test_short_name_not_padded(self):
        name = _make_job_name("dag", "ts")
        assert name == "dag-transform-ts"

    def test_no_leading_hyphen(self):
        # A dag_id starting with special chars should still produce no leading hyphen
        name = _make_job_name("dag", "20240115t120000")
        assert not name.startswith("-")

    def test_no_trailing_hyphen_on_normal_input(self):
        name = _make_job_name("my-dag", "20240115t120000")
        assert not name.endswith("-")

    def test_only_valid_sagemaker_chars(self):
        name = _make_job_name("my_dag.v2", "20240115T120000")
        assert re.match(r"^[a-z0-9-]+$", name), f"Invalid SageMaker chars in: {name!r}"


# ---------------------------------------------------------------------------
# build_transform_config
# ---------------------------------------------------------------------------

class TestBuildTransformConfig:

    def _build(self, bucket="my-bucket", date="2024-01-15", dag_id="my-dag", ts="20240115t120000"):
        return build_transform_config(MODEL_INFO, bucket, date, dag_id, ts)

    def test_all_expected_keys_present(self):
        config = self._build()
        assert set(config.keys()) == {
            "model_name", "model_version", "model_package_arn",
            "job_name",
            "input_s3", "output_s3", "output_bucket", "output_prefix",
            "processed_s3_key",
            "instance_type",
        }

    # --- model identity pass-through ---

    def test_model_name_passed_through(self):
        assert self._build()["model_name"] == "my-model"

    def test_model_version_passed_through(self):
        assert self._build()["model_version"] == "3"

    def test_model_package_arn_passed_through(self):
        assert self._build()["model_package_arn"] == MODEL_INFO["model_package_arn"]

    def test_instance_type_passed_through(self):
        assert self._build()["instance_type"] == "ml.m5.xlarge"

    # --- S3 path construction ---

    def test_input_s3_format(self):
        assert self._build()["input_s3"] == "s3://my-bucket/batch/input/2024-01-15/"

    def test_output_s3_format(self):
        assert self._build()["output_s3"] == "s3://my-bucket/batch/output/2024-01-15/"

    def test_output_bucket_is_bare_name(self):
        config = self._build()
        assert config["output_bucket"] == "my-bucket"
        assert not config["output_bucket"].startswith("s3://")

    def test_output_prefix_has_no_s3_scheme(self):
        config = self._build()
        assert config["output_prefix"] == "batch/output/2024-01-15/"
        assert not config["output_prefix"].startswith("s3://")

    def test_processed_s3_key_format(self):
        assert self._build()["processed_s3_key"] == "batch/processed/2024-01-15/combined.csv"

    def test_paths_use_execution_date(self):
        config = self._build(date="2024-06-30")
        assert "2024-06-30" in config["input_s3"]
        assert "2024-06-30" in config["output_s3"]
        assert "2024-06-30" in config["output_prefix"]
        assert "2024-06-30" in config["processed_s3_key"]

    def test_different_bucket_reflected_in_paths(self):
        config = self._build(bucket="prod-bucket")
        assert "prod-bucket" in config["input_s3"]
        assert "prod-bucket" in config["output_s3"]
        assert config["output_bucket"] == "prod-bucket"

    # --- job name ---

    def test_job_name_contains_dag_id(self):
        config = self._build(dag_id="my-dag")
        assert "my-dag" in config["job_name"]

    def test_job_name_contains_ts_nodash(self):
        config = self._build(ts="20240115t120000")
        assert "20240115t120000" in config["job_name"]

    def test_job_name_respects_63_char_limit(self):
        config = self._build(dag_id="a" * 100)
        assert len(config["job_name"]) <= 63

    def test_job_name_is_lowercase(self):
        config = self._build(dag_id="MY-DAG", ts="20240115T120000")
        assert config["job_name"] == config["job_name"].lower()
