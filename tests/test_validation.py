import pytest
import pandas as pd
from utils.validation import validate_dag_conf, validate_output_dataframe, REQUIRED_OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_df(n_rows: int = 10, null_count: int = 0) -> pd.DataFrame:
    """Build a structurally valid inference output DataFrame."""
    predictions = [float(i % 10) for i in range(n_rows - null_count)] + [None] * null_count
    return pd.DataFrame({
        "source_file": [f"file_{i}.csv.out" for i in range(n_rows)],
        "row_index":   list(range(n_rows)),
        "raw_output":  [f"0.{i % 10}" for i in range(n_rows)],
        "prediction":  predictions,
    })


# ---------------------------------------------------------------------------
# validate_dag_conf
# ---------------------------------------------------------------------------

class TestValidateDagConf:

    # --- None / empty inputs ---

    def test_none_returns_empty_dict(self):
        assert validate_dag_conf(None) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert validate_dag_conf({}) == {}

    # --- execution_date_override ---

    def test_valid_execution_date(self):
        result = validate_dag_conf({"execution_date_override": "2024-01-15"})
        assert result == {"execution_date_override": "2024-01-15"}

    @pytest.mark.parametrize("bad_date", [
        "2024/01/15",   # wrong separator
        "20240115",     # no separators
        "not-a-date",   # text
        "2024-1-5",     # single-digit month/day
        "01-15-2024",   # American format
        "",             # empty string
    ])
    def test_invalid_execution_date_raises(self, bad_date):
        with pytest.raises(ValueError, match="execution_date_override"):
            validate_dag_conf({"execution_date_override": bad_date})

    # --- model_package_group_override ---

    def test_valid_model_package_group_override(self):
        result = validate_dag_conf({"model_package_group_override": "my-model-group"})
        assert result == {"model_package_group_override": "my-model-group"}

    def test_model_package_group_override_strips_whitespace(self):
        result = validate_dag_conf({"model_package_group_override": "  my-group  "})
        assert result["model_package_group_override"] == "my-group"

    @pytest.mark.parametrize("bad_group", ["", "   ", "\t\n"])
    def test_blank_model_package_group_override_raises(self, bad_group):
        with pytest.raises(ValueError, match="model_package_group_override"):
            validate_dag_conf({"model_package_group_override": bad_group})

    # --- instance_type_override ---

    @pytest.mark.parametrize("valid_type", [
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.p3.2xlarge",
        "ml.c5.4xlarge",
        "ml.g4dn.xlarge",
    ])
    def test_valid_instance_type_override(self, valid_type):
        result = validate_dag_conf({"instance_type_override": valid_type})
        assert result["instance_type_override"] == valid_type

    @pytest.mark.parametrize("bad_type", [
        "m5.xlarge",            # missing ml. prefix
        "ml.m5",                # missing size segment
        "ec2.m5.xlarge",        # wrong prefix
        "ml.M5.XLARGE",         # uppercase not allowed by regex
        "ml.m5.xlarge.extra",   # extra segment makes fullmatch fail
    ])
    def test_invalid_instance_type_raises(self, bad_type):
        with pytest.raises(ValueError, match="instance_type_override"):
            validate_dag_conf({"instance_type_override": bad_type})

    # --- error accumulation ---

    def test_multiple_errors_raised_together(self):
        with pytest.raises(ValueError) as exc_info:
            validate_dag_conf({
                "execution_date_override": "bad-date",
                "instance_type_override":  "not-an-instance",
            })
        msg = str(exc_info.value)
        assert "execution_date_override" in msg
        assert "instance_type_override" in msg

    def test_all_three_errors_raised_together(self):
        with pytest.raises(ValueError) as exc_info:
            validate_dag_conf({
                "execution_date_override":       "bad-date",
                "model_package_group_override":  "   ",
                "instance_type_override":        "not-an-instance",
            })
        msg = str(exc_info.value)
        assert "execution_date_override" in msg
        assert "model_package_group_override" in msg
        assert "instance_type_override" in msg

    # --- miscellaneous ---

    def test_unknown_keys_are_ignored(self):
        result = validate_dag_conf({
            "execution_date_override": "2024-01-15",
            "some_unknown_key": "value",
        })
        assert "some_unknown_key" not in result

    def test_all_valid_keys_returned(self):
        conf = {
            "execution_date_override":      "2024-03-20",
            "model_package_group_override": "my-group",
            "instance_type_override":       "ml.m5.2xlarge",
        }
        assert validate_dag_conf(conf) == conf


# ---------------------------------------------------------------------------
# validate_output_dataframe
# ---------------------------------------------------------------------------

class TestValidateOutputDataframe:

    # --- happy path ---

    def test_valid_dataframe_passes(self):
        validate_output_dataframe(_make_output_df())  # must not raise

    def test_single_row_passes(self):
        validate_output_dataframe(_make_output_df(n_rows=1))

    # --- row count ---

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="Row count"):
            validate_output_dataframe(pd.DataFrame())

    def test_row_count_below_min_raises(self):
        with pytest.raises(ValueError, match="Row count"):
            validate_output_dataframe(_make_output_df(n_rows=3), min_rows=5)

    def test_row_count_at_min_passes(self):
        validate_output_dataframe(_make_output_df(n_rows=5), min_rows=5)

    # --- required columns ---

    def test_missing_one_column_raises(self):
        df = _make_output_df().drop(columns=["prediction"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_output_dataframe(df)

    def test_missing_multiple_columns_raises(self):
        df = _make_output_df().drop(columns=["prediction", "source_file"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_output_dataframe(df)

    def test_custom_required_columns_used(self):
        df = pd.DataFrame({"custom_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_output_dataframe(df, required_columns=frozenset({"custom_col", "other_col"}))

    def test_custom_required_columns_all_present_passes(self):
        df = pd.DataFrame({"custom_col": [1, 2, 3]})
        validate_output_dataframe(df, required_columns=frozenset({"custom_col"}))

    # --- null prediction ratio ---

    def test_null_ratio_below_threshold_passes(self):
        # 4/100 = 4% < 5%
        validate_output_dataframe(_make_output_df(n_rows=100, null_count=4))

    def test_null_ratio_at_threshold_passes(self):
        # exactly 5/100 = 5.0% == threshold, not strictly greater — must pass
        validate_output_dataframe(_make_output_df(n_rows=100, null_count=5))

    def test_null_ratio_above_threshold_raises(self):
        # 6/100 = 6% > 5%
        with pytest.raises(ValueError, match="Null prediction ratio"):
            validate_output_dataframe(_make_output_df(n_rows=100, null_count=6))

    def test_all_nulls_raises(self):
        with pytest.raises(ValueError, match="Null prediction ratio"):
            validate_output_dataframe(_make_output_df(n_rows=10, null_count=10))

    def test_no_prediction_column_skips_null_check(self):
        # When prediction is not in df.columns, the null check is bypassed
        df = pd.DataFrame({"custom_col": [1, 2, 3]})
        validate_output_dataframe(df, required_columns=frozenset({"custom_col"}))

    # --- error accumulation ---

    def test_multiple_errors_raised_together(self):
        # DataFrame with wrong columns and zero rows → both errors reported at once
        df = pd.DataFrame({"irrelevant_col": []})
        with pytest.raises(ValueError) as exc_info:
            validate_output_dataframe(df)
        msg = str(exc_info.value)
        assert "Row count" in msg
        assert "Missing required columns" in msg
