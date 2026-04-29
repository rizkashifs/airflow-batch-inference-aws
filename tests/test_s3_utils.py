from unittest.mock import patch, MagicMock
import io

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from utils.s3_utils import (
    list_input_files,
    list_output_files,
    validate_input_data,
    collect_inference_results,
    write_dataframe_to_s3,
    read_dataframe_from_s3,
    _parse_first_field,
    _split_s3_uri,
)

# ---------------------------------------------------------------------------
# Helpers for building boto3 mock responses
# ---------------------------------------------------------------------------

def _make_s3():
    """Return a fresh mock S3 client."""
    return MagicMock()


def _paginator(pages: list[dict]):
    """Build a mock paginator that yields *pages* when paginate() is called."""
    p = MagicMock()
    p.paginate.return_value = iter(pages)
    return p


def _s3_body(content: bytes) -> dict:
    """Build a mock get_object response with the given body bytes."""
    body = MagicMock()
    body.read.return_value = content
    return {"Body": body}


def _client_error(code: str = "NoSuchKey") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


# ---------------------------------------------------------------------------
# _parse_first_field  (pure, no AWS)
# ---------------------------------------------------------------------------

class TestParseFirstField:

    def test_simple_float(self):
        assert _parse_first_field("0.75") == 0.75

    def test_integer_string(self):
        assert _parse_first_field("1") == 1.0

    def test_negative_float(self):
        assert _parse_first_field("-0.5") == -0.5

    def test_scientific_notation(self):
        assert _parse_first_field("1e-5") == pytest.approx(1e-5)

    def test_csv_line_extracts_first_field(self):
        assert _parse_first_field("0.9,0.1,label") == 0.9

    def test_whitespace_around_value_is_stripped(self):
        assert _parse_first_field("  0.5  ") == 0.5

    def test_non_numeric_string_returns_none(self):
        assert _parse_first_field("prediction") is None

    def test_empty_string_returns_none(self):
        assert _parse_first_field("") is None

    def test_header_row_returns_none(self):
        assert _parse_first_field("score,label") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_first_field("   ") is None


# ---------------------------------------------------------------------------
# _split_s3_uri  (pure, no AWS)
# ---------------------------------------------------------------------------

class TestSplitS3Uri:

    def test_standard_uri(self):
        assert _split_s3_uri("s3://bucket/key/path") == ("bucket", "key/path")

    def test_single_level_key(self):
        assert _split_s3_uri("s3://bucket/key") == ("bucket", "key")

    def test_trailing_slash(self):
        bucket, key = _split_s3_uri("s3://bucket/prefix/")
        assert bucket == "bucket"
        assert key == "prefix/"

    def test_bucket_only_with_slash(self):
        bucket, _ = _split_s3_uri("s3://bucket/")
        assert bucket == "bucket"

    def test_http_scheme_raises(self):
        with pytest.raises(ValueError, match="s3://"):
            _split_s3_uri("http://bucket/key")

    def test_no_scheme_raises(self):
        with pytest.raises(ValueError, match="s3://"):
            _split_s3_uri("bucket/key")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="s3://"):
            _split_s3_uri("")


# ---------------------------------------------------------------------------
# list_input_files / list_output_files
# ---------------------------------------------------------------------------

class TestListFiles:

    def test_list_input_files_returns_csv_only(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "prefix/a.csv"},
                {"Key": "prefix/b.json"},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert list_input_files("bucket", "prefix/") == ["prefix/a.csv"]

    def test_list_output_files_returns_csv_out_only(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "out/a.csv.out"},
                {"Key": "out/b.csv"},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert list_output_files("bucket", "out/") == ["out/a.csv.out"]

    def test_list_input_files_are_sorted(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "prefix/c.csv"},
                {"Key": "prefix/a.csv"},
                {"Key": "prefix/b.csv"},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = list_input_files("bucket", "prefix/")
        assert result == sorted(result)

    def test_list_output_files_are_sorted(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "out/z.csv.out"},
                {"Key": "out/a.csv.out"},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = list_output_files("bucket", "out/")
        assert result == sorted(result)

    def test_empty_contents_returns_empty_list(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([{}])  # no "Contents" key
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert list_input_files("bucket", "no-such-prefix/") == []

    def test_multiple_pages_combined(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "input/file_0.csv"}, {"Key": "input/file_1.csv"}]},
            {"Contents": [{"Key": "input/file_2.csv"}]},
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = list_input_files("bucket", "input/")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# validate_input_data
# ---------------------------------------------------------------------------

class TestValidateInputData:

    def test_single_csv_returns_file_count(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "input/a.csv", "Size": 100}]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert validate_input_data("bucket", "input/") == {"file_count": 1}

    def test_multiple_csvs_correct_count(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "input/a.csv", "Size": 100},
                {"Key": "input/b.csv", "Size": 200},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert validate_input_data("bucket", "input/") == {"file_count": 2}

    def test_no_csv_files_raises(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "input/file.json", "Size": 10}]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            with pytest.raises(RuntimeError, match="No CSV files found"):
                validate_input_data("bucket", "input/")

    def test_empty_prefix_raises(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([{}])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            with pytest.raises(RuntimeError, match="No CSV files found"):
                validate_input_data("bucket", "no-such-prefix/")

    def test_empty_csv_file_raises(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "input/empty.csv", "Size": 0}]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            with pytest.raises(RuntimeError, match="0 bytes"):
                validate_input_data("bucket", "input/")

    def test_non_csv_objects_not_counted(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "input/data.csv",   "Size": 100},
                {"Key": "input/other.json", "Size": 20},
            ]}
        ])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            assert validate_input_data("bucket", "input/")["file_count"] == 1


# ---------------------------------------------------------------------------
# collect_inference_results
# ---------------------------------------------------------------------------

class TestCollectInferenceResults:

    def test_single_file_row_count(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/result.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"0.9\n0.2\n0.7\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert len(df) == 3

    def test_output_columns_present(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/r.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"0.5\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert set(df.columns) == {"source_file", "row_index", "raw_output", "prediction"}

    def test_row_index_is_global_across_files(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "output/a.csv.out"},
                {"Key": "output/b.csv.out"},
            ]}
        ])
        mock_s3.get_object.side_effect = [
            _s3_body(b"0.1\n0.2\n"),
            _s3_body(b"0.3\n0.4\n"),
        ]
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert list(df["row_index"]) == [0, 1, 2, 3]

    def test_source_file_column_identifies_origin(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "output/a.csv.out"},
                {"Key": "output/b.csv.out"},
            ]}
        ])
        mock_s3.get_object.side_effect = [
            _s3_body(b"0.5\n"),
            _s3_body(b"0.6\n"),
        ]
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert df.iloc[0]["source_file"] == "output/a.csv.out"
        assert df.iloc[1]["source_file"] == "output/b.csv.out"

    def test_files_processed_in_sorted_key_order(self):
        # Paginator returns keys in reverse order; results must be sorted by key
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [
                {"Key": "output/z.csv.out"},
                {"Key": "output/a.csv.out"},
            ]}
        ])
        mock_s3.get_object.side_effect = [
            _s3_body(b"0.1\n"),  # called for "a" first because keys are sorted
            _s3_body(b"0.9\n"),
        ]
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert df.iloc[0]["source_file"] == "output/a.csv.out"
        assert df.iloc[1]["source_file"] == "output/z.csv.out"

    def test_unparseable_line_produces_none_prediction(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/r.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"header\n0.5\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert pd.isna(df.iloc[0]["prediction"])
        assert df.iloc[1]["prediction"] == 0.5

    def test_empty_lines_are_skipped(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/r.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"0.5\n\n0.6\n\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert len(df) == 2

    def test_no_output_files_raises(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([{}])
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            with pytest.raises(RuntimeError, match=r"\.csv\.out"):
                collect_inference_results("bucket", "empty-output/")

    def test_prediction_values_parsed_correctly(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/r.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"0.1\n0.9\n0.5\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert list(df["prediction"]) == pytest.approx([0.1, 0.9, 0.5])

    def test_csv_output_extracts_first_field_as_prediction(self):
        mock_s3 = _make_s3()
        mock_s3.get_paginator.return_value = _paginator([
            {"Contents": [{"Key": "output/r.csv.out"}]}
        ])
        mock_s3.get_object.return_value = _s3_body(b"0.8,class_a\n0.3,class_b\n")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            df = collect_inference_results("bucket", "output/")
        assert list(df["prediction"]) == pytest.approx([0.8, 0.3])


# ---------------------------------------------------------------------------
# write_dataframe_to_s3 / read_dataframe_from_s3
# ---------------------------------------------------------------------------

class TestWriteAndReadDataframe:

    def _sample_df(self):
        return pd.DataFrame({
            "source_file": ["a.csv.out", "b.csv.out"],
            "row_index":   [0, 1],
            "raw_output":  ["0.5", "0.8"],
            "prediction":  [0.5, 0.8],
        })

    def test_write_returns_s3_uri(self):
        mock_s3 = _make_s3()
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            uri = write_dataframe_to_s3(self._sample_df(), "my-bucket", "staged/combined.csv")
        assert uri == "s3://my-bucket/staged/combined.csv"

    def test_write_calls_put_object_with_correct_bucket_and_key(self):
        mock_s3 = _make_s3()
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            write_dataframe_to_s3(self._sample_df(), "my-bucket", "staged/combined.csv")
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["Key"] == "staged/combined.csv"
        assert call_kwargs["ContentType"] == "text/csv"

    def test_write_body_is_valid_csv_bytes(self):
        mock_s3 = _make_s3()
        df = self._sample_df()
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            write_dataframe_to_s3(df, "my-bucket", "staged/combined.csv")
        body = mock_s3.put_object.call_args.kwargs["Body"]
        import io as _io
        result_df = pd.read_csv(_io.BytesIO(body))
        assert list(result_df.columns) == list(df.columns)
        assert len(result_df) == len(df)

    def test_roundtrip_row_count(self):
        df = self._sample_df()
        csv_bytes = df.to_csv(index=False).encode()
        mock_s3 = _make_s3()
        mock_s3.get_object.return_value = _s3_body(csv_bytes)
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = read_dataframe_from_s3("s3://my-bucket/staged/combined.csv")
        assert len(result) == len(df)

    def test_roundtrip_columns_preserved(self):
        df = self._sample_df()
        csv_bytes = df.to_csv(index=False).encode()
        mock_s3 = _make_s3()
        mock_s3.get_object.return_value = _s3_body(csv_bytes)
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = read_dataframe_from_s3("s3://my-bucket/staged/combined.csv")
        assert list(result.columns) == list(df.columns)

    def test_roundtrip_values_preserved(self):
        df = self._sample_df()
        csv_bytes = df.to_csv(index=False).encode()
        mock_s3 = _make_s3()
        mock_s3.get_object.return_value = _s3_body(csv_bytes)
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            result = read_dataframe_from_s3("s3://my-bucket/staged/combined.csv")
        assert list(result["prediction"]) == pytest.approx([0.5, 0.8])

    def test_read_nonexistent_key_raises_client_error(self):
        mock_s3 = _make_s3()
        mock_s3.get_object.side_effect = _client_error("NoSuchKey")
        with patch("utils.s3_utils.boto3.client", return_value=mock_s3):
            with pytest.raises(ClientError):
                read_dataframe_from_s3("s3://my-bucket/does/not/exist.csv")
