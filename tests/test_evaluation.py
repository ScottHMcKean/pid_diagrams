"""Tests for the evaluation module functions."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.evaluation import (
    clean_pid_tags,
    load_ground_truth,
    jaccard_similarity,
    normalized_levenshtein,
    boolean_accuracy,
    calculate_recall,
    calculate_precision,
    evaluate_parsed_vs_ground_truth,
)
from src.config import PIDConfig, EvaluateConfig, ParseConfig, PreprocessConfig


class TestCleanPidTags:
    """Test the clean_pid_tags function."""

    def test_clean_equipment_tags(self) -> None:
        """Test cleaning equipment tags with dashes."""
        input_tags = {
            "equipment_tags": ["P-101", "V-201", "invalid_tag", "T-401(backup)"]
        }
        result = clean_pid_tags(input_tags)

        expected = ["P-101", "V-201", "T-401"]
        assert result["equipment_tags"] == expected

    def test_clean_stream_tags(self) -> None:
        """Test cleaning stream tags with dots."""
        input_tags = {
            "incoming_streams": ["1.01", "2.15", "invalid", "3.20(main)"],
            "outgoing_streams": ["4.01", "invalid_stream"],
        }
        result = clean_pid_tags(input_tags)

        assert result["incoming_streams"] == ["1.01", "2.15", "3.20"]
        assert result["outgoing_streams"] == ["4.01"]

    def test_other_categories_unchanged(self) -> None:
        """Test that other categories pass through unchanged."""
        input_tags = {"legacy_numbers": ["123", "456"]}
        result = clean_pid_tags(input_tags)

        assert result["legacy_numbers"] == ["123", "456"]


class TestJaccardSimilarity:
    """Test the jaccard_similarity function."""

    def test_identical_lists(self) -> None:
        """Test Jaccard similarity with identical lists."""
        result = jaccard_similarity(["A", "B", "C"], ["A", "B", "C"])
        assert result == 1.0

    def test_partial_overlap(self) -> None:
        """Test Jaccard similarity with partial overlap."""
        result = jaccard_similarity(["A", "B", "C"], ["B", "C", "D"])
        # Intersection: {B, C} = 2, Union: {A, B, C, D} = 4
        assert result == 0.5

    def test_no_overlap(self) -> None:
        """Test Jaccard similarity with no overlap."""
        result = jaccard_similarity(["A", "B"], ["C", "D"])
        assert result == 0.0

    def test_empty_lists(self) -> None:
        """Test Jaccard similarity with empty lists."""
        result = jaccard_similarity([], [])
        assert result == 1.0


class TestNormalizedLevenshtein:
    """Test the normalized_levenshtein function."""

    def test_identical_strings(self) -> None:
        """Test normalized Levenshtein with identical strings."""
        result = normalized_levenshtein("hello", "hello")
        assert result == 1.0

    def test_partial_similarity(self) -> None:
        """Test normalized Levenshtein with partial similarity."""
        result = normalized_levenshtein("hello", "hallo")
        # Edit distance: 1, Max length: 5, Similarity: 1 - 1/5 = 0.8
        assert result == 0.8

    def test_empty_strings(self) -> None:
        """Test normalized Levenshtein with empty strings."""
        result = normalized_levenshtein("", "")
        assert result == 1.0


class TestBooleanAccuracy:
    """Test the boolean_accuracy function."""

    def test_matching_values(self) -> None:
        """Test boolean accuracy with matching values."""
        assert boolean_accuracy(True, True) == 1
        assert boolean_accuracy(False, False) == 1

    def test_different_values(self) -> None:
        """Test boolean accuracy with different values."""
        assert boolean_accuracy(True, False) == 0
        assert boolean_accuracy(False, True) == 0


class TestRecallAndPrecision:
    """Test the calculate_recall and calculate_precision functions."""

    def test_perfect_recall_and_precision(self) -> None:
        """Test with perfect matches."""
        ground_truth = ["A", "B", "C"]
        parsed = ["A", "B", "C"]

        assert calculate_recall(ground_truth, parsed) == 1.0
        assert calculate_precision(ground_truth, parsed) == 1.0

    def test_partial_recall_perfect_precision(self) -> None:
        """Test when parser finds subset of ground truth."""
        ground_truth = ["A", "B", "C", "D"]
        parsed = ["A", "B"]  # Missing C, D

        # Recall: 2 out of 4 found = 0.5
        assert calculate_recall(ground_truth, parsed) == 0.5
        # Precision: 2 out of 2 parsed are correct = 1.0
        assert calculate_precision(ground_truth, parsed) == 1.0

    def test_perfect_recall_partial_precision(self) -> None:
        """Test when parser finds all ground truth plus extras."""
        ground_truth = ["A", "B"]
        parsed = ["A", "B", "C", "D"]  # Extra C, D

        # Recall: 2 out of 2 found = 1.0
        assert calculate_recall(ground_truth, parsed) == 1.0
        # Precision: 2 out of 4 parsed are correct = 0.5
        assert calculate_precision(ground_truth, parsed) == 0.5

    def test_partial_recall_partial_precision(self) -> None:
        """Test with both missed and extra items."""
        ground_truth = ["A", "B", "C", "D"]
        parsed = ["A", "B", "E", "F"]  # Missing C, D; Extra E, F

        # Recall: 2 out of 4 found = 0.5
        assert calculate_recall(ground_truth, parsed) == 0.5
        # Precision: 2 out of 4 parsed are correct = 0.5
        assert calculate_precision(ground_truth, parsed) == 0.5

    def test_empty_lists(self) -> None:
        """Test with empty lists."""
        assert calculate_recall([], []) == 1.0
        assert calculate_precision([], []) == 1.0

        # Empty ground truth, some parsed
        assert calculate_recall([], ["A", "B"]) == 1.0
        assert calculate_precision([], ["A", "B"]) == 0.0

        # Some ground truth, empty parsed
        assert calculate_recall(["A", "B"], []) == 0.0
        assert calculate_precision(["A", "B"], []) == 1.0


def test_evaluate_parsed_vs_ground_truth():
    """Test high-level evaluation that uses pre-combined tags and streams."""
    from src.evaluation import evaluate_parsed_vs_ground_truth

    # Create test dataframes with pre-combined columns
    ground_truth_df = pd.DataFrame(
        [
            {
                "unique_key": "test1",
                "combined_tags": ["E-101", "P-102", "L-201", "L-202"],  # Pre-combined
                "combined_streams": ["1.01", "2.01", "3.01", "4.01"],  # Pre-combined
                "drawing_name": "Drawing A",
                "has_stamp": True,
            },
            {
                "unique_key": "test2",
                "combined_tags": ["E-201", "L-301"],  # Pre-combined
                "combined_streams": ["5.01", "6.01"],  # Pre-combined
                "drawing_name": "Drawing B",
                "has_stamp": False,
            },
        ]
    )

    parsed_df = pd.DataFrame(
        [
            {
                "unique_key": "test1",
                "combined_tags": [
                    "E-101",
                    "P-103",
                    "L-201",
                ],  # Pre-combined: 2 matches out of 5 total
                "combined_streams": [
                    "1.01",
                    "2.02",
                    "3.01",
                ],  # Pre-combined: 2 matches out of 5 total
                "drawing_name": "Drawing A",
                "has_stamp": True,
            },
            {
                "unique_key": "test2",
                "combined_tags": ["E-201", "L-301"],  # Pre-combined: perfect match
                "combined_streams": ["5.01", "6.01"],  # Pre-combined: perfect match
                "drawing_name": "Drawing B Modified",  # Different name
                "has_stamp": False,
            },
        ]
    )

    string_columns = ["drawing_name"]
    boolean_columns = ["has_stamp"]

    result_df = evaluate_parsed_vs_ground_truth(
        ground_truth_df, parsed_df, string_columns, boolean_columns
    )

    # Check that we have the right columns
    expected_columns = [
        "unique_key",
        "tags_jaccard",
        "tags_recall",
        "tags_precision",
        "streams_jaccard",
        "streams_recall",
        "streams_precision",
        "drawing_name_levenshtein",
        "has_stamp_boolean",
    ]
    assert all(col in result_df.columns for col in expected_columns)

    # Check first row calculations
    test1_row = result_df[result_df["unique_key"] == "test1"].iloc[0]

    # Combined tags: GT = ["E-101", "P-102", "L-201", "L-202"], Parsed = ["E-101", "P-103", "L-201"]
    # Intersection = ["E-101", "L-201"] = 2 items
    # Union = ["E-101", "P-102", "L-201", "L-202", "P-103"] = 5 items
    # Jaccard = 2/5 = 0.4
    assert abs(test1_row["tags_jaccard"] - 0.4) < 0.01

    # Tags Recall: 2 matches out of 4 ground truth = 0.5
    assert abs(test1_row["tags_recall"] - 0.5) < 0.01

    # Tags Precision: 2 matches out of 3 parsed = 0.667
    assert abs(test1_row["tags_precision"] - 0.667) < 0.01

    # Combined streams: GT = ["1.01", "2.01", "3.01", "4.01"], Parsed = ["1.01", "2.02", "3.01"]
    # Intersection = ["1.01", "3.01"] = 2 items
    # Union = ["1.01", "2.01", "3.01", "4.01", "2.02"] = 5 items
    # Jaccard = 2/5 = 0.4
    assert abs(test1_row["streams_jaccard"] - 0.4) < 0.01

    # Streams Recall: 2 matches out of 4 ground truth = 0.5
    assert abs(test1_row["streams_recall"] - 0.5) < 0.01

    # Streams Precision: 2 matches out of 3 parsed = 0.667
    assert abs(test1_row["streams_precision"] - 0.667) < 0.01

    # String comparison: exact match
    assert test1_row["drawing_name_levenshtein"] == 1.0

    # Boolean comparison: exact match (returns int now)
    assert test1_row["has_stamp_boolean"] == 1

    # Check second row
    test2_row = result_df[result_df["unique_key"] == "test2"].iloc[0]

    # Perfect matches for tags and streams
    assert test2_row["tags_jaccard"] == 1.0
    assert test2_row["tags_recall"] == 1.0
    assert test2_row["tags_precision"] == 1.0
    assert test2_row["streams_jaccard"] == 1.0
    assert test2_row["streams_recall"] == 1.0
    assert test2_row["streams_precision"] == 1.0

    # Different string
    assert test2_row["drawing_name_levenshtein"] < 1.0

    # Same boolean (returns int now)
    assert test2_row["has_stamp_boolean"] == 1


@pytest.mark.evaluation
class TestLoadGroundTruth:
    """Test the unified load_ground_truth function."""

    def _create_test_config(
        self, source: str, json_path: str = None, table: str = None
    ) -> PIDConfig:
        """Helper to create test config with different evaluation sources."""
        return PIDConfig(
            catalog="test_catalog",
            schema="test_schema",
            preprocess=PreprocessConfig(
                raw_path="/test/raw",
                processed_path="/test/processed",
                tile_table_name="test_tiles",
            ),
            parse=ParseConfig(
                parsed_path="/test/parsed",
                example_path="/test/examples",
                local_tables_path="/test/local",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                fm_endpoint="test-endpoint",
                metadata_prompt="test prompt",
                metadata_example="test example",
                tag_prompt="test prompt",
                tag_example="test example",
            ),
            evaluate=EvaluateConfig(
                ground_truth_source=source,
                ground_truth_json_path=json_path,
                ground_truth_table=table,
            ),
        )

    @patch("src.evaluation.load_ground_truth_json")
    def test_load_ground_truth_json_source(self, mock_json_loader):
        """Test loading ground truth from JSON source."""
        # Setup
        config = self._create_test_config("json", json_path="/test/examples")
        expected_df = pd.DataFrame({"test": ["data"]})
        mock_json_loader.return_value = expected_df

        # Execute
        result = load_ground_truth(config)

        # Verify
        assert result.equals(expected_df)
        mock_json_loader.assert_called_once()

        # Check that the temporary config was created with correct example_path
        call_args = mock_json_loader.call_args[0][0]
        assert call_args.parse.example_path == "/test/examples"

    @patch("src.evaluation.load_ground_truth_load_sheet")
    @patch("src.evaluation.get_spark")
    def test_load_ground_truth_load_sheet_source(
        self, mock_get_spark, mock_sheet_loader
    ):
        """Test loading ground truth from load_sheet source."""
        # Setup
        config = self._create_test_config("load_sheet", table="test.catalog.table")
        expected_df = pd.DataFrame({"test": ["data"]})
        mock_spark = Mock()
        mock_get_spark.return_value = mock_spark
        mock_sheet_loader.return_value = expected_df

        # Execute
        result = load_ground_truth(config)

        # Verify
        assert result.equals(expected_df)
        mock_sheet_loader.assert_called_once_with(mock_spark, "test.catalog.table")
        mock_get_spark.assert_called_once()

    @patch("src.evaluation.load_ground_truth_load_sheet")
    def test_load_ground_truth_load_sheet_source_with_spark(self, mock_sheet_loader):
        """Test loading ground truth from load_sheet source with provided spark session."""
        # Setup
        config = self._create_test_config("load_sheet", table="test.catalog.table")
        expected_df = pd.DataFrame({"test": ["data"]})
        mock_spark = Mock()
        mock_sheet_loader.return_value = expected_df

        # Execute
        result = load_ground_truth(config, spark=mock_spark)

        # Verify
        assert result.equals(expected_df)
        mock_sheet_loader.assert_called_once_with(mock_spark, "test.catalog.table")

    def test_load_ground_truth_invalid_source(self):
        """Test that invalid source raises ValidationError during config creation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="Input should be 'json' or 'load_sheet'"
        ):
            self._create_test_config("invalid_source")

    def test_load_ground_truth_json_missing_path(self):
        """Test that JSON source without path raises ValidationError during config creation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="ground_truth_json_path is required when source is 'json'",
        ):
            self._create_test_config("json")  # No json_path provided

    def test_load_ground_truth_load_sheet_missing_table(self):
        """Test that load_sheet source without table raises ValidationError during config creation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="ground_truth_table is required when source is 'load_sheet'",
        ):
            self._create_test_config("load_sheet")  # No table provided

    @patch("src.evaluation.load_ground_truth_json")
    def test_load_ground_truth_json_preserves_original_config(self, mock_json_loader):
        """Test that the original config is not modified when creating temp config."""
        # Setup
        original_example_path = "/original/examples"
        config = self._create_test_config("json", json_path="/test/examples")
        config.parse.example_path = original_example_path

        mock_json_loader.return_value = pd.DataFrame({"test": ["data"]})

        # Execute
        load_ground_truth(config)

        # Verify original config is unchanged
        assert config.parse.example_path == original_example_path

        # Verify temporary config was created correctly
        call_args = mock_json_loader.call_args[0][0]
        assert call_args.parse.example_path == "/test/examples"
