"""Tests for the evaluation module functions."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation import (
    clean_pid_tags,
    jaccard_similarity,
    normalized_levenshtein,
    boolean_accuracy,
    calculate_recall,
    calculate_precision,
    evaluate_parsed_vs_ground_truth,
)


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
