"""Tests for the evaluation module functions."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation import (
    clean_pid_tags,
    jaccard_similarity,
    normalized_levenshtein,
    boolean_accuracy,
    evaluate_dataframes,
    get_evaluation_summary,
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
        assert boolean_accuracy(True, True) is True
        assert boolean_accuracy(False, False) is True

    def test_different_values(self) -> None:
        """Test boolean accuracy with different values."""
        assert boolean_accuracy(True, False) is False
        assert boolean_accuracy(False, True) is False


class TestEvaluateDataframes:
    """Test the evaluate_dataframes function."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.ground_truth_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2"],
                "array_col": [["A", "B"], ["C", "D"]],
                "string_col": ["hello", "world"],
                "bool_col": [True, False],
            }
        )

        self.parsed_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2"],
                "array_col": [["A", "C"], ["C", "D", "E"]],
                "string_col": ["hallo", "world"],
                "bool_col": [True, True],
            }
        )

    def test_basic_evaluation(self) -> None:
        """Test basic dataframe evaluation."""
        result = evaluate_dataframes(
            self.ground_truth_df,
            self.parsed_df,
            array_columns=["array_col"],
            string_columns=["string_col"],
            boolean_columns=["bool_col"],
        )

        assert len(result) == 2
        assert "unique_key" in result.columns
        assert "array_col_jaccard" in result.columns
        assert "string_col_levenshtein" in result.columns
        assert "bool_col_boolean" in result.columns

    def test_jaccard_calculation(self) -> None:
        """Test Jaccard score calculation."""
        result = evaluate_dataframes(
            self.ground_truth_df,
            self.parsed_df,
            array_columns=["array_col"],
            string_columns=[],
            boolean_columns=[],
        )

        # test1: ["A", "B"] vs ["A", "C"] -> Intersection: {A}, Union: {A, B, C} -> 1/3
        assert abs(result.loc[0, "array_col_jaccard"] - 1 / 3) < 0.001


class TestGetEvaluationSummary:
    """Test the get_evaluation_summary function."""

    def test_summary_generation(self) -> None:
        """Test generating evaluation summary."""
        similarity_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2"],
                "array1_jaccard": [1.0, 0.5],
                "string1_levenshtein": [1.0, 0.9],
                "bool1_boolean": [True, False],
            }
        )

        result = get_evaluation_summary(similarity_df)

        assert "jaccard" in result
        assert "levenshtein" in result
        assert "boolean" in result

    def test_boolean_accuracy_calculation(self) -> None:
        """Test boolean accuracy in summary."""
        similarity_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2", "test3"],
                "bool1_boolean": [True, False, True],
            }
        )

        result = get_evaluation_summary(similarity_df)
        bool_summary = result["boolean"]

        # [True, False, True] -> 2/3 accuracy
        assert abs(bool_summary.loc["bool1_boolean", "accuracy"] - 2 / 3) < 0.001


class TestIntegration:
    """Integration test for the complete pipeline."""

    def test_full_pipeline(self) -> None:
        """Test the complete evaluation pipeline."""
        ground_truth_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2"],
                "equipment_tags": [["P-101", "V-201"], ["T-301"]],
                "drawing_name": ["Flow Diagram A", "Flow Diagram B"],
                "has_stamp": [True, False],
            }
        )

        parsed_df = pd.DataFrame(
            {
                "unique_key": ["test1", "test2"],
                "equipment_tags": [["P-101", "V-202"], ["T-301", "F-401"]],
                "drawing_name": ["Flow Diagram A", "Flow Diagram Beta"],
                "has_stamp": [True, True],
            }
        )

        # Run evaluation
        similarity_df = evaluate_dataframes(
            ground_truth_df,
            parsed_df,
            array_columns=["equipment_tags"],
            string_columns=["drawing_name"],
            boolean_columns=["has_stamp"],
        )

        # Get summary
        summary = get_evaluation_summary(similarity_df)

        # Verify results
        assert len(similarity_df) == 2
        assert "jaccard" in summary
        assert "levenshtein" in summary
        assert "boolean" in summary

        # Basic sanity checks
        assert all(0 <= val <= 1 for val in similarity_df["equipment_tags_jaccard"])
        assert all(0 <= val <= 1 for val in similarity_df["drawing_name_levenshtein"])
