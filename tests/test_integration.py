"""
Integration tests for the complete evaluation workflow.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from src.config import PIDConfig, EvaluateConfig, ParseConfig, PreprocessConfig
from src.evaluation import load_ground_truth


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for the evaluation pipeline."""

    def _create_test_config_with_evaluate(
        self, source: str, **eval_kwargs
    ) -> PIDConfig:
        """Helper to create a complete config for integration testing."""
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
            evaluate=EvaluateConfig(ground_truth_source=source, **eval_kwargs),
        )

    def test_complete_config_loading_json_source(self):
        """Test that we can load a complete config with JSON evaluation source."""
        # Test that the actual config.yaml works with our changes
        from src.config import load_config

        config = load_config("config.yaml")

        # Verify all sections are present
        assert hasattr(config, "catalog")
        assert hasattr(config, "preprocess")
        assert hasattr(config, "parse")
        assert hasattr(config, "evaluate")

        # Verify evaluation config is properly loaded
        assert config.evaluate.ground_truth_source == "load_sheet"
        assert config.evaluate.ground_truth_table == "shm.pid.alb_load_sheet"
        assert config.evaluate.ground_truth_json_path == "/Volumes/shm/pid/alb_examples"

    def test_complete_config_loading_different_sources(self):
        """Test that we can create configs for both evaluation sources."""
        # JSON source config
        json_config = self._create_test_config_with_evaluate(
            "json", ground_truth_json_path="/test/json/path"
        )
        assert json_config.evaluate.ground_truth_source == "json"
        assert json_config.evaluate.ground_truth_json_path == "/test/json/path"
        assert json_config.evaluate.ground_truth_table is None

        # Load sheet source config
        sheet_config = self._create_test_config_with_evaluate(
            "load_sheet", ground_truth_table="test.catalog.table"
        )
        assert sheet_config.evaluate.ground_truth_source == "load_sheet"
        assert sheet_config.evaluate.ground_truth_table == "test.catalog.table"
        assert sheet_config.evaluate.ground_truth_json_path is None

    @patch("src.evaluation.get_page_and_tag_files")
    def test_json_source_integration(self, mock_get_files):
        """Test JSON source loading integration."""
        # Skip this complex integration test and just test that the function can be called
        # The actual JSON loading is tested in the unit tests
        page_files = [Path("/test/examples/test_page.json")]
        tag_files = [Path("/test/examples/test_page_t1.json")]
        mock_get_files.return_value = (page_files, tag_files)

        # Create config
        config = self._create_test_config_with_evaluate(
            "json", ground_truth_json_path="/test/examples"
        )

        # Mock the actual load_ground_truth_json function to avoid file I/O
        with patch("src.evaluation.load_ground_truth_json") as mock_json_loader:
            test_df = pd.DataFrame({"test": ["data"]})
            mock_json_loader.return_value = test_df

            result_df = load_ground_truth(config)

            # Verify we get a DataFrame back and the function was called
            assert isinstance(result_df, pd.DataFrame)
            mock_json_loader.assert_called_once()

            # Verify the config was modified correctly
            call_args = mock_json_loader.call_args[0][0]
            assert call_args.parse.example_path == "/test/examples"

    @patch("src.evaluation.get_spark")
    def test_load_sheet_source_integration(self, mock_get_spark):
        """Test load sheet source integration."""
        # Setup mock spark and table
        mock_spark = Mock()
        mock_get_spark.return_value = mock_spark

        # Mock table data
        test_data = pd.DataFrame(
            {
                "drawing_name": ["TEST-001", "TEST-002"],
                "has_stamp": [True, False],
                "closest_filename": ["test1.pdf", "test2.pdf"],
            }
        )

        mock_spark.table.return_value.toPandas.return_value = test_data

        # Create config and test
        config = self._create_test_config_with_evaluate(
            "load_sheet", ground_truth_table="test.catalog.table"
        )

        result_df = load_ground_truth(config)

        # Verify we get the expected DataFrame back
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "drawing_name" in result_df.columns
        assert "has_stamp" in result_df.columns

        # Verify spark.table was called with correct table name
        mock_spark.table.assert_called_once_with("test.catalog.table")

    def test_evaluation_config_validation_integration(self):
        """Test that validation works properly in the complete pipeline."""
        from pydantic import ValidationError

        # Test invalid source
        with pytest.raises(ValidationError):
            self._create_test_config_with_evaluate("invalid_source")

        # Test JSON source without path
        with pytest.raises(ValidationError):
            self._create_test_config_with_evaluate("json")

        # Test load_sheet source without table
        with pytest.raises(ValidationError):
            self._create_test_config_with_evaluate("load_sheet")

        # Test valid configurations don't raise errors
        json_config = self._create_test_config_with_evaluate(
            "json", ground_truth_json_path="/test/path"
        )
        assert json_config.evaluate.ground_truth_source == "json"

        sheet_config = self._create_test_config_with_evaluate(
            "load_sheet", ground_truth_table="test.table"
        )
        assert sheet_config.evaluate.ground_truth_source == "load_sheet"

    def test_config_yaml_compatibility(self):
        """Test that the existing config.yaml is compatible with our changes."""
        from src.config import load_config, get_evaluate_config

        # Should be able to load the main config
        config = load_config("config.yaml")
        assert config.evaluate.ground_truth_source == "load_sheet"

        # Should be able to get just the evaluate config
        eval_config = get_evaluate_config("config.yaml")
        assert eval_config.ground_truth_source == "load_sheet"
        assert eval_config.ground_truth_table == "shm.pid.alb_load_sheet"

    @patch("src.evaluation.load_ground_truth_json")
    @patch("src.evaluation.load_ground_truth_load_sheet")
    @patch("src.evaluation.get_spark")
    def test_unified_function_routing(
        self, mock_get_spark, mock_sheet_loader, mock_json_loader
    ):
        """Test that the unified function correctly routes to the right implementation."""
        # Setup mocks
        json_df = pd.DataFrame({"json_data": [1, 2, 3]})
        sheet_df = pd.DataFrame({"sheet_data": [4, 5, 6]})
        mock_json_loader.return_value = json_df
        mock_sheet_loader.return_value = sheet_df
        mock_get_spark.return_value = Mock()

        # Test JSON routing
        json_config = self._create_test_config_with_evaluate(
            "json", ground_truth_json_path="/test/path"
        )
        result = load_ground_truth(json_config)
        assert result.equals(json_df)
        mock_json_loader.assert_called_once()

        # Reset mocks
        mock_json_loader.reset_mock()
        mock_sheet_loader.reset_mock()

        # Test load_sheet routing
        sheet_config = self._create_test_config_with_evaluate(
            "load_sheet", ground_truth_table="test.table"
        )
        result = load_ground_truth(sheet_config)
        assert result.equals(sheet_df)
        mock_sheet_loader.assert_called_once()
