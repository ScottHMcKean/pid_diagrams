"""
Test configuration loading and validation.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    load_config,
    get_preprocessing_config,
    get_parse_config,
    get_evaluate_config,
    PIDConfig,
    PreprocessConfig,
    ParseConfig,
    EvaluateConfig,
)


@pytest.mark.config
@pytest.mark.unit
class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_full_config(self):
        """Test loading the complete configuration."""
        config = load_config("config.yaml")

        assert isinstance(config, PIDConfig)
        assert config.catalog == "shm"
        assert config.schema == "pid"  # Using the property
        assert config.db_schema == "pid"  # Using the actual field
        assert isinstance(config.preprocess, PreprocessConfig)
        assert isinstance(config.parse, ParseConfig)

    def test_preprocessing_config(self):
        """Test preprocessing configuration loading and validation."""
        config = get_preprocessing_config("config.yaml")

        assert isinstance(config, PreprocessConfig)
        assert config.raw_path == "/Volumes/shm/pid/alb_raw_pdfs"
        assert config.dpi == 200
        assert config.tile_width_px == 4096
        assert config.tile_height_px == 2048
        assert config.overlap_px == 256
        assert config.tile_table_name == "alb_tile_info"

        # Test model_dump works
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "raw_path" in config_dict
        assert "dpi" in config_dict

    def test_parse_config(self):
        """Test parse configuration loading and validation."""
        config = get_parse_config("config.yaml")

        assert isinstance(config, ParseConfig)
        assert config.fm_endpoint == "databricks-claude-3-7-sonnet"
        assert config.temperature == 0.1
        assert config.thinking_budget_tokens == 2048
        assert config.max_retries == 2
        assert config.retry_delay_s == 1
        assert len(config.metadata_prompt) > 0
        assert len(config.tag_prompt) > 0

    def test_evaluate_config(self):
        """Test evaluation configuration loading and validation."""
        config = get_evaluate_config("config.yaml")

        assert isinstance(config, EvaluateConfig)
        assert config.ground_truth_source == "load_sheet"
        assert config.ground_truth_table == "shm.pid.alb_load_sheet"
        assert config.ground_truth_json_path == "/Volumes/shm/pid/alb_examples"

        # Test model_dump works
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "ground_truth_source" in config_dict

    def test_full_config_includes_evaluate(self):
        """Test that full config loading includes evaluation section."""
        config = load_config("config.yaml")

        assert hasattr(config, "evaluate")
        assert isinstance(config.evaluate, EvaluateConfig)
        assert config.evaluate.ground_truth_source == "load_sheet"

    def test_config_file_not_found(self):
        """Test behavior when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


@pytest.mark.config
@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    def test_preprocessing_dpi_validation(self):
        """Test DPI validation in preprocessing config."""
        with pytest.raises(ValidationError):
            PreprocessConfig(
                raw_path="test.pdf",
                processed_path="/tmp",
                tile_table_name="test_table",
                dpi=10,  # Too low
                tile_width_px=2048,
                tile_height_px=1024,
                overlap_px=256,
            )

        with pytest.raises(ValidationError):
            PreprocessConfig(
                raw_path="test.pdf",
                processed_path="/tmp",
                tile_table_name="test_table",
                dpi=1000,  # Too high
                tile_width_px=2048,
                tile_height_px=1024,
                overlap_px=256,
            )

    def test_preprocessing_tile_validation(self):
        """Test tile dimension validation."""
        with pytest.raises(ValidationError):
            PreprocessConfig(
                raw_path="test.pdf",
                processed_path="/tmp",
                tile_table_name="test_table",
                dpi=200,
                tile_width_px=100,  # Too small
                tile_height_px=1024,
                overlap_px=256,
            )

    def test_parse_probability_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            ParseConfig(
                parsed_path="/tmp",
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                temperature=2.1,  # Too high
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

    def test_parse_retry_validation(self):
        """Test retry configuration validation."""
        with pytest.raises(ValidationError):
            ParseConfig(
                parsed_path="/tmp",
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                max_retries=15,  # Too high
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

    def test_evaluate_config_validation(self):
        """Test evaluation configuration validation."""
        with pytest.raises(ValidationError):
            EvaluateConfig(
                ground_truth_source="invalid_source",
                ground_truth_table="invalid_table",
                ground_truth_json_path="invalid_path",
            )

    def test_evaluate_config_json_source_validation(self):
        """Test that JSON path is required when source is json."""
        with pytest.raises(ValidationError):
            EvaluateConfig(
                ground_truth_source="json",
                ground_truth_table="some_table",  # This should be ignored
                # Missing ground_truth_json_path
            )

        # This should work
        config = EvaluateConfig(
            ground_truth_source="json",
            ground_truth_json_path="/path/to/examples",
        )
        assert config.ground_truth_source == "json"
        assert config.ground_truth_json_path == "/path/to/examples"

    def test_evaluate_config_load_sheet_source_validation(self):
        """Test that table name is required when source is load_sheet."""
        with pytest.raises(ValidationError):
            EvaluateConfig(
                ground_truth_source="load_sheet",
                ground_truth_json_path="/some/path",  # This should be ignored
                # Missing ground_truth_table
            )

        # This should work
        config = EvaluateConfig(
            ground_truth_source="load_sheet",
            ground_truth_table="catalog.schema.table",
        )
        assert config.ground_truth_source == "load_sheet"
        assert config.ground_truth_table == "catalog.schema.table"


@pytest.mark.config
@pytest.mark.unit
class TestConfigCompatibility:
    """Test backward compatibility features."""

    def test_schema_property_access(self):
        """Test that schema can be accessed both ways."""
        config = load_config("config.yaml")

        # Both should work and return the same value
        assert config.schema == config.db_schema
        assert config.schema == "pid"

    def test_model_dump_backward_compatibility(self):
        """Test that model_dump works for backward compatibility."""
        preprocess_config = get_preprocessing_config("config.yaml")
        config_dict = preprocess_config.model_dump()

        # Should contain all expected keys
        expected_keys = {
            "raw_path",
            "processed_path",
            "dpi",
            "tile_width_px",
            "tile_height_px",
            "tile_table_name",
            "overlap_px",
        }
        assert set(config_dict.keys()) == expected_keys
