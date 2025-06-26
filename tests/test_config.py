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
    PIDConfig,
    PreprocessConfig,
    ParseConfig,
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
        assert config.raw_path == "./assets/shell_facility_drawings.pdf"
        assert config.dpi == 200
        assert config.tile_width_px == 4096
        assert config.tile_height_px == 2048
        assert config.overlap_px == 512

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
        assert config.max_retries == 3
        assert config.retry_delay_s == 1
        assert len(config.metadata_prompt) > 0
        assert len(config.tag_prompt) > 0

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
                dpi=10,  # Too low
                tile_width_px=2048,
                tile_height_px=1024,
                overlap_px=256,
            )

        with pytest.raises(ValidationError):
            PreprocessConfig(
                raw_path="test.pdf",
                processed_path="/tmp",
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
                dpi=200,
                tile_width_px=100,  # Too small
                tile_height_px=1024,
                overlap_px=256,
            )

    def test_parse_probability_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            ParseConfig(
                output_path="/tmp",
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
                output_path="/tmp",
                fm_endpoint="test-model",
                max_retries=15,  # Too high
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )


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
            "overlap_px",
        }
        assert set(config_dict.keys()) == expected_keys
