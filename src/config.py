"""
Configuration models for P&ID processing pipeline.
"""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class PreprocessConfig(BaseModel):
    """Configuration for PDF preprocessing pipeline."""

    raw_path: str = Field(..., description="Path to raw PDF file")
    processed_path: str = Field(
        ..., description="Output directory for processed images"
    )
    dpi: int = Field(200, description="DPI for PDF to image conversion")
    tile_width_px: int = Field(4096, description="Width of tiles in pixels")
    tile_height_px: int = Field(2048, description="Height of tiles in pixels")
    overlap_px: int = Field(512, description="Overlap between tiles in pixels")

    @field_validator("dpi")
    @classmethod
    def validate_dpi(cls, v: int) -> int:
        """Validate DPI is reasonable."""
        if v < 50 or v > 600:
            raise ValueError("DPI must be between 50 and 600")
        return v

    @field_validator("tile_width_px", "tile_height_px")
    @classmethod
    def validate_tile_dimensions(cls, v: int) -> int:
        """Validate tile dimensions are reasonable."""
        if v < 512 or v > 8192:
            raise ValueError("Tile dimensions must be between 512 and 8192 pixels")
        return v

    @field_validator("overlap_px")
    @classmethod
    def validate_overlap(cls, v: int) -> int:
        """Validate overlap is reasonable."""
        if v < 0 or v > 1024:
            raise ValueError("Overlap must be between 0 and 1024 pixels")
        return v


class ParseConfig(BaseModel):
    """Configuration for parsing pipeline."""

    output_path: str = Field(..., description="Output directory for parsed results")
    example_path: str = Field(..., description="Path to example images")
    fm_endpoint: str = Field(..., description="Foundation model endpoint name")
    temperature: float = Field(0.1, description="Model temperature for generation")
    top_p: float = Field(0.1, description="Top-p sampling parameter")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay_s: int = Field(1, description="Delay between retries in seconds")
    metadata_prompt: str = Field(
        ..., description="System prompt for metadata extraction"
    )
    metadata_example: str = Field(
        ..., description="Example output for metadata extraction"
    )
    tag_prompt: str = Field(..., description="System prompt for tag extraction")
    tag_example: str = Field(..., description="Example output for tag extraction")
    num_few_shot_examples: int = Field(
        0, description="Number of few shot examples to use"
    )

    @field_validator("temperature", "top_p")
    @classmethod
    def validate_probabilities(cls, v: float) -> float:
        """Validate probability parameters are in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Temperature and top_p must be between 0.0 and 1.0")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """Validate max retries is reasonable."""
        if v < 0 or v > 10:
            raise ValueError("Max retries must be between 0 and 10")
        return v

    @field_validator("retry_delay_s")
    @classmethod
    def validate_retry_delay(cls, v: int) -> int:
        """Validate retry delay is reasonable."""
        if v < 0 or v > 60:
            raise ValueError("Retry delay must be between 0 and 60 seconds")
        return v


class PIDConfig(BaseModel):
    """Main configuration for P&ID processing pipeline."""

    catalog: str = Field(..., description="Databricks catalog name")
    db_schema: str = Field(..., alias="schema", description="Databricks schema name")
    preprocess: PreprocessConfig = Field(..., description="Preprocessing configuration")
    parse: ParseConfig = Field(..., description="Parsing configuration")

    @field_validator("catalog", "db_schema")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        """Validate catalog and schema names are valid identifiers."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Catalog and schema names must be alphanumeric with underscores/hyphens"
            )
        return v

    @property
    def schema(self) -> str:
        """Backward compatibility property for schema access."""
        return self.db_schema


def load_config(config_path: str = "config.yaml") -> PIDConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Validated PIDConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return PIDConfig(**config_data)


def get_preprocessing_config(config_path: str = "config.yaml") -> PreprocessConfig:
    """
    Load only preprocessing configuration.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        PreprocessConfig instance
    """
    config = load_config(config_path)
    return config.preprocess


def get_parse_config(config_path: str = "config.yaml") -> ParseConfig:
    """
    Load only parsing configuration.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        ParseConfig instance
    """
    config = load_config(config_path)
    return config.parse
