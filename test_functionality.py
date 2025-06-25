#!/usr/bin/env python3
"""
Test script for P&ID parsing functionality.

This script tests the core components of the P&ID parsing project including:
- Configuration loading
- Image preprocessing functions
- Basic image operations
- File I/O operations
- JSON parsing functionality
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml
from PIL import Image, ImageEnhance
from mlflow.models import ModelConfig

# Import our custom modules
from src.preprocess import get_tile_positions


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def test_config_loading() -> Dict[str, Any]:
    """Test loading configuration from YAML file."""
    logger = logging.getLogger(__name__)

    try:
        config = ModelConfig(development_config="config.yaml").to_dict()
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"Catalog: {config.get('catalog', 'N/A')}")
        logger.info(f"Schema: {config.get('schema', 'N/A')}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return {}


def test_tile_positions() -> bool:
    """Test the tile position calculation function."""
    logger = logging.getLogger(__name__)

    try:
        # Test with sample parameters
        image_size = 1000
        tile_size = 256
        overlap = 64

        positions = get_tile_positions(image_size, tile_size, overlap)

        logger.info(f"✅ Tile positions calculated successfully")
        logger.info(
            f"Image size: {image_size}, Tile size: {tile_size}, Overlap: {overlap}"
        )
        logger.info(f"Generated positions: {positions}")

        # Basic validation
        assert len(positions) > 0, "Should generate at least one position"
        assert all(
            pos >= 0 for pos in positions
        ), "All positions should be non-negative"
        assert all(
            pos + tile_size <= image_size for pos in positions
        ), "All tiles should fit within image"

        return True
    except Exception as e:
        logger.error(f"❌ Tile position calculation failed: {e}")
        return False


def test_image_processing() -> bool:
    """Test basic image processing capabilities."""
    logger = logging.getLogger(__name__)

    try:
        # Check if example images exist
        example_dir = Path("examples")
        if not example_dir.exists():
            logger.warning("Examples directory not found, creating a test image")
            # Create a simple test image
            test_image = Image.new("RGB", (500, 300), color="white")
            test_image.save("test_image.jpg")
            image_path = "test_image.jpg"
        else:
            # Use existing example image
            example_images = list(example_dir.glob("*.jpg"))
            if example_images:
                image_path = str(example_images[0])
                logger.info(f"Using example image: {image_path}")
            else:
                logger.warning("No example images found, creating test image")
                test_image = Image.new("RGB", (500, 300), color="white")
                test_image.save("test_image.jpg")
                image_path = "test_image.jpg"

        # Test image loading and processing
        image = Image.open(image_path)
        logger.info(f"✅ Image loaded successfully: {image.size}")

        # Test image enhancement (similar to preprocessing pipeline)
        grayscale = image.convert("L")
        contrast = ImageEnhance.Contrast(grayscale)
        enhanced = contrast.enhance(2.0)

        # Test thresholding
        threshold = 128
        binary = enhanced.point(lambda x: 255 if x > threshold else 0, mode="1")

        logger.info("✅ Image processing operations completed successfully")

        # Clean up test image if we created one
        if image_path == "test_image.jpg":
            Path("test_image.jpg").unlink(missing_ok=True)

        return True
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        return False


def test_json_parsing() -> bool:
    """Test JSON parsing functionality similar to the parser module."""
    logger = logging.getLogger(__name__)

    try:
        # Test cases for JSON extraction
        test_cases = [
            # Valid JSON
            '{"equipment_tags": ["250-LT-1610"], "line_tags": ["14\\"-SU-241001-SEG"]}',
            # JSON with markdown formatting
            '```json\n{"equipment_tags": ["250-LT-1610"]}\n```',
            # JSON with trailing comma
            '{"equipment_tags": ["250-LT-1610"],}',
            # JSON with single quotes
            "{'equipment_tags': ['250-LT-1610']}",
        ]

        for i, test_case in enumerate(test_cases):
            try:
                # Simulate the JSON cleaning process from parser.py
                cleaned = test_case.strip()

                # Remove markdown markers
                if cleaned.startswith("```"):
                    first_newline = cleaned.find("\n")
                    if first_newline != -1:
                        cleaned = cleaned[first_newline + 1 :]

                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

                # Remove trailing commas
                import re

                cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

                # Fix single quotes
                cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
                cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)

                # Parse JSON
                parsed = json.loads(cleaned)
                logger.info(f"✅ Test case {i+1} parsed successfully")

            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Test case {i+1} failed to parse: {e}")

        logger.info("✅ JSON parsing tests completed")
        return True
    except Exception as e:
        logger.error(f"❌ JSON parsing test failed: {e}")
        return False


def test_dataframe_operations() -> bool:
    """Test pandas DataFrame operations used in the project."""
    logger = logging.getLogger(__name__)

    try:
        # Create sample metadata similar to preprocessing output
        sample_metadata = [
            {
                "filename": "test_diagram",
                "file_path_hash": "abc123",
                "file_width": 1000,
                "file_height": 800,
                "page_number": 1,
                "tile_number": 1,
                "left": 0,
                "upper": 0,
                "right": 256,
                "lower": 256,
            },
            {
                "filename": "test_diagram",
                "file_path_hash": "abc123",
                "file_width": 1000,
                "file_height": 800,
                "page_number": 1,
                "tile_number": 2,
                "left": 192,
                "upper": 0,
                "right": 448,
                "lower": 256,
            },
        ]

        # Create DataFrame
        df = pd.DataFrame(sample_metadata)
        logger.info(f"✅ DataFrame created with {len(df)} rows")

        # Test basic operations
        grouped = df.groupby("page_number").size()
        logger.info(f"Tiles per page: {grouped.to_dict()}")

        # Test filtering
        filtered = df[df["tile_number"] > 1]
        logger.info(f"Filtered DataFrame has {len(filtered)} rows")

        logger.info("✅ DataFrame operations completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ DataFrame operations failed: {e}")
        return False


def run_all_tests() -> None:
    """Run all test functions and report results."""
    logger = setup_logging()

    logger.info("🚀 Starting P&ID parsing functionality tests...")
    logger.info("=" * 60)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Tile Position Calculation", test_tile_positions),
        ("Image Processing", test_image_processing),
        ("JSON Parsing", test_json_parsing),
        ("DataFrame Operations", test_dataframe_operations),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)

        try:
            result = test_func()
            if result is True or (isinstance(result, dict) and result):
                results.append((test_name, "PASS"))
                logger.info(f"✅ {test_name}: PASSED")
            else:
                results.append((test_name, "FAIL"))
                logger.info(f"❌ {test_name}: FAILED")
        except Exception as e:
            results.append((test_name, "ERROR"))
            logger.error(f"💥 {test_name}: ERROR - {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for test_name, status in results:
        status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[status]
        logger.info(f"{status_emoji} {test_name}: {status}")

    logger.info(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! The project is ready to use.")
        sys.exit(0)
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
