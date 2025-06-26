"""
Test preprocessing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from PIL import Image
import tempfile
import os

from src.preprocess import get_tile_positions, preprocess_image, process_pdf_to_tiles


@pytest.mark.preprocess
@pytest.mark.unit
class TestGetTilePositions:
    """Test tile position calculation."""

    def test_simple_tiling(self):
        """Test basic tile position calculation."""
        positions = get_tile_positions(image_size=1000, tile_size=400, overlap=50)

        # Should return list of integers
        assert isinstance(positions, list)
        assert all(isinstance(pos, int) for pos in positions)

        # First position should be near 0
        assert positions[0] >= 0

        # Last position should allow full tile within image
        assert positions[-1] + 400 <= 1000

    def test_no_overlap_tiling(self):
        """Test tiling with no overlap."""
        positions = get_tile_positions(image_size=800, tile_size=200, overlap=0)

        # Should have 4 tiles (800/200)
        assert len(positions) == 4
        assert positions == [0, 200, 400, 600]

    def test_large_overlap(self):
        """Test tiling with large overlap."""
        positions = get_tile_positions(image_size=500, tile_size=300, overlap=200)

        assert len(positions) >= 2  # Should have at least 2 overlapping tiles

        # Check that positions are valid (all positions should be >= 0)
        assert all(pos >= 0 for pos in positions)

        # Check that last tile fits within image
        assert positions[-1] + 300 <= 500

    def test_edge_cases(self):
        """Test edge cases for tile positioning."""
        # Single tile case
        positions = get_tile_positions(image_size=100, tile_size=100, overlap=0)
        assert positions == [0]

        # Tile larger than image - should be adjusted to fit
        positions = get_tile_positions(image_size=50, tile_size=100, overlap=0)
        # Should still return a position, but adjusted to fit within image
        assert len(positions) == 1
        # Position should be adjusted so tile fits (image_size - tile_size = 50 - 100 = -50, so position should be 0 or negative offset handled)
        assert (
            positions[0] + 100 <= 50 or positions[0] <= 0
        )  # Either fits within or is offset


@pytest.mark.preprocess
@pytest.mark.unit
class TestPreprocessImage:
    """Test image preprocessing functionality."""

    def test_preprocess_image_basic(self):
        """Test basic image preprocessing."""
        # Create a simple test image
        test_image = Image.new("RGB", (100, 100), color="white")

        processed = preprocess_image(test_image)

        # Should return PIL Image in binary mode
        assert isinstance(processed, Image.Image)
        assert processed.mode == "1"  # Binary mode
        assert processed.size == (100, 100)

    def test_preprocess_image_with_params(self):
        """Test image preprocessing with custom parameters."""
        test_image = Image.new("RGB", (50, 50), color="gray")

        processed = preprocess_image(
            test_image, contrast_factor=1.5, threshold_value=100
        )

        assert isinstance(processed, Image.Image)
        assert processed.mode == "1"
        assert processed.size == (50, 50)

    def test_grayscale_input(self):
        """Test preprocessing with grayscale input."""
        test_image = Image.new("L", (75, 75), color=128)

        processed = preprocess_image(test_image)

        assert isinstance(processed, Image.Image)
        assert processed.mode == "1"


@pytest.mark.preprocess
@pytest.mark.integration
class TestProcessPdfToTiles:
    """Test complete PDF to tiles processing."""

    @patch("src.preprocess.pdfplumber.open")
    def test_process_pdf_mock(self, mock_pdf_open):
        """Test PDF processing with mocked pdfplumber."""
        # Create mock PDF and page
        mock_page = Mock()
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Mock the to_image method
        mock_image = Image.new("RGB", (2000, 1000), color="white")
        mock_page.to_image.return_value.original = mock_image

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.config import PreprocessConfig

            config = PreprocessConfig(
                raw_path="/tmp/test.pdf",
                processed_path=temp_dir,
                dpi=200,
                tile_width_px=1000,
                tile_height_px=512,
                overlap_px=100,
            )

            metadata = process_pdf_to_tiles(
                pdf_path="test.pdf", output_dir=temp_dir, config=config
            )

            # Should return list of metadata dicts
            assert isinstance(metadata, list)
            assert len(metadata) > 0

            # Check metadata structure
            first_tile = metadata[0]
            expected_keys = {
                "filename",
                "file_path_hash",
                "file_width",
                "file_height",
                "file_dpi",
                "page_number",
                "page_path",
                "tile_number",
                "left",
                "upper",
                "right",
                "lower",
                "tile_path",
                "unique_key",
            }
            assert set(first_tile.keys()) == expected_keys

            # Check that hash is generated
            assert len(first_tile["file_path_hash"]) == 32  # MD5 hash length

            # Check DPI is preserved
            assert first_tile["file_dpi"] == 200

    def test_invalid_config(self):
        """Test process_pdf_to_tiles with invalid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.config import PreprocessConfig

            # Create invalid config - this should cause validation error during creation
            with pytest.raises(Exception):  # Pydantic validation error
                config = PreprocessConfig(
                    raw_path="/tmp/test.pdf",
                    processed_path=temp_dir,
                    dpi=-1,  # Invalid DPI
                    tile_width_px=1000,
                    tile_height_px=500,
                    overlap_px=100,
                )

    @patch("src.preprocess.pdfplumber.open")
    def test_multiple_pages(self, mock_pdf_open):
        """Test PDF processing with multiple pages."""
        # Create mock PDF with 2 pages
        mock_page1 = Mock()
        mock_page2 = Mock()
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Mock images for both pages
        mock_image1 = Image.new("RGB", (1000, 500), color="white")
        mock_image2 = Image.new("RGB", (1000, 500), color="black")
        mock_page1.to_image.return_value.original = mock_image1
        mock_page2.to_image.return_value.original = mock_image2

        with tempfile.TemporaryDirectory() as temp_dir:
            from src.config import PreprocessConfig

            config = PreprocessConfig(
                raw_path="/tmp/multipage.pdf",
                processed_path=temp_dir,
                dpi=150,
                tile_width_px=512,
                tile_height_px=512,
                overlap_px=50,
            )

            metadata = process_pdf_to_tiles(
                pdf_path="multipage.pdf", output_dir=temp_dir, config=config
            )

            # Should have tiles from both pages
            page_numbers = {tile["page_number"] for tile in metadata}
            assert page_numbers == {1, 2}

            # Each page should have multiple tiles
            page1_tiles = [t for t in metadata if t["page_number"] == 1]
            page2_tiles = [t for t in metadata if t["page_number"] == 2]

            assert len(page1_tiles) > 0
            assert len(page2_tiles) > 0
