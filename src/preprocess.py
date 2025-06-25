import math
import hashlib
from pathlib import Path
from typing import Dict, List, Any

import pdfplumber
from src.config import PreprocessConfig
from PIL import Image, ImageEnhance
import base64
import io


def get_tile_positions(image_size: int, tile_size: int, overlap: int) -> List[int]:
    """
    Calculate tile positions for image tiling with overlap.

    Args:
        image_size: Total size of the image in pixels
        tile_size: Size of each tile in pixels
        overlap: Overlap between adjacent tiles in pixels

    Returns:
        List of sorted tile positions (start coordinates) for tiling the image
    """
    step = tile_size - overlap
    num_tiles = math.ceil((image_size - overlap) / step)
    total_coverage = step * num_tiles + overlap
    offset = max(0, (total_coverage - image_size) // 2)
    positions = []
    for i in range(num_tiles):
        pos = i * step - offset
        if pos < 0:
            pos = 0
        if pos + tile_size > image_size:
            pos = image_size - tile_size
        positions.append(pos)
    return sorted(set(positions))


def load_image_w_max_size(
    file_path: str,
    max_size_bytes: float = 0.5 * 1024 * 1024,
    target_dpi: int = 72,
    original_dpi: int = 200,
):
    """
    Load image and reduce DPI if file size exceeds maximum.

    Args:
        file_path: Path to the image file
        max_size_bytes: Maximum allowed file size in bytes (default: 0.5 * 1024 * 1024)
        target_dpi: Target DPI to reduce to (default: 72)
        original_dpi: Assumed original DPI (default: 200)

    Returns:
        str: Base64 encoded image data as a string
    """
    file_size = Path(file_path).stat().st_size

    if file_size <= max_size_bytes:
        # File is small enough, load normally
        return base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")

    # File is too large, reduce DPI by resizing
    with Image.open(file_path) as img:
        # Calculate new dimensions based on DPI reduction
        width, height = img.size
        dpi_ratio = target_dpi / original_dpi
        new_width = int(width * dpi_ratio)
        new_height = int(height * dpi_ratio)

        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save to bytes buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format=img.format, optimize=True, quality=85)
        buffer.seek(0)

        # Check if still too large, reduce quality further if needed
        if len(buffer.getvalue()) > max_size_bytes:
            buffer = io.BytesIO()
            resized_img.save(buffer, format=img.format, optimize=True, quality=60)
            buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(
    image: Image.Image, contrast_factor: float = 2.0, threshold_value: int = 128
) -> Image.Image:
    """
    Convert image to grayscale, enhance contrast, and apply thresholding.

    Args:
        image: Input PIL Image to preprocess
        contrast_factor: Factor to enhance contrast (default: 2.0)
        threshold_value: Threshold value for binary conversion (default: 128)

    Returns:
        Preprocessed PIL Image in binary mode
    """
    image = image.convert("L")
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(contrast_factor)
    return image.point(lambda x: 255 if x > threshold_value else 0, mode="1")


def process_pdf_to_tiles(
    pdf_path: str, output_dir: str, config: PreprocessConfig
) -> List[Dict[str, Any]]:
    """
    Process PDF file into tiled images and return metadata.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save processed images
        config: Preprocessing configuration with validated parameters

    Returns:
        List of metadata dictionaries for each tile
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Create hash and directories
    file_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
    tile_dir = output_dir / file_hash / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Convert page to image and preprocess
            page_img = page.to_image(resolution=config.dpi).original
            page_img = preprocess_image(page_img)

            # Save full page
            page_path = output_dir / file_hash / f"{file_hash}_p{page_num}.jpg"
            page_img.save(page_path, "JPEG")

            # Get tile positions
            width, height = page_img.size
            x_positions = get_tile_positions(
                width, config.tile_width_px, config.overlap_px
            )
            y_positions = get_tile_positions(
                height, config.tile_height_px, config.overlap_px
            )

            # Generate tiles
            tile_count = 1
            for upper in y_positions:
                for left in x_positions:
                    right = left + config.tile_width_px
                    lower = upper + config.tile_height_px

                    # Crop and save tile
                    tile = page_img.crop((left, upper, right, lower))
                    tile_filename = f"{file_hash}_p{page_num}_t{tile_count}.jpg"
                    tile_path = tile_dir / tile_filename
                    tile.save(tile_path, "JPEG")

                    # Store metadata
                    metadata.append(
                        {
                            "filename": pdf_path.stem,
                            "file_path_hash": file_hash,
                            "file_width": width,
                            "file_height": height,
                            "file_dpi": config.dpi,
                            "page_number": page_num,
                            "page_path": str(page_path),
                            "tile_number": tile_count,
                            "left": left,
                            "upper": upper,
                            "right": right,
                            "lower": lower,
                            "tile_path": str(tile_path),
                        }
                    )

                    tile_count += 1

    return metadata
