from pathlib import Path
from PIL import Image


def tile_image_with_overlap(image_path, output_dir, hash, overlap_percent=10):
    img = Image.open(image_path)
    img_name_suffix = image_path.stem
    width, height = img.size

    cols, rows = 4, 2  # 8 tiles: 4 columns x 2 rows

    # Compute base tile size (without overlap)
    base_tile_width = width // cols
    base_tile_height = height // rows

    # Compute overlap in pixels
    overlap_x = int(base_tile_width * overlap_percent / 100)
    overlap_y = int(base_tile_height * overlap_percent / 100)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile = 1
    for row in range(rows):
        for col in range(cols):
            # Calculate the starting x/y
            left = col * (base_tile_width - overlap_x)
            upper = row * (base_tile_height - overlap_y)

            # For the last column/row, ensure we reach the image edge
            if col == cols - 1:
                right = width
            else:
                right = left + base_tile_width

            if row == rows - 1:
                lower = height
            else:
                lower = upper + base_tile_height

            # Clamp to image boundaries
            left = max(0, left)
            upper = max(0, upper)
            right = min(width, right)
            lower = min(height, lower)

            tile = img.crop((left, upper, right, lower))
            tile_path = output_dir / hash / f"{img_name_suffix}_tile_{str(tile)}.webp"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(tile_path)
            tile += 1
