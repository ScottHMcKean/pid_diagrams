import math

def get_tile_positions(image_size, tile_size, overlap):
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