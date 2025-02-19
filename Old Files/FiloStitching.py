import os
import re
import cv2
import numpy as np

def parse_filename(filename):
    pattern = r"x-?(\d+)_y-?(\d+)"
    match = re.search(pattern, filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None

def collect_tiles(input_folder):
    tiles = {}
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            coords = parse_filename(filename)
            if coords:
                x, y = coords
                if y not in tiles:
                    tiles[y] = {}
                tiles[y][x] = filename

    if not tiles:
        raise ValueError("No valid tiles found in the input folder")

    return tiles

def stitch_tiles(tiles, input_folder, output_path):
    tile_width, tile_height = None, None

    rows = len(tiles)
    columns = max(len(row) for row in tiles.values())

    print(f"Detected grid size: {columns} columns x {rows} rows")

    # Determine tile size from the first tile
    for row in tiles.values():
        for filename in row.values():
            tile_path = os.path.join(input_folder, filename)
            tile_image = cv2.imread(tile_path)
            if tile_image is None:
                raise ValueError(f"Failed to read image: {tile_path}")
            if tile_width is None or tile_height is None:
                tile_height, tile_width = tile_image.shape[:2]
                print(f"Tile size detected: {tile_width} x {tile_height}")
            break
        break

    if tile_width is None or tile_height is None:
        raise ValueError("Failed to determine tile size")

    stitched_image = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    print(f"Creating new image of size: {columns * tile_width} x {rows * tile_height}")

    for row_idx, (y, row) in enumerate(sorted(tiles.items())):
        for col_idx, (x, filename) in enumerate(sorted(row.items())):
            try:
                tile_path = os.path.join(input_folder, filename)
                print(f"Pasting tile {filename} from path {tile_path} at position (column {col_idx}, row {row_idx})")
                tile_image = cv2.imread(tile_path)
                if tile_image is None:
                    raise ValueError(f"Failed to read image: {tile_path}")
                stitched_image[row_idx * tile_height:(row_idx + 1) * tile_height,
                               col_idx * tile_width:(col_idx + 1) * tile_width] = tile_image
            except Exception as e:
                print(f"Error while processing tile {filename} at (column {col_idx}, row {row_idx}): {e}")
                raise

    cv2.imwrite(output_path, stitched_image)
    print(f"Stitched image saved at: {output_path}")

input_folder = '/Users/filippocolella/Desktop/Digital Image Analysis/thresholded'
output_image_path = '/Users/filippocolella/Desktop/Digital Image Analysis/stitched_image.png'  # Ensure this path ends with a valid image extension

try:
    tiles = collect_tiles(input_folder)
    print(f"Tiles collected successfully. Total rows: {len(tiles)}")
    stitch_tiles(tiles, input_folder, output_image_path)
    print(f"Image stitched successfully. Output saved at: {output_image_path}")
except Exception as e:
    print(f"An error occurred: {e}")
