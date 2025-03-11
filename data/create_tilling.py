import numpy as np
from skimage.util import view_as_windows
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse

import pdb


def generate_tiling(image_path, w_size, mask_path=None):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))

    msk_bg = None
    if mask_path:
        msk_bg = np.array(Image.open(mask_path))
        if msk_bg is None:
            raise ValueError(f"Mask file {mask_path} cannot be read")
        if msk_bg.shape != in_img.shape[:2]:
            raise ValueError(f"GT and mask shapes don't match: {in_img.shape[:2]} vs {msk_bg.shape}")
        # Create boolean mask (0=masked) and pad it same as image
        msk_bg = msk_bg == 0
        msk_bg = np.pad(msk_bg, [(pad_px,pad_px), (pad_px,pad_px)], 'edge')
        mask_tiles = view_as_windows(msk_bg, (win_size,win_size), step=pad_px) if msk_bg is not None else None

    if len(in_img.shape) == 2:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    else:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    tiles_lst = []
    tile_positions = []  # Track positions of valid tiles
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            # Check mask if provided
            if mask_tiles is not None:
                mask_tile = mask_tiles[row, col]
                # Skip tile if any part is masked
                masked_ratio = np.mean(mask_tile)
                if masked_ratio > 0:
                    continue
                
            if len(in_img.shape) == 2:
                tt = tiles[row, col, ...].copy()
            else:
                tt = tiles[row, col, 0, ...].copy()
            
            tiles_lst.append(tt)
            tile_positions.append((row, col))  # Store position of valid tile
            
    return tiles_lst, tile_positions

def main():
    parser = argparse.ArgumentParser(description='Create Tillings.')
    parser.add_argument('input_path', help='Path of original image.')
    parser.add_argument('gt_path', help='Path of ground truth.')
    parser.add_argument('save_image_path', help='Directory of images for saving the tilings.')
    parser.add_argument('save_gt_path', help='Directory of ground truth for saving the tilings.')
    parser.add_argument('width', help='Width of the tillings')

    args = parser.parse_args()
    generate_tiling(args.input_path, args.gt_path, args.save_image_path, args.save_gt_path, args.width)


if __name__ == '__main__':
    main()
