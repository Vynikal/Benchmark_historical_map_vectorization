import numpy as np
from skimage.util import view_as_windows
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse


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
            if mask_path is not None:
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

def generate_tiling_gdal(image_path, w_size, mask_path=None, batch_size=1000):
    """Ultra memory-efficient tiling using GDAL and disk storage"""
    from osgeo import gdal
    import os
    import tempfile
    import pickle
    import gc
    
    gdal.UseExceptions()  # Enable exceptions
    win_size = w_size
    pad_px = win_size // 2
    
    # Create temporary directory for tile storage
    temp_dir = tempfile.mkdtemp(prefix="image_tiles_")
    print(f"Storing tiles temporarily in {temp_dir}")
    
    # Create a metadata file to track tile positions
    metadata_file = os.path.join(temp_dir, "tile_metadata.pkl")
    tile_positions = []
    
    # Open the image dataset without loading into memory
    img_ds = gdal.Open(image_path)
    if img_ds is None:
        raise ValueError(f"Cannot open image file {image_path}")
    
    # Get image dimensions and band count
    img_width = img_ds.RasterXSize
    img_height = img_ds.RasterYSize
    num_bands = img_ds.RasterCount
    
    # Open mask if provided
    mask_ds = None
    if mask_path:
        mask_ds = gdal.Open(mask_path)
        if mask_ds is None:
            raise ValueError(f"Cannot open mask file {mask_path}")
    
    # Calculate number of tiles
    tiles_y = (img_height) // pad_px - 1
    tiles_x = (img_width) // pad_px - 1
    
    print(f"Processing {tiles_y * tiles_x} potential tiles...")
    
    # Counter for valid tiles
    tile_counter = 0
    
    # Process directly tile by tile
    for row in range(tiles_y):
        y_pos = row * pad_px
        
        # Skip if out of bounds
        if y_pos + win_size > img_height:
            continue
            
        for col in range(tiles_x):
            x_pos = col * pad_px
            
            # Skip if out of bounds
            if x_pos + win_size > img_width:
                continue
            
            # Check mask first if provided (using GDAL to avoid loading the whole mask)
            if mask_ds:
                mask_data = mask_ds.ReadAsArray(x_pos, y_pos, win_size, win_size)
                # Convert to boolean mask based on your mask criteria
                if mask_data is not None:
                    # Handle multi-band masks
                    if len(mask_data.shape) == 3:
                        mask_data = mask_data[0]
                    
                    # Create boolean mask (0=masked)
                    mask_bool = mask_data == 0
                    
                    # Skip tile if any part is masked
                    if np.any(mask_bool):
                        continue
            
            # Read just this tile from the image
            if num_bands == 1:
                # Single band image (grayscale)
                tile_data = img_ds.ReadAsArray(x_pos, y_pos, win_size, win_size)
                # Add a dummy dimension if needed for consistency
                if len(tile_data.shape) == 2:
                    tile_data = tile_data[..., np.newaxis]
            else:
                # Multi-band image (RGB, etc.)
                tile_data = np.zeros((win_size, win_size, num_bands), dtype=np.uint8)
                for band in range(num_bands):
                    band_data = img_ds.GetRasterBand(band+1).ReadAsArray(
                        xoff=x_pos,
                        yoff=y_pos,
                        win_xsize=win_size,
                        win_ysize=win_size
                    )
                    tile_data[:, :, band] = band_data
            
            # Save tile to disk
            tile_file = os.path.join(temp_dir, f"tile_{tile_counter:08d}.npy")
            np.save(tile_file, tile_data, allow_pickle=False)
            
            # Store position info
            tile_positions.append((row, col, tile_file))
            tile_counter += 1
            
            # Periodically save metadata and clean memory
            if tile_counter % batch_size == 0:
                with open(metadata_file, 'wb') as f:
                    pickle.dump(tile_positions, f)
                print(f"Processed {tile_counter} tiles so far...")
                gc.collect()
    
    # Save final metadata
    with open(metadata_file, 'wb') as f:
        pickle.dump(tile_positions, f)
    
    # Clean up GDAL resources
    img_ds = None
    if mask_ds:
        mask_ds = None
    
    # Create a wrapper class to handle tile loading
    class DiskTileDataset:
        def __init__(self, metadata_file, win_size):
            with open(metadata_file, 'rb') as f:
                self.tile_metadata = pickle.load(f)
            self.win_size = win_size
            self.temp_dir = os.path.dirname(metadata_file)
            
        def __len__(self):
            return len(self.tile_metadata)
            
        def __getitem__(self, idx):
            row, col, tile_path = self.tile_metadata[idx]
            return np.load(tile_path)
            
        def get_patch_positions(self):
            # Return just the row, col pairs without file paths
            return [(row, col) for row, col, _ in self.tile_metadata]
            
        def cleanup(self):
            """Remove temporary files"""
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory {self.temp_dir}")
            except:
                print(f"Warning: Could not clean up {self.temp_dir}")
    
    print(f"Successfully processed and stored {tile_counter} tiles")
    return DiskTileDataset(metadata_file, win_size)

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
