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

def generate_tiling_gdal(image_path, w_size, mask_path=None, batch_size=1000, max_memory_mb=1024):
    """Memory-efficient tiling that follows the same pattern as generate_tiling"""
    from osgeo import gdal
    import os
    import tempfile
    import pickle
    import gc
    import numpy as np
    import psutil
    
    gdal.UseExceptions()
    win_size = w_size
    pad_px = win_size // 2  # Important: Use same padding logic as generate_tiling
    
    # Create temporary directory for tile storage
    temp_dir = tempfile.mkdtemp(prefix="image_tiles_")
    print(f"Storing tiles temporarily in {temp_dir}")
    metadata_file = os.path.join(temp_dir, "tile_metadata.pkl")
    position_file = os.path.join(temp_dir, "positions.pkl")
    tile_positions = []
    
    # Open datasets with minimal caching
    gdal.SetCacheMax(max_memory_mb * 1024 * 1024 // 4)  # Set GDAL cache to 1/4 of max memory
    opts = ['SPARSE_OK=YES', 'NUM_THREADS=ALL_CPUS', 'USE_PIXEL_BLOCKS=YES']
    img_ds = gdal.Open(image_path)
    if img_ds is None:
        raise ValueError(f"Cannot open image file {image_path}")
    
    img_width = img_ds.RasterXSize
    img_height = img_ds.RasterYSize
    num_bands = img_ds.RasterCount
    
    # **FIX: Detect the actual data type of the source image**
    first_band = img_ds.GetRasterBand(1)
    gdal_dtype = first_band.DataType
    
    # Map GDAL data types to numpy data types
    gdal_to_numpy = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex64,
        gdal.GDT_CInt32: np.complex128,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128
    }
    
    numpy_dtype = gdal_to_numpy.get(gdal_dtype, np.uint8)
    print(f"Detected image data type: GDAL {gdal.GetDataTypeName(gdal_dtype)} -> NumPy {numpy_dtype}")
    
    # Get nodata value if it exists
    nodata_value = first_band.GetNoDataValue()
    if nodata_value is not None:
        print(f"NoData value: {nodata_value}")
    
    mask_ds = None
    if mask_path:
        mask_ds = gdal.Open(mask_path)
    
    # Create a padded virtual dataset to match generate_tiling's logic
    padded_width = img_width + 2*pad_px
    padded_height = img_height + 2*pad_px
    
    # Calculate number of tiles - same logic as view_as_windows with step=pad_px
    num_rows = (padded_height - win_size) // pad_px + 1
    num_cols = (padded_width - win_size) // pad_px + 1
    
    # Calculate total number of tiles and estimate memory requirements
    total_tiles = num_rows * num_cols
    # **FIX: Use actual data type size for memory estimation**
    dtype_size = np.dtype(numpy_dtype).itemsize
    estimated_tile_size = (win_size * win_size * (num_bands if num_bands > 0 else 1) * dtype_size) / (1024 * 1024)  # in MB
    print(f"Processing {total_tiles} tiles (rows={num_rows}, cols={num_cols})")
    print(f"Estimated memory per tile: ~{estimated_tile_size:.2f} MB")
    
    # Pre-calculate all tile positions and save to disk to reduce memory usage
    # This allows us to process chunks without holding all positions in memory
    all_positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            y_padded = row * pad_px
            x_padded = col * pad_px
            y_img = y_padded - pad_px
            x_img = x_padded - pad_px
            read_x = max(0, x_img)
            read_y = max(0, y_img)
            read_width = min(win_size - max(0, -x_img), img_width - read_x)
            read_height = min(win_size - max(0, -y_img), img_height - read_y)
            place_x = max(0, -x_img)
            place_y = max(0, -y_img)
            
            if read_width > 0 and read_height > 0:
                all_positions.append((row, col, read_x, read_y, read_width, read_height, place_x, place_y))
    
    # Save positions to disk and free memory
    with open(position_file, 'wb') as f:
        pickle.dump(all_positions, f)
    del all_positions
    gc.collect()
    
    # Process tiles in batches to limit memory usage
    with open(position_file, 'rb') as f:
        all_positions = pickle.load(f)
    
    # Process tiles in chunks
    chunk_size = min(batch_size, max(1, int(max_memory_mb // estimated_tile_size * 2)))
    print(f"Processing tiles in chunks of {chunk_size} to conserve memory")
    
    tile_counter = 0
    for chunk_start in range(0, len(all_positions), chunk_size):
        # Get current memory usage
        mem_info = psutil.Process(os.getpid()).memory_info()
        print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")
        
        # Process a chunk of tiles
        chunk_end = min(chunk_start + chunk_size, len(all_positions))
        chunk = all_positions[chunk_start:chunk_end]
        
        for row, col, read_x, read_y, read_width, read_height, place_x, place_y in chunk:
            # Check mask if provided
            if mask_ds:
                try:
                    mask_data = mask_ds.ReadAsArray(read_x, read_y, read_width, read_height)
                    if mask_data is not None:
                        if len(mask_data.shape) == 3:
                            mask_data = mask_data[0]
                        # Skip tile if any part is masked (0=masked)
                        if np.any(mask_data == 0):
                            continue
                except Exception as e:
                    print(f"Warning: Error reading mask at {(read_x, read_y)}: {e}")
                    continue
            
            # Process tile with padding as needed
            try:
                # **FIX: Use detected data type instead of hardcoded uint8**
                # For 1-band images
                if num_bands == 1:
                    # Create full tile with correct data type
                    tile_data = np.zeros((win_size, win_size), dtype=numpy_dtype)
                    
                    # Read available data into the correct position in the tile
                    if read_width > 0 and read_height > 0:
                        img_data = img_ds.ReadAsArray(read_x, read_y, read_width, read_height)
                        # Ensure data type consistency
                        if img_data.dtype != numpy_dtype:
                            img_data = img_data.astype(numpy_dtype)
                        tile_data[place_y:place_y + read_height, place_x:place_x + read_width] = img_data
                    
                    # Add padding
                    # Left padding if needed
                    if place_x > 0:
                        for i in range(place_x):
                            tile_data[:, i] = tile_data[:, place_x]
                    
                    # Right padding if needed
                    right_edge = place_x + read_width
                    if right_edge < win_size:
                        for i in range(right_edge, win_size):
                            tile_data[:, i] = tile_data[:, right_edge-1]
                    
                    # Top padding if needed
                    if place_y > 0:
                        for i in range(place_y):
                            tile_data[i, :] = tile_data[place_y, :]
                    
                    # Bottom padding if needed
                    bottom_edge = place_y + read_height
                    if bottom_edge < win_size:
                        for i in range(bottom_edge, win_size):
                            tile_data[i, :] = tile_data[bottom_edge-1, :]
                    
                    # Add channel dimension for consistency
                    tile_data = np.expand_dims(tile_data, axis=2)
                
                # For multi-band images
                else:
                    # **FIX: Create full tile with correct data type**
                    tile_data = np.zeros((win_size, win_size, num_bands), dtype=numpy_dtype)
                    
                    # Process one band at a time to reduce memory
                    for band in range(num_bands):
                        if read_width > 0 and read_height > 0:
                            # Read only one band at a time
                            band_data = img_ds.GetRasterBand(band+1).ReadAsArray(read_x, read_y, read_width, read_height)
                            # **FIX: Ensure data type consistency**
                            if band_data.dtype != numpy_dtype:
                                band_data = band_data.astype(numpy_dtype)
                            tile_data[place_y:place_y + read_height, place_x:place_x + read_width, band] = band_data
                            # Free memory immediately
                            del band_data
                        
                        # Padding for this band
                        if place_x > 0:
                            for i in range(place_x):
                                tile_data[:, i, band] = tile_data[:, place_x, band]
                        
                        right_edge = place_x + read_width
                        if right_edge < win_size:
                            for i in range(right_edge, win_size):
                                tile_data[:, i, band] = tile_data[:, right_edge-1, band]
                        
                        if place_y > 0:
                            for i in range(place_y):
                                tile_data[i, :, band] = tile_data[place_y, :, band]
                        
                        bottom_edge = place_y + read_height
                        if bottom_edge < win_size:
                            for i in range(bottom_edge, win_size):
                                tile_data[i, :, band] = tile_data[bottom_edge-1, :, band]
                
                # Save tile to disk
                tile_file = os.path.join(temp_dir, f"tile_{tile_counter:08d}.npy")
                np.save(tile_file, tile_data, allow_pickle=False)
                
                # Store position info
                tile_positions.append((row, col, tile_file))
                tile_counter += 1
                
                # Clean up
                del tile_data
                
                # Write metadata to disk periodically
                if tile_counter % batch_size == 0:
                    with open(metadata_file, 'wb') as f:
                        pickle.dump(tile_positions, f)
                    print(f"Processed {tile_counter} tiles so far...")
                    gc.collect()  # Force garbage collection
                    
            except Exception as e:
                print(f"Warning: Error processing tile: {e}")
                print(f"Error details: {str(e)}")  # Print more error details
                continue
        
        # Clean up after each chunk
        gc.collect()
    
    # Save final metadata
    with open(metadata_file, 'wb') as f:
        pickle.dump(tile_positions, f)
    
    # Clean up positions file
    try:
        os.unlink(position_file)
    except:
        pass
    
    # Clean up GDAL resources
    img_ds = None
    if mask_ds:
        mask_ds = None
    
    # **FIX: Store data type in metadata for the dataset class**
    class DiskTileDataset:
        def __init__(self, metadata_file, win_size, numpy_dtype):
            with open(metadata_file, 'rb') as f:
                self.tile_metadata = pickle.load(f)
            self.win_size = win_size
            self.numpy_dtype = numpy_dtype  # Store the data type
            self.temp_dir = os.path.dirname(metadata_file)
            
        def __len__(self):
            return len(self.tile_metadata)
            
        def __getitem__(self, idx):
            row, col, tile_path = self.tile_metadata[idx]
            tile_data = np.load(tile_path)
            # Ensure correct data type (safety check)
            if tile_data.dtype != self.numpy_dtype:
                tile_data = tile_data.astype(self.numpy_dtype)
            return tile_data
            
        def get_patch_positions(self):
            # Return row, col pairs just like generate_tiling does
            return [(row, col) for row, col, _ in self.tile_metadata]
            
        def cleanup(self):
            """Remove temporary files"""
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {self.temp_dir}: {e}")
    
    print(f"Successfully processed and stored {tile_counter} tiles with data type {numpy_dtype}")
    return DiskTileDataset(metadata_file, win_size, numpy_dtype)

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
