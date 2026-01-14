import numpy as np
import cv2
import os
from tempfile import mkdtemp


def reconstruct_tiling(original_image_path, test_pred_dict, tile_save_path, w_size, image_debug=None, save_image=True):
    in_patches = list(test_pred_dict.keys())
    in_patches.sort()
    patches_images =  list(test_pred_dict.values())

    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)

    if save_image:
        cv2.imwrite(tile_save_path, new_img)

    if image_debug is not None:
        cv2.imwrite(image_debug, new_img * 255)
    return new_img


def reconstruct_tiling_array(original_image_path, patches_images, w_size):
    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)
    return new_img


def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype, patch_positions=None, batch_size=100):
    '''Memory-efficient reconstruction from patches using a single memory-mapped array
    
    Args:
        patches_images: List of patch arrays
        patch_size: Size of each patch
        step_size: Step size between patches (usually patch_size//2)
        image_size_2d: Original image dimensions (H,W) or (H,W,C)
        image_dtype: Data type for output image
        patch_positions: List of (row,col) positions for each patch. If None, assumes sequential filling
        batch_size: Number of patches to process at once
    '''
    import math
    
    # Get original dimensions
    out_h = image_size_2d[0]
    out_w = image_size_2d[1]
    has_channels = len(patches_images.shape) > 3 and patches_images.shape[3] > 1
    out_channels = 3 if has_channels else 1
    
    # Calculate dimensions needed for the patches (same padding as tiling)
    i_h = out_h + patch_size
    i_w = out_w + patch_size
    p_h = p_w = patch_size
    
    # Calculate offsets and overlap handling parameters
    patch_offset = step_size//2
    patch_inner = p_h - step_size
    
    # Create a single memory-mapped array for the final result
    result_filename = os.path.join(mkdtemp(), 'final_result.npy')
    out_shape = (out_h, out_w, out_channels) if has_channels else (out_h, out_w)
    result = np.memmap(result_filename, dtype=image_dtype, mode='w+', shape=out_shape)
    
    # Calculate number of rows and columns in patch grid - MUST match tiling logic
    # Use ceiling division to ensure we can handle all tiles from the tiling function
    numrows = math.ceil((i_h - patch_size) / step_size) + 1
    numcols = math.ceil((i_w - patch_size) / step_size) + 1
    
    # If no positions provided, create position mapping
    if patch_positions is None:
        patch_positions = [(r,c) for r in range(numrows) for c in range(numcols)]
    
    # Validate number of positions matches patches
    if len(patch_positions) != len(patches_images):
        raise ValueError(f"Number of positions ({len(patch_positions)}) must match number of patches ({len(patches_images)})")
    
    # Process patches in batches to manage memory
    total_patches = len(patches_images)
    for start_idx in range(0, total_patches, batch_size):
        end_idx = min(start_idx + batch_size, total_patches)
        batch_patches = patches_images[start_idx:end_idx]
        batch_positions = patch_positions[start_idx:end_idx]
        
        for patch, (row, col) in zip(batch_patches, batch_positions):
            # Skip if position is out of bounds
            if row >= numrows or col >= numcols:
                continue
                
            # Extract inner region of patch (non-overlapping part)
            inner_patch = patch[patch_offset:-patch_offset, patch_offset:-patch_offset]
            
            # Get the actual size of inner_patch
            actual_patch_h, actual_patch_w = inner_patch.shape[:2]
            
            # Calculate destination position in final array
            # Account for the step_size//2 offset by subtracting it from the position
            dest_y = row * step_size - step_size//2
            dest_x = col * step_size - step_size//2
            
            # Skip patches that are completely outside the output bounds
            if dest_y + actual_patch_h <= 0 or dest_y >= out_h or dest_x + actual_patch_w <= 0 or dest_x >= out_w:
                continue
            
            # Ensure destination is within bounds of final array
            if dest_y < 0 or dest_x < 0 or dest_y + actual_patch_h > out_h or dest_x + actual_patch_w > out_w:
                # Calculate the overlap region in output coordinates
                dst_y_start = max(0, dest_y)
                dst_x_start = max(0, dest_x)
                dst_y_end = min(out_h, dest_y + actual_patch_h)
                dst_x_end = min(out_w, dest_x + actual_patch_w)
                
                # Calculate corresponding region in patch coordinates (same size as dst)
                src_y_start = dst_y_start - dest_y
                src_x_start = dst_x_start - dest_x
                src_y_end = src_y_start + (dst_y_end - dst_y_start)
                src_x_end = src_x_start + (dst_x_end - dst_x_start)
                
                # Copy valid part of patch to result
                if has_channels:
                    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = inner_patch[
                        src_y_start:src_y_end, src_x_start:src_x_end, :]
                else:
                    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = inner_patch[
                        src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                # Normal case - patch fits entirely within bounds
                if has_channels:
                    result[dest_y:dest_y+actual_patch_h, dest_x:dest_x+actual_patch_w, :] = inner_patch
                else:
                    result[dest_y:dest_y+actual_patch_h, dest_x:dest_x+actual_patch_w] = inner_patch
        
        # Force write to disk after each batch
        result.flush()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return result

def save_random_chips(dataset, save_path, prefix, num_chips=5):
    """Save random chips from a dataset
    
    Args:
        dataset: Data instance from smart_data_loader
        save_path: Directory to save chips
        prefix: Prefix for saved files (e.g. 'train' or 'val')
        num_chips: Number of random chips to save
    """
    import random
    import cv2
    import os
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_chips)
    
    for i, idx in enumerate(indices):
        if dataset.unseen:
            img, pos = dataset[idx]
            # Convert from tensor to numpy
            img = img.numpy().transpose(1,2,0) * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_r{pos[0]}_c{pos[1]}.png'), img)
        else:
            img, labels = dataset[idx]
            # Convert from tensor to numpy
            img = img.numpy().transpose(1,2,0) * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.astype(np.uint8)
            if isinstance(labels, dict):
                labels = labels['labels'].numpy()[0] * 255
            else:
                labels = labels.numpy()[0] * 255
            labels = labels.astype(np.uint8)
            
            # Save both image and label
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_img.png'), img)
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_gt.png'), labels)