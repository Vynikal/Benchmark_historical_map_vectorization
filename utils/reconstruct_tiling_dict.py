import numpy as np
import cv2
import argparse


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


def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype, patch_positions=None):
    '''Reconstruct image from patches, placing them in their original positions
    
    Args:
        patches_images: List of patch arrays
        patch_size: Size of each patch
        step_size: Step size between patches (usually patch_size//2)
        image_size_2d: Original image dimensions (H,W) or (H,W,C)
        image_dtype: Data type for output image
        patch_positions: List of (row,col) positions for each patch. If None, assumes sequential filling
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    if len(patches_images.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1

    patch_offset = step_size//2
    patch_inner = p_h-step_size

    # If no positions provided, create position mapping
    if patch_positions is None:
        patch_positions = [(r,c) for r in range(numrows) for c in range(numcols)]
    
    # Validate number of positions matches patches
    if len(patch_positions) != len(patches_images):
        raise ValueError(f"Number of positions ({len(patch_positions)}) must match number of patches ({len(patches_images)})")

    # Place each patch in its specified position
    for patch, (row, col) in zip(patches_images, patch_positions):
        if row >= numrows or col >= numcols:
            continue
        tt_roi = patch[patch_offset:-patch_offset, patch_offset:-patch_offset]
        img[row*step_size:row*step_size+patch_inner,
            col*step_size:col*step_size+patch_inner] = tt_roi

    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]

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
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_r{pos[0]}_c{pos[1]}.png'), img)
        else:
            img, labels = dataset[idx]
            # Convert from tensor to numpy
            img = img.numpy().transpose(1,2,0) * 255
            img = img.astype(np.uint8)
            if isinstance(labels, dict):
                labels = labels['labels'].numpy()[0] * 255
            else:
                labels = labels.numpy()[0] * 255
            labels = labels.astype(np.uint8)
            
            # Save both image and label
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_img.png'), img)
            cv2.imwrite(os.path.join(save_path, f'{prefix}_chip_{i}_gt.png'), labels)