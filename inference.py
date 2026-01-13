import os
import gc
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm

from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation

# Import dataloader
from data.smart_data_loader import Data
from data.reconstruct_tiling_dict import reconstruct_from_patches

# Import model
from model.unet import UNET

from osgeo import gdal
import tempfile
from evaluation.p_eval import corr_comp_qual, clDice

def test(model, win_size, args):
    input = args.input_map_path
    test_mask_path = None
    gt = None

    # Create a disk-based dataset to handle large images
    test_img = Data(input, gt, win_size, unseen=args.unseen, mask_path=test_mask_path)
    test_img_pos = test_img.get_patch_positions()
    
    # Use a smaller batch size to reduce memory usage
    batch_size = 8  # Reduce this if you encounter memory issues
    testloader = torch.utils.data.DataLoader(test_img, batch_size=batch_size, 
                                            shuffle=False, num_workers=0, pin_memory=False)

    if args.cuda:
        model.to(args.device)

    # Create memory-mapped array for storing all patch results
    result_dir = tempfile.mkdtemp(prefix="patch_results_")
    patch_results_file = os.path.join(result_dir, "patch_results.dat")
    
    # Estimate the total number of patches
    total_patches = len(test_img)
    patches_results = np.memmap(patch_results_file, dtype=np.float32, mode='w+',
                              shape=(total_patches, win_size, win_size))
    
    model.eval()
    start_idx = 0
    
    for i, (images, _) in enumerate(tqdm(testloader, desc="Processing image tiles")):
        if args.cuda:
            images = images.to(args.device)

        with torch.no_grad():
            out = model(images)
            fuse = torch.sigmoid(out).cpu().numpy()
                
        # Store results in the memory-mapped array
        batch_size = fuse.shape[0]
        patches_results[start_idx:start_idx+batch_size] = fuse[:, 0, ...]
        start_idx += batch_size
        
        # Force cleanup
        del images, out, fuse
        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()
    
    # Clean up any temporary files from the disk dataset
    if hasattr(test_img, 'disk_dataset') and test_img.disk_dataset is not None:
        if hasattr(test_img.disk_dataset, 'cleanup'):
            test_img.disk_dataset.cleanup()
    
    # Return the memory-mapped array and positions
    return patches_results, test_img_pos


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model as before
    if args.model_type == 'unet':
        model = UNET(n_channels=args.channels, n_classes=args.classes)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500

    # Process tiles and get memory-mapped results
    patches_results, test_img_pos = test(model, win_size, args)
    
    # Get image dimensions
    input_ds = gdal.Open(args.input_map_path)
    if input_ds is None:
        raise ValueError(f"Cannot open image file {args.input_map_path}")
    img_width = input_ds.RasterXSize
    img_height = input_ds.RasterYSize
    input_ds = None  # Close dataset

    # Set up output directories
    name = str(args.input_map_path).split('/')[-1].split('.')[0]
    output_dir = os.path.join(str(Path(args.model).parent), name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate window and padding sizes
    pad_px = win_size // 2

    print("Reconstructing image from patches...")

    # Use the memory-efficient reconstruction function
    image_shape = (img_height, img_width, 1)  # Add channel dimension for consistency
    reconstructed = reconstruct_from_patches(
        patches_results, 
        win_size, 
        pad_px, 
        image_shape, 
        np.float32, 
        patch_positions=test_img_pos,
        batch_size=100  # Process 100 patches at a time to manage memory
    )

    # Initialize these variables before any conditional blocks
    base_filename = f"{str(args.model_type)}_test"
    tile_save_image_path = os.path.join(output_dir, f"{base_filename}.tif")
    mask_applied = False

    # If you have ground truth, evaluate the edge prediction before saving
    if args.gt_edge_path and os.path.exists(args.gt_edge_path):
        print("Evaluating edge prediction against ground truth...")
        # The base_filename is already defined above
        
        # Load ground truth edge map
        gt_edge = cv2.imread(args.gt_edge_path, cv2.IMREAD_GRAYSCALE)


        gt_edge = gt_edge / 255.0  # Normalize to 0-1
        gt_bg = 1 - gt_edge  # Invert to get background
        reconstructed_bg = 1 - reconstructed  # Invert prediction to get background

        if args.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            gt_edge = binary_dilation(gt_edge, structure=struct1).astype(np.uint8)

        
        # Resize ground truth if needed to match prediction
        if gt_edge.shape != reconstructed.shape[:2]:
            gt_edge = cv2.resize(gt_edge, (reconstructed.shape[1], reconstructed.shape[0]))
        
        # Load and apply mask if provided
        if args.input_mask_path and os.path.exists(args.input_mask_path):
            print("Using mask to clip evaluation area...")
            mask = cv2.imread(args.input_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize mask if needed
            if mask.shape != reconstructed.shape[:2]:
                mask = cv2.resize(mask, (reconstructed.shape[1], reconstructed.shape[0]))
            
            # Normalize mask to binary (0 or 1)
            mask = (mask > 127).astype(np.uint8)
            
            # Apply mask to both ground truth and prediction
            gt_edge = gt_edge * mask
            gt_bg = gt_bg * mask
            reconstructed = reconstructed * mask
            reconstructed_bg = reconstructed_bg * mask
            tile_save_image_path = os.path.join(output_dir, f"{base_filename}_masked.tif")
            mask_applied = True
        # No need for an else clause since we've already set default values above
    
        # Convert to boolean
        gt_bool = (gt_edge > 0.5).astype(bool)
        gt_bool_bg = (gt_bg > 0.5).astype(bool)

        pred_bool = (reconstructed > args.threshold).astype(bool)
        pred_bool_bg = (reconstructed_bg > args.threshold).astype(bool)
        
        # Calculate metrics for EDGE class (foreground)
        print("Calculating metrics for EDGE class...")
        edge_corr, edge_comp, edge_qual, TP_edge, TP_p, FN, FP = corr_comp_qual(gt_bool, pred_bool, slack=0)
        edge_f1 = (2 * edge_corr * edge_comp) / (edge_corr + edge_comp) if (edge_corr + edge_comp) > 0 else 0

        # Calculate metrics for BACKGROUND class by inverting the masks
        print("Calculating metrics for BACKGROUND class...")

        # Calculate background metrics using the same function
        bg_corr, bg_comp, bg_qual, TP_bg, _, _, _ = corr_comp_qual(gt_bool_bg, pred_bool_bg, slack=0)
        bg_f1 = (2 * bg_corr * bg_comp) / (bg_corr + bg_comp) if (bg_corr + bg_comp) > 0 else 0

        # Calculate class distribution statistics
        total_pixels = gt_bool.size
        if args.input_mask_path and os.path.exists(args.input_mask_path):
            # Count only pixels in mask
            total_pixels = np.sum(mask)

        edge_pixels = np.sum(gt_bool)
        bg_pixels = total_pixels - edge_pixels
        edge_percent = (edge_pixels / total_pixels) * 100
        bg_percent = (bg_pixels / total_pixels) * 100

        # Calculate clDice for edge class (connectivity-preserving Dice)
        score_clDice = clDice(pred_bool, gt_bool)

        # Calculate balanced accuracy (average of both recalls)
        balanced_acc = (edge_comp + bg_comp) / 2

        # Print evaluation results for both classes
        print('Edge Prediction Evaluation:')
        print(f'Class distribution: Edge={edge_percent:.2f}%, Background={bg_percent:.2f}%')
        print('EDGE CLASS METRICS:')
        print(f'Precision: {edge_corr*100:.2f}%')
        print(f'Recall: {edge_comp*100:.2f}%')
        print(f'F1-score: {edge_f1*100:.2f}%')
        print(f'Quality: {edge_qual*100:.2f}%')
        print(f'clDice: {score_clDice*100:.2f}%')

        print('BACKGROUND CLASS METRICS:')
        print(f'Precision: {bg_corr*100:.2f}%')
        print(f'Recall: {bg_comp*100:.2f}%')
        print(f'F1-score: {bg_f1*100:.2f}%')
        print(f'Quality: {bg_qual*100:.2f}%')

        print('COMBINED METRICS:')
        print(f'Balanced Accuracy: {balanced_acc*100:.2f}%')

        # Save comprehensive evaluation results to a CSV file
        eval_results = {
            'Edge_Percentage': edge_percent,
            'Background_Percentage': bg_percent,
            'Edge_Precision': edge_corr*100,
            'Edge_Recall': edge_comp*100,
            'Edge_F1': edge_f1*100,
            'Edge_Quality': edge_qual*100,
            'Edge_clDice': score_clDice*100,
            'Background_Precision': bg_corr*100,
            'Background_Recall': bg_comp*100,
            'Background_F1': bg_f1*100,
            'Background_Quality': bg_qual*100,
            'Balanced_Accuracy': balanced_acc*100,
            'Masked': args.input_mask_path is not None
        }
        pd.DataFrame([eval_results]).to_csv(os.path.join(output_dir, 'edge_evaluation.csv'), index=False)

        # After calculating the metrics but before saving the reconstructed image:
        if args.gt_edge_path and os.path.exists(args.gt_edge_path):
            
            # Create visualization image for TP/FP/FN
            print("Generating visualization of True/False Positives/Negatives...")
            
            # Create a blank RGB visualization image
            h, w = gt_bool.shape
            visualization = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Assign colors to different categories:
            # - True Positives (TP): Green (correctly detected edges)
            # - False Positives (FP): Red (false alarms)
            # - False Negatives (FN): Blue (missed edges)
            # - True Negatives (TN): Black (correctly identified background)
            
            # True positives (where both GT and prediction are True) - Green
            true_positives = np.logical_and(gt_bool, pred_bool)
            visualization[true_positives] = [0, 255, 0]
            
            # False positives (where GT is False but prediction is True) - Red
            false_positives = np.logical_and(np.logical_not(gt_bool), pred_bool)
            visualization[false_positives] = [255, 0, 0]
            
            # False negatives (where GT is True but prediction is False) - Blue
            false_negatives = np.logical_and(gt_bool, np.logical_not(pred_bool))
            visualization[false_negatives] = [0, 0, 255]
            
            # Count pixels in each category
            tp_count = np.sum(true_positives)
            fp_count = np.sum(false_positives)
            fn_count = np.sum(false_negatives)
            tn_count = np.sum(np.logical_and(np.logical_not(gt_bool), np.logical_not(pred_bool)))
            
            # Add counts to visualization metadata
            error_counts = {
                'True_Positives': int(tp_count),
                'False_Positives': int(fp_count),
                'False_Negatives': int(fn_count),
                'True_Negatives': int(tn_count),
                'TP_Percentage': float(tp_count / total_pixels * 100),
                'FP_Percentage': float(fp_count / total_pixels * 100),
                'FN_Percentage': float(fn_count / total_pixels * 100),
                'TN_Percentage': float(tn_count / total_pixels * 100)
            }
            
            # Add these counts to the existing evaluation results
            eval_results.update(error_counts)
            
            # Save the visualization image
            vis_path = os.path.join(output_dir, f"{base_filename}_errors{'_masked' if mask_applied else ''}.png")
            cv2.imwrite(vis_path, visualization)
            
            print(f"Saved error visualization to {vis_path}")
            print(f"TP: {tp_count} ({tp_count/total_pixels*100:.2f}%) | "
                  f"FP: {fp_count} ({fp_count/total_pixels*100:.2f}%) | "
                  f"FN: {fn_count} ({fn_count/total_pixels*100:.2f}%)")
        
            # After the existing visualization code, add the following to create a background visualization:
            print("Generating background class visualization...")
    
            # Create a second RGB visualization image specifically for background
            bg_visualization = np.zeros((h, w, 3), dtype=np.uint8)
    
            # Identify the four categories
            true_positives = np.logical_and(gt_bool_bg, pred_bool_bg)                       # Correctly detected edges
            false_positives = np.logical_and(np.logical_not(gt_bool_bg), pred_bool_bg)      # False edges detected
            false_negatives = np.logical_and(gt_bool_bg, np.logical_not(pred_bool_bg))      # Missed edges
            true_negatives = np.logical_and(np.logical_not(gt_bool_bg), np.logical_not(pred_bool_bg))  # Correctly identified background
    
            # For background visualization:
            # - True Negatives (TN): Green (correctly identified background)
            # - False Positives (FP): Red (background incorrectly marked as edge)
            # - False Negatives (FN): Blue (edge incorrectly marked as background)
            # - True Positives (TP): Black (correctly identified edges)
            bg_visualization[true_positives] = [0, 255, 0]     # Green for TN (correct background)
            bg_visualization[false_positives] = [255, 0, 0]    # Red for FP (background incorrectly identified as edge)
            bg_visualization[false_negatives] = [0, 0, 255]    # Blue for FN (edge incorrectly identified as background)
    
            # Save the background visualization
            bg_vis_path = os.path.join(output_dir, f"{base_filename}_background{'_masked' if mask_applied else ''}.png")
            cv2.imwrite(bg_vis_path, bg_visualization)
    
            print(f"Saved background visualization to {bg_vis_path}")
            print(f"TN: {tn_count} ({tn_count/total_pixels*100:.2f}%) | "
                  f"FP: {fp_count} ({fp_count/total_pixels*100:.2f}%) | "
                  f"FN: {fn_count} ({fn_count/total_pixels*100:.2f}%)")

    # Apply inversion if needed
    if args.invert_label_map:
        reconstructed = 1/reconstructed

    # Save using GDAL to avoid memory issues
    print(f"Writing {'masked ' if mask_applied else ''}output to disk...")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tile_save_image_path, img_width, img_height, 1, gdal.GDT_Byte)

    if out_ds is None:
        raise ValueError(f"Cannot create output file {tile_save_image_path}")

    # Process in chunks to avoid memory issues
    chunk_size = 1000
    for y in range(0, img_height, chunk_size):
        y_end = min(y + chunk_size, img_height)
        chunk = reconstructed[y:y_end, :]
        # Convert to uint8
        chunk = (chunk * 255).astype(np.uint8)
        # Write to GDAL dataset
        out_ds.GetRasterBand(1).WriteArray(chunk, 0, y)

    # Close GDAL dataset to flush changes
    out_ds = None

    print(f"Saved {'masked ' if mask_applied else ''}reconstruction image to {tile_save_image_path}")

    # Clean up memory maps
    del reconstructed
    del patches_results
        
    print('Done')
    
    # Clean up all temporary directories created by tempfile.mkdtemp
    print("Cleaning up temporary files...")
    import shutil
    import glob
    
    # Find all temp directories created by this script
    temp_patterns = [
        "/tmp/patch_results_*",   # Patch results
        "/tmp/tmp*",              # Generic temp dirs
        "/tmp/reconstructed*"     # Reconstruction dirs
    ]
    
    cleaned_dirs = 0
    for pattern in temp_patterns:
        for temp_dir in glob.glob(pattern):
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    cleaned_dirs += 1
                except Exception as e:
                    print(f"Warning: Failed to clean up {temp_dir}: {e}")
    
    print(f"Cleaned up {cleaned_dirs} temporary directories")


def parse_args():
    parser = argparse.ArgumentParser('Test UNET')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--unseen', action='store_true', default=True,
                        help='Unseen dataset')

    parser.add_argument('-c', '--cuda', action='store_true', default=True,
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='models/base.pth',
                        help='the model to test')

    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')

    parser.add_argument('--input_map_path', type=str, default='dataset/TM/Test/TM25_sample.tif',
                        help='Input map image.')
    parser.add_argument('--input_mask_path', type=str, default=None,
                        help='Input map image.')
    
    parser.add_argument('--invert_label_map', action='store_true', default=False,
                        help='use negative pixels')
    parser.add_argument('--dilation', action='store_true', default=False,
                        help='dilate ground truth by one pixel')
    parser.add_argument('--threshold', action='store_true', default=0.5,
                        help='CC thresholding')
    
    parser.add_argument('--gt_edge_path', type=str, default=None,
                        help='Path to ground truth edge ZZmap for evaluation')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
