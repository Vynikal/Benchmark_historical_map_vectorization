import sys
import os
import gc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

# Import shapely -> geo tool
from shapely.ops import polygonize_full
from shapely.geometry import mapping
from skimage.measure import approximate_polygon
import fiona
from skimage import measure
from shapely.geometry import Polygon
import geopandas as gpd

# Import dataloader
from data.smart_data_loader import Data

# Import model
from model.unet import UNET
from model.hed import hed
from model.bdcn import bdcn
from model.segmenter.factory import create_segmenter
from model.pvt import pvt_2
from model.dws import watershed_net_combine
from model.mosin import mosin, VGGNet

# Import Utils
from utils.reconstruct_tiling_dict import reconstruct_from_patches

import pdb

# Add these imports at the top
from osgeo import gdal
import tempfile

# Add this import at the top of your file
from evaluation.all_eval.run_eval import evaluation
from evaluation.all_eval.pixel_eval.p_eval import corr_comp_qual, clDice
import cv2

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, win_size, args):
    input = args.input_map_path
    test_mask_path = args.input_mask_path
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
    
    for i, (images) in enumerate(tqdm(testloader, desc="Processing image tiles")):
        if args.cuda:
            images = images.to(args.device)

        with torch.no_grad():
            if args.model_type == 'mosin':
                init_labels = torch.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
                init_labels = init_labels.to(args.device)
                out = model(images, init_labels)
                fuse = out[0][0][-1].cpu().numpy()
            elif args.model_type == 'hed' or args.model_type == 'bdcn' or args.model_type == 'hed_pretrain' or args.model_type == 'bdcn_pretrain':
                out = model(images)
                fuse = torch.sigmoid(out[-1]).cpu().numpy()
            elif args.model_type == 'dws':
                out = model(images)
                out = torch.softmax(out, 1).squeeze()
                out = torch.argmax(out, 0)
                fuse = (out == 0).type(torch.uint8).cpu().numpy()
            else:
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


def meyer_watershed(image_path, dynamic, area, output_path, out_visu_path):
    print('watershed/histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))
    os.system('watershed/histmapseg/build/bin/histmapseg {} {} {} {} {}'.format(image_path, int(dynamic), int(area), output_path, out_visu_path))

# Replace the main() function with this memory-efficient version
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model as before
    if args.model_type == 'unet' or args.model_type == 'bal' or args.model_type == 'topo' or args.model_type == 'pathloss' or args.model_type == 'unet_bri' or args.model_type == 'unet_aff' or args.model_type == 'unet_hom' or args.model_type == 'unet_tps' or args.model_type == 'unet_bri_aff' or args.model_type == 'unet_bri_hom' or args.model_type == 'unet_bri_tps':
        model = UNET(n_channels=args.channels, n_classes=args.classes)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 512
    elif args.model_type == 'mini-unet':
        model = UNET(n_channels=args.channels, n_classes=args.classes, mode='mini')
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 512
    elif args.model_type == 'vit':
        def load_config():
            return yaml.load(open('../config/config.yml', 'r'), Loader=yaml.FullLoader)
        model_cfg = load_config()['net_kwargs']
        model_cfg['dropout'] = 0.0
        model = create_segmenter(model_cfg, mode='epm')
        pretrain_weight = torch.load('%s' % (args.model))
        model.load_state_dict(pretrain_weight)
        print('Load model {}'.format(args.model))
        win_size = 256
    elif args.model_type == 'mosin':
        model = UNET(n_channels=args.channels, n_classes=args.classes)
        vggnet = VGGNet(args.vgg, args.layers)
        model = mosin(model, vggnet, args)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
    elif args.model_type == 'pvt':
        model = pvt_2()
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 256
    elif args.model_type == 'hed' or args.model_type == 'hed_pretrain':
        model = hed(pretrain=None)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
    elif args.model_type == 'bdcn' or args.model_type == 'bdcn_pretrain':
        model = bdcn(pretrain=None)
        model.load_state_dict(torch.load('%s' % (args.model)))
        print('Load model {}'.format(args.model))
        win_size = 500
    elif args.model_type == 'dws':
        model = watershed_net_combine('Inference')
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

    # If you have ground truth, evaluate the edge prediction before saving
    if args.gt_edge_path and os.path.exists(args.gt_edge_path):
        print("Evaluating edge prediction against ground truth...")
        
        # Load ground truth edge map
        gt_edge = cv2.imread(args.gt_edge_path, cv2.IMREAD_GRAYSCALE)
        gt_edge = gt_edge / 255.0  # Normalize to 0-1
        
        # Resize ground truth if needed to match prediction
        if gt_edge.shape != reconstructed.shape[:2]:
            gt_edge = cv2.resize(gt_edge, (reconstructed.shape[1], reconstructed.shape[0]))
        
        # Convert to boolean
        gt_bool = (gt_edge > 0.5).astype(bool)
        pred_bool = (reconstructed > 0.5).astype(bool)
        
        # Calculate metrics with a tolerance of 8 pixels
        corr, comp, qual, TP_g, TP_p, FN, FP = corr_comp_qual(gt_bool, pred_bool, slack=8)
        
        # Calculate clDice (connectivity-preserving Dice)
        score_clDice = clDice(pred_bool, gt_bool)
        
        # Print evaluation results
        print('Edge Prediction Evaluation:')
        print(f'Precision: {corr*100:.2f}%')
        print(f'Recall: {comp*100:.2f}%')
        print(f'F1-score: {(2*corr*comp/(corr+comp))*100:.2f}%')
        print(f'Quality: {qual*100:.2f}%')
        print(f'clDice: {score_clDice*100:.2f}%')
        
        # Save evaluation results to a CSV file
        eval_results = {
            'Precision': corr*100,
            'Recall': comp*100,
            'F1-score': (2*corr*comp/(corr+comp))*100,
            'Quality': qual*100,
            'clDice': score_clDice*100
        }
        pd.DataFrame([eval_results]).to_csv(os.path.join(output_dir, 'edge_evaluation.csv'), index=False)

    # Apply inversion if needed
    if args.invert_label_map:
        reconstructed = 1/reconstructed

    # Create output file
    tile_save_image_path = os.path.join(output_dir, f"{str(args.model_type)}_test.tif")

    # Save using GDAL to avoid memory issues
    print("Writing final output to disk...")
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

    print(f'Saved reconstruction image to {tile_save_image_path}')

    # Clean up memory maps
    del reconstructed
    del patches_results
        
    # Continue with watershed as before
    # output_path = os.path.join(output_dir, 'label_map.tif')
    # meyer_watershed(tile_save_image_path, args.dynamic, args.area, output_path, './out.png')
    
    # Continue with vectorization if requested
    if args.vectorization:
        save_vector = os.path.join(output_dir, 'vector_output')
        if not os.path.exists(save_vector): 
            os.makedirs(save_vector)
        watershed_label_path = output_path
        sal_2_polygon(watershed_label_path, save_vector)

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

def di(lines, i):
    return lines[lines[..., 2] == i, :2]

def sal_2_polygon(img_path, save_vector):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    binary = img
    binary_image = ((binary == 0)).astype(np.uint8)
    contours = measure.find_contours(binary_image, 0.5)
    shapely_polygons = [Polygon(contour) for contour in contours]

    save_shapely_polygons = []
    for original_polygon in shapely_polygons:
        coordinates = list(original_polygon.exterior.coords)
        modified_coordinates = [(y, -x) for x, y in coordinates]
        modified_polygon = Polygon(modified_coordinates)
        save_shapely_polygons.append(modified_polygon)

    data = {'geometry': save_shapely_polygons}
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')  # 'EPSG:4326' is the coordinate reference system (CRS)
    output_shapefile = os.path.join(save_vector, 'output.shp')
    gdf.to_file(output_shapefile)

# def sal_2_polygon(img, vector_path, res_dir, dp_tol=2):
#     lines = np.load(vector_path)
#     schema = {
#         'geometry': 'Polygon',
#         'properties': {'id': 'int'},
#     }

#     save_vector_path = os.path.join(res_dir, 'shape_file')
#     if not os.path.exists(save_vector_path): 
#         os.makedirs(save_vector_path)

#     lineShp = fiona.open(os.path.join(save_vector_path, 'Polygon.shp'), mode='w', driver='ESRI Shapefile', schema = schema, crs = "EPSG:4326")
#     x, y = img.shape
#     all_lines = []

#     # Add border lines
#     left_top     = (0,  0)
#     left_bottom  = (0,  -x)
#     right_top    = (y, 0)
#     right_bottom = (y, -x)

#     for i in np.unique(lines[..., 2]):
#         d_plot = di(lines, i)
#         d_plot = approximate_polygon(d_plot, tolerance=dp_tol)
#         all_lines += [((int(d_plot[d][0]/2), -int(d_plot[d][1]/2)), (int(d_plot[d+1][0]/2), -int(d_plot[d+1][1]/2))) for d in range(0, len(d_plot)-1)]
    
#     # Add border lines
#     all_lines.append((left_top, left_bottom))
#     all_lines.append((left_bottom, right_bottom))
#     all_lines.append((right_bottom, right_top))
#     all_lines.append((right_top, left_top))
#     result, dangles, cuts, invalids = polygonize_full(all_lines)

#     print('Number of valid geometry: {}'.format(len(result.geoms)))
#     print('Number of dangles geometry: {}'.format(len(dangles.geoms)))
#     print('Number of cuts geometry: {}'.format(len(cuts.geoms)))
#     print('Number of invalids geometry: {}'.format(len(invalids.geoms)))
    
#     for index in range(0, len(result.geoms)):
#         lineShp.write({
#             'geometry': mapping(result.geoms[index]),
#             'properties': {'id': index},
#         })

#     lineShp.close()

def parse_args():
    parser = argparse.ArgumentParser('Test UNET')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--unseen', action='store_true', default=True,
                        help='Unseen dataset')
    parser.add_argument('--vectorization', action='store_true',
                        help='Vectorization the maps')

    parser.add_argument('-c', '--cuda', action='store_true', default=True,
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='../training_info/Vltava_SMO/unet/2025-02-07_11-22-04_lr_0.0001_train_unet_bs_4_aug_ctr+aff_hard_thindilate/params/topo_best_val_49.pth',#'../training_info/kameny/unet/2025-03-12_19-04-32_lr_0.0001_train_unet_bs_4_aug_ctr+aff/params/topo_best_val_49.pth',#
                        help='the model to test')

    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')

    parser.add_argument('-a', '--area', type=int, default=100,
                        help='Area of the meyer watershed.')
    parser.add_argument('-d', '--dynamic', type=int, default=7,
                        help='Dynamics of the meyer watershed.')

    parser.add_argument('--vgg', type=str, default='vgg19',
						help='pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', default=[4, 9, 18], type=int,
						help='the extracted features from vgg [4, 9, 27]')
    parser.add_argument('--K', type=int, default=3,
						help='number of iterative steps')
    parser.add_argument('--mu', type=float, default=10,
						help='loss coeff for vgg features')

    parser.add_argument('--input_map_path', type=str, default='dataset/Vltava_SMO/raster_hard_insane.jpg',#None,#
                        help='Input map image.')
    parser.add_argument('--input_mask_path', type=str, default=None,#'dataset/Vltava_SMO/mask_1970.tif',#
                        help='Input map image.')
    
    parser.add_argument('--invert_label_map', action='store_true', default=False,
                        help='use negative pixels')
    
    parser.add_argument('--gt_edge_path', type=str, default=None,
                        help='Path to ground truth edge ZZmap for evaluation')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
