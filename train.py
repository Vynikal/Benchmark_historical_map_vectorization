import os

from pathlib import Path
import numpy as np
import torch
from torch import optim
import argparse
import datetime
from tqdm.auto import tqdm
import glob
import cv2

# Import dataloader
from data.smart_data_loader import Data
from data import log
from data.reconstruct_tiling_dict import reconstruct_from_patches, save_random_chips

# Import model
from model.unet import UNET as unet

# Import loss function
from loss.bce_loss import cross_entropy_loss2d_sigmoid
from loss.dice_loss import dice_loss as dice
from loss.MBD_BAL.BALoss import boundary_awareness_loss


def load_datasets_from_directory(data_dir, w_size, data_aug=None, aug_mode=None, dilation=False, mode='loss'):
    """Load all .tif images from a directory and create datasets"""
    tif_files = sorted([f for f in glob.glob(os.path.join(data_dir, "*.tif")) 
                       if not f.endswith("_GT.tif") and not f.endswith("_mask.tif")])
    
    if not tif_files:
        raise ValueError(f"No images found in {data_dir}")
    
    datasets = []
    img_paths = []
    img_positions = []
    
    for img_path in tif_files:
        base_name = os.path.splitext(img_path)[0]
        gt_path = f"{base_name}_GT.tif"
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT file not found for {img_path}, skipping...")
            continue
        
        mask_path = f"{base_name}_mask.tif" if os.path.exists(f"{base_name}_mask.tif") else None
        
        print(f"Loading: {os.path.basename(img_path)}")
        dataset = Data(img_path, gt_path, w_size, data_aug, aug_mode=aug_mode, 
                      dilation=dilation, mode=mode, mask_path=mask_path)
        datasets.append(dataset)
        img_paths.append(img_path)
        img_positions.append(dataset.get_patch_positions())
    
    if not datasets:
        raise ValueError(f"No valid image-GT pairs found in {data_dir}")
    
    return datasets, img_paths, img_positions


def compute_losses(out, labels, seeds, args):
    """Compute all losses (BCE, Dice, Topo)"""
    device = out.device
    
    bce_loss = cross_entropy_loss2d_sigmoid(out, labels) if args.main_loss_type in ['bce', 'both'] else torch.tensor(0.0, device=device)
    dice_loss = dice(out, labels) if args.main_loss_type in ['dice', 'both'] else torch.tensor(0.0, device=device)
    
    topo_loss = torch.tensor(0.0, device=device)
    if args.topo_loss_type == 'baloss':
        batch_size = out.shape[0]
        for b in range(batch_size):
            topo_loss += args.alpha * boundary_awareness_loss(
                out[b].unsqueeze(0), seeds[b], labels[b].unsqueeze(0)
            )
    
    return bce_loss, dice_loss, topo_loss


def train(args):
    # Initialize the model
    if args.model_type == 'unet':
        model = unet(n_channels=args.channels, n_classes=args.classes)
    w_size = args.w_size
    if args.topo_loss_type == 'baloss':
        model.load_state_dict(torch.load(args.pretrain))
        print('Load pretrain: {}'.format(args.pretrain))

    if not(args.topo_loss_type):
        args.topo_loss_type = 'no_topo'

    print('Training with model: {}'.format(args.model_type))
    print('Training with losses: {}, {}'.format(args.main_loss_type, args.topo_loss_type))

    aug_mode = args.data_aug_mode
    if args.data_aug:
        data_aug_stat = 'aug_' + aug_mode
    else:
        data_aug_stat = 'no_aug'

    # Load training data from directory
    print(f"Loading training data from directory: {args.train_data_dir}")
    train_datasets, _, _ = load_datasets_from_directory(
        args.train_data_dir, w_size, args.data_aug, aug_mode, args.dilation, 'loss'
    )
    
    # Combine all training datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    n_train = len(trainloader)

    print(f"Loading validation data from directory: {args.val_data_dir}")
    val_datasets, val_img_paths, val_img_positions = load_datasets_from_directory(
        args.val_data_dir, w_size, data_aug=None, dilation=args.dilation, mode='loss'
    )
    
    # Combine all validation datasets
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    n_val = len(valloader)


    # Change it to adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5)

    if args.cuda:
        model.cuda()

    if args.resume:
        model_pretrain = torch.load(args.resume)
        model.load_state_dict(model_pretrain)
        print('Resume pretrain {}'.format(args.resume))

        res_dir = Path(args.resume).parent.parent
        logger = log.get_logger(os.path.join(res_dir, '{}.txt'.format(args.model_type)), mode='a')
        start_epoch = int(args.resume.split('_')[-1].split('.')[0]) + 1
        recon_save_path = os.path.join(res_dir, 'reconstruction_png')
        parm_save_path = os.path.join(res_dir, 'params')
    else:
        model_name = args.model_type
        loss_type = 'train_{}'.format(model_name) + '_bs_'+ str(args.batch_size) + '_{}_'.format(args.main_loss_type)

        # Create res directory
        res_dir = os.path.join(args.res_dir + args.dataset, model_name, str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0] + '_lr_' + str(args.base_lr)) + '_' + loss_type + '_' + data_aug_stat
        print('Model save in {}'.format(res_dir))

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        recon_save_path = os.path.join(res_dir, 'reconstruction_png')
        if not os.path.exists(recon_save_path):
            os.makedirs(recon_save_path)

        # Create params folder
        parm_save_path = os.path.join(res_dir, 'params')
        if not os.path.exists(parm_save_path):
            os.makedirs(parm_save_path)

        # Create Logger 
        logger = log.get_logger(os.path.join(res_dir, '{}.txt'.format(args.model_type)))

        start_epoch = 0

    # Create chips directory
    chips_dir = os.path.join(res_dir, 'random_chips')
    if not os.path.exists(chips_dir):
        os.makedirs(chips_dir)

    # Save random chips from datasets
    if train_datasets:
        save_random_chips(train_datasets[0], chips_dir, 'train', num_chips=50)
    
    if val_datasets:
        save_random_chips(val_datasets[0], chips_dir, 'val', num_chips=20)


    epochs = args.epochs
    for epoch in range(start_epoch, start_epoch+epochs):
        model.train()
        
        # Preallocate arrays for training losses
        train_losses = np.zeros((n_train, 4))  # [total, bce, dice, topo]
        
        with tqdm(total=len(trainloader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (img, labels) in enumerate(trainloader):
                labels, seeds = labels['labels'], labels['seeds']
                optimizer.zero_grad()

                if args.cuda:
                    img, labels, seeds  = img.cuda(), labels.cuda(), seeds.cuda()

                out = model(img)
                bce_loss, dice_loss, topo_loss = compute_losses(out, labels, seeds, args)
                total_loss = bce_loss + dice_loss + topo_loss

                # Back calculating loss
                total_loss.backward()

                # update parameter, gradient descent, back propagation
                optimizer.step()

                train_losses[i] = [total_loss.item(), bce_loss.item(), dice_loss.item(), topo_loss.item()]

                # Update the pbar
                pbar.update(labels.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'total_loss': total_loss.item(), 'bce_loss': bce_loss.item(), 'dice_loss': dice_loss.item(), 'topo_loss': topo_loss.item()})
            
            train_mean_loss = train_losses[:, 0].mean()
            train_bce_loss = train_losses[:, 1].mean()
            train_dice_loss = train_losses[:, 2].mean()
            train_topo_loss = train_losses[:, 3].mean()

        model.eval()
        
        # Preallocate arrays for validation losses and patches
        val_losses = np.zeros((n_val, 4))  # [total, bce, dice, topo]
        patches_images_ws = np.zeros((len(valloader.dataset), w_size, w_size))
        global_idx = 0
        
        with tqdm(total=len(valloader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, (val_img, val_labels) in enumerate(valloader):
                val_labels, val_seeds = val_labels['labels'], val_labels['seeds']

                if args.cuda:
                    val_img, val_labels, val_seeds  = val_img.cuda(), val_labels.cuda(), val_seeds.cuda()

                with torch.no_grad():
                    val_out = model(val_img)
                    val_bce_loss, val_dice_loss, val_topo_loss = compute_losses(val_out, val_labels, val_seeds, args)

                val_out = torch.sigmoid(val_out)
                batch, _, _, _ = val_out.shape
                for b in range(batch):
                    fuse_ws = val_out[b, ...].cpu().numpy()[0,...]
                    patches_images_ws[global_idx] = fuse_ws
                    global_idx += 1

                val_total_loss = val_bce_loss + val_dice_loss + val_topo_loss

                val_losses[i] = [val_total_loss.item(), val_bce_loss.item(), val_dice_loss.item(), val_topo_loss.item()]

                # Update the pbar
                pbar.update(val_img.shape[0])

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'val_total_loss': val_total_loss.item(), 'val_bce_loss': val_bce_loss.item(), 'val_topo_loss': val_topo_loss.item(), 'val_dice_loss': val_dice_loss.item()})
            
            val_mean_loss = val_losses[:, 0].mean()
            val_bce_loss = val_losses[:, 1].mean()
            val_dice_loss = val_losses[:, 2].mean()
            val_topo_loss = val_losses[:, 3].mean()
        logger.info('lr: %e, train_total_loss: %f, train_bce_loss: %f, train_dice_loss: %f, train_topo_loss: %f, val_total_loss: %f, val_bce_loss: %f, val_dice_loss: %f, val_topo_loss: %f' %
                    (optimizer.param_groups[0]['lr'], train_mean_loss, train_bce_loss, train_dice_loss, train_topo_loss,
                     val_mean_loss, val_bce_loss, val_dice_loss, val_topo_loss))

        # Reconstruct all validation images
        pad_px = w_size // 2
        patch_idx = 0
        
        for img_idx, (val_img_path, val_pos) in enumerate(zip(val_img_paths, val_img_positions)):
            # Get number of patches for this image
            num_patches = len(val_pos)
            
            # Extract patches for this specific image
            img_patches = patches_images_ws[patch_idx:patch_idx + num_patches]
            
            # Read original image to get dimensions
            in_img = cv2.imread(val_img_path)
            
            # Reconstruct image from patches
            new_img = reconstruct_from_patches(img_patches, w_size, pad_px, in_img.shape, np.float32, val_pos)
            
            # Save reconstructed image
            img_basename = os.path.splitext(os.path.basename(val_img_path))[0]
            tile_save_image_path = os.path.join(recon_save_path, 
                f'{epoch}_{img_basename}_reconstruct.png')
            
            new_img = (new_img * 255).astype(np.uint8)
            cv2.imwrite(tile_save_image_path, new_img)
            
            # Move to next image's patches
            patch_idx += num_patches
        
        # Save model checkpoint
        torch.save(model.state_dict(), '{}/epoch_{}.pth'.format(parm_save_path, epoch))

        # Learning rate schedular to change learning
        scheduler.step(val_mean_loss)
        print('Current learning rate             {}'.format(optimizer.param_groups[0]['lr']))


def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)

    train(args)

def parse_args():
    def path_exists(p):
        if(os.path.exists(p)):
            return p
        else:
            return None

    parser = argparse.ArgumentParser(
        description='Train leakage-loss for different args')
    parser.add_argument('-p', '--pretrain', type=path_exists, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='The type of the model')
    parser.add_argument('--main_loss_type', type=str, default='bce',
                        help='The type of the model')
    parser.add_argument('--topo_loss_type', type=str, default=None,
                        help='The type of the model')
    parser.add_argument('--alpha', type=float, default=100,
                        help='the coefficient for topo loss')
    parser.add_argument('-d', '--dataset', type=str,
                        default='BOTH', help='The dataset to train (only the name)')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-4,
                        help='the base learning rate of model')
    parser.add_argument('-c', '--cuda', action='store_true', default=True,
                        help='whether use gpu to train network')
    parser.add_argument('--data_aug', action='store_true', default=True,
                        help='Augmentation the data or not')
    parser.add_argument('--data_aug_mode', type=str, default='ctr+aff',
                        help='Augmentation mode')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='whether resume from some, default is None')
    parser.add_argument('--dilation', type=int, default=False,
                        help='Dilate the ground truth by 1px')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')
    parser.add_argument('--res_dir', type=str, default='training_info/',
                        help='the dir to store result')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch to train network, default is 100')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size of one iteration, default 4')
    parser.add_argument('--w_size', type=int, default=256,
                        help='Patch size for training')
    parser.add_argument('--train_data_dir', type=str, default='dataset/TM/Train',
                        help='Directory containing training .tif files and corresponding _GT.tif files')
    parser.add_argument('--val_data_dir', type=str, default='dataset/TM/Val',
                        help='Directory containing validation .tif files and corresponding _GT.tif files')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()

