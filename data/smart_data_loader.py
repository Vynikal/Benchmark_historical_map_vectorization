import numpy as np
import torch
from torch.utils import data
from skimage.measure import label
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt
import cv2

from data.data_aug import transformation
from data.create_tilling import generate_tiling, generate_tiling_gdal


class Data(data.Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, data_aug=None, aug_mode=None, dilation=False, mode=None, unseen=False, mask_path=None):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size = w_size
        self.dilation = dilation
        self.data_aug = data_aug
        self.aug_mode = aug_mode
        self.mode = mode
        self.unseen = unseen
        
        # Use GDAL-based disk tiling for memory efficiency
        try:
            self.disk_dataset = generate_tiling_gdal(self.image_path, w_size=self.w_size, mask_path=mask_path)
            self.patch_pos = self.disk_dataset.get_patch_positions()
            if not self.unseen:
                self.gt_path = generate_tiling_gdal(self.gt_path, w_size=self.w_size, mask_path=mask_path)
        except (ImportError, ModuleNotFoundError):
            print("GDAL not available, falling back to standard tiling (may use more memory)")
            self.gt_path, self.patch_pos = generate_tiling(self.image_path, w_size=self.w_size, mask_path=mask_path)
            self.gt_path = np.array(self.gt_path)
            self.disk_dataset = None
            if not self.unseen:
                self.gt_path = generate_tiling(self.gt_path, w_size=self.w_size, mask_path=mask_path)
            
        print(f"Window_size: {w_size}, Generated {len(self)} image patches")
        
        if not self.unseen and large_gt_path:
            # Handle ground truth if needed
            # [your existing code for ground truth]
            pass

    def __len__(self):
        if hasattr(self, 'disk_dataset') and self.disk_dataset is not None:
            return len(self.disk_dataset)
        else:
            return len(self.image_path)

    def __getitem__(self, index):
        if hasattr(self, 'disk_dataset') and self.disk_dataset is not None:
            img = self.disk_dataset[index]
        else:
            img = self.image_path[index]
            
        if self.unseen:
            img = img / 255.
            img = np.array(img, dtype=np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()
            return img

        labels = self.gt_path[index].squeeze()
        labels = labels/255.
        labels = labels.astype(np.uint8)

        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        if self.data_aug:
            img, labels = transformation(img, labels, self.aug_mode)
            
        img = img / 255.
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        
        if self.mode == 'loss':
            seeds = get_seed(labels)
            labels = torch.from_numpy(np.array([labels])).float()
            labels = {'labels': labels, 'seeds': seeds}
            return img, labels
        elif self.mode == 'direction':
            dist = ndimage.distance_transform_edt(1-labels)
            x_grad = cv2.Sobel(dist, cv2.CV_64F, 1, 0) / 8 # Order of derivative X
            y_grad = cv2.Sobel(dist, cv2.CV_64F, 0, 1) / 8 # Order of derivative Y
            grad_img = np.stack([x_grad, y_grad], 2)
            grad_img = np.transpose(grad_img, (2, 0, 1))
            labels = {'grad_img': grad_img, 'distance_map': dist, 'labels': labels}
            return img, labels
        elif self.mode == 'learned_watershed':
            dist = ndimage.distance_transform_edt(1-labels)
            # Transfer gt into discrete level
            depth_bin = 16
            for i in range(0, len(depth_bin)-1):
                dist[np.bitwise_and(dist > depth_bin[i], dist <= depth_bin[i+1])] = i
            labels = {'distance_map': dist, 'labels': labels}
            return img, labels
        else:
            labels = torch.from_numpy(np.array([labels])).float()
            return img, labels
        
    def get_patch_positions(self):
        return self.patch_pos

def get_seed(labels):
    labels = np.ascontiguousarray(labels)
    label_invert = 1-labels

    cc_epm = label(label_invert, connectivity=1)
    dm_label = distance_transform_edt(label_invert)
    seed_image = np.zeros(labels.shape).astype(np.int16)
    for index, c in enumerate(np.unique(cc_epm)):
        if index == 0:
            continue
        cc_binary = (cc_epm == c).astype(np.uint8)
        dm_cc = cc_binary * dm_label
        x, y = np.nonzero(dm_cc)
        middle_index = int(len(x) / 2)
        seed_coordinate = (x[middle_index], y[middle_index])
        seed_image[seed_coordinate] = index
    return seed_image
