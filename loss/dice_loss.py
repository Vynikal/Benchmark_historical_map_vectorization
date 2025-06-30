import torch
import numpy as np

def dice_loss(inputs, targets, weighted=True, smooth=1.0, sigmoid=True, balance=1.0):
    """
    Weighted Dice Loss for handling class imbalance in segmentation
    """
    if weighted:
        n, c, h, w = inputs.size()
        weights = np.zeros((n, c, h, w))
        for i in range(n):
            t = targets[i, :, :, :].cpu().data.numpy()
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            valid = neg + pos
            weights[i, t == 1] = neg * 1. / valid
            weights[i, t == 0] = pos * balance / valid
        weights = torch.Tensor(weights)
        if inputs.is_cuda:
            weights = weights.cuda()

    # Apply sigmoid if needed (for raw network outputs)
    if sigmoid:
        inputs = torch.sigmoid(inputs)
    
    # Calculate per-sample Dice scores
    n = inputs.size(0)
    inputs_flat = inputs.view(n, -1)
    targets_flat = targets.view(n, -1)
    
    # Calculate intersection and union for each sample
    intersection = (inputs_flat * targets_flat).sum(1)
    total = inputs_flat.sum(1) + targets_flat.sum(1)
    
    # Calculate Dice coefficient for each sample
    dice = (2. * intersection + smooth) / (total + smooth)
    
    # Apply weights if provided - but we need to adapt weights first
    if weighted:
        # Calculate per-sample weight average instead of pixel-wise multiplication
        weights_flat = weights.view(n, -1)
        # Get the average weight per sample
        weights_avg = weights_flat.mean(dim=1)
        # Apply weights to dice scores
        dice = dice * weights_avg
    
    # Return mean Dice loss
    return 1 - dice.mean()
