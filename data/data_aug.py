import numpy as np
import random
import cv2
import math
import itertools


def transformation(img, targets, mode):
    if mode == 'ctr' or mode == 'ctr+aff' or mode == 'ctr+hom' or mode == 'ctr+tps':
        img = random_contrast(img)

    if mode == 'ctr+aff' or mode == 'aff':
        img, targets = random_affine(img, targets)
    elif mode == 'ctr+hom' or mode == 'hom':
        img, targets = random_homography(img, targets)
    elif mode == 'ctr+tps' or mode == 'tps':
        img, targets = random_tps(img, targets)
    else:
        assert mode == 'ctr' or mode == 'aff' or mode == 'hom' or mode == 'tps' or mode == 'ctr+aff' or mode == 'ctr+hom' or mode == 'ctr+tps'
    return img, targets

def random_contrast(img, low=0.8, high=1.25, beta=0):
    """Bit-depth agnostic contrast augmentation"""
    alpha = np.random.uniform(low, high)
    
    # Preserve original data type
    original_dtype = img.dtype
    
    # Convert to float for processing to avoid overflow/clipping
    if img.dtype in [np.uint8, np.uint16]:
        # For integer types, normalize to [0,1] first
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img_float = img.astype(np.float32) / 65535.0
        else:
            img_float = img.astype(np.float32)
            
        # Apply contrast
        img_float = np.clip(alpha * img_float + beta, 0.0, 1.0)
        
        # Convert back to original range
        if original_dtype == np.uint8:
            img = (img_float * 255.0).astype(np.uint8)
        elif original_dtype == np.uint16:
            img = (img_float * 65535.0).astype(np.uint16)
    else:
        # For float types, apply directly
        img = np.clip(alpha * img.astype(np.float32) + beta, 0.0, 1.0).astype(original_dtype)
    
    return img


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    if targets is None:
        targets = []
    
    # Store original data type
    original_dtype = img.dtype
    
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Combined rotation matrix
    M = S @ T @ R
    if (border != 0) or (M != np.eye(3)).any():
        # **FIX: Preserve data type with explicit dtype parameter**
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), 
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        # Ensure output matches input type
        img = img.astype(original_dtype)
        
        if len(targets) > 0:
            targets = cv2.warpAffine(targets, M[:2], dsize=(width, height), 
                                   flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    
    return img, targets


def random_homography(img, targets, random_t_tps=0.5):
    # Store original data type
    original_dtype = img.dtype
    
    x_max, y_max = img.shape[0: 2]
    X = np.array([0, 0, x_max, 0, y_max, x_max, 0, y_max])
    Y = X + (np.random.rand(8)-0.5) * x_max * random_t_tps

    X = X.reshape((4, 2))
    Y = Y.reshape((4, 2))

    # Linear direct transform
    A = np.zeros((8, 9))
    for i in range(4):
        x, y, x_, y_ = X[i][0], X[i][1], Y[i][0], Y[i][1]
        A[2 * i ] = np.stack([-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_])
        A[2 * i + 1] = np.stack([0, 0, 0, -x, -y, -1, x*y_, y*y_, y_])

    A = np.matrix(A)
    u, s, v = np.linalg.svd(A)
    H21 = np.reshape(v[8], (3, 3))
    H21 = (1/H21[2,2]) * H21

    if len(img.shape) == 3:
        x, y, _ = img.shape
    else:
        x, y = img.shape

    # **FIX: Preserve data type**
    img = cv2.warpPerspective(img, H21, (x, y))
    img = img.astype(original_dtype)
    
    if len(targets) > 0:
        targets = cv2.warpPerspective(targets, H21, (x, y))
    
    return img, targets


def random_tps(img, targets):
    # creat control points
    c_dst = np.array(list(itertools.product(
        np.arange(-1, 1.00001, 2.0 / 4),
        np.arange(-1, 1.00001, 2.0 / 4),
    )))

    # low to high from -0.05 to 0.05 without creating much deformation which does not exist in maps
    c_src = c_dst + np.random.uniform(low=-0.05, high=0.05, size=c_dst.shape)
    img = warp_image_cv(img, c_src, c_dst, dshape=img.shape)
    targets = warp_image_cv(targets, c_src, c_dst, dshape=img.shape)
    return img, targets


class TPS:
    # Source: https://github.com/cheind/py-thin-plate-spline
    @staticmethod
    def fit(c, lambd=0., reduced=False):        
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b

def uniform_grid(shape):
    '''Uniform grid coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid
    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H,W = shape[:2]    
    c = np.empty((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c

def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))

    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)

def tps_grid(theta, c_dst, dshape):    
    ugrid = uniform_grid(dshape)

    reduced = c_dst.shape[0] + 2 == theta.shape[0]

    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid
    
    return grid # H'xW'x2 grid[i,j] in range [0..1]

def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    
    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.
    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my

def warp_image_cv(img, c_src, c_dst, dshape=None):
    # Store original data type
    original_dtype = img.dtype
    
    dshape = dshape or img.shape
    theta = tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps_grid_to_remap(grid, img.shape)
    
    # **FIX: Preserve data type**
    result = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
    return result.astype(original_dtype)
