from easydict import EasyDict as edict
import cv2
import torch
import einx
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as VF
from torchvision.models import ConvNeXt_Tiny_Weights


# TODO these stats were computed for d_min = 0.01 and d_max = 1000.0
# if the range changes, these need to be recomputed for the new range
# fix this so that it can be normalized for range shifts (they were computed from the formula in loss.py)
# The stats from CO3D were estimated from the processed dataset, so they are different from the ones from the full raw dataset
def get_input_norm_stats(input_type, is_diff=False):
    if input_type == 'image':
        # imagenet_1k
        convnext_transforms = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        m, s = convnext_transforms.mean, convnext_transforms.std
    elif input_type == 'log_depth':
        # co3d log_depth
        m, s = [-0.2749], [0.9187]
    elif input_type == 'depth_mask':
        # co3d depth_mask
        m, s = [0.8591], [0.3479]
    else:
        assert False, f'Invalid input type "{input_type}"'
    
    if is_diff:
        m, s = [0.0] * len(m), [i * (2 ** 0.5) for i in s]
    
    return m, s


def create_input_normalizer(input_type, is_diff=False):
    m, s = get_input_norm_stats(input_type, is_diff)
    
    return T.Normalize(mean=m, std=s)


def normalize_depths(depths, log_min, log_max):
    eps = 1e-8
    depths = (depths + eps).log()
    depths = (depths - log_min) / (log_max - log_min)
    
    return depths


def denormalize_depths(depths, log_min, log_max):
    depths = (log_max - log_min) * depths + log_min
    depths = depths.exp()
    
    return depths


def normalize_depths2(depths, min, max):
    # TODO check if its more precise
    # depths = depths - min
    # d_range = torch.tensor(max - min)
    # depths = ((torch.e - 1.0) * depths + d_range).log() - d_range.log()
    
    depths = (depths - min) / (max - min)
    depths = ((torch.e - 1.0) * depths).log1p()
    
    return depths


def denormalize_depths2(depths, min, max):
    # TODO check if its more precise
    # d_range = torch.tensor(max - min)
    # depths = ((depths + d_range.log()).exp() - d_range) / (torch.e - 1.0)
    # depths = depths + min
    
    depths = depths.expm1() / (torch.e - 1.0)
    depths = (max - min) * depths + min
    
    return depths


def get_image(path, is_depth, output_dims):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if is_depth else cv2.imread(path)
    
    original_dim = (img.shape[0], img.shape[1])
    
    ar = img.shape[0] / img.shape[1]
    out_ar = output_dims[0] / output_dims[1]
    if out_ar > ar:
        new_dim = output_dims[0] / ar
        new_dim = max(output_dims[1], round(new_dim))
        new_shape = (output_dims[0], new_dim)
    else:
        new_dim = output_dims[1] * ar
        new_dim = max(output_dims[0], round(new_dim))
        new_shape = (new_dim, output_dims[1])
    
    # CV2 resize shape is (w, h)
    new_shape = new_shape[::-1]
    if is_depth:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
        img = img[:, :, None]
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    resize_ratio = (img.shape[0] / original_dim[0], img.shape[1] / original_dim[1])
    before_crop_dim = img.shape[:2]
    
    img = torch.from_numpy(img)
    img = einx.id('h w c -> c h w', img)
    img = VF.center_crop(img, output_size=output_dims)
    new_center_displacement = (img.shape[-2] / 2 - before_crop_dim[-2] / 2, img.shape[-1] / 2 - before_crop_dim[-1] / 2)
    
    return img, resize_ratio, original_dim, new_center_displacement


def process_data(c2ws, K, image_paths, depth_paths, output_dims):
    (images, resize_ratios, image_dims, center_displacements), (depths, _, depth_dims, _) = [
        [list(i) for i in zip(*(get_image(path, is_depth, output_dims) for path in paths))]
        for paths, is_depth in ((image_paths, False), (depth_paths, True))
    ]
    assert image_dims == depth_dims, f'Inconsistency between image sizes and depth sizes in dataset in scene "{image_paths[0]}"'
    
    images, depths = [torch.stack(i) for i in (images, depths)]
    resize_ratios, center_displacements = [torch.stack([torch.tensor(i) for i in p]) for p in (resize_ratios, center_displacements)]
    
    images = images / 255.0 # convert from uint8 to 0.0-1.0
    depths = depths.int()
    depth_masks = depths > 0
    depths = depths / 1000.0 # convert from mm to m
    
    R, t = c2ws[..., :3, :3], c2ws[..., :3, 3]
    
    # Batching and normalizing intrinsic matrices
    if len(K.shape) == 2:
        K = einx.id('m n -> b m n', K, b=len(images)).clone()
    K[:, 0, :] = resize_ratios[:, 1, None] * K[:, 0, :]
    K[:, 1, :] = resize_ratios[:, 0, None] * K[:, 1, :]
    K[:, 0, 2] = K[:, 0, 2] + center_displacements[:, 1]
    K[:, 1, 2] = K[:, 1, 2] + center_displacements[:, 0]
    
    views = edict(
        K=K,
        R=R,
        t=t,
        images=images,
        depths=depths,
        depth_masks=depth_masks
    )
    return views


def random_rotation_matrix(device=None, dtype=torch.float32):
    # Uniform random numbers in [0, 1)
    u1, u2, u3 = torch.rand(3, device=device, dtype=dtype)

    # Random unit quaternion (x, y, z, w)
    q = torch.stack([
        torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2),
        torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * torch.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * torch.pi * u3),
    ])

    x, y, z, w = q

    R = torch.tensor([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z),     2 * (x*z + w*y)],
        [2 * (x*y + w*z),     1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y),     2 * (y*z + w*x),     1 - 2 * (x*x + y*y)],
    ], device=device, dtype=dtype)

    return R
