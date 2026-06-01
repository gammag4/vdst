from easydict import EasyDict as edict
import cv2
import torch
import einx
import torchvision.transforms.functional as VF


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
    new_center_displacement = (img.shape[0] / 2 - before_crop_dim[0] / 2, img.shape[1] / 2 - before_crop_dim[1] / 2)
    
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
