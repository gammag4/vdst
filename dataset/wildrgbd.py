import os
import json
import random
import cv2
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
import torchvision.transforms.functional as VF
from torchvision.transforms import InterpolationMode
import numpy as np
import einx


class WildRGBDDataset(Dataset):
    def __init__(self, path, n_sources, n_targets, output_dims, use_constrained_views, val_split, test_split, split='train', test_category='truck', seed=42):
        self.path = path
        self.n_sources = n_sources
        self.n_targets = n_targets
        self.output_dims = output_dims
        self.use_constrained_views = use_constrained_views
        self.current_epoch = 0
        
        if use_constrained_views:
            assert self.n_sources == 2, f'use_constrained_views cannot be used with only 2 sources per scene, but {self.n_sources} were requested'
        
        self.seed = seed
        self.random = random.Random(seed)
        
        assert split in ['train', 'val', 'test', 'test_new_category'], f'Invalid dataset split "{split}"'
        
        if split == 'test_new_category':
            cpaths = [(test_category, os.path.join(path, test_category))]
        else:
            cpaths = [(c, os.path.join(path, c)) for c in os.listdir(path) if c != test_category]
            cpaths = [(c, cpath) for c, cpath in cpaths if os.path.isdir(cpath)]
        
        cpaths.sort()
        # cpaths = [i for i in cpaths if i[0] != 'pineapple'] # TODO
        
        spaths = []
        for c, cpath in cpaths:
            scenes = os.listdir(os.path.join(cpath, 'scenes'))
            scenes.sort()
            self.random.shuffle(scenes)
            
            s_val_split = max(1, round(len(scenes) * val_split))
            s_test_split = max(1, round(len(scenes) * test_split))
            
            if split == 'train':
                scenes = scenes[s_test_split + s_val_split:]
            elif split == 'val':
                scenes = scenes[s_test_split:s_test_split + s_val_split]
            elif split == 'test':
                scenes = scenes[:s_test_split]
            
            spaths.extend([(f'{c}_{s}', os.path.join(cpath, 'scenes', s)) for s in scenes])
        
        spaths = [(sname, spath) for sname, spath in spaths if os.path.isdir(spath)]
        self.random.shuffle(spaths)
        
        spaths = [
            (
                f'{sname}',
                spath,
                # Only uses cone 0 for val and test
                [os.path.join(spath, 'cones', c) for c in sorted(os.listdir(os.path.join(spath, 'cones')) if split == 'train' else ['0']) if os.path.isdir(os.path.join(spath, 'cones', c))]
            )
            for (sname, spath) in spaths
        ]
        
        self.spaths = spaths
    
    def __len__(self):
        return len(self.spaths)
    
    def get_image(self, path, is_depth):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if is_depth else cv2.imread(path)
        
        original_dim = (img.shape[0], img.shape[1])
        
        ar = img.shape[0] / img.shape[1]
        out_ar = self.output_dims[0] / self.output_dims[1]
        if out_ar > ar:
            new_dim = self.output_dims[0] / ar
            new_dim = max(self.output_dims[1], round(new_dim))
            new_shape = (self.output_dims[0], new_dim)
        else:
            new_dim = self.output_dims[1] * ar
            new_dim = max(self.output_dims[0], round(new_dim))
            new_shape = (new_dim, self.output_dims[1])
        
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
        img = VF.center_crop(img, output_size=self.output_dims)
        new_center_displacement = (img.shape[0] / 2 - before_crop_dim[0] / 2, img.shape[1] / 2 - before_crop_dim[1] / 2)
        
        return img, resize_ratio, original_dim, new_center_displacement
    
    def get_image_paths(self, cpath, type):
        return [os.path.join(cpath, type, p) for p in os.listdir(os.path.join(cpath, type))]
    
    def get_cam_pose_strs(self, cpath):
        with open(os.path.join(cpath, 'cam_poses.txt'), 'r', encoding='utf8') as f:
            return f.read().strip().split('\n')
    
    def __getitem__(self, i):
        self.random.seed(self.seed + i + len(self) * self.current_epoch)
        sname, spath, cpaths = self.spaths[i]
        
        with open(os.path.join(spath, 'metadata'), 'r', encoding='utf8') as f:
            data = edict(json.load(f))
            K = torch.tensor(data.K).reshape(3, 3).T
        
        if self.use_constrained_views:
            cpath = self.random.choice(cpaths)
            
            images, depths = [sorted(self.get_image_paths(cpath, t)) for t in ('rgb', 'depth')]
            pose_strs = self.get_cam_pose_strs(cpath)
            
            s_pose_strs = sorted(self.get_cam_pose_strs(cpath), key=lambda l: int(l.split(' ')[0]))
            source_indices = sorted([s_pose_strs.index(pose_strs[0]), s_pose_strs.index(pose_strs[-1])], reverse=True)
            pose_strs = s_pose_strs
            
            images2, depths2, pose_strs2 = [[p[i] for i in source_indices] for p in (images, depths, pose_strs)]
            for p in images, depths, pose_strs:
                for i in source_indices:
                    p.pop(i)
            
            images2, depths2, pose_strs2 = [list(i) for i in zip(*self.random.sample(list(zip(images2, depths2, pose_strs2)), self.n_sources))]
            images, depths, pose_strs = [list(i) for i in zip(*self.random.sample(list(zip(images, depths, pose_strs)), self.n_targets))]
            images, depths, pose_strs = [p2 + p1 for p1, p2 in ((images, images2), (depths, depths2), (pose_strs, pose_strs2))]
        else:
            images, depths = [[p for cpath in cpaths for p in sorted(self.get_image_paths(cpath, t))] for t in ('rgb', 'depth')]
            pose_strs = [pose for cpath in cpaths for pose in sorted(self.get_cam_pose_strs(cpath), key=lambda l: int(l.split(' ')[0]))]
            
            images, depths, pose_strs = [list(i) for i in zip(*self.random.sample(list(zip(images, depths, pose_strs)), self.n_sources + self.n_targets))]
        
        c2ws = torch.stack([torch.tensor([float(i) for i in l.strip().split()[1:]]).reshape(4, 4) for l in pose_strs])
        R, t = c2ws[..., :3, :3], c2ws[..., :3, 3]
        (images, resize_ratios, image_dims, center_displacements), (depths, _, depth_dims, _) = [
            [list(i) for i in zip(*(self.get_image(path, is_depth) for path in paths))]
            for paths, is_depth in ((images, False), (depths, True))
        ]
        assert image_dims == depth_dims, f'Inconsistency between image sizes and depth sizes in dataset in scene "{spath}"'
        
        images, depths = [torch.stack(i) for i in (images, depths)]
        resize_ratios, center_displacements = [torch.stack([torch.tensor(i) for i in p]) for p in (resize_ratios, center_displacements)]
        
        images = images / 255.0 # convert from uint8 to 0.0-1.0
        depths = depths.int()
        depth_masks = depths > 0
        depths = depths / 1000.0 # convert from mm to m
        
        # Batching and normalizing intrinsic matrices
        K = einx.id('m n -> b m n', K, b=images.shape[0]).clone()
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
        
        sources = edict({k: v[:2] for k, v in views.items()})
        targets = edict({k: v[2:] for k, v in views.items()})
        
        return edict(
            scene_name=sname,
            sources=sources,
            targets=targets
        )
