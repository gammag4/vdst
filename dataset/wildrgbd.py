import os
import json
import random
import PIL
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
import torchvision.transforms.functional as VF
from torchvision.transforms import InterpolationMode
import numpy as np
import einx


class WildRGBDDatasetConstrainedViews(Dataset):
    def __init__(self, path, n_sources, n_targets, output_dims, train_val_split_index, test_category='truck', split='train', seed=42):
        self.path = path
        self.n_sources = n_sources
        self.n_targets = n_targets
        self.output_dims = output_dims

        self.seed = seed
        self.random = random.Random(seed)
        
        assert split in ['train', 'val', 'test'], f'Invalid dataset split "{split}"'
        
        if split == 'test':
            cpaths = [(test_category, os.path.join(path, test_category))]
        else:
            cpaths = [(c, os.path.join(path, c)) for c in os.listdir(path)]
            cpaths = [(c, cpath) for c, cpath in cpaths if os.path.isdir(cpath)]
        
        spaths = [(f'{c}_{s}', os.path.join(cpath, 'scenes', s)) for c, cpath in cpaths for s in os.listdir(os.path.join(cpath, 'scenes'))]
        self.spaths = [(sname, spath) for sname, spath in spaths if os.path.isdir(spath)]
        
        self.random.shuffle(spaths)
        if split == 'train':
            self.spaths = self.spaths[train_val_split_index:]
        elif split == 'val':
            self.spaths = self.spaths[:train_val_split_index]
        
        self.spaths = [
            (
                f'{sname}_{c}',
                spath,
                os.path.join(spath, 'cones', c)
            )
            # Only uses cone 0 for val and test
            for (sname, spath) in self.spaths for c in (os.listdir(os.path.join(spath, 'cones') if split == 'train' else '0'))
        ]
        self.spaths = [(sname, spath, cpath) for sname, spath, cpath in self.spaths if os.path.isdir(cpath)]
        self.random.shuffle(spaths)

    def __len__(self):
        return len(self.spaths)
    
    def __getitem__(self, i):
        self.random.seed(self.seed + i)
        sname, spath, cpath = self.spaths[i]
        
        with open(os.path.join(spath, 'metadata'), 'r', encoding='utf8') as f:
            data = edict(json.load(f))
            K = torch.tensor(data.K).reshape(3, 3).T
        
        images, depths = [sorted([os.path.join(cpath, p, p2) for p2 in os.listdir(os.path.join(cpath, p))]) for p in ('rgb', 'depth')]
        
        with open(os.path.join(cpath, 'cam_poses.txt'), 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')
            c2ws = [torch.tensor([float(i) for i in l.strip().split()[1:]]).reshape(4, 4) for l in lines]
            R, t = zip(*((c2w[:3, :3], c2w[:3, 3]) for c2w in c2ws))
        
        images, depths, R, t = zip(*self.random.sample(list(zip(images, depths, R, t)), self.n_sources + self.n_targets))
        images, depths = [[torch.from_numpy(np.array(PIL.Image.open(path))) for path in paths] for paths in (images, depths)]
        
        images = [einx.id('h w c -> c h w', image).float() / 255.0 for image in images]
        depths = [einx.id('h w -> 1 h w', depth).int() for depth in depths]
        
        images = [
            VF.resize(
                VF.center_crop(t, output_size=[min(t.shape[-2:])] * 2),
                size=self.output_dims,
                interpolation=InterpolationMode.BICUBIC, # TODO lanczos
                antialias=True
            ).clamp(0, 1)
            for t in images
        ]
        depths = [
            VF.resize(
                VF.center_crop(t, output_size=[min(t.shape[-2:])] * 2),
                size=self.output_dims,
                interpolation=InterpolationMode.NEAREST,
                antialias=False
            )
            for t in depths
        ]
        
        depth_masks = [depth > 0 for depth in depths]
        depths = [depth / 1000.0 for depth in depths]
        
        
        images, depths, depth_masks, R, t = [torch.stack(t, dim=0) for t in (images, depths, depth_masks, R, t)]
        K = einx.id('m n -> b m n', K, b=images.shape[0])
        
        views = edict(
            K=K,
            R=R,
            t=t,
            images=images,
            depths=depths,
            depth_masks=depth_masks
        )
        
        sources = edict({k: torch.stack(v[0], v[-1]) for k, v in views.items()})
        targets = edict({k: v[1:-1] for k, v in views.items()})
        
        return edict(
            scene_name=sname,
            sources=sources,
            targets=targets
        )


class WildRGBDDataset(Dataset):
    def __init__(self, path, n_sources, n_targets, output_dims, train_val_split_index, test_category='truck', split='train', seed=42):
        self.path = path
        self.n_sources = n_sources
        self.n_targets = n_targets
        self.output_dims = output_dims

        self.seed = seed
        self.random = random.Random(seed)
        
        assert split in ['train', 'val', 'test'], f'Invalid dataset split "{split}"'
        
        if split == 'test':
            cpaths = [(test_category, os.path.join(path, test_category))]
        else:
            cpaths = [(c, os.path.join(path, c)) for c in os.listdir(path)]
            cpaths = [(c, cpath) for c, cpath in cpaths if os.path.isdir(cpath)]
        
        spaths = [(f'{c}_{s}', os.path.join(cpath, 'scenes', s)) for c, cpath in cpaths for s in os.listdir(os.path.join(cpath, 'scenes'))]
        self.spaths = [(sname, spath) for sname, spath in spaths if os.path.isdir(spath)]
        
        self.random.shuffle(spaths)
        if split == 'train':
            self.spaths = self.spaths[train_val_split_index:]
        elif split == 'val':
            self.spaths = self.spaths[:train_val_split_index]
    
    def __len__(self):
        return len(self.spaths)
    
    def __getitem__(self, i):
        self.random.seed(self.seed + i)
        sname, spath = self.spaths[i]
        
        with open(os.path.join(spath, 'metadata'), 'r', encoding='utf8') as f:
            data = edict(json.load(f))
            K = torch.tensor(data.K).reshape(3, 3).T
        
        images, depths = [sorted([os.path.join(spath, p, p2) for p2 in os.listdir(os.path.join(spath, p))]) for p in ('rgb', 'depth')]
        
        with open(os.path.join(spath, 'cam_poses.txt'), 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')
            c2ws = [torch.tensor([float(i) for i in l.strip().split()[1:]]).reshape(4, 4) for l in lines]
            R, t = zip(*((c2w[:3, :3], c2w[:3, 3]) for c2w in c2ws))
        
        images, depths, R, t = zip(*self.random.sample(list(zip(images, depths, R, t)), self.n_sources + self.n_targets))
        images, depths = [[torch.from_numpy(np.array(PIL.Image.open(path))) for path in paths] for paths in (images, depths)]
        
        images = [einx.id('h w c -> c h w', image).float() / 255.0 for image in images]
        depths = [einx.id('h w -> 1 h w', depth).int() for depth in depths]
        
        images = [
            VF.resize(
                VF.center_crop(t, output_size=[min(t.shape[-2:])] * 2),
                size=self.output_dims,
                interpolation=InterpolationMode.BICUBIC, # TODO lanczos
                antialias=True
            ).clamp(0, 1)
            for t in images
        ]
        depths = [
            VF.resize(
                VF.center_crop(t, output_size=[min(t.shape[-2:])] * 2),
                size=self.output_dims,
                interpolation=InterpolationMode.NEAREST,
                antialias=False
            )
            for t in depths
        ]
        
        depth_masks = [depth > 0 for depth in depths]
        depths = [depth / 1000.0 for depth in depths]
        
        images, depths, depth_masks, R, t = [torch.stack(t, dim=0) for t in (images, depths, depth_masks, R, t)]
        
        sources, targets = [
            edict(
                K=einx.id('m n -> b m n', K, b=t[s:e].shape[0]),
                R=R[s:e],
                t=t[s:e],
                images=images[s:e],
                depths=depths[s:e],
                depth_masks=depth_masks[s:e]
            )
            for s, e in [(None, self.n_sources), (self.n_sources, None)]
        ]

        return edict(
            scene_name=sname,
            sources=sources,
            targets=targets
        )
