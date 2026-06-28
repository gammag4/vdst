import os
import json
import random
import cv2
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
import torchvision.transforms.functional as VF
import einx
from utils.data import process_data


class WildRGBDDataset(Dataset):
    def __init__(self, path, n_sources, n_targets, output_dims, dataset_type, val_split, test_split, split='train', test_category='truck', seed=42):
        self.path = path
        self.n_sources = n_sources
        self.n_targets = n_targets
        self.output_dims = output_dims
        self.dataset_type = dataset_type
        self.current_epoch = 0
        
        if dataset_type == 'cones_constrained':
            assert self.n_sources == 2, f'dataset_type cannot be used with only 2 sources per scene, but {self.n_sources} were requested'
        
        self.seed = seed
        self.random = random.Random(seed)
        
        assert split in ['train', 'val', 'test', 'test_new_category'], f'Invalid dataset split "{split}"'
        
        if split == 'test_new_category':
            cpaths = [(test_category, os.path.join(path, test_category))]
        else:
            cpaths = [(c, os.path.join(path, c)) for c in os.listdir(path) if c != test_category]
            cpaths = [(c, cpath) for c, cpath in cpaths if os.path.isdir(cpath)]
        
        cpaths.sort()
        
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
        
        if dataset_type == 'raw':
            spaths = [(sname, spath, [spath]) for (sname, spath) in spaths]
        else:
            spaths = [
                (
                    sname,
                    spath,
                    [
                        os.path.join(spath, 'cones', c)
                        # Only uses cone 0 for val and test
                        for c in sorted(os.listdir(os.path.join(spath, 'cones')) if split == 'train' else ['0'])
                        if os.path.isdir(os.path.join(spath, 'cones', c))
                    ]
                )
                for (sname, spath) in spaths
            ]
        
        self.spaths = spaths
    
    def __len__(self):
        return len(self.spaths)
    
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
        
        if self.dataset_type == 'cones_constrained':
            cpath = self.random.choice(cpaths)
            sname = f'{sname}_{os.path.split(cpath)[1]}'
            
            images, depths = [sorted(self.get_image_paths(cpath, t)) for t in ('rgb', 'depth')]
            pose_strs = self.get_cam_pose_strs(cpath)
            
            s_pose_strs = sorted(self.get_cam_pose_strs(cpath), key=lambda l: int(l.split(' ')[0]))
            source_indices = sorted([s_pose_strs.index(pose_strs[0]), s_pose_strs.index(pose_strs[-1])], reverse=True)
            pose_strs = s_pose_strs
            
            s_images, s_depths, s_pose_strs = [[p[i] for i in source_indices] for p in (images, depths, pose_strs)]
            for p in images, depths, pose_strs:
                for i in source_indices:
                    p.pop(i)
            
            s_images, s_depths, s_pose_strs = [list(i) for i in zip(*self.random.sample(list(zip(s_images, s_depths, s_pose_strs)), self.n_sources))]
            images, depths, pose_strs = [list(i) for i in zip(*self.random.sample(list(zip(images, depths, pose_strs)), self.n_targets))]
            images, depths, pose_strs = s_images + images, s_depths + depths, s_pose_strs + pose_strs
        else:
            images, depths = [[p for cpath in cpaths for p in sorted(self.get_image_paths(cpath, t))] for t in ('rgb', 'depth')]
            pose_strs = [pose for cpath in cpaths for pose in sorted(self.get_cam_pose_strs(cpath), key=lambda l: int(l.split(' ')[0]))]
            
            images, depths, pose_strs = [list(i) for i in zip(*self.random.sample(list(zip(images, depths, pose_strs)), self.n_sources + self.n_targets))]
        
        c2ws = torch.stack([torch.tensor([float(i) for i in l.strip().split()[1:]]).reshape(4, 4) for l in pose_strs])
        views = process_data(c2ws, K, images, depths, self.output_dims)
        
        sources = edict({k: v[:2] for k, v in views.items()})
        targets = edict({k: v[2:] for k, v in views.items()})
        
        images, depths = [[os.path.split(i)[1] for i in p] for p in (images, depths)]
        (sources.images_ids, targets.images_ids), (sources.depths_ids, targets.depths_ids) = [(p[:2], p[2:]) for p in (images, depths)]
        
        return edict(
            scene_name=sname,
            sources=sources,
            targets=targets
        )
