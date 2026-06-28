import os
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torch
from utils.data import process_data


class CO3DEvalDataset(Dataset):
    def __init__(self, path, split='3'):
        self.path = path
        
        assert split in ['3', '6', '9'], f'Wrong split "{split}"'
        self.split_path = f'train_test_split_{split}.json'
        
        cpaths = [(c, os.path.join(path, c)) for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))]
        spaths = [(f'{c}_{s}', os.path.join(cpath, s)) for c, cpath in cpaths for s in os.listdir(cpath) if os.path.isdir(os.path.join(cpath, s))]
        self.spaths = spaths
    
    def __len__(self):
        return len(self.spaths)
    
    def get_image_paths(self, cpath, type):
        return [os.path.join(cpath, type, p) for p in os.listdir(os.path.join(cpath, type))]
    
    def get_cam_pose_strs(self, cpath):
        with open(os.path.join(cpath, 'cam_poses.txt'), 'r', encoding='utf8') as f:
            return f.read().strip().split('\n')
    
    def __getitem__(self, i):
        sname, spath = self.spaths[i]
        
        images, depths = [sorted([os.path.join(spath, c, i) for i in os.listdir(os.path.join(spath, c))]) for c in ('images', 'depths')]
        with open(os.path.join(spath, 'transforms.json'), 'r', encoding='utf8') as f:
            transforms = json.load(f)
        K = torch.tensor([[transforms['fl_x'], 0.0, transforms['cx']], [0.0, transforms['fl_y'], transforms['cy']], [0.0, 0.0, 1.0]])
        output_dims = (transforms['h'], transforms['w'])
        frames_paths = [f['file_path'] for f in transforms['frames']]
        c2ws = torch.stack([torch.tensor(f['transform_matrix']) for f in transforms['frames']])
        valid_views = [f['is_valid'] for f in transforms['frames']]
        
        # Transforming matrices so that it uses the same convention as WildRGBD
        P1 = torch.eye(4)
        P1[1, 1], P1[2, 2] = -1, -1
        P2 = torch.eye(4)
        P2[1, 1], P2[2, 2] = 0, 0
        P2[2, 1], P2[1, 2] = 1, -1
        c2ws = P2 @ c2ws @ P1
        
        for i, d, fp in zip(images, depths, frames_paths):
            i, d, fp = [os.path.split(t)[1].split('.')[0] for t in (i, d, fp)]
            assert i == d and i == fp, f'Inconsistency between images and depths in dataset in scene "{spath}"'
        
        with open(os.path.join(spath, self.split_path), 'r', encoding='utf8') as f:
            splits = json.load(f)
        source_ids, target_ids = splits['train_ids'], splits['test_ids']
        source_ids = source_ids[:2] # TODO remove after testing stuff
        view_ids = source_ids + target_ids
        
        assert False not in [valid_views[i] for i in view_ids], 'Trying to use invalid views for evaluation'
        
        output_dims = (256, 256)  # TODO remove after testing stuff
        c2ws = c2ws[view_ids]
        images, depths = [[t[i] for i in view_ids] for t in (images, depths)]
        views = process_data(c2ws, K, images, depths, output_dims)
        
        sources = edict({k: v[:len(source_ids)] for k, v in views.items()})
        targets = edict({k: v[len(source_ids):] for k, v in views.items()})
        
        return edict(
            scene_name=sname,
            sources=sources,
            targets=targets
        )
