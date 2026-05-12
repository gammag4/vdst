import os
import PIL
import yaml
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.optim.lr_scheduler import ConstantLR, ChainedScheduler, LinearLR, CosineAnnealingLR
import torchvision.transforms.functional as VF
import einx
import matplotlib.pyplot as plt

from utils.module_initialization import init_module_weights, init_transformer_weights
from utils.logger import WandbLogger
from utils.other import edict_to_dict
from run.trainer import DistributedTrainer
from model.model import VDST
from loss.loss import Loss
from loss.eval_metrics import EvalMetrics
from dataset.wildrgbd import WildRGBDDataset


class VDSTTrainer(DistributedTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        if self.rank == 0:
            self.eval_metrics = EvalMetrics().to(self.device)
    
    def _create_dataset(self, config):
        dataset = WildRGBDDataset(config.datasets.wildrgbd.path, config.n_sources, config.n_targets, seed=self.config.setup.seed)
        
        return dataset
    
    def _create_loss(self, model_config, loss_config, n_steps):
        loss = Loss(model_config, loss_config)
        
        return (loss, None)
    
    def _create_model(self, config, loss):
        model = VDST(config, loss)

        init_module_weights(model)
        init_transformer_weights(model.transformer)
        
        return model
    
    def _create_optimizer(self, config, model, n_steps):
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=config.fused
        )
        
        if n_steps is None:
            n_steps = config.n_warmup_steps
        
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-12, end_factor=1, total_iters=config.n_warmup_steps)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=n_steps - config.n_warmup_steps)
        lr_scheduler = ChainedScheduler([warmup_scheduler, decay_scheduler], optimizer=optimizer)
        # lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=n_steps)
        
        return (optimizer, lr_scheduler)
    
    def _create_logger(self):
        return WandbLogger(self.config.train.logger.project_name, self.config.train.logger.run_name, self.config, self.rank == 0)
    
    def _init_training(self):
        dataset = self._create_dataset(self.config.train.data)
        loss, loss_scheduler = self._create_loss(self.config.model, self.config.train.loss, self.n_steps)
        model = self._create_model(self.config.model, loss)
        optimizer, lr_scheduler = self._create_optimizer(self.config.train.optimizer, model, self.n_steps)
        logger = self._create_logger()
        
        return edict(
            dataset=dataset,
            loss_scheduler=loss_scheduler,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
        )
    
    def _save_intermediate_results(self, res):
        source_images, source_depths = res.sources.images, res.sources.depths
        target_gen_images, target_gen_depths = res.gen_targets.images, res.gen_targets.depths
        target_gt_images, target_gt_depths = res.targets.images, res.targets.depths
        
        source_images, source_depths = [einx.rearrange('b v c h w -> (b h) (v w) c', t) for t in (source_images, source_depths)]
        
        target_images, target_depths = [torch.stack(tp, dim=0) for tp in [[target_gen_images, target_gt_images], [target_gen_depths, target_gt_depths]]]
        target_images, target_depths = [einx.rearrange('l b v c h w -> (b h) (v l w) c', t) for t in (target_images, target_depths)]
        
        source_images, target_images = [t.detach().cpu().numpy() for t in (source_images, target_images)]
        
        cmap = plt.get_cmap('jet')
        source_depths, target_depths = [einx.rearrange('h w c -> h (w c)', t) for t in (source_depths, target_depths)]
        source_depths, target_depths = [cmap(((t - t.min()) / (t.max() - t.min())).detach().cpu().numpy()) for t in (source_depths, target_depths)]
        
        source_images, target_images, source_depths, target_depths = [PIL.Image.fromarray((t * 255.0).astype(np.uint8)) for t in (source_images, target_images, source_depths, target_depths)]
        
        path = os.path.join(self.config.train.checkpoints.path, 'intermediate_results', f'{self.logger.current_step}')
        os.makedirs(path, exist_ok=True)
        
        for img, name in [
            (source_images, 'source_images'),
            (source_depths, 'source_depths'),
            (target_images, 'target_images'),
            (target_depths, 'target_depths')
        ]:
            img_path = os.path.join(path, f'{name}.png')
            img.save(img_path)
            self.logger.log_image(img_path, name)
        
        with open(os.path.join(path, 'scenes.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(res.scene_name))
        
        eval_metrics = self.eval_metrics(res.gen_targets, res.targets, valid_depth_range=(0.001, 20))
        
        eval_metrics.num_images = eval_metrics.images.psnr.numel()
        
        for t in (eval_metrics.images, eval_metrics.depths):
            for k in t.keys():
                t[k] = t[k].mean().item()
        
        self.logger.log({'eval_metrics': eval_metrics})
        
        with open(os.path.join(path, 'eval_metrics.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(edict_to_dict(eval_metrics), f, default_flow_style=False, sort_keys=True)
        
        self.logger.message(f'Saved intermediate results at {path}')
    
    def _run_forward(self, batch):
        res = self.model(batch)
        
        self.logger.log({
            'scene_names': batch.scene_name,
            'optimizer_lrs': {f'{i}': p['lr'] for i, p in enumerate(self.optimizer.param_groups)},
            'weighted_losses': {f'{i}': w for i, w in enumerate(res.loss.weighted_losses.detach().tolist())},
            'weighted_image_perceptual_losses': {f'{i}': w for i, w in enumerate(res.loss.weighted_image_perceptual_losses.detach().tolist())},
            'weighted_depth_perceptual_losses': {f'{i}': w for i, w in enumerate(res.loss.weighted_depth_perceptual_losses.detach().tolist())}
        })
        
        if self.rank == 0 and self.logger.current_step % self.config.train.checkpoints.results_steps_interval == 0:
            self._save_intermediate_results(res)
        
        return res.loss.loss
