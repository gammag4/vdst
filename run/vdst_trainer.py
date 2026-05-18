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
from loss.scheduler import PerceptualLossScheduler
from loss.eval_metrics import EvalMetrics
from dataset.wildrgbd import WildRGBDDataset


class VDSTTrainer(DistributedTrainer):
    def __init__(self, config, config_raw):
        super().__init__(config, config_raw)
        
        self.val_intermediate_results_steps = 0
        
        self.val_batch_size = self.config.train.data.val_batch_size
        self.val_split = 1 * self.val_batch_size # Picking n scenes for validation
        
        if self.is_main_process:
            self.eval_metrics = EvalMetrics().to(self.device)
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['val_intermediate_results_steps'] = self.val_intermediate_results_steps
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.val_intermediate_results_steps = state_dict['val_intermediate_results_steps']
        return super().load_state_dict(state_dict)
    
    def _create_datasets(self, config):
        train_dataset, val_dataset = [
            WildRGBDDataset(
                config.datasets.wildrgbd.path,
                config.n_sources,
                config.n_targets,
                output_dims=config.output_dims,
                seed=self.config.setup.seed
            ) for _ in range(2)
        ]
        train_dataset.random.shuffle(train_dataset.spaths)
        val_dataset.random.shuffle(val_dataset.spaths)
        
        train_dataset.spaths, val_dataset.spaths = train_dataset.spaths[self.val_split:], val_dataset.spaths[:self.val_split * 2] # One part for scenes not in training another for scenes in training
        
        return train_dataset, val_dataset
    
    def _create_loss(self, model_config, loss_config, n_steps):
        loss = Loss(model_config, loss_config)
        loss_scheduler = PerceptualLossScheduler(loss, n_steps, loss_config.perceptual.scheduler)
        
        return loss, loss_scheduler
    
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
        return WandbLogger(self.config.train.logger, self.config_raw, self.is_main_process)
    
    def _init_training(self):
        train_dataset, val_dataset = self._create_datasets(self.config.train.data)
        loss, loss_scheduler = self._create_loss(self.config.model, self.config.train.loss, self.n_steps)
        model = self._create_model(self.config.model, loss)
        optimizer, lr_scheduler = self._create_optimizer(self.config.train.optimizer, model, self.n_steps)
        logger = self._create_logger()
        
        return edict(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_scheduler=loss_scheduler,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
        )
    
    def _try_fit_power_law(self):
        pl_config = self.config.train.metric_power_law_fitting

        if pl_config.should_fit and ((self.is_last and -1 in pl_config.instants) or self.logger.current_iter in pl_config.instants):
            start_step = self.config.train.optimizer.n_warmup_steps if pl_config.skip_warmup_steps else 0
            points = []  # TODO
            points = points[start_step:]
            # Does not fit if there is too little data
            if len(points) >= 10:
                # TODO fit power law
                pass
    
    def _after_step(self):
        self._try_fit_power_law()
    
    def _save_intermediate_results(self, path, batch_index, batch_res):
        source_images, source_depths = batch_res.sources.images, batch_res.sources.depths
        target_gen_images, target_gen_depths = batch_res.gen_targets.images, batch_res.gen_targets.depths
        target_gt_images, target_gt_depths = batch_res.targets.images, batch_res.targets.depths
        
        source_images, source_depths = [einx.id('b v c h w -> b v h w c', t) for t in (source_images, source_depths)]
        
        target_images, target_depths = [torch.stack(tp, dim=0) for tp in [[target_gen_images, target_gt_images], [target_gen_depths, target_gt_depths]]]
        target_images, target_depths = [einx.id('l b v c h w -> l b v h w c', t) for t in (target_images, target_depths)]
        
        source_images, target_images = [t.detach().cpu() for t in (source_images, target_images)]
        
        cmap = plt.get_cmap('jet')
        source_depths, target_depths = [(einx.id('... h w c -> (... h) (w c)', t), t.shape) for t in (source_depths, target_depths)]
        source_depths, target_depths = [torch.from_numpy(cmap(((t - t.min()) / (t.max() - t.min())).detach().cpu().numpy())).reshape(*shape[:-1], 4)[..., :3] for t, shape in (source_depths, target_depths)]
        # source_depths, target_depths = [einx.id('... h w c -> ... h (w c)', t) for t in (source_depths, target_depths)]
        
        sources = einx.id('b v h1 w c, b v h2 w c -> (b (h1 + h2)) (v w) c', source_images, source_depths)
        targets = einx.id('l b v h1 w c, l b v h2 w c -> (b (h1 + h2)) (v l w) c', target_images, target_depths)
        
        is_val_str = 'val' if batch_index < self.val_split // self.val_batch_size else 'train'
        for img, name in [
            (sources, f'sources_{batch_index}_{is_val_str}'),
            (targets, f'targets_{batch_index}_{is_val_str}')
        ]:
            print((img.numpy() * 255.0).astype(np.uint8).shape)
            img = (img * 255.0).numpy().astype(np.uint8)
            img = PIL.Image.fromarray(img)
            img_path = os.path.join(path, f'{name}.png')
            img.save(img_path)
            self.logger.log_image(img_path, name)
        
        with open(os.path.join(path, 'scenes.txt'), 'a', encoding='utf8') as f:
            f.write('\n'.join(batch_res.scene_name) + '\n')
    
    def _run_eval(self, data_iter):
        path = os.path.join(self.config.train.checkpoints.path, 'intermediate_results', f'{self.logger.current_step}')
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'scenes.txt'), 'w', encoding='utf8') as f:
            f.write('')
        
        eval_metricss = []
        for i, batch in enumerate(data_iter):
            batch_res = self.model(batch)
            
            if self.is_last or self.val_intermediate_results_steps % 10 == 0: # only saves results every 1/10th of the time # TODO refactor
                self._save_intermediate_results(path, i, batch_res)
                self.val_intermediate_results_steps = 0
            self.val_intermediate_results_steps += 1
            
            eval_metrics = self.eval_metrics(batch_res.gen_targets, batch_res.targets, valid_depth_range=(0.001, 20))
            eval_metrics.num_images = eval_metrics.images.psnr.numel()
            
            eval_metricss.append(eval_metrics)
        
        eval_metrics = edict()
        
        for k1 in [i for i in eval_metricss[0].keys() if i != 'num_images']:
            for k2 in eval_metricss[0][k1].keys():
                eval_metrics[k1] = eval_metrics.get(k1, edict())
                eval_metrics[k1][k2] = torch.concat([e[k1][k2] for e in eval_metricss], dim=0).mean().item()
        
        eval_metrics.num_images = sum([e.num_images for e in eval_metricss])
        
        self.logger.log({'metrics/eval': eval_metrics})
        
        with open(os.path.join(path, 'eval_metrics.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(edict_to_dict(eval_metrics), f, default_flow_style=False, sort_keys=True)
        
        self.logger.message(f'Saved evaluation results at {path}')
    
    def _run_forward(self, batch):
        res = self.model(batch)
        
        self.logger.log({
            'info/scene_names': batch.scene_name,
            'info/optimizer_lrs': {f'{i}': p['lr'] for i, p in enumerate(self.optimizer.param_groups)},
            'info/loss_weights': {f'{i}': w for i, w in enumerate(res.loss.loss_weights.detach().tolist())},
            'metrics/raw_losses': {f'{i}': w for i, w in enumerate(res.loss.raw_losses.detach().tolist())},
            'metrics/weighted_losses': {f'{i}': w for i, w in enumerate(res.loss.weighted_losses.detach().tolist())},
            'perceptual_metrics/weighted_perceptual_losses': {
                'image': {f'{i}': w for i, w in enumerate(res.loss.weighted_image_perceptual_losses.detach().tolist())},
                'depth': {f'{i}': w for i, w in enumerate(res.loss.weighted_depth_perceptual_losses.detach().tolist())}
            }
        })
        
        return res.loss.loss
