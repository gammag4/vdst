import os
import yaml
from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
from torch.optim.lr_scheduler import ConstantLR, ChainedScheduler, LinearLR, CosineAnnealingLR
import torchvision.transforms.functional as VF
import einx
import matplotlib.pyplot as plt
import shutil

from utils.module_initialization import init_module_weights, init_transformer_weights
from utils.logger import WandbLogger
from utils.other import edict_to_dict
from run.trainer import DistributedTrainer
from model.model import VDST
from loss.loss import Loss
from loss.scheduler import PerceptualLossScheduler
from loss.eval_metrics import EvalMetrics
from dataset.wildrgbd import WildRGBDDataset
from dataset.co3d_eval import CO3DEvalDataset


class VDSTTrainer(DistributedTrainer):
    def __init__(self, config, config_raw):
        super().__init__(config, config_raw)
        
        self.curr_val_step = 0
        
        # only saves results every 1/10th of the time that eval metrics are computed
        self.intermediate_results_num_batches = 1 # should be <= intermediate_val_num_batches
        self.intermediate_results_interval = 1
        
        # only evals entire dataset every 1/1th of the time that eval metrics are computed
        self.intermediate_val_num_batches = 1
        self.use_entire_val_datset_interval = 1
        
        if self.is_main_process:
            self.eval_metrics = EvalMetrics().to(self.device)
        
        self.training = True
        self.test_dataloader = None
        self.test_new_category_dataloader = None
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['curr_val_step'] = self.curr_val_step
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.curr_val_step = state_dict['curr_val_step']
        
        return super().load_state_dict(state_dict)
    
    def _create_datasets(self, config):
        # Picking n scenes for validation
        val_split = 0.005
        test_split = 0.02
        
        train_dataset, val_dataset, test_dataset, test_new_category_dataset = [
            WildRGBDDataset(
                config.datasets.wildrgbd.path,
                config.n_sources,
                config.n_targets,
                output_dims=config.output_dims,
                dataset_type=config.datasets.wildrgbd.dataset_type,
                val_split=val_split,
                test_split=test_split,
                split=split,
                test_category='truck',
                seed=self.config.setup.seed
            )
            for split in ('train', 'val', 'test', 'test_new_category')
        ]
        
        # val_dataset = CO3DEvalDataset(
        #     path='/media/gabriel/6d735c7f-5832-4134-afa6-9e50454ca09c/co3d_data/co3d_eval/',
        #     split='3'
        # )
        
        return train_dataset, val_dataset, test_dataset, test_new_category_dataset
    
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
        train_dataset, val_dataset, test_dataset, test_new_category_dataset = self._create_datasets(self.config.train.data)
        train_dataloader = self._create_dataloader(train_dataset, train_dataloader=True)
        val_dataloader = self._create_dataloader(val_dataset, train_dataloader=False)
        self.test_dataloader = self._create_dataloader(test_dataset, train_dataloader=False)
        self.test_new_category_dataloader = self._create_dataloader(test_new_category_dataset, train_dataloader=False)
        
        loss, loss_scheduler = self._create_loss(self.config.model, self.config.train.loss, self.n_steps)
        model = self._create_model(self.config.model, loss)
        optimizer, lr_scheduler = self._create_optimizer(self.config.train.optimizer, model, self.n_steps)
        logger = self._create_logger()
        
        return edict(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_scheduler=loss_scheduler,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
        )
    
    def _try_fit_power_law(self):
        pl_config = self.config.train.metric_power_law_fitting
        
        if pl_config.should_fit and ((self.is_last and -1 in pl_config.instants) or self.current_step in pl_config.instants):
            start_step = self.config.train.optimizer.n_warmup_steps if pl_config.skip_warmup_steps else 0
            points = []  # TODO
            points = points[start_step:]
            # Does not fit if there is too little data
            if len(points) >= 10:
                # TODO fit power law
                pass
    
    def _after_step(self):
        self._try_fit_power_law()
    
    def _save_val_results_scene_names(self, scenes_file_path, batch_res):
        with open(scenes_file_path, 'a+', encoding='utf8') as f:
            f.seek(0)
            data = yaml.safe_load(f) or {}
            
            data['sources'] = data.get('sources', {})
            data['targets'] = data.get('targets', {})
            data['scene_names'] = data.get('scene_names', []) + ['_'.join(i.split('_')[:-1]) for i in batch_res.scene_name]
            data['sources']['images_ids'] = data['sources'].get('images_ids', []) + batch_res.sources.images_ids
            data['sources']['depths_ids'] = data['sources'].get('depths_ids', []) + batch_res.sources.depths_ids
            data['targets']['images_ids'] = data['targets'].get('images_ids', []) + batch_res.targets.images_ids
            data['targets']['depths_ids'] = data['targets'].get('depths_ids', []) + batch_res.targets.depths_ids
            
            f.truncate(0)
            f.seek(0)
            yaml.safe_dump(data, f)
    
    def _save_intermediate_results(self, path, batch_index, batch_res, is_diff=False, is_train=False):
        source_images, source_depths = batch_res.sources.images, batch_res.sources.depths
        target_gen_images, target_gen_depths = batch_res.gen_targets.images, batch_res.gen_targets.depths
        target_gt_images, target_gt_depths = batch_res.targets.images, batch_res.targets.depths
        
        smin, tmin1, tmin2 = [t.min().detach().cpu().item() for t in [source_depths, target_gen_depths, target_gt_depths]]
        smax, tmax1, tmax2 = [t.max().detach().cpu().item() for t in [source_depths, target_gen_depths, target_gt_depths]]
        tmin, tmax = min(tmin1, tmin2), max(tmax1, tmax2)
        
        source_images, source_depths = [einx.id('b v c h w -> b v h w c', t) for t in (source_images, source_depths)]
        
        imgs = [[target_gen_images, target_gt_images], [target_gen_depths, target_gt_depths]]
        imgs = [[torch.where(batch_res.targets.depth_masks, (t[0] - t[1]).abs(), 0)] if is_diff else t for t in imgs]
        target_images, target_depths = [torch.stack(tp, dim=0) for tp in imgs]
        target_images, target_depths = [einx.id('l b v c h w -> l b v h w c', t) for t in (target_images, target_depths)]
        
        source_images, target_images = [t.detach().cpu() for t in (source_images, target_images)]

        cmap = plt.get_cmap('jet')
        source_depths, target_depths = [(einx.id('... h w c -> (... h) (w c)', t), t.shape) for t in (source_depths, target_depths)]
        source_depths, target_depths = [torch.from_numpy(cmap(((t - mi) / (ma - mi)).detach().cpu().numpy())).reshape(*shape[:-1], 4)[..., :3] for (t, shape), (mi, ma) in ((source_depths, (smin, smax)), (target_depths, (tmin, tmax)))]
        # source_depths, target_depths = [einx.id('... h w c -> ... h (w c)', t) for t in (source_depths, target_depths)]
        
        sources = einx.id('b v h1 w c, b v h2 w c -> (b (h1 + h2)) (v w) c', source_images, source_depths)
        targets = einx.id('l b v h1 w c, l b v h2 w c -> (b (h1 + h2)) (v l w) c', target_images, target_depths)
        
        is_val_str = 'train' if is_train else 'val'
        for img, name in [
            (targets, f'diff_targets_{batch_index}_{is_val_str}')
        ] if is_diff else [
            (sources, f'sources_{batch_index}_{is_val_str}'),
            (targets, f'targets_{batch_index}_{is_val_str}')
        ]:
            img = (img * 255.0).numpy().astype(np.uint8)
            img_path = os.path.join(path, f'{name}.png')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)
            self.logger.log_image(img_path, name)
    
    def _run_eval(self, data_iter, out_path, should_log=True):
        if self.training:
            train_scenes_path = os.path.join(out_path, 'train_scenes.yaml')
            if os.path.isfile(train_scenes_path):
                os.remove(train_scenes_path)
        val_scenes_path = os.path.join(out_path, 'val_scenes.yaml' if self.training else 'scenes.yaml')
        val_scenes_rendered_path = os.path.join(out_path, 'val_scenes_rendered.yaml' if self.training else 'scenes_rendered.yaml')
        if os.path.isfile(val_scenes_path):
            os.remove(val_scenes_path)
        if os.path.isfile(val_scenes_rendered_path):
            os.remove(val_scenes_rendered_path)
        
        if self.training:
            should_save_intermediate_results = self.is_last or self.curr_val_step % self.intermediate_results_interval == 0
            should_use_entire_val_dataset = self.is_last or self.curr_val_step % self.use_entire_val_datset_interval == 0
            self.curr_val_step += 1
        else:
            should_save_intermediate_results = True
            should_use_entire_val_dataset = True
        
        # Always uses only number specified of batches for intermediate results of scenes used in training
        if self.training and should_save_intermediate_results:
            for j in range(self.intermediate_results_num_batches):
                train_batch = []
                for i in range(j * self.config.train.data.train_batch_size, (j + 1) * self.config.train.data.train_batch_size):
                    train_batch.append(self.train_dataloader.dataset[i])
                
                sources, targets = [edict({k: (lambda x: torch.stack(x) if isinstance(train_batch[0][p][k], torch.Tensor) else x)([i[p][k] for i in train_batch]) for k in train_batch[0][p].keys()}) for p in ('sources', 'targets')]
                train_batch = edict(
                    scene_name=[i.scene_name for i in train_batch],
                    sources=sources,
                    targets=targets
                )
                
                train_res = self.model(train_batch)
                self._save_intermediate_results(out_path, 0, train_res, is_train=True)
                self._save_intermediate_results(out_path, 0, train_res, is_diff=True, is_train=True)
                self._save_val_results_scene_names(train_scenes_path, train_res)
        
        eval_metricss = []
        for i, batch in enumerate(data_iter):
            if not should_use_entire_val_dataset and i >= self.intermediate_val_num_batches:
                break
            
            batch.sources.images_ids, batch.sources.depths_ids, batch.targets.images_ids, batch.targets.depths_ids = [
                [[j[i] for j in p] for i in range(len(p[0]))]
                for p in (batch.sources.images_ids, batch.sources.depths_ids, batch.targets.images_ids, batch.targets.depths_ids)
            ]
            batch_res = self.model(batch)
            
            # Always uses only number specified of batches for intermediate results of validation scenes
            if should_save_intermediate_results and i < self.intermediate_results_num_batches:
                self._save_intermediate_results(out_path, i, batch_res)
                self._save_intermediate_results(out_path, i, batch_res, is_diff=True)
                self._save_val_results_scene_names(val_scenes_rendered_path, batch_res)
            self._save_val_results_scene_names(val_scenes_path, batch_res)
            
            eval_metrics = self.eval_metrics(batch_res.gen_targets, batch_res.targets, valid_depth_range=self.config.model.d_range)
            eval_metrics.num_images = eval_metrics.images.psnr.numel()
            
            eval_metricss.append(eval_metrics)
        
        eval_metrics = edict()
        
        for k1 in [i for i in eval_metricss[0].keys() if i != 'num_images']:
            for k2 in eval_metricss[0][k1].keys():
                eval_metrics[k1] = eval_metrics.get(k1, edict())
                t = [e[k1][k2] for e in eval_metricss]
                eval_metrics[k1][k2] = torch.stack([i.sum() for i in t]).sum().item() / sum([i.numel() for i in t])
        
        eval_metrics.num_images = sum([e.num_images for e in eval_metricss])
        
        if should_log:
            self.logger.log({'metrics/eval': eval_metrics})
        
        with open(os.path.join(out_path, 'eval_metrics.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(edict_to_dict(eval_metrics), f, default_flow_style=False, sort_keys=True)
        
        if should_log:
            self.logger.message(f'Saved evaluation results at {out_path}')
    
    def _post_train(self):
        self.training = False
        shutil.rmtree(os.path.join(self.config.train.checkpoints.path, 'final_eval'), ignore_errors=True)
        # self._eval(self.train_dataloader, os.path.join(self.config.train.checkpoints.path, 'final_eval', 'train'), False)
        self._eval(self.val_dataloader, os.path.join(self.config.train.checkpoints.path, 'final_eval', 'val'), False)
        self._eval(self.test_dataloader, os.path.join(self.config.train.checkpoints.path, 'final_eval', 'test'), False)
        self._eval(self.test_new_category_dataloader, os.path.join(self.config.train.checkpoints.path, 'final_eval', 'test_new_category'), False)
        self.training = True # TODO remove
    
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
