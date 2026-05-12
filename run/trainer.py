import os
import datetime
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
# Model that takes in data and distributes across GPUs
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper
import torch.distributed as dist
import torch.amp as amp
from utils.grad_scaler import GradScaler

from utils.other import print_model_stats, find_unused_model_params
from utils.timer import Timer


class DistributedTrainer(ABC):
    def __init__(self, config):
        self.config = config

        self.device = config.setup.distributed.device
        self.local_rank = config.setup.distributed.local_rank
        self.rank = config.setup.distributed.rank
        self.amp_config = config.setup.amp
        self.grad_scaler_enabled = config.setup.grad_scaler
        self.grad_clipping_config = config.train.grad_clipping
        self.n_steps = self.config.train.n_steps
        
        self.last_grad_norms = torch.tensor([], dtype=torch.float32)
        self.train_data = None
        self.val_data = None
        self.loss_scheduler = None
        self.lr_scheduler = None
        self.optimizer = None
        self.grad_scaler = None
        self.model = None
        self.logger = None
        self.timer = None
        self.current_epoch = 0
        self.current_epoch_step = 0

    def _create_dataloader(self, dataset: Dataset, train_dataloader=True):
        config = self.config.train.data

        # TODO check if this loader works with ddp. seems to work, but needs to check with multiple gpus
        return StatefulDataLoader(
            dataset,
            batch_size=config.train_batch_size if train_dataloader else config.val_batch_size,
            # Shuffle should be defined in sampler when using DistributedSampler
            shuffle=False,
            # Sampler that sends different batches to different gpus
            sampler=DistributedSampler(dataset, shuffle=config.shuffle),
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=True,
            pin_memory=config.pin_memory,
            drop_last=False,
        )

    def state_dict(self):
        state_dict = {
            'train_data': self.train_data.state_dict(),
            'val_data': None if self.val_data is None else self.val_data.state_dict(),
            'loss_scheduler': None if self.loss_scheduler is None else self.loss_scheduler.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'logger': self.logger.state_dict(),
            'timer': self.timer.state_dict(),
            'last_grad_norms': self.last_grad_norms,
            'current_epoch': self.current_epoch,
            'current_epoch_step': self.current_epoch_step,
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.train_data.load_state_dict(state_dict['train_data'])
        self.val_data.load_state_dict(state_dict['val_data'])
        if state_dict['loss_scheduler'] is not None:
            self.loss_scheduler.load_state_dict(state_dict['loss_scheduler'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
        self.logger.load_state_dict(state_dict['logger'])
        self.timer.load_state_dict(state_dict['timer'])
        self.last_grad_norms = state_dict['last_grad_norms']
        self.current_epoch = state_dict['current_epoch']
        self.current_epoch_step = state_dict['current_epoch_step']

    def _get_last_checkpoint_path(self, path):
        os.makedirs(path, exist_ok=True)

        checkpoints = os.listdir(path)
        
        if len(checkpoints) == 0:
            return None
        
        checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
        checkpoint_path = os.path.join(path, checkpoint)
        
        return checkpoint_path
    
    def _try_load_checkpoint(self):
        config = self.config.train.checkpoints
        
        model_checkpoint_path = self._get_last_checkpoint_path(os.path.join(config.path, 'checkpoints'))
        train_checkpoint_path = self._get_last_checkpoint_path(os.path.join(config.path, 'train_checkpoints'))
        
        if model_checkpoint_path is not None:
            # Maps to the specific device
            # This prevents processes from using others' devices (when set to accelerator:local_rank)
            self.model.module.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu', weights_only=config.weights_only))
            self.logger.message(f'Resumed training with model from {model_checkpoint_path}')
        
        if train_checkpoint_path is not None:
            # Maps to the specific device
            # This prevents processes from using others' devices (when set to accelerator:local_rank)
            self.load_state_dict(torch.load(train_checkpoint_path, map_location='cpu', weights_only=config.weights_only))
            self.logger.message(f'Resumed training with training data from {train_checkpoint_path}')

    def _try_save_checkpoint(self):
        config = self.config.train.checkpoints
        
        current_step = self.logger.current_step - 1 # Runs after updating step
        
        # Ensures only saves from first GPU to prevent redundancy
        if current_step == 0 or self.rank != 0 or current_step % self.config.train.checkpoints.checkpoint_steps_interval != 0:
            return
        
        torch.accelerator.synchronize(self.device)

        model_checkpoint_path = os.path.join(config.path, 'checkpoints', f'{current_step}.pt')
        torch.save(self.model.module.state_dict(), model_checkpoint_path)
        self.logger.message(f'Saved trained model at {model_checkpoint_path}')

        train_checkpoint_path = os.path.join(config.path, 'train_checkpoints', f'{current_step}.pt')
        torch.save(self.state_dict(), train_checkpoint_path)
        self.logger.message(f'Saved training data at {train_checkpoint_path}')
    
    @abstractmethod
    def _run_forward(self, batch):
        pass
    
    def _try_grad_clip_skip(self):
        if self.grad_clipping_config.type is None:
            return False
        
        assert self.grad_clipping_config.type in ['abs', 'std'], 'Invalid grad clipping config'
        
        if len(self.last_grad_norms) > 50:
            self.last_grad_norms = self.last_grad_norms[-50:]
        
        if len(self.last_grad_norms) >= 5:
            grad_norm_mean, grad_norm_std = self.last_grad_norms.mean().item(), self.last_grad_norms.std().item()
        else:
            grad_norm_mean, grad_norm_std = 1.0, 0.6
        
        grad_clip_norm = 0.0
        grad_skip_norm = 0.0
        if self.grad_clipping_config.type == 'std':
            grad_clip_norm = grad_norm_mean + self.grad_clipping_config.clip_std * grad_norm_std
            grad_skip_norm = grad_norm_mean + self.grad_clipping_config.skip_std * grad_norm_std
        if self.grad_clipping_config.type == 'abs':
            grad_clip_norm = self.grad_clipping_config.clip_norm
            grad_skip_norm = self.grad_clipping_config.skip_norm
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm).detach()
        grad_norm = grad_norm.cpu()
        self.last_grad_norms = torch.concat([self.last_grad_norms, grad_norm.reshape(-1)])
        grad_norm = grad_norm.item()

        should_skip_step = grad_norm > grad_skip_norm
        clipped = grad_norm > grad_clip_norm
        
        if should_skip_step:
            self.logger.message(f'Skipped norm: {grad_norm:.4f} > {grad_skip_norm:.4f} ({grad_norm_mean:.4f} + {grad_skip_norm - grad_norm_mean:.4f})')
        elif clipped:
            self.logger.message(f'Clipped norm: {grad_norm:.4f} > {grad_clip_norm:.4f} ({grad_norm_mean:.4f} + {grad_clip_norm - grad_norm_mean:.4f})')
        
        self.logger.log({'grad': {
            'norm': grad_norm,
            'norm_mean': grad_norm_mean,
            'norm_std': grad_norm_std,
            'clip_norm': grad_clip_norm,
            'skip_norm': grad_skip_norm,
            'clipped': clipped,
            'skipped': should_skip_step
        }})
        
        return should_skip_step

    # Use this function to run a batch for a generic model
    def _run_pass_once(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        # AMP: Casts operations to mixed precision
        with amp.autocast(device_type=self.device, dtype=self.amp_config.dtype, enabled=self.amp_config.enabled):
            # output.dtype is bfloat16 because linear layers autocast to bfloat16
            # loss.dtype is float32 because mse_loss layers autocast to float32
            loss = self._run_forward(batch)

        # Exits autocast before backward()
        # Backward passes under autocast are not recommended
        # Backward ops run in the same dtype autocast chose for corresponding forward ops

        # Scales the loss, and calls backward()
        # to create scaled gradients
        self.grad_scaler.scale(loss).backward()  # Already called in model

        # find_unused_params(self.model) # TODO

        # All gradients are scaled in this region up to scaler.step(optimizer), so they need to be unscaled to be used
        # Unscales the gradients of optimizer's assigned params in-place
        self.grad_scaler.unscale_(self.optimizer)

        # Gradient clipping/skipping
        should_skip_step = self._try_grad_clip_skip()

        # Unscales gradients (if not unscaled before) and calls or skips optimizer.step()
        # It skips if there are infs or NaNs in grads
        # Since we called unscale_ before, it will not unscale gradients again
        if not should_skip_step:
            self.grad_scaler.step(self.optimizer)

        # Updates the scale for next iteration
        self.grad_scaler.update()

        self.logger.log({'loss': loss.detach().item()})

    # This method is run after each pass to update stuff
    def _step(self):
        self.timer.update()
        
        self.logger.log({'time': {
            'total': self.timer.total,
            'delta': self.timer.delta,
            'avg_delta': self.timer.avg_delta,
            'eta': self.timer.eta,
            'total_str': str(datetime.timedelta(seconds=self.timer.total)),
            'eta_str': str(datetime.timedelta(seconds=self.timer.eta))
        }})
        
        if self.loss_scheduler is not None:
            self.loss_scheduler.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
    
    # Use this method to run one forward/backward pass for a generic model
    def _run_pass(self, batch):
        self._run_pass_once(batch)
        
        self._step()

        if self.logger.current_step % self.config.train.val_steps_interval == 0:
            self._val()
        
        if self.rank == 0 and self.logger.current_step % self.config.train.display_log_steps_interval == 0:
            torch.accelerator.synchronize(self.device)
            self.logger.display_current()
            print('')
        self.logger.update()
        
        self.current_epoch_step += 1
        
        self._try_save_checkpoint()
    
    def _train(self):
        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(self.current_epoch)
        it = iter(self.train_data)
        
        for _ in range(self.n_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_data)
                batch = next(it)
                self.current_epoch += 1
                self.current_epoch_step = 0
                self.train_data.sampler.set_epoch(self.current_epoch)
            
            self.logger.log({
                'epoch': self.current_epoch,
                'epoch_step': self.current_epoch_step
            })
            
            self._run_pass(batch)
    
    # Should check and only save stuff at rank 0
    @abstractmethod
    def _run_eval(self, data_iter):
        pass
    
    def _val(self):
        self.model.eval()
        
        self.val_data.sampler.set_epoch(0)

        with amp.autocast(device_type=self.device, dtype=self.amp_config.dtype, enabled=self.amp_config.enabled), torch.no_grad():
            if self.rank == 0:
                self._run_eval(iter(self.val_data))
        
        self.model.train()
    
    @abstractmethod
    def _init_training(self):
        pass
    
    async def run(self):
        print(f'Starting run "{self.config.train.logger.project_name} - {self.config.train.logger.run_name}"\n')
        
        training_args = self._init_training()
        
        self.train_data = self._create_dataloader(training_args.train_dataset, train_dataloader=True)
        self.val_data = self._create_dataloader(training_args.val_dataset, train_dataloader=False)
        self.loss_scheduler = training_args.loss_scheduler
        self.optimizer = training_args.optimizer
        self.lr_scheduler = training_args.lr_scheduler
        self.logger = training_args.logger
        
        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(training_args.model.to(self.device), device_ids=[self.local_rank])
        
        # Gradient scaler for AMP (probably not needed if using bfloat16)
        self.grad_scaler = GradScaler(
            device=self.device,
            enabled=self.amp_config.enabled and self.grad_scaler_enabled
        )
        
        self.timer = Timer(self.n_steps)
        
        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing checkpoint
        # Starts from checkpoint if exists
        self._try_load_checkpoint()
        
        if self.rank == 0:
            print_model_stats(self.model, print_all_named_params=False)
        
        return self._train()
