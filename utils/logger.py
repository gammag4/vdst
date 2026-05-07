from abc import ABC, abstractmethod
from pprint import pprint
import wandb


class Logger(ABC):
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process
        self.current_step = 0
        self.iteration_vars = {}
        self.global_vars = {}
        
        self.log({'current_step': self.current_step})
    
    @property
    def vars(self):
        return {**self.global_vars, **self.iteration_vars}

    @abstractmethod
    def _log_vars(self, vars):
        pass

    # Should be called after every pass
    def update(self):
        if self.is_main_process:
            self._log_vars(self.vars)
            self.iteration_vars = {}
            self.current_step += 1
            
            self.log({'current_step': self.current_step})
    
    def log(self, vars: dict):
        if self.is_main_process:
            self.iteration_vars = {**self.iteration_vars, **vars}
    
    @abstractmethod
    def log_images(self, paths, captions):
        pass
    
    def log_global(self, vars: dict):
        if self.is_main_process:
            self.global_vars = {**self.global_vars, **vars}
    
    @abstractmethod
    def _display_message(self, msg):
        pass
    
    def message(self, msg):
        if self.is_main_process:
            self._display_message(msg)

    @abstractmethod
    def _display_vars(self, vars):
        pass

    def display_current(self):
        if self.is_main_process:
            self._display_vars(self.vars)
    
    def state_dict(self):
        return {
            'iteration': self.current_step,
            'iteration_vars': self.iteration_vars,
            'global_vars': self.global_vars
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['iteration']
        self.iteration_vars = state_dict['iteration_vars']
        self.global_vars = state_dict['global_vars']


class PrintLogger(Logger):
    def _display_message(self, msg):
        print(msg)
    
    def _display_vars(self, vars):
        pprint(vars)


class StandardLogger(PrintLogger):
    def __init__(self, is_main_process):
        super().__init__(is_main_process)
        self.logs = []

    def _log_vars(self, vars):
        self.logs.append(vars)
    
    def log_images(self, paths, captions):
        raise Exception('Cant log images')

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['logs'] = self.logs
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.logs = state_dict['logs']


class WandbLogger(PrintLogger):
    def __init__(self, project_name, run_name, config, is_main_process):
        super().__init__(is_main_process)
        wandb.init(project=project_name, name=run_name, config={}) # TODO config
    
    def _log_vars(self, vars):
        wandb.log(vars, step=self.current_step)
    
    def log_images(self, paths, captions):
        images = [wandb.Image(p, caption=c) for p, c in zip(paths, captions)]
        wandb.log({'images': images}, step=self.current_step)


# TODO ???
class Stateful(ABC):
    @abstractmethod
    def load_default_state(self):
        pass
    
    @abstractmethod
    def state_dict(self) -> dict:
        pass
    
    @abstractmethod
    def load_state_dict(self, state: dict):
        pass
