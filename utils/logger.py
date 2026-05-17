from abc import ABC, abstractmethod
from pprint import pprint
import wandb


class Logger(ABC):
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process
        self.current_step = 0
        self.iteration_vars = {}
        self.global_vars = {}
    
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
    
    def start(self):
        pass
    
    def end(self):
        pass
    
    def log(self, vars: dict):
        if self.is_main_process:
            self.iteration_vars = {**self.iteration_vars, **vars}
    
    @abstractmethod
    def log_image(self, path, name):
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
    
    def log_image(self, path, name):
        raise Exception('Cant log images')

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['logs'] = self.logs
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.logs = state_dict['logs']


class WandbLogger(PrintLogger):
    def __init__(self, logger_config, config, is_main_process):
        super().__init__(is_main_process)
        
        self.logger_config = logger_config
        self.config = config
    
    def start(self):
        group_msg = f' - {self.logger_config.run_group_name}' if self.logger_config.run_group_name is not None else ''
        self.message(f'Starting run "{self.logger_config.project_name}{group_msg} - {self.logger_config.run_name}"\n')
        
        wandb.init(
            project=self.logger_config.project_name,
            group=self.logger_config.run_group_name,
            name=self.logger_config.run_name,
            config=self.config
        )
    
    def _log_vars(self, vars):
        wandb.log(vars, step=self.current_step)
    
    def log_image(self, path, name):
        wandb.log({name: wandb.Image(path, caption=name)}, step=self.current_step)
    
    def end(self):
        wandb.finish()
