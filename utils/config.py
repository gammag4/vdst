import os
import importlib
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import torch

from .other import edict_to_dict


def import_object(full_name):
    # Use [(obj), 'path.to.Object'] to return object
    s = full_name.split('.')
    path, name = '.'.join(s[:-1]), s[-1]
    return importlib.import_module(path).__dict__[name]


def import_and_run_object(full_name, *args):
    obj = import_object(full_name)

    # Use [(obj), 'path.to.Object', null] to run without arguments
    if len(args) == 1 and args[0] is None:
        return obj()
    # Use [(obj), 'path.to.Object', arg1, arg2, ...] to run with arguments
    elif len(args) >= 1:
        return obj(*args)
    # Use [(obj), 'path.to.Object'] to return without running
    else:
        return obj


def parse_env(var, default):
    res = os.environ.get(var, default)
    try:
        res = float(res)
        res = int(res)
    except ValueError:
        pass

    return res


def parse_env_default(var_name, current_value, default_value):
    return parse_env(var_name, default_value) if current_value is None else current_value


def parse_prefix(prefix, f, v):
    return f(*v[1:]) if type(v) is list and v[0] == prefix else v


def parse_config_item(v):
    v = parse_prefix('(env)', parse_env, v)
    v = parse_prefix('(obj)', import_and_run_object, v)

    return v


def parse_config(config):
    if isinstance(config, dict):
        return {k: parse_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return parse_config_item([parse_config(v) for v in config])
    else:
        return config


def process_config(config):
    acc = torch.accelerator.current_accelerator()
    config.setup.distributed.device = f'{acc}:{config.setup.distributed.local_rank}'

    num_cpus = os.cpu_count()
    config.setup.distributed.num_threads = num_cpus // config.setup.distributed.local_world_size + \
        (1 if config.setup.distributed.local_rank > num_cpus % config.setup.distributed.local_world_size else 0)

    return config

def parse_omega_config(config):
    config = OmegaConf.to_container(config, resolve=True)
    config = parse_config(config)
    config = edict(config)

    return config


def load_config(path, cli_args):
    config = OmegaConf.load(path)
    extra_config = OmegaConf.from_cli(cli_args)
    config = OmegaConf.merge(config, extra_config)
    config = parse_omega_config(config)
    config_raw = edict_to_dict(config)
    config = process_config(config)

    return config, config_raw


def load_experiments_config(path, experiments_path, cli_args):
    global_config = OmegaConf.load(path)
    experiments_config = OmegaConf.load(experiments_path)
    extra_config = OmegaConf.from_cli(cli_args)

    extra_config = OmegaConf.merge(experiments_config['overrides'], extra_config)
    global_config = OmegaConf.merge(global_config, extra_config)

    setup_config = parse_omega_config(experiments_config['setup'])

    experiments = []
    for k, v in experiments_config['experiments'].items():
        group_name = f'e {k}'
        for k2, v2 in v.items():
            name = f'e {k2}'

            config = OmegaConf.merge(global_config, v2)
            config = parse_omega_config(config)

            config.train.logger.run_group_name = group_name
            config.train.logger.run_name = name
            config.train.checkpoints.path = os.path.join(setup_config.out_path, group_name, name)
            config.train.n_real_steps = setup_config.total_experiment_steps

            config_raw = edict_to_dict(config)
            config = process_config(config)

            experiments.append((config, config_raw))

    return experiments, setup_config
