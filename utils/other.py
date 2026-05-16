import math

from .io import try_run_cmd


def edict_to_dict(ed):
    if isinstance(ed, dict):
        return {k: edict_to_dict(v) for k, v in ed.items()}
    elif isinstance(ed, tuple):
        return tuple(edict_to_dict(v) for v in ed)
    elif isinstance(ed, list):
        return [edict_to_dict(v) for v in ed]
    else:
        return ed


def format_big_number(num):
    if num < 1:
        return f'{num}'

    suffixes = ['', 'K', 'M', 'B', 'T']
    i = min(len(suffixes) - 1, int(math.floor(math.log10(num) / 3)))
    unit = suffixes[i]
    num = num / (10 ** (3 * i))

    return f'{num:.2f}{unit}'


def print_model_stats(model, print_all_named_params=False):
    # TODO change to return model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    res = f'\nTotal params: {format_big_number(total_params)}; Trainable params: {format_big_number(trainable_params)}'
    if print_all_named_params:
        res += '\nTrainable params list:'
        for n, p in model.named_parameters():
            if p.requires_grad:
                res += f'\n\t{n}: {format_big_number(p.numel())}'

    res += '\n'

    print(res)


def find_unused_model_params(model):
    print('\nUnused parameters:')

    for n, p in model.named_parameters():
        if p.grad is None:
            print(n)


def get_folder_size(path):
    # Neither os or shutil give accurate results
    # TODO add windows equivalent
    return int(try_run_cmd(f'du -sb "{path}"', raise_err=True).split()[0])
