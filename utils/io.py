import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import subprocess
import aiohttp
from easydict import EasyDict as edict


class _SafeOpenWrite:
    def __init__(self, path, mode='w+b', encoding='utf-8'):
        self.path = Path(path)
        os.makedirs(self.path.parent, exist_ok=True)
        self.file = NamedTemporaryFile(mode=mode, encoding=encoding, dir=self.path.parent, prefix=self.path.name, suffix='.tmp', delete=False)
    
    def __enter__(self):
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            os.replace(self.file.name, self.path)
        except:
            try:
                os.remove(self.file.name)
            except:
                pass

        return False


def safe_open_write(path, mode='w+b', encoding='utf-8'):
    return _SafeOpenWrite(path, mode, encoding)


def json_load(path, use_edict=True):
    with open(path, mode='r', encoding='utf-8') as f:
        res = json.load(f)
        if use_edict:
            if type(res) is list:
                return edict({'a': res}).a
            else:
                return edict(res)
        else:
            return res


def json_dump(path, data: edict | dict):
    with safe_open_write(path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)


async def json_get(url, use_edict=True):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            res = await resp.json()
            return edict(res) if use_edict else res


async def text_get(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()


def try_run_cmd(cmd, verbose=False, raise_err=False):
    # If raise_err, raises if error and only returns output if succeeds
    if verbose:
        print(f'Running "{cmd}"')
    
    if raise_err:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        return res.stdout
    
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print('Error:')
            print(e.stderr)
            print('Command output:')
            print(e.stdout)
        return False, None
    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return False, None

    return True, res.stdout
