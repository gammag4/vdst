import os
import asyncio
import argparse
from dotenv import load_dotenv
import torch
import gc

from utils.config import load_config, load_experiments_config
from run.runner import run_distributed
from run.vdst_trainer import VDSTTrainer


async def run_experiment(config, config_raw):
    assert 'cuda' in config.setup.distributed.device, 'non-CUDA devices not supported'
    
    trainer = VDSTTrainer(config, config_raw)
    await run_distributed(config, trainer.run)


async def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', help='Config file path', required=True)
    parser.add_argument('--experiments', help='Experiments file path')
    parser.add_argument('other', nargs='*')
    args = parser.parse_args()
    
    load_dotenv()
    
    if args.experiments:
        experiments, setup_config = load_experiments_config(args.config, args.experiments, args.other)
        
        os.makedirs(setup_config.out_path, exist_ok=True)
        experiments_checkpoint_path = os.path.join(setup_config.out_path, 'checkpoint.txt')
        
        try:
            with open(experiments_checkpoint_path, 'r', encoding='utf8') as f:
                res = f.read()
                if res == 'ended':
                    print('Experiments already ran.')
                    return
                run_group_name, run_name = res.split('\n')[:2]
                start_experiment_id = [e.train.logger.run_group_name == run_group_name and e.train.logger.run_name == run_name for e, c in experiments].index(True)
        except FileNotFoundError:
            start_experiment_id = 0
        
        print('Running experiments ...\n')
        
        for config, config_raw in experiments[start_experiment_id:]:
            with open(experiments_checkpoint_path, 'w', encoding='utf8') as f:
                f.write(f'{config.train.logger.run_group_name}\n{config.train.logger.run_name}')
            
            await run_experiment(config, config_raw)
            gc.collect()
            torch.cuda.empty_cache()
        
        with open(experiments_checkpoint_path, 'w', encoding='utf8') as f:
            f.write('ended')
        
        print('\nExperiments ended.')
    else:
        config, config_raw = load_config(args.config, args.other)
        await run_experiment(config, config_raw)


if __name__ == '__main__':
    asyncio.run(main())
    
    # TODO add dist.barrier()
    
    # torchrun already handles setting up env variables and launching processes on the appropriate nodes

    # TODO On using numactl with torchrun:
    # https://github.com/pytorch/pytorch/issues/115305#issuecomment-1845957682
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#numactl

    # Running single node:
    # torchrun --standalone --nproc-per-node=gpu main.py
    # --standalone: tells it is a single-machine setup
    # --nproc-per-node: num processes per node, can be a number, "gpu" which will create a process per gpu
    # train.py
    #   config_file: res/config.yaml

    # Running multi-node:
    # We run the command in each node, specifying how many nodes in total and the global rank of the current node
    # We also need to set rendezvous arguments to allow them to synchronize and communicate
    # torchrun --nproc-per-node=gpu --nnodes=2 --node-rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.31.43.139:29603 train.py
    # --nnodes: total num of nodes (can also be in format "min_nodes:max_nodes" where it looks for at least min_nodes and for at most max_nodes, also called elastic launch)
    # --node-rank: current node rank (between 0 and nnodes - 1)
    #   it seems like it does not need to be specified when using SLURM bc it already passes $SLURM_NODEID down
    # --rdzv-id: id for rendezvous protocol, random number
    # --rdzv-backend: backend for rendezvous protocol
    # --rdzv-endpoint: ip and port of any of the participating nodes
    #   the rendezvous backend is hosted here, so its best to choose the one with best bandwidth
    #   it should also be the same for all nodes
    # --max-restarts (optional): number of allowed failures or membership changes (details here: https://docs.pytorch.org/docs/stable/elastic/run.html#membership-changes)

    # E.g. on SLURM enabled cluster, we can find master endpoint by:
    # export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
    # Then we use that like:
    # --rdzv-endpoint=$MASTER_ADDR:29603
