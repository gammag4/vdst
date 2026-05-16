# View-Depth Synthesis Transformer: A Transformer-based Model for RGB-D Novel View Synthesis

| English | [Português](README_PT.md) |

![Results (images)](results.png)
![Results (depths)](results_depths.png)

> [!NOTE]
> This model is currently under development/experimentation and is not yet complete.
> The results above are from the first experimental runs (after training for two days on an RTX 4060 Ti with 8GB vRAM).
> More details about these and other experiments can be found [here](https://wandb.ai/gammag9-none/vdst/runs/ob9jxu15).

This is the implementation of the VDST model, together with code to train it with the datasets originally used.

It is an RGB-D Novel View Synthesis model, where given a set of images and depths from a 3D scene with their respective camera properties/poses,
the model aims to generate a new view/depth in the scene, given the camera properties and pose of the target view.

We propose VDST to investigate the capability of Transformer-based models in solving the task of RGB-D Novel View Synthesis.
It is based primarily on the architecture of [LVSM](https://haian-jin.github.io/projects/LVSM/) and, just like it,
is also generalizable to novel scenes and also follows its philosophy of minimizing inductive bias.

## Training

To train it, download the datasets to `datsets/` folder and run:

```bash
conda create -n vdst python=3.13
conda activate vdst
pip install -r requirements.txt
torchrun --standalone --nproc-per-node=gpu train.py --config config.yaml
```
