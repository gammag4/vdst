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
This type of Transformer-based architecture for Novel View Synthesis is not new and was not originally proposed by us.
In our case, the model architecture and philosophy was inspired primarily on [LVSM](https://haian-jin.github.io/projects/LVSM/),
though there are other similar architectures as well, with [SRT](https://srt-paper.github.io/) being another relevant example.

These are the main advantages this model has in comparison to other methods:

- Due to the generalization ability of Transformer-based models across domains, it is capable of:
  - Generalizing to novel scenes that follow a similar distribution to the original training data;
  - Doing few-shot NVS, needing only two source images or even just a single one in most cases for generating new views;
- Following the same philosophy as LVSM, our architecture also tries to minimize the inductive bias of the model,
  and we hypothesize that this allows it to achieve better results than other methods when trained for longer periods with sufficiently large amounts of data,
  although we do not have enough computational resources to verify this, leaving it for future work;
- It can be trained under constrained resources without diverging (the author used a single RTX 4060 Ti with 8GB vRAM).

## Training

### Requirements

You need:

- Some conda distribution (we recommend using [Miniforge](https://conda-forge.org/download/))
- NVIDIA drivers that support CUDA >= 13.0

### Downloading and processing datasets

Download and process the WildRGB-D dataset using the script provided [here](https://github.com/gammag4/nvs_datasets) to the folder `datasets/wildrgbd`.

### Creating environment

Create the Python environment and install dependencies:

```sh
conda create -n vdst python=3.13
conda activate vdst
pip install -r requirements.txt
```

### Training the model

Run the train script:

```bash
torchrun --standalone --nproc-per-node=gpu train.py --config config.yaml
```

## Rendering

We also made a renderer that you can use to navigate on the scenes using this model, you can check it out [here](https://github.com/gammag4/nvs_renderer).

