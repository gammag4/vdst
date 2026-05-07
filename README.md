# View-Depth Synthesis Transformer: A model for Novel View Synthesis with depth estimation

| English | [Português](README_PT.md) |

This is the implementation of the VDST model, together with code to train it with the datasets originally used.

This is an RGB-D Novel View Synthesis model, where given a set of images and depths from a 3D scene with their respective camera properties/poses,
the model aims to generate a new view/depth in the scene, given the camera properties and pose of the target view.

This model has two things in special in comparison to others:

- It is generalizable, meaning while older models needed to be retrained for every new scene, this one can be used in new scenes without being retrained;
- It minimizes inductive bias by using just a simple vision transformer right after stacking the camera views/depths together with their poses and breaking down into patches. That is why when it is trained with high-resolution images, the results look way better than other models.

## Training

To train it, download the datasets to `datsets/` folder and run:

```bash
conda create -n vdst python=3.13
conda activate vdst
pip install -r requirements.txt
torchrun --standalone --nproc-per-node=gpu train.py --config config.yaml
```
