# Chip Placement Using Diffusion Models

Vint Lee, Minh Nguyen, Leena Elzeiny, Chun Deng, Pieter Abbeel, John Wawrzynek
This repository is adapted from the original work at https://github.com/vint-1/chipdiffusion (“Chip Placement Using Diffusion Models” by Vint Lee, Minh Nguyen, Leena Elzeiny, Chun Deng, Pieter Abbeel, and John Wawrzynek).

[Paper](https://arxiv.org/abs/2407.12282)

![Teaser](/media/teaser.png "Diffusion process used to generate placement")

## Installation
Use conda environment found in `environment.yaml`
```
conda env create -f environment.yaml
conda activate chipdiffusion
```
On Windows PowerShell, set `PYTHONPATH` before running any repo commands:
```
$env:PYTHONPATH="."
```

Training and evaluation experiments will log data to Weights & Biases by default. Set your name and W&B project in the [config](diffusion/configs) files using the `logger.wandb_entity` and `logger.wandb_project` options before running the commands below. Turn off W&B logging by appending `logger.wandb=False` to the commands below.

For running evaluations that require clustering, download [shmetis and hmetis](http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview) and place in the repo's root directory.

## Directory Structure
* [diffusion](diffusion) contains code for training, fine-tuning, and evaluating models
* [data-gen](data-gen) contains code for generating synthetic datasets
* [data-gen/outputs](data-gen/outputs) will be used to store the generated datasets
* [datasets](datasets) is used for other datasets (like IBM and ISPD benchmarks).
* [notebooks](notebooks) has useful scripts and functions for evaluating placements (measuring congestion in particular) and inspecting benchmark files.
* [parsing](parsing) has scripts for converting and clustering benchmarks in the DEF/LEF format (such as IBM).

## Local Setup Status
* Pre-trained `large-v2` checkpoint is already available locally; ensure PyTorch and torchvision are installed with CUDA-compatible versions before running GPU jobs.
* ISPD2005 benchmark files are installed and ready for evaluation.
* Generated datasets currently live under `data-gen/outputs`: `v0.61` and `vertex_0.7x.61`.

## Usage

### Data Generation
Generate `v0`, `v1`, and `v2` datasets for training:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v0


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v1


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=5000 num_val_samples=2500
```

Configs are also provided for running dataset design experiments. Since we only use these for evaluation, not for training, we only need to generate a few circuits:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=vertex-0.7x num_train_samples=0 num_val_samples=200


PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=distribution-linear num_train_samples=0 num_val_samples=200
```

For experiments on scale factor, the scale factor has to be specified by including `gen_params.edge_dist.dist_params.scale=<SCALE_FACTOR>`. For example:
```
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=scale gen_params.edge_dist.dist_params.scale=0.8 num_train_samples=0 num_val_samples=200
```

For easier debugging, use `data-gen/generate.py`.

### Training Models
After generating data, we train models on the `v1` dataset:

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_large task=v1.61
```

We can train smaller models using:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_medium task=v1.61 model/size@model.backbone_params=medium


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=train_small task=v1.61 model/size@model.backbone_params=small
```

### Fine-tuning
Once the models have been trained, we can fine-tune them on `v2`:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/train_graph.py method=finetune_large task=v2.61 mode@_global_=finetune from_checkpoint=v1.61.train_large.61/step_3000000.ckpt
```

### Generating Samples

Evaluating on `v1` dataset without guidance:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py task=v1.61 method=eval from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=none guidance@_global_=none num_output_samples=128
```

Evaluating zero-shot on clustered IBM benchmark with guidance:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt num_output_samples=18
```

Macro-only evaluation for IBM and ISPD benchmarks:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ibm.cluster512.v1 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 from_checkpoint=v2.61.finetune_large.61/step_250000.ckpt legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True
```

Running the installed pre-trained checkpoint on the ISPD2005 benchmark (macro-only):
```
python diffusion/eval.py method=eval_macro_only task=ispd2005-s0 from_checkpoint=large-v2/large-v2.ckpt legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True
```

Examples of generated placements, for both clustered and macro-only settings, can be found [here](placements).

### Simulated Annealing + Diffusion Hybrid
`diffusion/sa_hybrid.py` runs simulated annealing with optional diffusion moves (if you provide a checkpoint that supports `reverse_samples`). Outputs live in `logs/<task>.sa_hybrid.<seed>/samples`.

Example (PowerShell):
```
$env:PYTHONPATH="."
python diffusion/sa_hybrid.py task=ispd2005-s0 sa_mode=hybrid sa_steps=500 temp_init=1.0 temp_decay=0.995 diffusion.k_steps=8 diffusion.noise_level=0.1
```
Example (PowerShell, explicit model path):
```
python diffusion/sa_hybrid.py task=vertex_0.7x.61 sa_mode=hybrid sa_steps=500 temp_init=1.0 temp_decay=0.995 diffusion.mode=cont diffusion.k_steps=8 diffusion.noise_level=0.1 diffusion.model_path="d:/eda_final/chip_diffusion/logs/diffusion_debug/large-v2/large-v2.ckpt"
```
Use `sa_mode=sa_only` to disable diffusion moves, or set `diffusion.model_path` to point at a saved model.

### Visualize Placement Pickles
`scripts/visualize_placement_pkls.py` renders placement `.pkl` files (from `eval.py` or `sa_hybrid.py`) to PNGs, matching them with the corresponding graph metadata.

Example (PowerShell):
```
$env:PYTHONPATH="."
python scripts/visualize_placement_pkls.py --task vertex_0.7x.61 --pkl logs/vertex_0.7x.61.sa_hybrid.0/samples/best0.pkl --save-ref
```
Use `--pkl-dir` with `--pattern "best*.pkl"` to batch-convert a folder.

## Dataset Format
Input netlist is stored using PyTorch-Geometric's [Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch-geometric-data-data) object.

Input placements for training are stored as numpy arrays.

Placement outputs are saved as pickle files containing a single numpy array with (x, y) coordinates for each object.

## Benchmarks
To obtain the IBM dataset, download the benchmark in DEF/LEF format to `benchmarks/ibm` and run the code in [parsing](parsing):
```
PYTHONPATH=. python parsing/cluster.py

PYTHONPATH=. python parsing/cluster.py num_clusters=0
```
The code will parse the DEF/LEFs, cluster the netlists if `num_clusters` is non-zero, then output the dataset as pickle files to `datasets/clustered` directory.

To obtain the ISPD dataset for running evaluations, download the ISPD benchmark in bookshelf format to `benchmarks/ispd2005`, then use [this notebook](notebooks/parse_bookshelf.ipynb).

Once the benchmark files have been generated, copy [this config](datasets/graph/config.yaml) into the benchmark directory, and change `val_samples` as needed.

## Pre-trained Models
For convenience, we provide the training checkpoint for the *Large+v2* model at [this link](https://drive.google.com/drive/folders/16b8RkVwMqcrlV_55JKwgprv-DevZOX8v?usp=sharing). To use it, copy the `large-v2` directory into your `logs` directory and specify `from_checkpoint` accordingly when running the commands above. 

Note that if the checkpoint loads correctly, the code will print `successfully loaded state dict for model` before running training or evaluation; otherwise, `successfully loaded model` will be printed instead, and the code will default to random model weights. Hyperparameter mismatch is a common cause of failure, and we provide the training config used for reference.

## Citation
If you found our work useful, please cite:
```
@inproceedings{
      lee2025chipdiffusion,
      title={Chip Placement with Diffusion Models},
      author={Vint Lee and Minh Nguyen and Leena Elzeiny and Chun Deng and Pieter Abbeel and John Wawrzynek},
      booktitle={Forty-second International Conference on Machine Learning},
      year={2025},
      url={https://arxiv.org/abs/2407.12282}
}
```
