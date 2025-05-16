# Omnifall Experiments

This repository contains the code for running benchmarking experiments on the Omnifall dataset, a comprehensive fall detection dataset superset combining multiple sources like cmdfall, up_fall, le2i, OOPS, and others.

**Currently this repository is provided for review. We provide additional instructions on how to run the experiments here in the supplementary material of our submitted paper.**

**There we also provide links to download the features of all datasets which are needed for these experiments.**

We might provide further convencience commits which do not change the experiments or functionality, but further improved documentation and facilitated setup [here](https://github.com/simplexsigil/omnifall-experiments).

## Overview

Omnifall Experiments provides a unified framework for:
- Training and evaluating fall detection models across multiple datasets
- Testing generalization capabilities across different environments
- Benchmarking various feature extractors (I3D, DINOv2, InternVideo2, VideoMAE)
- Detailed performance analysis with fall-specific metrics

The system uses pre-extracted features from various backbone models and trains a feature transformer to classify action segments into 10 classes (walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing, other).

## Setup

### Environment Setup

1. Create a conda environment using the provided environment file:
```bash
conda create -n omnifall python=3.12
conda activate omnifall
pip install -r requirements.txt
```


### Dataset Setup

1. Set the environment variable for the Omnifall root directory:
```bash
export OMNIFALL_ROOT="/path/to/omnifall/dataset"
```

2. The Omnifall dataset should have the following structure:
```
$OMNIFALL_ROOT/
├── cmdfall/
│   ├── features/
│   │   ├── i3d/
│   │   ├── dinov2/
│   │   ├── invid2-2fps/
│   │   ├── videomae-base-7d5fps-1f/
│   │   └── videomae-kinetics-7d5fps-1f/
├── up_fall/
│   ├── features/
│   │   ├── i3d/
│   │   └── ...
├── le2i/
│   ├── features/
│   │   ├── i3d/
│   │   └── ...
# ... other datasets following the same pattern
├── labels/
│   ├── dataset/
│   │   ├── cmdfall.csv
│   │   ├── up_fall.csv
│   │   ├── le2i.csv
│   │   └── ...
└── splits/
    ├── cs/
    │   ├── cmdfall/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── test.csv
    │   ├── up_fall/
    │   └── ...
```

## Running Experiments

### Basic Usage

To run an experiment, use the `feature_train_eval.py` script with a configuration:

```bash
python -m scripts.feature_train_eval \
    -cn feature_train_config.yaml \
    "+experiments/omnifall/i3d@_global_=omnifall_lab-all_i3d_18_layer_2"
```

This trains a 2-layer transformer model on staged fall datasets using I3D features with 18 frames per segment and evaluates on all datasets.

### Configuration Options

The system uses Hydra for configuration management. You can override configuration options on the command line:

```bash
python -m scripts.feature_train_eval \
    -cn feature_train_config.yaml \
    "+experiments/omnifall/i3d@_global_=omnifall_lab-all_i3d_18_layer_2" \
    "training.num_epochs=50" \
    "training.batch_size=128" \
    "model.num_layers=4"
```

### Available Experiments

Several pre-configured experiments are available in the [experiments config directory](configs/experiments).

#### I3D Experiments
- `omnifall_lab-all_i3d_18_layer_2`: 2-layer model with 18 features, cross-subject split.
- `omnifall_lab-all_i3d_18_layer_2_cv`: 2-layer model with 18 features, cross-view split.

#### VideoMAE Experiments
- `omnifall_lab-all_vmae-base_18_layer_2`: VideoMAE Base, 2-layer model with 18 features, cross-subject split.
- `omnifall_lab-all_vmae-base_18_layer_2_cv`: VideoMAE Base, 2-layer model with 18 features, cross-view split.
- 
- `omnifall_lab-all_vmae-kinetics_18_layer_2`: VideoMAE Kinetics, 2-layer model with 18 features, cross-subject split.
- `omnifall_lab-all_vmae-kinetics_18_layer_2_cv`: VideoMAE Kinetics, 2-layer model with 18 features, cross-view split.

### Creating Custom Configurations

To create a custom experiment configuration:

1. Create a new YAML file in `configs/experiments/omnifall/{feature_type}/`
2. Define your configuration by extending existing ones:

```yaml
defaults:
  - /dataset/omnifall/{feature_type}@dataset: feature-dataset-labonly
  - /dataset/omnifall/{feature_type}@dataset_val: feature-dataset-all
  - /dataset/omnifall/{feature_type}@dataset_test: feature-dataset-all
  - _self_

dataset:
  num_features: 24
  
model:
  num_layers: 4
  hidden_dim: 1024

wandb:
  project: "Omnifall Feature Transformer"
  name: "Custom Experiment Name"
```

## Evaluation

The system automatically evaluates models on the test set and reports:

- Multi-class metrics: accuracy, balanced accuracy, macro F1
- Binary fall detection metrics: sensitivity, specificity, F1 score
- Per-class F1 scores
- Domain-specific metrics for each dataset

Results are logged to Weights & Biases and saved to the output directory.

## Advanced Usage

### Cross-Dataset Evaluation

To train on one dataset and test on another:

```bash
python scripts/feature_train_eval.py \
    -cn feature_train_config.yaml \
    "+experiments/omnifall/i3d@_global_=omnifall_lab-all_i3d_18_layer_2" \
    "dataset.feature_datasets=[{name: 'cmdfall', ...}]" \
    "dataset_test.feature_datasets=[{name: 'OOPS', ...}]"
```

### Class-Balanced Loss

Enable class-balanced loss to handle class imbalance:

```bash
python scripts/feature_train_eval.py \
    -cn feature_train_config.yaml \
    "+experiments/omnifall/i3d@_global_=omnifall_lab-all_i3d_18_layer_2" \
    "training.use_class_balanced_loss=true" \
    "training.class_balanced_loss_beta=0.999"
```

### Domain-Weighted Sampling

Enable domain-weighted sampling to balance datasets:

```bash
python scripts/feature_train_eval.py \
    -cn feature_train_config.yaml \
    "+experiments/omnifall/i3d@_global_=omnifall_lab-all_i3d_18_layer_2" \
    "training.use_domain_weighted_sampler=true" \
    "training.domain_sampler_max_cap=10.0"
```

## Citation

Hopefully coming soon.