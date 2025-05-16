#!/usr/bin/env python

import logging
import os
import sys
import psutil
import math
import time
import torch
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
from torchinfo import summary

from utils.utils import generate_latex_rows

# ---------- autodetect + heuristics for better threading ----------
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)
reserve = max(2, math.ceil(logical_cores * 0.04))  # 4% headroom (≥2)
threads = max(2, (logical_cores - reserve) // (16 + 1))

# ---------- apply settings ----------
torch.set_num_threads(threads)
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)

import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb
import yaml
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    TrainerCallback,
)
import evaluate
from evaluate.module import EvaluationModule
from torch.utils.data import WeightedRandomSampler

from training.ta_trainer import ThreadAwareTrainer as Trainer
from data.feature_dataset import FeatureVideoDataset, MultiFeatureDataset, label2idx, idx2label
from models.feature_transformer import FeatureTransformer, PositionalEncoding, FeatureTransformerWrapper
from models.cb_loss import ClassBalancedLoss
import utils.assertions as assertions
from utils.sampling import get_domain_weighted_sampler
from balanced_accuracy.balanced_accuracy import BalancedAccuracy
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

# ---------- SETUP LOGGING ----------
logging.basicConfig(
    filename="local_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- DATASET FUNCTIONS ----------
def get_feature_datasets(
    cfg: DictConfig, mode: str, run: Optional[Any] = None, return_individual: bool = False, split="cs"
) -> Union[MultiFeatureDataset, Dict[str, Any]]:
    """
    Create and return feature datasets based on configuration.

    Args:
        cfg: Hydra configuration
        mode: Dataset mode ('train', 'val', 'test')
        run: WandB run for logging
        return_individual: Whether to return individual datasets as dict (for evaluation)

    Returns:
        If return_individual is False:
            MultiFeatureDataset instance
        If return_individual is True:
            Dict with:
                'combined': MultiFeatureDataset instance
                'individual': Dict of dataset_name -> FeatureVideoDataset instances
    """
    datasets = {}

    # Select the appropriate dataset configuration based on mode
    # Fall back to dataset if specific mode config is not available
    if mode == "train" and hasattr(cfg, "dataset_train"):
        dataset_config = cfg.dataset_train
        logger.info(f"Using specific {mode} dataset configuration")
    elif mode == "val" and hasattr(cfg, "dataset_val"):
        dataset_config = cfg.dataset_val
        logger.info(f"Using specific {mode} dataset configuration")
    elif mode == "test" and hasattr(cfg, "dataset_test"):
        dataset_config = cfg.dataset_test
        logger.info(f"Using specific {mode} dataset configuration")
    else:
        dataset_config = cfg.dataset
        logger.info(f"Using default dataset configuration for {mode}")

    for ds_config in dataset_config.feature_datasets:
        dataset = FeatureVideoDataset(
            feature_root=ds_config.feature_root,
            annotations_file=ds_config.annotations_file,
            feature_fps=ds_config.feature_fps,
            feature_frames=ds_config.frames_per_feature,
            feature_stride=ds_config.feature_stride,
            feature_centered=ds_config.feature_centered,
            feature_type=ds_config.feature_type,
            dataset_name=ds_config.name,
            mode=mode,
            feature_ext=dataset_config.feature_ext,
            split_root=ds_config.split_root,
            num_features=dataset_config.num_features,
            get_features=True,
            split=split,
        )
        if len(dataset) > 0:
            datasets[ds_config.name] = dataset
            logger.info(f"Added dataset {ds_config.name} with {len(dataset)} segments for {mode} split")

            if run is not None and run.id:
                run.log({f"{ds_config.name}_{mode}_size": len(dataset)})
        else:
            logger.warning(f"Dataset {ds_config.name} is empty for {mode} split. Skipping.")
            if run is not None and run.id:
                run.log({f"{ds_config.name}_{mode}_size": 0})

    if not datasets:
        raise ValueError(f"No datasets could be loaded for {mode} split")

    # Convert dict_values to list to make it picklable
    dataset_list = list(datasets.values())
    multi_dataset = MultiFeatureDataset(dataset_list)
    logger.info(f"Created combined dataset with {len(multi_dataset)} total segments for {mode} split")

    if return_individual:
        individual_datasets = {k: v for k, v in datasets.items()}
        return {"combined": multi_dataset, "individual": individual_datasets}
    else:
        return multi_dataset


# ---------- MODEL FUNCTIONS ----------
def get_model(
    cfg: DictConfig, mode: str = "train", strict_loading: bool = False, return_wrapped: bool = True
) -> Union[FeatureTransformerWrapper, FeatureTransformer]:
    """
    Create and return the feature transformer model based on configuration.

    Args:
        cfg: Hydra configuration
        mode: Dataset mode ('train', 'val', 'test') to determine which dataset config to use
        strict_loading: Whether to use strict state dict loading (True for evaluation, False for training)
        return_wrapped: Whether to return a wrapped model (compatible with Trainer) or the base model

    Returns:
        FeatureTransformerWrapper or FeatureTransformer model based on return_wrapped
    """
    # Select the appropriate dataset configuration based on mode
    if mode == "train" and hasattr(cfg, "dataset_train"):
        dataset_config = cfg.dataset_train
    elif mode == "val" and hasattr(cfg, "dataset_val"):
        dataset_config = cfg.dataset_val
    elif mode == "test" and hasattr(cfg, "dataset_test"):
        dataset_config = cfg.dataset_test
    else:
        dataset_config = cfg.dataset

    # Create the base model with configuration
    model = FeatureTransformer(
        feature_dim=cfg.model.feature_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=dataset_config.num_classes,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        mlp_ratio=cfg.model.mlp_ratio,
        num_features=dataset_config.num_features * dataset_config.tokens_per_feature,
    )

    # Create wrapper model if requested
    wrapped_model = FeatureTransformerWrapper(model) if return_wrapped else None
    target_model = wrapped_model if return_wrapped else model

    # Load checkpoint if specified
    if cfg.model.get("checkpoint_path", None) is not None:
        checkpoint_path = cfg.model.checkpoint_path
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Load state dict with safetensors if possible
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(checkpoint_path)
                logger.info("Loaded checkpoint using safetensors")
            else:
                # Fall back to PyTorch loading
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Loaded model_state_dict from checkpoint")
                else:
                    state_dict = checkpoint
                    logger.info("Loaded state_dict directly from checkpoint")

            # Try loading the state dict to the wrapped model first (for Trainer-saved checkpoints)
            if return_wrapped:
                try:
                    result = target_model.load_state_dict(state_dict, strict=strict_loading)
                    logger.info("Successfully loaded state dict to wrapped model")
                except Exception as e:
                    # If loading to wrapper fails, try to load to the base model instead
                    logger.info(f"Could not load state dict to wrapped model: {e}")
                    logger.info("Attempting to load state dict to the base model")
                    result = model.load_state_dict(state_dict, strict=strict_loading)
                    logger.info("Successfully loaded state dict to base model")
            else:
                # Load directly to the base model
                result = model.load_state_dict(state_dict, strict=strict_loading)
                logger.info("Successfully loaded state dict to base model")

            # Log warning for non-strict loading with missing or unexpected keys
            if not strict_loading and (result.missing_keys or result.unexpected_keys):
                if result.missing_keys:
                    logger.warning(f"Missing keys when loading checkpoint: {result.missing_keys}")
                if result.unexpected_keys:
                    logger.warning(f"Unexpected keys when loading checkpoint: {result.unexpected_keys}")

            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")

            # Ensure the model has the correct number of classes based on the current dataset
            if model.num_classes != dataset_config.num_classes:
                logger.info(
                    f"Replacing classifier to match dataset classes: {model.num_classes} -> {dataset_config.num_classes}"
                )
                model.classifier = nn.Linear(model.hidden_dim, dataset_config.num_classes)
                model.num_classes = dataset_config.num_classes

            # Ensure the model has the correct number of frames based on the current dataset
            if model.num_features != dataset_config.num_features:
                logger.info(f"Updating num_frames parameter: {model.num_features} -> {dataset_config.num_features}")
                model.num_features = dataset_config.num_features
                # Update positional encoding if needed
                model.pos_encoder = PositionalEncoding(model.hidden_dim, max_len=dataset_config.num_features)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Using initialized model without loading weights")

    return target_model


# ---------- DATA PROCESSING FUNCTIONS ----------
def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for feature batches.

    Args:
        examples: List of examples

    Returns:
        Dictionary with collated tensors and other data
    """
    if not examples:
        return {}

    batch = {}
    # Iterate over all keys in the first example, assuming all examples have the same keys
    for key in examples[0].keys():
        if key == "rgb_features":
            batch["features"] = torch.stack([example[key] for example in examples])
        elif key == "label":
            batch["labels"] = torch.tensor([example[key] for example in examples], dtype=torch.long)
        else:
            # For other keys, collect their values
            values = [example[key] for example in examples]
            # Check if the values are numeric (int, float) and convert to tensor
            if values and isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            # Check if the values are already tensors (e.g., some pre-processed metadata)
            elif values and isinstance(values[0], torch.Tensor):
                try:
                    # Attempt to stack if they are tensors (e.g. each example has a 1D tensor for this key)
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # If stacking fails (e.g., tensors of different shapes), keep as a list
                    batch[key] = values
            else:
                # For non-numeric types (e.g., strings) or if unsure, keep as a list
                batch[key] = values

    # Ensure 'features' and 'labels' are present if 'rgb_features' and 'label' were in the input
    # This handles the case where the input might not have 'rgb_features' but the model expects 'features'
    if "rgb_features" in examples[0] and "features" not in batch:
        batch["features"] = torch.stack([example["rgb_features"] for example in examples])
    if "label" in examples[0] and "labels" not in batch:
        batch["labels"] = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return batch


# ---------- TRAINING CONFIGURATION FUNCTIONS ----------
def scaled_lr(cfg: DictConfig, accelerator: Optional[Accelerator] = None) -> float:
    """
    Calculate the scaled learning rate based on batch size.

    Args:
        cfg: Hydra configuration
        accelerator: Accelerator instance

    Returns:
        Scaled learning rate
    """
    # Calculate the scaled learning rate
    base_lr = cfg.training.optimizer.get("BASE_LR", 1e-4)

    # Get global batch size (accounting for all processes)
    per_device_batch_size = cfg.training.batch_size
    if accelerator is not None:
        # When using accelerator, use its process count
        num_processes = accelerator.num_processes
    else:
        # Fallback if accelerator not provided
        num_processes = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    global_batch_size = per_device_batch_size * num_processes
    scaled_lr = base_lr * global_batch_size / 256

    if not accelerator or (accelerator and accelerator.is_main_process):
        logger.info(f"Base learning rate: {base_lr}")
        logger.info(f"Local batch size: {per_device_batch_size}")
        logger.info(f"Global batch size: {global_batch_size}")
        if scaled_lr < base_lr:
            logger.warning(
                f"Scaled learning rate ({scaled_lr}) is less than base learning rate ({base_lr}). Using base_lr as lower bound."
            )
        else:
            logger.info(f"Scaled learning rate (base * global_batch_size / 256): {scaled_lr}")

    # Ensure LR doesn't go below base value
    if scaled_lr < base_lr:
        scaled_lr = base_lr

    return scaled_lr


def get_training_args(
    cfg: DictConfig, output_dir: str, accelerator: Optional[Accelerator] = None, run_name: Optional[str] = None
) -> TrainingArguments:
    """
    Create and return training arguments.

    Args:
        cfg: Hydra configuration
        output_dir: Output directory for checkpoints
        accelerator: Accelerator instance
        run_name: Optional run name for wandb logging

    Returns:
        TrainingArguments instance
    """
    lr = scaled_lr(cfg, accelerator)

    args = TrainingArguments(
        output_dir,
        do_train=cfg.get("do_train", True),
        do_eval=cfg.get("do_train", True),
        remove_unused_columns=False,
        eval_strategy="steps" if cfg.get("do_train", True) else "no",
        save_strategy="steps",
        eval_steps=cfg.training.eval_stride,
        save_steps=cfg.training.save_stride,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        warmup_ratio=0.05,
        logging_steps=10,
        load_best_model_at_end=True if cfg.get("do_train", True) else False,
        metric_for_best_model="eval_OOPS_fall_f1",  # accuracy
        push_to_hub=False,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=cfg.hardware.get("num_data_workers", 4),
        ddp_find_unused_parameters=False,
        seed=42,
        lr_scheduler_type=cfg.training.scheduler.name,
        max_grad_norm=cfg.training.optimizer.get("max_grad_norm", 1.0),
        label_smoothing_factor=0.1,
        dataloader_persistent_workers=True,
        weight_decay=cfg.training.optimizer.get("weight_decay", 0.01),
    )

    return args


# ---------- METRICS COMPUTATION FUNCTIONS ----------


def compute_metrics_factory(node_id) -> Callable:
    """
    Factory function to create a compute_metrics function with access to metrics objects.

    Args:
        metric_acc: Accuracy metric instance
        metric_bacc: Balanced accuracy metric instance

    Returns:
        Compute metrics function to be used by the Trainer
    """

    def create_metrics(node_id: str) -> Tuple[EvaluationModule, BalancedAccuracy]:
        """
        Create metrics objects with unique experiment IDs.

        Args:
            node_id: Unique node identifier for metrics

        Returns:
            Tuple of (accuracy metric, balanced accuracy metric)
        """
        metric_acc = evaluate.load("accuracy", experiment_id=f"accuracy_{node_id}")
        metric_bacc = BalancedAccuracy(experiment_id=f"balanced_accuracy_{node_id}")
        return metric_acc, metric_bacc

    # Create metrics objects
    metric_acc, metric_bacc = create_metrics(node_id)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute metrics for evaluation:
        - 10-class: accuracy, balanced accuracy, macro F1
        - Per-class: F1 score for each class
        - Binary (fall vs non-fall): sensitivity, specificity, F1
        - Binary (fallen vs non-fallen): sensitivity, specificity, F1
        - Binary (fall ∪ fallen vs others): sensitivity, specificity, F1
        - Class distribution (percentage of each class)
        This function is used by the Trainer for both combined and individual dataset evaluation.
        Dataset-specific metrics will be handled by prepending the dataset name in the evaluation code.
        Args:
            eval_pred: EvalPrediction object containing predictions and labels
        Returns:
            Dictionary of metrics
        """
        # Get predictions and references
        predictions = np.argmax(eval_pred.predictions, axis=1)
        references = eval_pred.label_ids
        # Multi-class metrics (10-class)
        acc = metric_acc.compute(predictions=predictions, references=references)["accuracy"]
        bacc = metric_bacc.compute(predictions=predictions, references=references)["balanced_accuracy"]
        macro_f1 = f1_score(references, predictions, average="macro", zero_division=0)
        # Class distribution (percentage of each class)
        unique, counts = np.unique(references, return_counts=True)
        class_distribution = {
            idx2label.get(int(label), f"unknown_{label}"): count / len(references)
            for label, count in zip(unique, counts)
        }
        # Per-class F1 scores
        per_class_f1_scores = f1_score(references, predictions, average=None, zero_division=0)
        per_class_f1_dict = {
            idx2label.get(i, f"unknown_{i}_f1"): score for i, score in enumerate(per_class_f1_scores) if i in unique
        }
        # Binary metrics for fall (class 1)
        # Create binary representations
        binary_fall_pred = (predictions == 1).astype(int)
        binary_fall_ref = (references == 1).astype(int)
        # Check if we have any fall examples in the dataset
        if 1 in unique:
            # Use sklearn for binary metrics
            # precision_recall_fscore_support returns: precision, recall, fbeta_score, support
            _, fall_sensitivity, fall_f1, _ = precision_recall_fscore_support(
                binary_fall_ref, binary_fall_pred, average="binary", beta=1, zero_division=0
            )
            # Specificity = TN / (TN + FP) = recall of the negative class
            fall_specificity = recall_score(1 - binary_fall_ref, 1 - binary_fall_pred, zero_division=0)
        else:
            # No fall examples in this dataset
            fall_sensitivity, fall_f1, fall_specificity = 0, 0, 0
        # Binary metrics for fallen (class 2)
        binary_fallen_pred = (predictions == 2).astype(int)
        binary_fallen_ref = (references == 2).astype(int)
        # Check if we have any fallen examples in the dataset
        if 2 in unique:
            _, fallen_sensitivity, fallen_f1, _ = precision_recall_fscore_support(
                binary_fallen_ref, binary_fallen_pred, average="binary", beta=1, zero_division=0
            )
            fallen_specificity = recall_score(1 - binary_fallen_ref, 1 - binary_fallen_pred, zero_division=0)
        else:
            # No fallen examples in this dataset
            fallen_sensitivity, fallen_f1, fallen_specificity = 0, 0, 0

        # Binary metrics for fall ∪ fallen (classes 1 or 2)
        binary_fall_union_fallen_pred = ((predictions == 1) | (predictions == 2)).astype(int)
        binary_fall_union_fallen_ref = ((references == 1) | (references == 2)).astype(int)
        # Check if we have any fall or fallen examples in the dataset
        if 1 in unique or 2 in unique:
            _, fall_union_fallen_sensitivity, fall_union_fallen_f1, _ = precision_recall_fscore_support(
                binary_fall_union_fallen_ref, binary_fall_union_fallen_pred, average="binary", beta=1, zero_division=0
            )
            # Specificity = TN / (TN + FP) = recall of the negative class
            fall_union_fallen_specificity = recall_score(
                1 - binary_fall_union_fallen_ref, 1 - binary_fall_union_fallen_pred, zero_division=0
            )
        else:
            # No fall or fallen examples in this dataset
            fall_union_fallen_sensitivity = 0
            fall_union_fallen_f1 = 0
            fall_union_fallen_specificity = 0

        # Add sample count to metrics
        metrics_dict = {
            # Multi-class metrics
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            # Binary metrics: fall vs non-fall
            "fall_sensitivity": fall_sensitivity,
            "fall_specificity": fall_specificity,
            "fall_f1": fall_f1,
            # Binary metrics: fallen vs non-fallen
            "fallen_sensitivity": fallen_sensitivity,
            "fallen_specificity": fallen_specificity,
            "fallen_f1": fallen_f1,
            # Binary metrics: fall ∪ fallen vs others
            "fall_union_fallen_sensitivity": fall_union_fallen_sensitivity,
            "fall_union_fallen_specificity": fall_union_fallen_specificity,
            "fall_union_fallen_f1": fall_union_fallen_f1,
        }
        # Add class distribution and per-class F1 to the metrics dictionary
        metrics_dict.update({f"dist_{k}": v for k, v in class_distribution.items()})
        metrics_dict.update(per_class_f1_dict)  # This already contains per-class F1s
        metrics_dict["sample_count"] = len(references)

        return metrics_dict

    return compute_metrics


# ---------- EVALUATION FUNCTIONS ----------
def test_model(
    cfg: DictConfig, model: FeatureTransformer, trainer: Trainer, accelerator: Accelerator, run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation function for the model.

    Args:
        cfg: Hydra configuration
        model: Model to evaluate
        trainer: Trainer instance with model loaded
        accelerator: Accelerator instance
        run: WandB run for logging

    Returns:
        Dictionary with evaluation results
    """
    # Get both combined and individual test datasets
    test_datasets = get_feature_datasets(cfg, "test", run, return_individual=True, split=cfg.dataset.get("split", "cs"))

    # TODO: Issue / pull request: trainer can not handle multiple datasets with persistent workers
    # I persists the workers for the first dataset and then loads always data for that one, despite
    # reporting the other dataset
    # Persistent workers are useless in that case anyways, but the option is mainly set for training.
    trainer.args.dataloader_persistent_workers = False

    # First evaluate on the combined dataset
    logger.info("Starting evaluation on combined test dataset")
    combined_results = trainer.evaluate(test_datasets["combined"])

    # Then evaluate on individual datasets
    individual_results = {}
    for dataset_name, dataset in test_datasets["individual"].items():
        logger.info(f"Evaluating on individual dataset: {dataset_name}")
        try:
            results = trainer.evaluate(dataset)
            individual_results[dataset_name] = results
            logger.info(f"Test results for {dataset_name}: {results}")

            # Log individual dataset results
            if accelerator.is_main_process and run:
                wandb.log({f"Test results {dataset_name}": str(results)})
                # Log individual metrics for each dataset
                for k, v in results.items():
                    wandb.log({f"{dataset_name}_{k}": v})
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_name}: {e}")
            individual_results[dataset_name] = {"error": str(e)}

    # Calculate per-class metrics
    # This can be extended further as needed

    # Compile all results
    all_results = {"combined": combined_results, **individual_results}

    logger.info(f"Results:\n{all_results}")

    # Save results to file
    try:
        if accelerator.is_main_process:
            results_dir = os.path.join(trainer.args.output_dir, "evaluation_results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"test_results_{time.strftime('%Y%m%d-%H%M%S')}.yaml")
            with open(results_file, "w") as f:
                yaml.dump(all_results, f, default_flow_style=False)
            logger.info(f"Saved evaluation results to {results_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")

    # Log all results together as a dictionary for easy comparison
    if accelerator.is_main_process and run:
        wandb.log(data={"All test results": str(all_results)})

    generate_latex_rows(all_results)

    return all_results


def load_best_model_for_evaluation(cfg: DictConfig, trainer: Trainer, output_dir: str) -> None:
    """
    Load the best model for evaluation based on configuration.

    Args:
        cfg: Hydra configuration
        trainer: Trainer instance
        output_dir: Output directory where checkpoints are saved

    Returns:
        None - Updates trainer's model in-place
    """
    # If a specific evaluation checkpoint is provided, use it
    if cfg.get("eval_checkpoint_path"):
        try:
            logger.info(f"Loading specified checkpoint for evaluation: {cfg.eval_checkpoint_path}")
            # Get a wrapped model directly using our enhanced get_model function
            model = get_model(cfg, mode="test", strict_loading=False, return_wrapped=True)
            trainer.model = model
            logger.info(f"Successfully loaded checkpoint from {cfg.eval_checkpoint_path}")
            return
        except Exception as e:
            logger.error(f"Failed to load specified evaluation checkpoint: {e}")
            logger.info("Falling back to best checkpoint from training")

    # Try to find the best checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint-best")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading best checkpoint from {checkpoint_path}")
        trainer.model = trainer._load_best_model()
    else:
        # If no best checkpoint, try the latest
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            logger.info(f"No best checkpoint found. Loading latest checkpoint: {latest_checkpoint}")

            try:
                # Get the wrapped model with default options
                model = get_model(cfg, mode="test", strict_loading=False, return_wrapped=True)

                # Attempt to load the checkpoint file directly
                state_dict_path = os.path.join(latest_checkpoint, "pytorch_model.bin")
                if os.path.exists(state_dict_path):
                    checkpoint = torch.load(state_dict_path, map_location="cpu")
                    result = model.load_state_dict(checkpoint, strict=False)

                    # Log warning for missing or unexpected keys
                    if result.missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {result.missing_keys}")
                    if result.unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {result.unexpected_keys}")

                    trainer.model = model
                    logger.info(f"Successfully loaded checkpoint from {state_dict_path}")
                    return

                # Try safetensors if available
                state_dict_path = os.path.join(latest_checkpoint, "model.safetensors")
                if os.path.exists(state_dict_path):
                    from safetensors.torch import load_file

                    state_dict = load_file(state_dict_path)
                    result = model.load_state_dict(state_dict, strict=False)

                    # Log warning for missing or unexpected keys
                    if result.missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {result.missing_keys}")
                    if result.unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {result.unexpected_keys}")

                    trainer.model = model
                    logger.info(f"Successfully loaded checkpoint from {state_dict_path}")
                    return

                # If we couldn't load the checkpoint directly, use the original trainer method
                logger.info("Falling back to trainer's method to load the model")
                trainer.model = model
                trainer.model.load_state_dict(
                    torch.load(os.path.join(latest_checkpoint, "pytorch_model.bin"), map_location=trainer.args.device),
                    strict=False,
                )
                logger.info(f"Loaded model using fallback method from {latest_checkpoint}")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.warning("Using current model for evaluation.")
        else:
            logger.warning("No checkpoints found. Using current model for evaluation.")


def describe(model, input_size=(1, 9, 2048), depth=7):
    """Pretty‑print a readable model summary."""
    return summary(
        model,
        input_size=input_size,  # batch‑first input shape
        col_names=("input_size", "output_size", "num_params", "mult_adds"),  # what to show
        depth=depth,  # full module depth
        row_settings=("var_names",),  # show layer names
        verbose=0,
    )


# ---------- MAIN EXECUTION FUNCTION ----------
def run(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Main training and evaluation function.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with evaluation results if evaluation was performed
    """
    try:
        # Instantiate Accelerator
        accelerator = Accelerator()

        # Initialize wandb for the main process
        if accelerator.is_main_process:
            # Initialize wandb run
            job_cfg = HydraConfig.get().job

            # Create a descriptive run name
            if cfg.wandb.get("name", None):
                # Use name from config if available
                base_name = cfg.wandb.name
            else:
                # Create name from config parameters
                model_info = f"L{cfg.model.num_layers}_H{cfg.model.hidden_dim}"
                dataset_info = f"F{cfg.dataset.num_features}"
                base_name = f"ftransformer_{model_info}_{dataset_info}"

            # Add unique ID to prevent collisions
            run_name = f"{base_name} {wandb.util.generate_id()}"

            run = wandb.init(
                project=str(cfg.wandb.project),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                settings=wandb.Settings(start_method="thread"),
            )
            wandb.log(data={"Working Directory": os.getcwd()})
        else:
            run = None  # Ensure run is None on non-main processes

        # Log working directory
        logger.info(f"Current working directory: {os.getcwd()}")

        logger.info("Configuration summary:")
        logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

        # Log accelerator setup information
        logger.info("Accelerator setup:")
        logger.info(f"- Number of processes: {accelerator.num_processes}")
        logger.info(f"- Process index: {accelerator.process_index}")
        logger.info(f"- Device: {accelerator.device}")
        logger.info(f"- Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"- Distributed type: {accelerator.distributed_type}")

        if accelerator.distributed_type != "NO":
            logger.info(f"- Local process index: {accelerator.local_process_index}")
            logger.info(f"- Is local main process: {accelerator.is_local_main_process}")
            logger.info(f"- Is main process: {accelerator.is_main_process}")
            if hasattr(accelerator.state, "num_processes_per_node"):
                logger.info(f"- Processes per node: {accelerator.state.num_processes_per_node}")
            if hasattr(accelerator.state, "num_nodes"):
                logger.info(f"- Number of nodes: {accelerator.state.num_nodes}")

        # Create model with appropriate configuration
        # Use train mode config by default if training, test mode config if only testing
        model_mode = "train" if cfg.get("do_train", True) else "test"
        # Use strict loading for evaluation, non-strict for training
        strict_loading = not cfg.get("do_train", True)
        # Get the model, already wrapped for Trainer compatibility
        wrapped_model = get_model(cfg, mode=model_mode, strict_loading=strict_loading, return_wrapped=True)

        logging.info(
            f"Model:\n {describe(wrapped_model.model, input_size=(cfg.training.batch_size, cfg.dataset.num_features, cfg.model.feature_dim), depth=7)}"
        )

        # Get datasets for training and validation if needed
        train_dataset = None
        val_dataset = None

        if cfg.get("do_train", True):
            try:
                logger.info("Loading training dataset")
                train_dataset = get_feature_datasets(cfg, "train", run, split=cfg.dataset.get("split", "cs"))
                logger.info(f"Training dataset loaded with {len(train_dataset)} samples")
            except Exception as e:
                logger.error(f"Error loading training dataset: {e}")
                if not cfg.get("do_test", True):  # Only raise if we're not in test-only mode
                    raise

            try:
                logger.info("Loading validation dataset")
                val_dataset = get_feature_datasets(
                    cfg, "val", run, return_individual=True, split=cfg.dataset.get("split", "cs")
                )
                logger.info(f"Validation dataset loaded with {len(val_dataset['combined'])} samples")
                logger.info(f"Loaded individual validation datasets: {list(val_dataset.keys())}")

                if "individual" in val_dataset and "OOPS" in val_dataset["individual"]:
                    val_dataset = {"combined": val_dataset["combined"], "OOPS": val_dataset["individual"]["OOPS"]}

            except Exception as e:
                logger.error(f"Error loading validation dataset: {e}")
                if not cfg.get("do_test", True):  # Only raise if we're not in test-only mode
                    raise
        else:
            logger.info("Skipping training dataset loading as do_train=False")

        # Create output directory
        if cfg.get("output_dir", None) is None:
            output_dir = f"feature_transformer_{cfg.model.hidden_dim}_{cfg.model.num_layers}l"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = cfg.output_dir
            logger.info(f"Using specified output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Get node information for metrics
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_id = f"{rank}_{local_rank}"

        # Create metrics
        compute_metrics = compute_metrics_factory(node_id)

        # Get training arguments
        # Pass run_name to the training args if it's available (for main process)
        run_name_arg = run_name if accelerator.is_main_process and "run_name" in locals() else None
        args = get_training_args(cfg, output_dir, accelerator, run_name_arg)

        # Create domain-weighted sampler for training if enabled in config
        train_sampler = None
        if train_dataset and cfg.get("do_train", True) and cfg.training.get("use_domain_weighted_sampler", False):
            logger.info("Creating domain-weighted sampler for training")
            try:
                train_sampler = get_domain_weighted_sampler(
                    train_dataset, max_cap=cfg.training.get("domain_sampler_max_cap", 10.0)
                )
                logger.info("Domain-weighted sampler created successfully")
                if accelerator.is_main_process and run:
                    wandb.log(data={"Using domain-weighted sampler": True})
            except Exception as e:
                logger.error(f"Failed to create domain-weighted sampler: {e}")
                train_sampler = None

        # Create class-balanced loss if enabled in config
        compute_loss_func = None
        if cfg.training.get("use_class_balanced_loss", False):
            logger.info("Using class-balanced loss function")
            try:
                # Calculate class weights once from the training dataset
                beta = cfg.training.get("class_balanced_loss_beta", 0.999)
                class_weights = ClassBalancedLoss.get_class_weights_from_dataset(train_dataset)
                logger.info(f"Calculated class weights: {class_weights}")

                # Create the loss function with pre-computed weights
                cb_loss = ClassBalancedLoss(beta=beta, class_weights=class_weights)

                # Create the compute_loss_func that will be used by the trainer
                def class_balanced_loss_func(outputs, labels, **kwargs):
                    return cb_loss(outputs["logits"], labels)

                compute_loss_func = class_balanced_loss_func
                if accelerator.is_main_process and run:
                    wandb.log(data={"Using class-balanced loss": True})
                    wandb.log(data={"Class weights": class_weights.tolist()})
            except Exception as e:
                logger.error(f"Failed to create class-balanced loss: {e}")
                logger.info("Falling back to default cross-entropy loss")
                compute_loss_func = None

        # Create trainer
        trainer = Trainer(
            wrapped_model,
            args,
            train_dataset=train_dataset if cfg.get("do_train", True) else None,
            eval_dataset=val_dataset if val_dataset and cfg.get("do_train", True) else None,
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
            data_collator=collate_fn,
            fourier=False,
            train_sampler=train_sampler,  # Add sampler to trainer
        )

        # Training process
        if cfg.get("do_train", True):
            try:
                if cfg.training.get("resume_from_checkpoint", None):
                    checkpoint_path = cfg.training.resume_from_checkpoint
                    logger.info(f"Resuming training from checkpoint at {checkpoint_path}")
                    if accelerator.is_main_process and run:
                        wandb.log(data={"Resumed from checkpoint": checkpoint_path})
                    train_results = trainer.train(resume_from_checkpoint=checkpoint_path)
                else:
                    logger.info("Starting training from scratch")
                    if accelerator.is_main_process and run:
                        wandb.log(data={"Starting training from scratch": True})
                        wandb.log(data={"Resumed from checkpoint": False})
                    train_results = trainer.train()

                logger.info("Training completed successfully")
                if accelerator.is_main_process and run:
                    wandb.log(data={"Training completed": True})
            except Exception as e:
                logger.error(f"Error during training: {e}")
                if not cfg.get("do_test", True):  # Only raise if we're not in test-only mode
                    raise

        # Evaluation process
        all_results = None
        if cfg.get("do_test", True):
            logger.info("Starting model evaluation on test set.")
            if not cfg.get("do_train", True):
                assert cfg.model.checkpoint_path, "Evalation only mode but no checkoint provided!"

            if accelerator.is_main_process and run:
                wandb.log(data={"Starting evaluation": True})

            # Use the comprehensive evaluation function
            all_results = test_model(cfg, wrapped_model, trainer, accelerator, run)

            logger.info("Evaluation completed successfully")
            if accelerator.is_main_process and run:
                wandb.log(data={"Evaluation completed": True})

        return all_results

    except KeyboardInterrupt:
        _handle_keyboard_interrupt(accelerator, trainer, output_dir, run)
    except Exception as e:
        _handle_exception(e, accelerator, trainer, output_dir, run)
    finally:
        # Clean up wandb
        if "accelerator" in locals() and "run" in locals() and accelerator.is_main_process and run:
            run.finish()


def _handle_keyboard_interrupt(
    accelerator: Optional[Accelerator], trainer: Optional[Trainer], output_dir: str, run: Optional[Any]
) -> None:
    """
    Handle keyboard interrupt gracefully.

    Args:
        accelerator: Accelerator instance
        trainer: Trainer instance
        output_dir: Output directory for checkpoints
        run: WandB run
    """
    try:
        logger.info("Training interrupted. Saving model.")
        # Use accelerator.is_main_process for WandB logging
        if accelerator and accelerator.is_main_process and run:
            wandb.log(data={"Training interrupted": True})
        # Saving should be done carefully in distributed setting, Trainer handles this
        if trainer:
            interrupt_dir = f"{output_dir}_interrupted_{int(time.time())}"
            trainer.save_model(interrupt_dir)
            logger.info(f"Saved interrupted model to {interrupt_dir}")
        print("Interrupted run, press Ctrl+C again to exit.")
        time.sleep(2)
        if accelerator and accelerator.is_main_process and wandb.run:
            wandb.finish(11)
    except KeyboardInterrupt:
        if accelerator and accelerator.is_main_process and wandb.run:
            wandb.finish(111)
        raise


def _handle_exception(
    exception: Exception,
    accelerator: Optional[Accelerator],
    trainer: Optional[Trainer],
    output_dir: str,
    run: Optional[Any],
) -> None:
    """
    Handle exceptions gracefully.

    Args:
        exception: The exception that was raised
        accelerator: Accelerator instance
        trainer: Trainer instance
        output_dir: Output directory for checkpoints
        run: WandB run
    """
    logger.error(f"An error occurred: {exception}")
    # Check if trainer exists before saving model
    if trainer is not None:
        error_dir = f"{output_dir}_error_{int(time.time())}"
        # Saving should be done carefully in distributed setting, Trainer handles this
        trainer.save_model(error_dir)
        logger.info(f"Saved model at error point to {error_dir}")
    # Use accelerator.is_main_process for WandB logging
    if accelerator and accelerator.is_main_process and run:
        wandb.log(data={"Error": str(exception)})
        wandb.finish(2)
    raise exception


# ---------- MAIN ENTRY POINT ----------
@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name=None,
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the script.

    Args:
        cfg: Hydra configuration
    """
    OmegaConf.resolve(cfg)
    run(cfg)


if __name__ == "__main__":
    # Set start method before any CUDA/DataLoader usage
    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # Might have already been set by Accelerate/torchrun
        logger.warning(f"Could not set multiprocessing start method (might be already set): {e}")

    main()
    print("Arguments for completed run/sweep:")
    print(" ".join(sys.argv))
