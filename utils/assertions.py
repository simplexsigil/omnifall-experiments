import logging

from omegaconf import DictConfig
from configs.config import Config
from data.dataset import GenericVideoDataset
import torch
from transformers import (
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
)
import numpy as np


def check_dataset_model_cfg(
    dataset: GenericVideoDataset,
    model: VideoMAEForPreTraining | VideoMAEForVideoClassification,
    cfg: Config,
):
    """
    Check the dataset, model, and configuration for consistency.
    """
    # Check if all inputs match the expected types
    # assert isinstance(dataset, (GenericVideoDataset, DatasetView)), "dataset is not a GenericVideoDataset"
    assert isinstance(model, (VideoMAEForPreTraining, VideoMAEForVideoClassification)), "model is not a VideoMAE model"
    assert isinstance(cfg, (Config, dict, DictConfig)), "cfg is not a Config object"

    # Setup
    pretraining = isinstance(model, VideoMAEForPreTraining)
    classification = isinstance(model, VideoMAEForVideoClassification)

    # Check dataset num classes match model num classes
    if classification:
        dataset_classes = set(dataset.annotations.values())
        model_classes = set(range(model.config.num_labels))

        assert dataset_classes.issubset(
            model_classes
        ), f"Dataset classes {dataset_classes} do not match model classes {model_classes}"


def check_batch_statistics(batch, threshold_magnitude=1):
    """
    Check the batch statistics for consistency.
    """
    # Check if all inputs match the expected types
    assert isinstance(batch, dict), "batch is not a dictionary"

    # Check if batch contains the expected keys
    assert "pixel_values" in batch, "batch does not contain pixel_values"
    assert "labels" in batch or "bool_masked_pos" in batch, "batch does not contain labels nor bool_masked_pos"

    pretrain = "bool_masked_pos" in batch
    classification = "labels" in batch

    # Check if pixel_values and labels are tensors
    assert isinstance(batch["pixel_values"], torch.Tensor), "pixel_values is not a tensor"
    assert not classification or isinstance(batch["labels"], torch.Tensor), "labels is not a tensor"
    assert not pretrain or isinstance(batch["bool_masked_pos"], torch.Tensor), "bool_masked_pos is not a tensor"

    # Check pixel_values mean and std
    if batch["pixel_values"].mean().abs() > 10 ** (threshold_magnitude - 1):
        logging.warning(f"pixel_values mean is not near 0, mean: {batch['pixel_values'].mean()}")
    # Check that pixel_values std is within m orders of magnitude of 1
    std = batch["pixel_values"].std()
    if torch.log10(std).abs() > threshold_magnitude:
        logging.warning(f"pixel_values std is not near 1, std: {std}")


def check_predictions(labels, predictions):
    """
    Check the predictions for consistency.
    """
    # Check if all inputs match the expected types
    assert isinstance(labels, (torch.Tensor, np.ndarray)), "labels is not a tensor"
    assert isinstance(predictions, (torch.Tensor, np.ndarray)), "predictions is not a tensor"

    # Check if labels and predictions have the same shape
    assert (
        labels.shape == predictions.shape
    ), f"labels shape {labels.shape} does not match predictions shape {predictions.shape}"

    # Check if labels and predictions are within the expected range
    assert (labels >= 0).all(), "labels contain negative values"
    assert (predictions >= 0).all(), "predictions contain negative values"

    # Warn if all predictions are the same unless the GT labels are mostly the same
    if np.unique(labels).shape[0] > 2:
        if np.unique(predictions).shape[0] == 1:
            logging.warning(f"All predictions are the same: {np.unique(predictions)}")
