#!/usr/bin/env python

import argparse
import logging
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
import yaml

from models.feature_transformer import FeatureTransformer
from data.feature_dataset import FeatureVideoDataset, MultiFeatureDataset, label2idx, idx2label

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, device="cuda"):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model = FeatureTransformer(
        feature_dim=config["model"]["feature_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["dataset"]["num_classes"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        mlp_ratio=config["model"]["mlp_ratio"],
        num_features=config["dataset"]["num_features"],
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def run_inference(model, dataset, device="cuda"):
    """
    Run inference on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to run inference on
        device: Device to run inference on

    Returns:
        Dictionary with predictions and ground truth
    """
    logger.info(f"Running inference on {len(dataset)} samples")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_predictions = []
    all_labels = []
    all_video_ids = []
    all_dataset_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            features = batch["rgb_features"].to(device)
            labels = batch["label"]

            outputs = model(features)
            _, predicted = outputs["logits"].max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_video_ids.extend(batch["video_id"])
            all_dataset_names.extend(batch["dataset_name"])

    return {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "video_ids": all_video_ids,
        "dataset_names": all_dataset_names,
    }


def calculate_metrics(results):
    """
    Calculate performance metrics.

    Args:
        results: Dictionary with predictions and ground truth

    Returns:
        Dictionary with metrics
    """
    predictions = results["predictions"]
    labels = results["labels"]

    # Overall accuracy
    accuracy = (predictions == labels).mean() * 100

    # Per-class accuracy
    class_accuracies = {}
    for class_idx in np.unique(labels):
        if class_idx >= 0:  # Skip unlabeled (-1)
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == class_idx).mean() * 100
                class_name = idx2label.get(class_idx, f"Unknown-{class_idx}")
                class_accuracies[class_name] = class_acc

    # Per-dataset accuracy
    dataset_accuracies = {}
    for dataset_name in np.unique(results["dataset_names"]):
        dataset_mask = np.array(results["dataset_names"]) == dataset_name
        if dataset_mask.sum() > 0:
            dataset_acc = (predictions[dataset_mask] == labels[dataset_mask]).mean() * 100
            dataset_accuracies[dataset_name] = dataset_acc

    return {
        "overall_accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "dataset_accuracies": dataset_accuracies,
    }


@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Main entry point.

    Args:
        config: Hydra config
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained feature transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save results")

    # Convert Hydra's config to command line args for backwards compatibility
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, model_config = load_model(args.checkpoint, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use same dataset config as the trained model
    datasets = []
    for ds_config in config.dataset.feature_datasets:
        dataset = FeatureVideoDataset(
            feature_root=ds_config.feature_root,
            annotations_file=ds_config.annotations_file,
            feature_fps=ds_config.feature_fps,
            feature_frames=ds_config.frames_per_feature,
            feature_stride=ds_config.feature_stride,
            feature_type=ds_config.feature_type,
            dataset_name=ds_config.name,
            mode="test",  # Use test split for inference
            feature_ext=config.dataset.feature_ext,
            split_root=ds_config.split_root,
            num_features=config.dataset.num_features,
            get_features=True,
        )
        datasets.append(dataset)
        logger.info(f"Added dataset {ds_config.name} with {len(dataset)} segments")

    # Combine into multi-dataset
    multi_dataset = MultiFeatureDataset(datasets)
    logger.info(f"Created combined dataset with {len(multi_dataset)} total segments")

    # Run inference
    results = run_inference(model, multi_dataset, device)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.2f}%")
    logger.info("\nPer-class accuracies:")
    for class_name, acc in metrics["class_accuracies"].items():
        logger.info(f"  {class_name}: {acc:.2f}%")

    logger.info("\nPer-dataset accuracies:")
    for dataset_name, acc in metrics["dataset_accuracies"].items():
        logger.info(f"  {dataset_name}: {acc:.2f}%")

    # Save results
    with open(output_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
