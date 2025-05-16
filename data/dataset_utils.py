import logging
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np

from data.cvhci_dataset import CVHCIDataset
from data.dataset import GenericVideoDataset
import data.fourier as fourier
import cProfile
import pstats
from pathlib import Path
from omegaconf import OmegaConf


def denormalize(pixel_values, mean, std):
    """
    Denormalizes the pixel values using the provided mean and std.

    Args:
        pixel_values: Tensor of shape (num_frames, channels, H, W)
        mean: List of means for each channel (e.g., [0.485, 0.456, 0.406])
        std: List of std deviations for each channel (e.g., [0.229, 0.224, 0.225])

    Returns:
        Denormalized tensor in the same shape as input.
    """
    # Get device from input tensor
    device = pixel_values.device
    # Create mean and std tensors on the same device
    mean = torch.tensor(mean, device=device, dtype=pixel_values.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device, dtype=pixel_values.dtype).view(1, -1, 1, 1)
    return pixel_values * std + mean


def log_dataset_samples(dataset, name="dataset_samples_video"):
    """
    Logs a video to WandB created from 5 samples of the dataset.

    Args:
        dataset: The dataset object to sample data from.
    """
    sample_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    sample_batch = next(iter(sample_loader))  # Get 5 random samples

    # Extract pixel values and labels (adapt to your dataset structure)
    pixel_values = sample_batch["pixel_values"]  # Shape: (batch_size, num_frames, channels, H, W)
    # labels = sample_batch["label"] # Assuming shape: (batch_size,) # Keep if needed elsewhere

    # Denormalize pixel values (on original device, potentially GPU)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # pixel_values = denormalize(pixel_values, mean, std)
    pixel_values = torch.clamp(pixel_values, 0, 1)
    pixel_values = (pixel_values * 255).byte()

    # Concatenate along the batch dimension first, then potentially frames if needed
    # Assuming wandb.Video expects (T, C, H, W) or (T, H, W, C)
    # Let's reshape to (batch_size * num_frames, channels, H, W)
    b, t, c, h, w = pixel_values.shape
    video_data_gpu = pixel_values.reshape(b * t, c, h, w)

    # Move the single concatenated tensor to CPU and convert to numpy
    video_data_np = video_data_gpu.cpu().numpy()

    # Log the video
    wandb.log({name: wandb.Video(video_data_np, fps=7.5, format="mp4")})

    fourier_video = fourier.PhaseOnly(in_arrangement="t c h w")(pixel_values)
    fourier_video = fourier_video.reshape(b * t, c, h, w).cpu().numpy()
    print(f"fourier_mean: {fourier_video.mean()}, fourier_std: {fourier_video.std()}")
    fourier_video = (fourier_video - fourier_video.min()) / (fourier_video.max() - fourier_video.min()) * 255.0
    wandb.log({"fourier_video": wandb.Video(fourier_video, fps=7.5, format="mp4")})


def _augment_with(dataset, augmentation_str, cfg={}):
    """
    Apply the specified augmentation to the dataset.
    """

    match augmentation_str:
        case "fourier_phase_only":
            return dataset.map(fourier.PhaseOnly(in_arrangement="t c h w", **cfg))
        case "tube_masking":
            input_size = cfg.get("input_size", (16, 16, 16))  # Patches in (T, H, W)
            mask_ratio = cfg.get("mask_ratio", 0.75)  # Ratio of patches to mask
            tube_masker = TubeMasker(input_size, mask_ratio)
            return dataset.map(tube_masker)
        case _:
            logging.warning(f"Unknown augmentation: {augmentation_str}, ignoring it.")
            return dataset


def _recursive_augment(dataset, augmentations):
    """
    Recursively apply augmentations to the dataset.
    """
    if isinstance(augmentations, list) or isinstance(augmentations, ListConfig):
        for aug in augmentations:
            dataset = _recursive_augment(dataset, aug)
        return dataset
    elif isinstance(augmentations, dict) or isinstance(augmentations, DictConfig):
        for key, value in augmentations.items():
            dataset = _augment_with(dataset, key, value)
        return dataset
    elif isinstance(augmentations, str):
        return _augment_with(dataset, augmentations)
    else:
        logging.warning(f"Unknown augmentation type: {type(augmentations)}, ignoring it.")
        return dataset


def apply_augmentation(cfg: DictConfig, mode: str, dataset: GenericVideoDataset):
    """
    Apply data augmentation to the dataset.
    """
    augmentation = (cfg.dataset.get("augmentation", {}) or {}).get(mode, None)
    if augmentation is not None:
        logging.info(f"Applying augmentation: {augmentation}")
        dataset = _recursive_augment(dataset, augmentation)
    return dataset


def get_dataset(cfg, mode, run=None):
    """
    Get the dataset based on the configuration.
    """
    logging.info(f"Loading GenericVideoDataset for {mode} mode.")
    normalize = {}
    if "normalize" in cfg.dataset:
        normalize["normalize"] = cfg.dataset.normalize

    DatasetClass = {
        None: GenericVideoDataset,
        "cvhci": CVHCIDataset,
    }[cfg.dataset.get("dataset_type", None)]

    dataset = DatasetClass(
        cfg.dataset.get(f"video_root_{mode}"),
        cfg.dataset.annotations.get(f"annf_{mode}"),
        target_fps=cfg.dataset.model_fps,
        vid_frame_count=cfg.dataset.num_frames,
        path_format=cfg.dataset.get(f"path_format_{mode}"),
        mode=mode,
        **normalize,
    )
    if cfg.dataset.get("small_subset", False):
        logging.info(f"Using a small subset of the dataset for {mode}.")
        dataset = dataset.subset(100)
    # dataset = apply_augmentation(cfg, mode, dataset)
    if run:
        logging.info(f"Logging dataset samples to WandB for {mode}.")
        log_dataset_samples(dataset, name=f"{mode.capitalize()} Samples")
    logging.info(f"Dataset for {mode} loaded with {len(dataset)} samples.")
    return dataset


def main():

    # Example configuration similar to the one in dataset.py
    cfg = {
        "dataset": {
            "annotations": {
                "ann_type": "single-label-class",
                "ann_name": "action",
                "annf_train": "/lsdf/data/activity/Kinetics/k400_full_labels/train.csv",
                "annf_val": "/lsdf/data/activity/Kinetics/k400_full_labels/val.csv",
                "annf_test": "/lsdf/data/activity/Kinetics/k400_full_labels/test.csv",
                "attributes": "action",
                "num_classes": 400,
            },
            "augmentation": {
                "all": "fourier_phase_only",
                "train": "fourier_phase_only",
                "val": "fourier_phase_only",
                "test": "fourier_phase_only",
            },
            "name": "Kinetics400",
            "dataset_fps": [30],
            "model_fps": 7.5,
            "num_frames": 16,
            "root": "/lsdf/data/activity/Kinetics/",
            "videos": "k400_full",
            "video_root_train": "/lsdf/data/activity/Kinetics/k400_full",
            "video_root_val": "/lsdf/data/activity/Kinetics/k400_full",
            "video_root_test": "/lsdf/data/activity/Kinetics/k400_full",
            "path_format_train": "{video_root}/{filename}",
            "path_format_val": "{video_root}/{filename}",
            "path_format_test": "{video_root}/{filename}",
        }
    }
    cfg = DictConfig(cfg)

    mode = "train"  # Example mode
    dataset = get_dataset(cfg, mode)

    # Profile the __getitem__ method
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(100):  # Profile 100 samples
        _ = dataset[i]

    profiler.disable()

    # Create directory for profile output if it doesn't exist
    profile_dir = Path("profile_output")
    profile_dir.mkdir(exist_ok=True)

    # Write profiling data to a binary file
    profile_path = profile_dir / "getitem_profiling_results.prof"
    profiler.dump_stats(str(profile_path))

    print(f"Profile data written to {profile_path}")
    print(f"You can visualize it with: snakeviz {profile_path}")

    # Also create a text report for convenience
    stats_path = profile_dir / "getitem_profiling_stats.txt"
    with open(stats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumtime")
        stats.print_stats()

    print(f"Text stats written to {stats_path}")


if __name__ == "__main__":
    main()
