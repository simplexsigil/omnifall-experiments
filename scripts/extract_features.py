from collections import defaultdict
import logging
import os
import h5py
import hydra
from omegaconf import OmegaConf
import torch
import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
import yaml
from data.dataset_utils import get_dataset
from data.feature_extraction_dataset import SlidingClipIterableDataset
import numpy as np
from accelerate import Accelerator

from data.transforms.fourier import PhaseOnly
from training.ta_trainer import default_worker_init_fn


def get_model(cfg):
    model = VideoMAEModel.from_pretrained(cfg.model.name_or_path)
    if cfg.model.get("checpoint_path", None) is not None:
        logging.info(f"Loading checkpoint from {cfg.model.checkpoint_path}.")
        checkpoint = torch.load(cfg.model.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def yield_video_features(model, dataloader):
    """
    Yield video features from the dataset using the model.
    Args:
        model: The model to use for feature extraction.
        dataloader: The dataset to extract features from, already wrapped in a DataLoader.
    Yields:
        dict: A dictionary containing the video path, extracted features, and number of clips.
    """
    open_features = defaultdict(list)
    clip_counts = {}

    model = torch.nn.DataParallel(model)
    model = model.eval().cuda()
    fourier_fn = PhaseOnly(in_arrangement="t c h w")

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            # Apply Fourier transform
            inputs["pixel_values"] = fourier_fn(inputs["pixel_values"])
            outputs = model(pixel_values=inputs["pixel_values"], return_dict=True)
            batch_features = outputs.last_hidden_state
            batch_features = batch_features.cpu().numpy()
            for path, clip_idx, last_clip, features in zip(
                inputs["path"], inputs["clip_idx"], inputs["last_clip"], batch_features
            ):
                clip_idx = int(clip_idx)
                open_features[path].append((clip_idx, features))
                if last_clip:
                    clip_counts[path] = clip_idx + 1

            finished_paths = [p for p in clip_counts if len(open_features[p]) >= clip_counts[p]]
            for path in finished_paths:
                clips = sorted(open_features[path], key=lambda x: x[0])
                stacked = np.stack([clip[1] for clip in clips], axis=0)  # [N_clips, D]
                yield {"path": path, "features": stacked, "num_clips": clip_counts[path], "time": inputs["time"]}
                del open_features[path]
                del clip_counts[path]

        if len(open_features) > 0:
            logging.warning(f"Incomplete features for {len(open_features)} videos. Skipping these videos.")


def save_video_features_to_hdf5(
    model,
    dataloader,
    output_replacement,
    overwrite=False,
):
    """
    Extracts features using a model and writes one .h5 file per video using yield_video_features().

    Args:
        model: A PyTorch model with forward() -> output.last_hidden_state.
        dataloader: A DataLoader yielding dicts with keys: 'pixel_values', 'path', 'clip_idx', 'last_clip'.
        output_dir: Directory where HDF5 files will be saved.
        overwrite: If False, will skip videos that already exist.
    """
    for video_data in tqdm.tqdm(yield_video_features(model, dataloader), desc="Saving features"):
        path = video_data["path"]
        features = video_data["features"]  # Shape: [N_clips, D]
        num_clips = video_data["num_clips"]

        # Derive output filename (you may customize this)
        video_id = os.path.splitext(path)[0]
        save_path = video_id.replace(*output_replacement) + ".h5"

        if os.path.exists(save_path) and not overwrite:
            continue

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logging.debug(f"Saving features for {path} to {save_path}.")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("features", data=features)  # shape: [N_clips, D]
            f.attrs["num_clips"] = num_clips
            f.attrs["original_path"] = path
            f.create_dataset("time", data=video_data["time"].numpy())  # Assuming time is a torch tensor


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config",
)
def main(cfg):
    OmegaConf.resolve(cfg)

    model = get_model(cfg)
    model = model.eval().cuda()

    base_dataset = get_dataset(cfg, "full")
    dataset = SlidingClipIterableDataset(
        generic_video_dataset=base_dataset,
        stride=cfg.dataset.get("stride", 4),
        path_replacement_for_skipping=cfg.features.output_replacement if cfg.dataset.get("overwrite", False) else None,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.hardware.num_data_workers,
        pin_memory=True,
        prefetch_factor=cfg.dataset.get("prefetch_factor", 2),
        persistent_workers=True,
        shuffle=False,
        # worker_init_fn=default_worker_init_fn, # Init thread-aware workers
    )

    save_video_features_to_hdf5(
        model=model,
        dataloader=dataloader,
        output_replacement=cfg.features.output_replacement,
        overwrite=cfg.dataset.get("overwrite", False),
    )


if __name__ == "__main__":
    # Set logging to info level
    logging.basicConfig(level=logging.INFO)
    main()
