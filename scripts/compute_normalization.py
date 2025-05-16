import logging
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from configs.config import Config
from data.dataset_utils import get_dataset
import data.fourier as fourier
from torch.utils.data import DataLoader


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config",
)
def main(cfg: Config):
    OmegaConf.resolve(cfg)
    logging.info(OmegaConf.to_yaml(cfg))

    # Get the dataset (train split is typical for normalization)
    dataset = get_dataset(cfg, "train", run=None)
    # Create a DataLoader with 96 parallel workers
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Assuming each sample is processed individually
        shuffle=False,
        num_workers=96,
        pin_memory=True,
    )

    n_samples = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)

    fourier_augment = fourier.PhaseOnly(in_arrangement="t c h w")

    for idx, sample in enumerate(dataloader):
        # sample["pixel_values"]: Tensor [num_frames, num_channels, H, W] or [num_channels, H, W]
        pixel_values = sample["pixel_values"].squeeze(0)  # Remove batch dimension
        pixel_values = fourier_augment(pixel_values)

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.float()
        else:
            pixel_values = torch.tensor(pixel_values).float()

        channel_sum += pixel_values.sum(dim=(0, 2, 3)).double()
        channel_squared_sum += (pixel_values**2).sum(dim=(0, 2, 3)).double()
        n_samples += pixel_values.shape[0] * pixel_values.shape[2] * pixel_values.shape[3]

        # Log every 1000 samples
        if (idx + 1) % 1000 == 0:
            mean = (channel_sum / n_samples).cpu().numpy()
            std = (channel_squared_sum / n_samples - mean**2) ** 0.5
            logging.info(f"After {idx+1} samples: mean={mean}, std={std}")

    # Final mean and std
    mean = (channel_sum / n_samples).cpu().numpy()
    std = (channel_squared_sum / n_samples - mean**2) ** 0.5
    print(f"Final mean: {mean}, std: {std}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
