import torch
import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        n_zeros = self.total_patches - self.total_masks
        n_ones = self.total_masks
        mask_per_frame = np.hstack([np.zeros(n_zeros), np.ones(n_ones)])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class TubeMasker:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)

    def __call__(self, x):
        # Determine the device from the input dictionary (assuming 'pixel_values' exists)
        # Fallback to CPU if 'pixel_values' is not present or not a tensor
        input_tensor = x.get("pixel_values")
        device = input_tensor.device if isinstance(input_tensor, torch.Tensor) else torch.device("cpu")

        # Check if we have a batch dimension
        has_batch = len(input_tensor.shape) > 4  # For videos, shape would be [t,c,h,w] or [b,t,c,h,w]
        batch_size = input_tensor.shape[0] if has_batch else 1

        # Generate masks for each item in the batch
        n_zeros = self.num_patches_per_frame - self.num_masks_per_frame
        n_ones = self.num_masks_per_frame

        if has_batch:
            # Create masks for each item in the batch
            all_masks = []
            for _ in range(batch_size):
                mask_per_frame = np.hstack([np.zeros(n_zeros), np.ones(n_ones)])
                np.random.shuffle(mask_per_frame)
                mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
                all_masks.append(mask)

            # Stack masks into a batch
            batch_mask = np.stack(all_masks)
            x.update({"bool_masked_pos": torch.tensor(batch_mask, dtype=torch.bool, device=device)})
        else:
            # Original behavior for single video
            mask_per_frame = np.hstack([np.zeros(n_zeros), np.ones(n_ones)])
            np.random.shuffle(mask_per_frame)
            mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
            x.update({"bool_masked_pos": torch.tensor(mask, dtype=torch.bool, device=device)})

        return x

    def __repr__(self):
        repr_str = f"Mask: total patches per frame {self.num_patches_per_frame}, mask patches per frame {self.num_masks_per_frame}"
        return repr_str
