import torch
from einops import rearrange as rea


class SpectralTransform(torch.nn.Module):
    """
    A class to apply spectral transformations to video frames.
    """

    def __init__(
        self,
        dim=(-3, -2, -1),
        in_arrangement="t h w c",
        denormalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        normalize={
            # video-wise mean/std
            "mean": [1.6358e-07, 1.6358e-07, 1.6358e-07],
            "std": [0.0004, 0.0004, 0.0004],
            # frame-wise mean/std
            # "mean": [6.6732e-09, 6.6732e-09, 6.6732e-09],
            # "std": [7.5445e-05, 7.5329e-05, 7.5546e-05],
        },
        padding_mode="reflect",
        padding=1,
    ):
        """
        Initialize the SpectralTransform class.
        Args:
            dim (tuple): Dimensions to apply the Fourier transform.
            in_arrangement (str): Arrangement of input frames.
            denormalize (dict): Dictionary with 'mean' and 'std' for denormalization.
            normalize (dict): Dictionary with 'mean' and 'std' for normalization.
            padding_mode (str): Padding mode ('reflect' or 'constant').
            padding (int): Amount of padding.
        """
        super().__init__()
        self.dim = dim
        self.in_arrangement = in_arrangement
        self.padding_mode = padding_mode
        self.padding = padding
        # Store values, don't create tensors yet
        self.denormalize_vals = denormalize
        self.normalize_vals = normalize
        self.fp16 = True

    def forward(self, x):
        if type(x) is dict:
            frames = x["pixel_values"]  # frames could also be a batch, e.g,  torch.Size([4, 16, 3, 224, 224])
            if self.fp16:
                # Convert to float16 if needed
                frames = frames.half()

            x["pixel_values"] = self.spectral_apply_video(frames, self.spectral_modification)

            if self.fp16:
                # Convert back to float32 if needed
                x["pixel_values"] = x["pixel_values"].float()

            return x
        if type(x) is torch.Tensor:
            return self.spectral_apply_video(x, self.spectral_modification)
        raise ValueError("Input must be a dictionary or a tensor.")

    def spectral_modification(self, phase, magnitude):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _pad(self, frames, time_dim=1):
        """
        Pad the input frames.
        Args:
            frames (torch.Tensor): Input frames.
            time_dim (int): Dimension index for time (frames).
        Returns:
            torch.Tensor: Padded frames.
        """
        match self.padding_mode:
            case "reflect":
                return torch.cat(
                    [
                        torch.flip(
                            frames.index_select(time_dim, torch.arange(0, self.padding, device=frames.device)),
                            dims=[time_dim],
                        ),
                        frames,
                        torch.flip(frames, dims=[time_dim]).index_select(
                            time_dim, torch.arange(0, self.padding, device=frames.device)
                        ),
                    ],
                    dim=time_dim,
                )
            case "constant" | _:
                padding_shape = list(frames.shape)
                padding_shape[time_dim] = self.padding
                zeros = torch.zeros(padding_shape, device=frames.device, dtype=frames.dtype)
                return torch.cat([zeros, frames, zeros], dim=time_dim)

    def spectral_apply_video(self, frames, fn):
        """
        Apply a function in the spectral domain to video frames.
        Args:
            frames (torch.Tensor): Input frames (assumed to be on the correct device by accelerate).
                Can be a single video or a batch of videos.
            fn (callable): Function to apply in the spectral domain.
        Returns:
            torch.Tensor: Transformed frames.
        """
        # The input 'frames' tensor determines the device
        current_device = frames.device
        original_dtype = frames.dtype  # Store original dtype
        original_shape = frames.shape

        # Check if we have a batch dimension
        has_batch = len(original_shape) > 4  # More than 4 dims likely means batch dimension exists

        # Ensure float type for FFT
        frames = frames.to(dtype=torch.float32)  # Keep on current_device

        # Rearrange based on whether we have a batch dimension
        if has_batch:
            # Assuming shape is [batch, t, h, w, c] or [batch, t, c, h, w]
            if "c" in self.in_arrangement and self.in_arrangement.index("c") < 2:
                # If channels come before spatial dimensions
                frames = rea(frames, f"b {self.in_arrangement} -> b c t h w")
            else:
                frames = rea(frames, f"b {self.in_arrangement} -> b c t h w")
        else:
            frames = rea(frames, f"{self.in_arrangement} -> c t h w")

        if self.denormalize_vals is not None:
            # Create tensors on the same device as frames
            mean = torch.tensor(self.denormalize_vals["mean"], device=current_device, dtype=frames.dtype)
            std = torch.tensor(self.denormalize_vals["std"], device=current_device, dtype=frames.dtype)

            if has_batch:
                mean = mean.view(1, -1, 1, 1, 1)
                std = std.view(1, -1, 1, 1, 1)
            else:
                mean = mean.view(-1, 1, 1, 1)
                std = std.view(-1, 1, 1, 1)

            frames = frames * std + mean

        if self.padding > 0:
            # Determine the time dimension for padding
            time_dim = 2 if has_batch else 1
            # Padding happens on the current_device
            frames = self._pad(frames, time_dim)

        # Apply FFT - need to handle batch dimension properly
        if has_batch:
            # For batched inputs, we need to adjust the dimensions for FFT
            # Shift dimensions by 1 because of batch dimension
            fft_dims = tuple(d + 1 if d < 0 else d for d in self.dim)
        else:
            fft_dims = self.dim

        # FFT operates on float, returns complex (on current_device)
        fft = torch.fft.fftn(frames, dim=fft_dims)
        phase = torch.angle(fft)
        magnitude = torch.abs(fft)
        phase, magnitude = fn(phase, magnitude)  # fn should handle tensors on current_device

        # Ensure complex type for exp operation if needed
        fft = magnitude.to(fft.dtype) * torch.exp(1j * phase.to(fft.dtype))

        # IFFT operates on complex, returns complex (on current_device)
        frames = torch.fft.ifftn(fft, dim=fft_dims)
        # Take real part, should result in float again (on current_device)
        frames = frames.real

        if self.padding > 0:
            # Unpadding happens on the current_device
            # Use the same time dimension as for padding
            time_dim = 2 if has_batch else 1
            if has_batch:
                frames = frames[:, :, self.padding : -self.padding]
            else:
                frames = frames[:, self.padding : -self.padding]

        if self.normalize_vals is not None:
            # Create tensors on the same device as frames
            mean = torch.tensor(self.normalize_vals["mean"], device=current_device, dtype=frames.dtype)
            std = torch.tensor(self.normalize_vals["std"], device=current_device, dtype=frames.dtype)

            if has_batch:
                mean = mean.view(1, -1, 1, 1, 1)
                std = std.view(1, -1, 1, 1, 1)
            else:
                mean = mean.view(-1, 1, 1, 1)
                std = std.view(-1, 1, 1, 1)

            # Add small epsilon to std to prevent division by zero
            frames = (frames - mean) / (std + 1e-6)

        # Cast back to original dtype if needed, but stay on the current_device
        frames = frames.to(dtype=original_dtype)

        # Rearrange back to original format
        if has_batch:
            frames = rea(frames, f"b c t h w -> b {self.in_arrangement}").contiguous()
        else:
            frames = rea(frames, f"c t h w -> {self.in_arrangement}").contiguous()

        # No need to move device, accelerate handles it.
        return frames


class PhaseOnly(SpectralTransform):
    """
    A class to apply phase-only transformations to video frames.
    """

    def spectral_modification(self, phase, magnitude):
        """
        Apply phase-only modification.
        Args:
            phase (torch.Tensor): Phase of the FFT.
            magnitude (torch.Tensor): Magnitude of the FFT.
        Returns:
            tuple: Modified phase and magnitude.
        """
        return phase, torch.ones_like(magnitude)
