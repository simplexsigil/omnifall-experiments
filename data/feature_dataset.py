import csv
import h5py
import logging
import random
import time
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import bisect
from collections import OrderedDict
import pandas as pd
from einops import rearrange

from data.utils import FeatureHelper

label2idx = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
    "no_fall": -1,
}

idx2label = {v: k for k, v in label2idx.items()}


class FeatureVideoDataset(Dataset):

    def __repr__(self):
        return f"FeatureVideoDataset(name='{self.name}', split='{self.split}', mode='{self.mode}', samples={len(self.samples)}, segments={len(self.video_segments)}, blacklisted={self.blacklisted_segments_count})"

    @property
    def targets(self):
        """Return all class labels for segments in this dataset."""
        return torch.tensor([segment["label"] for segment in self.video_segments])

    def __init__(
        self,
        feature_root,
        feature_fps,
        feature_frames,
        feature_stride,
        feature_centered,
        num_features,
        feature_type,
        annotations_file,
        split_root=None,
        dataset_name="UnnamedFeatureDataset",
        mode="train",
        split="cs",
        feature_ext=".h5",
        get_features=True,
        **kwargs,
    ):
        """
        Feature-based video dataset that handles temporal segmentation.

        Args:
            feature_root: Root directory for feature files
            annotations_file: CSV file with temporal labels (path,label,start,end)
            feature_fps: FPS at which features were extracted
            feature_frames: Number of frames covered by each feature
            feature_stride: Stride (in seconds) between consecutive features
            feature_centered: Whether features are centered around their timestamp
            num_features: Number of features to extract per segment
            feature_type: Type of feature (e.g., "i3d", "dinov2", "invid2")
            dataset_name: Name of the dataset (for multi-dataset scenarios)
            mode: Dataset mode ("train", "val", "test", or "all")
            split: Split name (typically cross-validation split)
            feature_ext: Extension of feature files (default: ".h5")
            split_root: Root directory for split files
            get_features: Whether to load features (False for metadata-only)
            **kwargs: Additional arguments
        """
        self.feature_root = feature_root
        self.feature_fps = feature_fps
        self.feature_frames = feature_frames
        self.feature_stride = feature_stride
        self.feature_centered = feature_centered
        self.feature_type = feature_type
        self.feature_ext = feature_ext
        self.name = dataset_name
        self.get_features = get_features
        self.split = split
        self.mode = mode
        self.max_retries = 2000

        # Initialize the feature helper for time calculations
        self.feature_helper = FeatureHelper(
            extractor_fps=feature_fps,
            extractor_frames=feature_frames,
            time_stride=feature_stride,
            features_centered=feature_centered,
        )

        # Index-based blacklist for segments
        self.blacklist = set()
        # Map from video paths to their segment indices
        self.video_to_indices = {}
        # Count of blacklisted segments for reporting
        self.blacklisted_segments_count = 0

        assert mode == "all" or (split_root), f"Split root not provided."

        self.split_file = os.path.join(split_root, split, dataset_name, f"{mode}.csv") if not mode == "all" else None
        self.num_features = num_features

        self.samples = OrderedDict()

        # Time duration covered by each feature in seconds
        # Using feature_helper instead of direct calculation
        self.feature_duration = self.feature_helper.window_duration

        # Load split if provided

        if self.split_file:
            with open(self.split_file, "r") as f:
                paths = sorted(list(f.read().splitlines()))
                for p in paths:
                    self.samples[p] = {"id": p}

        # Load temporal segmentation labels and initialize paths
        self._load_temporal_labels(annotations_file)

    def _load_temporal_labels(self, annotations_file):
        """Load temporal segmentation labels from CSV and create segment index."""

        df = pd.read_csv(annotations_file)

        for _, row in df.iterrows():
            path = row.iloc[0]  # Assuming first column is path
            label = row.iloc[1]
            start = row.iloc[2]
            end = row.iloc[3]
            subsect = row.iloc[4]
            cam = row.iloc[5]

            # Only process videos that are in our split
            if path in self.samples or self.mode == "all":
                if path not in self.samples:
                    self.samples[path] = {"id": path}

                if "segments" not in self.samples[path]:
                    self.samples[path]["segments"] = []

                # Add to path-based dictionary
                segment = {
                    "video_id": path,
                    "start": float(start),
                    "end": float(end),
                    "label": int(label),  # "labels" is a magic key in hugginface trainer.
                    "subject": int(subsect),
                    "camera": int(cam),
                }
                self.samples[path]["segments"].append(segment)

        # Sort segments by start time for each video
        for path in self.samples:
            if "segments" in self.samples[path]:
                self.samples[path]["segments"].sort(key=lambda x: x["start"])
            else:
                logging.warning(f"{self.name}: No annotations for video {path}")

        # Create flat segment index
        self.video_segments = []
        for path in self.samples:
            if "segments" in self.samples[path]:
                self.video_to_indices[path] = []
                for segment in self.samples[path]["segments"]:
                    segment_idx = len(self.video_segments)
                    self.video_to_indices[path].append(segment_idx)
                    self.video_segments.append(segment)

    def get_feature_path(self, video_path):
        """Convert video path to feature path."""
        # Extract filename without extension
        filename = os.path.splitext(video_path)[0]
        # Create feature path
        return os.path.join(self.feature_root, filename + self.feature_ext)

    def blacklist_file(self, video_path):
        """
        Blacklist all segments from a specific video file.

        Args:
            video_path: Path to the video file to blacklist

        Returns:
            int: Number of newly blacklisted segments
        """
        if video_path not in self.video_to_indices:
            return 0

        indices_to_blacklist = self.video_to_indices[video_path]
        original_blacklist_size = len(self.blacklist)
        self.blacklist.update(indices_to_blacklist)
        newly_blacklisted = len(self.blacklist) - original_blacklist_size
        self.blacklisted_segments_count += newly_blacklisted

        return newly_blacklisted

    def get_label_at_time(self, video_path, timestamp):
        """Get the action label at a specific timestamp."""
        if video_path not in self.samples or "segments" not in self.samples[video_path]:
            return -1  # No labels for this video

        for segment in self.samples[video_path]["segments"]:
            if segment["start"] <= timestamp < segment["end"]:
                return segment["label"]

        return -1  # No label found for this timestamp

    def get_features_for_segment(self, segment: dict):
        """
        Get features that correspond to a specific segment defined by start and end time.

        Args:
            segment: Segment dictionary containing video_id, start and end time

        Returns:
            dict: Contains features and metadata for the segment
        """
        start_time = segment["start"]
        end_time = segment["end"]
        video_path = segment["video_id"]
        feature_path = self.get_feature_path(video_path)

        if not os.path.exists(feature_path):
            print(f"Feature file not found: {feature_path}")
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        try:
            with h5py.File(feature_path, "r") as f:
                # Get feature count without loading all features
                feature_count = f["features"].shape[0]

                segment_duration = end_time - start_time

                # Calculate the time span covered by the feature sequence
                first_feature_idx = 0
                last_feature_idx = self.num_features - 1
                sequence_timespan = self.feature_helper.center_time_at(
                    last_feature_idx
                ) - self.feature_helper.center_time_at(first_feature_idx)

                # Calculate feature indices based on different scenarios
                if sequence_timespan <= segment_duration:
                    # Sequence is smaller than segment, randomly position within segment
                    # Calculate how much room we have to position the sequence
                    positioning_margin = segment_duration - sequence_timespan
                    # Random offset within the margin
                    time_offset = random.uniform(0, positioning_margin)
                    # Adjusted start time for the feature sequence
                    adjusted_start_time = start_time + time_offset

                    # Get feature index at the adjusted start time
                    start_idx = self.feature_helper.feature_idx_at(adjusted_start_time)
                    end_idx = start_idx + self.num_features
                else:
                    # Goal: Keep the segment in the middle of the feature span, with jitter
                    positioning_margin = sequence_timespan - segment_duration
                    ideal_feature_start_time = start_time - (positioning_margin / 2.0)
                    max_deviation_from_center_for_coverage = positioning_margin / 2.0
                    desired_jitter_half_range = 0.25 * segment_duration
                    actual_jitter_half_range = min(desired_jitter_half_range, max_deviation_from_center_for_coverage)
                    random_shift = random.uniform(-actual_jitter_half_range, actual_jitter_half_range)
                    adjusted_start_time = ideal_feature_start_time + random_shift
                    start_idx = max(0, self.feature_helper.feature_idx_at(adjusted_start_time))
                    end_idx = start_idx + self.num_features

                # Ensure we stay within feature bounds
                if end_idx > feature_count:
                    end_idx = feature_count
                    start_idx = max(0, end_idx - self.num_features)

                # Ensure we have at least one feature
                if start_idx >= feature_count:
                    start_idx = max(0, feature_count - 1)
                    end_idx = start_idx + 1

                # Calculate the actual number of features to load
                load_count = min(self.num_features, end_idx - start_idx)

                # Load only the required features
                segment_features = f["features"][start_idx : start_idx + load_count]

                # Pad by repeating the last feature if necessary
                if len(segment_features) < self.num_features:
                    if len(segment_features) > 0:
                        padding = np.tile(segment_features[-1], (self.num_features - len(segment_features), 1))
                        segment_features = np.vstack([segment_features, padding])
                    else:
                        # In case there are no features at all
                        raise ValueError("No features available for this segment")

                assert len(segment_features) == self.num_features, "Feature length mismatch"

                if len(segment_features.shape) > 2:
                    # If features are 3D, we have a token dimension. we average over it.
                    # (seq, token, dim)
                    segment_features = np.mean(segment_features, axis=1)

                segment.update(
                    {
                        "feature_type": self.feature_type,
                        "features": torch.tensor(segment_features, dtype=torch.float32),
                        "feature_start_time": self.feature_helper.start_time_at(start_idx),
                        "feature_end_time": self.feature_helper.end_time_at(start_idx + len(segment_features) - 1),
                    }
                )

                return segment
        except Exception as e:
            print(f"Error loading features from {feature_path}")
            raise e

    def __len__(self):
        """Return the number of segments rather than videos."""
        return len(self.video_segments)

    def __getitem__(self, idx, recursion_depth=0):
        """Get a feature segment with its label."""
        blacklist_ctr = 0
        while idx in self.blacklist:
            idx = idx + 1 if idx + 1 < len(self) else 0
            blacklist_ctr += 1
            if blacklist_ctr > len(self):
                raise ValueError("All segments are blacklisted. Please check your dataset.")

        seg = self.video_segments[idx]

        if self.get_features:
            # Load features for the segment
            try:
                seg = self.get_features_for_segment(seg)
            except Exception as e:
                video_path = seg["video_id"]
                print(f"Error loading features for {video_path} at recursion {recursion_depth}/{self.max_retries}")
                if recursion_depth < self.max_retries:
                    # Blacklist all segments from this video path
                    newly_blacklisted = self.blacklist_file(video_path)
                    print(f"Blacklisted {newly_blacklisted} segments from file {video_path}")

                    rand_indx = random.randint(0, len(self) - 1)
                    return self.__getitem__(rand_indx, recursion_depth + 1)
                else:
                    raise e

        seg["dataset_name"] = self.name

        return seg

    def sample_random_segment(self):
        """Sample a random segment from the dataset."""
        idx = random.randint(0, len(self) - 1)
        return self[idx]

    def get_blacklisted_count(self):
        """Get the number of blacklisted segments."""
        return self.blacklisted_segments_count


class MultiFeatureDataset(Dataset):
    def __init__(self, datasets):
        """
        Wrapper for multiple FeatureVideoDataset instances.

        Args:
            datasets: List of FeatureVideoDataset instances
        """
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum(self.dataset_sizes)

    @property
    def targets(self):
        """Return all class labels across all datasets."""
        return torch.cat([dataset.targets for dataset in self.datasets])

    @property
    def domain_ids(self):
        """Return domain ID for each segment (dataset index)."""
        domain_ids = []
        for i, dataset in enumerate(self.datasets):
            domain_ids.extend([i] * len(dataset))
        return torch.tensor(domain_ids)

    def __len__(self):
        """Total number of segments across all datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        """Get item from the appropriate dataset."""
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    def get_dataset_by_name(self, name):
        """Get a dataset by its name."""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None

    def get_blacklisted_count(self):
        """Get the total number of blacklisted segments across all datasets."""
        return sum(dataset.get_blacklisted_count() for dataset in self.datasets)

    def __repr__(self):
        dataset_info = ", ".join(
            [
                f"{dataset.name}({len(dataset)} segments, {dataset.get_blacklisted_count()} blacklisted)"
                for dataset in self.datasets
            ]
        )
        return f"MultiFeatureDataset({len(self.datasets)} datasets: {dataset_info})"
