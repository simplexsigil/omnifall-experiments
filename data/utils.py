class FeatureHelper:
    def __init__(self, extractor_fps=2, extractor_frames=4, time_stride=1 / 12, features_centered=False):
        """
        Helper class for time-wise feature extraction.

        Feature extraction happens at frame 0 and advances by time_stride seconds.
        If features are centered, these time points define the center of the feature window,
        meaning that the feature window starts at time -window_duration/2 and ends at time +window_duration/2.
        If features are not centered, the time points define the start of the feature window,
        meaning that the feature window starts at time and ends at time + window_duration.
        Args:
            extractor_fps (int): Frames per second of the feature extractor
            extractor_frames (int): Number of frames in each feature window
            time_stride (float): Time stride in seconds between consecutive features
            features_centered (bool): Whether the features are centered around the timestamp
        """
        self.extractor_fps = extractor_fps
        self.extractor_frames = extractor_frames
        self.window_duration = extractor_frames * 1 / extractor_fps
        self.features_centered = features_centered
        self.time_stride = time_stride

    def feature_idx_at(self, timestamp):
        """
        Get the index of the feature whose center is closest to the given timestamp

        Args:
            timestamp (float): Timestamp in seconds
        Returns:
            int: Index of the feature whose center is closest to the timestamp
        """
        if self.features_centered:
            # For centered features, the feature index directly corresponds to time/stride
            return max(0, round(timestamp / self.time_stride))
        else:
            # For non-centered features, we need to adjust the timestamp to account for
            # the fact that the center of the feature is at time + window_duration/2
            adjusted_timestamp = timestamp - (self.window_duration / 2)
            return max(0, round(adjusted_timestamp / self.time_stride))

    def center_time_at(self, feature_index):
        """
        Get the center timestamp of a feature at a given index.
        Args:
            feature_index (int): Feature index
        Returns:
            float: Center timestamp of the feature
        """
        if self.features_centered:
            return feature_index * self.time_stride
        else:
            return feature_index * self.time_stride + self.window_duration / 2

    def start_time_at(self, feature_index):
        """
        Get the start timestamp of a feature at a given index.
        Args:
            feature_index (int): Feature index
        Returns:
            float: Start timestamp of the feature
        """
        if self.features_centered:
            return feature_index * self.time_stride - self.window_duration / 2
        else:
            return feature_index * self.time_stride

    def end_time_at(self, feature_index):
        """
        Get the end timestamp of a feature at a given index.
        Args:
            feature_index (int): Feature index
        Returns:
            float: End timestamp of the feature
        """
        if self.features_centered:
            return feature_index * self.time_stride + self.window_duration / 2
        else:
            return feature_index * self.time_stride + self.window_duration

    def last_frame_start_time_at(self, feature_index):
        """
        This differs from end_time_at_index, since it does not consider the last frame duration,
        so it is the timestamp of the last frame rather than the end of the feature window.
        It is useful for approximating the time of the last frame of the video.
        With accuracy of roughly time_stride.
        """
        # The last frame starts at (extractor_frames - 1) / extractor_fps before the end
        frame_duration = 1 / self.extractor_fps
        last_frame_offset = (self.extractor_frames - 1) * frame_duration

        if self.features_centered:
            # If centered, the end is at time + window_duration/2
            return feature_index * self.time_stride + self.window_duration / 2 - frame_duration
        else:
            # If not centered, the end is at time + window_duration
            return feature_index * self.time_stride + last_frame_offset


if __name__ == "__main__":
    print("\n===== I3D Feature Helper (centered features) =====")
    feature_helper = FeatureHelper(extractor_fps=30, extractor_frames=16, time_stride=1 / 30, features_centered=True)

    print(f"\nConfiguration:")
    print(f"  FPS: {feature_helper.extractor_fps}")
    print(f"  Frames per window: {feature_helper.extractor_frames}")
    print(f"  Time stride: {feature_helper.time_stride:.4f} seconds")
    print(f"  Window duration: {feature_helper.window_duration:.4f} seconds")
    print(f"  Features centered: {feature_helper.features_centered}")

    # Example with timestamp 0.5s
    for timestamp in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        feature_idx = feature_helper.feature_idx_at(timestamp)
        print(f"\nFor timestamp {timestamp}s:")
        print(f"  Feature index: {feature_idx}")
        print(f"  Feature center: {feature_helper.center_time_at(feature_idx):.4f}s")
        print(f"  Feature start: {feature_helper.start_time_at(feature_idx):.4f}s")
        print(f"  Feature end: {feature_helper.end_time_at(feature_idx):.4f}s")

    for idx in [0, 1, 2]:
        print(f"\nFor feature index {idx}:")
        print(f"  Center time: {feature_helper.center_time_at(idx):.4f}s")
        print(f"  Start time: {feature_helper.start_time_at(idx):.4f}s")
        print(f"  End time: {feature_helper.end_time_at(idx):.4f}s")
        print(f"  Last frame time: {feature_helper.last_frame_start_time_at(idx):.4f}s")

    print("\n===== Invid Feature Helper =====")
    feature_helper = FeatureHelper(extractor_fps=2, extractor_frames=4, time_stride=1 / 12, features_centered=False)

    print(f"\nConfiguration:")
    print(f"  FPS: {feature_helper.extractor_fps}")
    print(f"  Frames per window: {feature_helper.extractor_frames}")
    print(f"  Time stride: {feature_helper.time_stride:.4f} seconds")
    print(f"  Window duration: {feature_helper.window_duration:.4f} seconds")
    print(f"  Features centered: {feature_helper.features_centered}")

    # Example with timestamp 0.5s
    for timestamp in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        feature_idx = feature_helper.feature_idx_at(timestamp)
        print(f"\nFor timestamp {timestamp}s:")
        print(f"  Feature index: {feature_idx}")
        print(f"  Feature center: {feature_helper.center_time_at(feature_idx):.4f}s")
        print(f"  Feature start: {feature_helper.start_time_at(feature_idx):.4f}s")
        print(f"  Feature end: {feature_helper.end_time_at(feature_idx):.4f}s")

    for idx in [0, 1, 2]:
        print(f"\nFor feature index {idx}:")
        print(f"  Center time: {feature_helper.center_time_at(idx):.4f}s")
        print(f"  Start time: {feature_helper.start_time_at(idx):.4f}s")
        print(f"  End time: {feature_helper.end_time_at(idx):.4f}s")
        print(f"  Last frame time: {feature_helper.last_frame_start_time_at(idx):.4f}s")
