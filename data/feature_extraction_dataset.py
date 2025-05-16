import os
import av
from torch.utils.data import IterableDataset, get_worker_info
import itertools
import numpy as np
import logging
from data.dataset import GenericVideoDataset
import torch


class SlidingClipIterableDataset(IterableDataset):
    """
    An IterableDataset that yields clips of video frames from a GenericVideoDataset.
    In contrast to the original dataset, this class loads all frames of a video with a given stride and yields
    clips of a specified length. This is useful for tasks where you want to process the entire video in chunks.
    """

    def __init__(
        self,
        generic_video_dataset: GenericVideoDataset,
        stride,
        path_replacement_for_skipping: str = None,
    ):
        """
        Args:
            generic_video_dataset (GenericVideoDataset): The dataset to iterate over.
            stride (int): The number of frames to skip after yielding a clip.
            path_replacement_for_skipping (str, optional): Path replacement for skipping videos.
        """
        self.generic_video_dataset = generic_video_dataset
        self.stride = stride
        self.video_paths = list(set(generic_video_dataset.paths))
        self.vid_frame_count = self.generic_video_dataset.vid_frame_count
        self.path_replacement_for_skipping = path_replacement_for_skipping

    def get_clip(self, frame_buffer, time_buffer, video_path, clip_idx):
        clip = frame_buffer[: self.vid_frame_count]
        clip = np.stack(clip, axis=0)  # [T, H, W, C]
        clip = torch.from_numpy(clip)
        inputs = self.generic_video_dataset.transform_frames(clip)
        inputs |= {
            "path": video_path,
            "clip_idx": clip_idx,
            "last_clip": False,
            "time": torch.from_numpy(np.stack(time_buffer[: self.vid_frame_count], axis=0)),
        }
        return inputs

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process loading
            iter_start = 0
            iter_end = len(self.video_paths)
        else:
            # Split workload
            per_worker = int(np.ceil(len(self.generic_video_dataset) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.video_paths))

        for idx in range(iter_start, iter_end):
            try:
                video_path = os.path.join(self.generic_video_dataset.video_root, self.video_paths[idx])
                if self.path_replacement_for_skipping is not None:
                    save_path = os.path.splitext(video_path)[0].replace(*self.path_replacement_for_skipping) + ".h5"
                    if os.path.exists(save_path):
                        logging.warning(f"Skipping video {video_path} as it already exists at {save_path}.")
                        continue

                frame_buffer = []
                time_buffer = []
                clip_idx = 0
                for frame, time in self.iter_frames(video_path, fps=self.generic_video_dataset.target_fps):
                    frame_buffer.append(frame)
                    time_buffer.append(time)
                    # Load self.vid_frame_count + self.stride frames, then yield the first self.vid_frame_count frames
                    if len(frame_buffer) == self.vid_frame_count + self.stride:
                        yield self.get_clip(frame_buffer, time_buffer, video_path, clip_idx)
                        frame_buffer = frame_buffer[self.stride :]
                        time_buffer = time_buffer[self.stride :]
                        clip_idx += 1
                # Handle the last clip
                if len(frame_buffer) < self.vid_frame_count:
                    # Original video clip was too short, cycle through the buffer to fill it
                    while len(frame_buffer) < self.vid_frame_count:
                        frame_buffer.append(frame_buffer[len(frame_buffer) % len(frame_buffer)])
                        time_buffer.append(time_buffer[len(time_buffer) % len(time_buffer)])

                inputs = self.get_clip(frame_buffer, time_buffer, video_path, clip_idx)
                inputs["last_clip"] = True
                yield inputs

            except Exception as e:
                logging.warning(f"Skipping video {self.video_paths[idx]} due to error: {e}")
                continue

    def iter_frames(self, video_path, fps=None):
        with av.open(video_path) as container:
            video_stream = next(s for s in container.streams if s.type == "video")
            i, delta = 0, 1.0 / float(fps) if fps is not None else 0.0
            for frame in container.decode(video_stream):
                if frame.time < i * delta:
                    continue
                i += 1
                yield frame.to_ndarray(format="rgb24"), frame.time
