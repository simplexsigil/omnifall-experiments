import concurrent.futures
import logging
import os

import time
import av
import hydra
import numpy as np
import torch
from av.video.frame import VideoFrame
from omegaconf import OmegaConf

from configs.config import Config
from data.dataset import GenericVideoDataset
from data.fourier import PhaseOnly


def get_dataset_paths(cfg, mode, run=None):
    """
    Get the dataset based on the configuration.
    """
    logging.info(f"Loading GenericVideoDataset for {mode} mode.")
    dataset = GenericVideoDataset(
        cfg.dataset.get(f"video_root_{mode}"),
        cfg.dataset.annotations.get(f"annf_{mode}"),
        target_fps=cfg.dataset.model_fps,
        vid_frame_count=cfg.dataset.num_frames,
        path_format=cfg.dataset.get(f"path_format_{mode}"),
        mode=mode,
    )
    return {
        dataset.path_format.format(video_root=dataset.video_root, filename=path) for path in dataset.annotations.keys()
    }


def load_video(video_path):
    try:
        with av.open(video_path) as container:
            video_stream = next(s for s in container.streams if s.type == "video")
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video_stream)]
            return (
                torch.tensor(np.array(frames), dtype=torch.uint8),
                video_stream.average_rate,
            )
    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return None


def transform_video(video_tensor, transform):
    if video_tensor is None:
        return None
    return transform(video_tensor.float())


def write_video(video_tensor, average_rate, output_path):
    if video_tensor is None:
        return
    video_tensor = video_tensor.clamp(0.0, 255.0).byte().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with av.open(output_path, "w") as container:
        stream = container.add_stream("h264", rate=average_rate or 30)
        stream.options = {
            "crf": "0",  # Constant Rate Factor 0 (lossless)
            "preset": "veryslow",  # Better compression at cost of speed
            "x264-params": "qp=0",  # Also enforce QP=0 for true lossless
        }
        for frame in video_tensor:
            frame = VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_tensor_h5(video_tensor, average_rate, output_path):
    import h5py

    if video_tensor is None:
        return
    video_tensor = video_tensor.numpy()
    output_path = output_path.replace(".avi", ".h5").replace(".mp4", ".h5")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        # Chunk by frame
        chunks = (1, *video_tensor.shape[1:])
        f.create_dataset(
            "video",
            shape=video_tensor.shape,
            dtype=video_tensor.dtype,
            # chunks=chunks,
            compression="lzf",
        )
        f["video"][:] = video_tensor
        if average_rate is not None:
            f.create_dataset("average_rate", data=average_rate)


def process_video(cfg, video_path, transform, denorm_mean=None, denorm_std=None, write_h5=False):

    # Time load_video
    start_time = time.time()
    result = load_video(video_path)
    if result is None:
        return None
    video_tensor, average_rate = result
    load_time = time.time() - start_time

    # Time transform_video
    start_time = time.time()
    transformed_video = transform_video(video_tensor, transform)
    transform_time = time.time() - start_time

    if transformed_video is None:
        return None

    # Denormalize
    if denorm_mean is not None and denorm_std is not None:
        transformed_video = transformed_video * denorm_std + denorm_mean
    max_val = transformed_video.max().item()
    min_val = transformed_video.min().item()

    # Time write_* operation
    start_time = time.time()
    output_path = video_path.replace(cfg.dataset.video_root_train, cfg.dataset.fourier_root)
    if write_h5:
        write_tensor_h5(transformed_video, average_rate, output_path)
    else:
        write_video(transformed_video, average_rate, output_path)
    write_time = time.time() - start_time

    return max_val, min_val, load_time, transform_time, write_time


def run_parallel(cfg, transform, files, denorm_mean, denorm_std, h5=False):

    max_val, min_val = float("-inf"), float("inf")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_video,
                cfg,
                path,
                transform,
                denorm_mean,
                denorm_std,
                write_h5=h5,
            ): path
            for path in files
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                continue
            local_max, local_min = result[:2]
            if local_max > max_val:
                max_val = local_max
                print(f"New Max value: {max_val}")
            if local_min < min_val:
                min_val = local_min
                print(f"New Min value: {min_val}")
            load_time, transform_time, write_time = result[2:]
            print(
                f"Processed {futures[future]}: Load time: {load_time:.2f}s, "
                f"Transform time: {transform_time:.2f}s, Write time: {write_time:.2f}s"
            )
        return max_val, min_val


def process_sequential(cfg, transform, files, denorm_mean, denorm_std, h5=False):
    max_val, min_val = float("-inf"), float("inf")

    for path in files:
        result = process_video(cfg, path, transform, denorm_mean, denorm_std, write_h5=h5)
        if result is None:
            continue
        local_max, local_min = result[:2]
        if local_max > max_val:
            max_val = local_max
            print(f"New Max value: {max_val}")
        if local_min < min_val:
            min_val = local_min
            print(f"New Min value: {min_val}")
        load_time, transform_time, write_time = result[2:]
        print(
            f"Processed {path}: Load time: {load_time:.2f}s, "
            f"Transform time: {transform_time:.2f}s, Write time: {write_time:.2f}s"
        )

    return max_val, min_val


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: Config):
    OmegaConf.resolve(cfg)

    # Collect all video file paths
    files = get_dataset_paths(cfg, "train") | get_dataset_paths(cfg, "val") | get_dataset_paths(cfg, "test")

    transform = PhaseOnly(denormalize=None, in_arrangement="t h w c", padding=1, log_first=False)
    denorm_mean = torch.tensor([127, 127, 127]).view(1, 1, 1, 3)
    denorm_std = torch.tensor([2.5, 2.5, 2.5]).view(1, 1, 1, 3)
    # denorm_mean = None
    # denorm_std = None

    parallel = cfg.get("parallel", True)
    if parallel:
        min_val, max_val = run_parallel(cfg, transform, files, denorm_mean, denorm_std)
    else:
        min_val, max_val = process_sequential(cfg, transform, files, denorm_mean, denorm_std)

    print(f"Final Max value: {max_val}, Min value: {min_val}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
