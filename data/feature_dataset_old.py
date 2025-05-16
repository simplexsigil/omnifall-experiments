import random
import h5py
import copy
from data.dataset import GenericVideoDataset


def make_h5py_dataset(dataset: GenericVideoDataset, path_replacement: tuple[str, str]) -> GenericVideoDataset:
    """
    Create an HDF5 dataset from a GenericVideoDataset.
    Args:
        dataset (GenericVideoDataset): The dataset to convert.
        path_replacement (str): Path replacement for saving the HDF5 file.
    Returns:
        H5FeatureDataset: The converted dataset.
    """

    def get_h5_item(self, index):
        return self.load_item(index)

    def load_h5py(path, idx):
        """
        Load a video from an HDF5 file.
        Args:
            path (str): Path to the HDF5 file.
        Returns:
            np.ndarray: Loaded video frames.
        """
        if not path.endswith(".h5"):
            path = path.replace(".avi", ".h5").replace(".mp4", ".h5")
            path = path.replace(*path_replacement)
        with h5py.File(path, "r") as f:
            video_data = f["features"][()]
            return {"features": video_data}

    ds_copy = copy.copy(dataset)
    ds_copy.load_video = load_h5py
    ds_copy.transform_frames = lambda x: x
    # Use setattr to dynamically replace the method of the class object
    setattr(ds_copy, "__getitem__", get_h5_item)
    return ds_copy


def get_random_timestamps(ds, n, length=4.0):
    """
    Get random timestamps from the dataset.
    Args:
        ds (GenericVideoDataset): The dataset to get timestamps from.
        n (int): Number of timestamps to get.
    Returns:
        list: List of random timestamps and video path.
    """
    annotations = [ds]
