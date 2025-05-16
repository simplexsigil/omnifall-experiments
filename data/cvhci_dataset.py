from data.dataset import GenericVideoDataset
import csv


class CVHCIDataset(GenericVideoDataset):

    def _load_annotations_from_file(self, annotations_file):
        """
        Load annotations from the given file.
        Args:
            annotations_file (str): Path to the annotations file.
        Returns:
            list: List of annotations.
        """
        with open(annotations_file, "r") as f:
            reader = csv.reader(f)
            return [(path, int(label), float(start), float(end)) for path, label, start, end in reader]

    def _id2label(self, idx):
        return self.annotations[idx][:2]

    def load_annotations(self, annotations_file):
        """
        Load annotations from the given file or list files.
        Args:
            annotations_file (str | list): Path to the annotations file.
        Returns:
            dict: Dictionary of annotations.
        """
        if isinstance(annotations_file, list):
            annotations = {}
            for file in annotations_file:
                annotations |= self._load_annotations_from_file(file)
            self.annotations = annotations
        else:
            self.annotations = self._load_annotations_from_file(annotations_file)
        self.paths = [a[0] for a in self.annotations]

    def get_random_offset(self, length, target_interval, idx, fps, start=0):
        # For using video clip annotations rather than the whole video
        # Calculate the length and start frames
        start_time, end_time = self.annotations[idx][2:4]
        duration_time = end_time - start_time

        start_frames = start_time * fps
        duration_frames = duration_time * fps

        return super().get_random_offset(duration_frames, target_interval, idx, fps, start=start_frames)
