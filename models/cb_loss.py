import torch
import torch.nn.functional as F


class ClassBalancedLoss:
    """
    Class-balanced loss function from Cui et al., 2019.
    Handles imbalanced datasets by weighting losses based on effective number of samples.

    Args:
        beta: Hyperparameter for computing effective number of samples (default: 0.999)
        class_weights: Optional pre-computed class weights tensor
        num_classes: Number of classes (used if class_weights is None)
    """

    def __init__(self, beta=0.999, class_weights=None, labels=None, label_smoothing=0.1):
        self.beta = beta
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        # If labels are provided but not weights, calculate weights once
        if class_weights is None and labels is not None:
            self.class_weights = self.get_class_weights(labels)
            print(f"Initialized class weights: {self.class_weights}")

    @staticmethod
    def get_class_weights_from_dataset(dataset):
        """
        Calculate class weights from a dataset with a 'targets' property.

        Args:
            dataset: Dataset with a targets property returning class indices

        Returns:
            Tensor of class weights
        """
        if not hasattr(dataset, "targets"):
            raise ValueError("Dataset must have a 'targets' property")
        return ClassBalancedLoss.get_class_weights(dataset.targets)

    @staticmethod
    def get_class_weights(labels, beta=0.999):
        """
        Calculate class weights based on effective number of samples.

        Args:
            labels: Tensor of class labels
            beta: Hyperparameter for computing effective number of samples

        Returns:
            Tensor of class weights
        """
        class_counts = torch.bincount(labels)
        effective_num = 1.0 - torch.pow(beta, class_counts.float()).clamp_min(1e-8)
        cls_weights = (1.0 - beta) / effective_num
        cls_weights = cls_weights / cls_weights.sum() * len(class_counts)  # Scale ≈ C
        return cls_weights

    def __call__(self, logits, targets):
        """
        Compute class-balanced cross-entropy loss.

        Args:
            logits: Model output logits
            targets: Ground truth class indices

        Returns:
            Loss tensor
        """
        # Calculate weights if not pre-computed
        if self.class_weights is None:
            self.class_weights = self.get_class_weights(targets, self.beta)

        weight = self.class_weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weight, label_smoothing=self.label_smoothing)
