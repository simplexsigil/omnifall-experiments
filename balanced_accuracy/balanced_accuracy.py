"""Balanced Accuracy Metric"""

import datasets
from sklearn.metrics import balanced_accuracy_score
import evaluate


_CITATION = """\
@article{article,
    author = {Brodersen, K. H. and Ong, C. S. and Stephan, K. E. and Buhmann, J. M.},
    title = {The balanced accuracy and its posterior distribution},
    year = {2010},
    journal = {Proceedings of the 20th International Conference on Pattern Recognition}
}
"""

_DESCRIPTION = """\
Balanced Accuracy is a metric used for binary and multiclass classification problems to deal with imbalanced datasets.
It is defined as the average of recall obtained on each class. It is particularly useful when the dataset classes are imbalanced.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: array-like of shape (n_samples,)
        Predicted labels.
    references: array-like of shape (n_samples,)
        Ground truth (correct) labels.
    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.
Returns:
    balanced_accuracy: float
        Balanced accuracy score. The best value is 1.0, and the worst value is 0.0.
Examples:
    >>> balanced_accuracy_metric = BalancedAccuracy()
    >>> predictions = [0, 1, 0, 1]
    >>> references = [0, 1, 1, 1]
    >>> results = balanced_accuracy_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    >>> {'balanced_accuracy': np.float64(0.8333333333333333)}
"""


def compute_balanced_accuracy(y_true, y_pred, *, sample_weight=None):
    """Compute balanced accuracy using sklearn's balanced_accuracy_score."""
    return balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BalancedAccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html"
            ],
        )

    def _get_feature_types(self):
        return {
            "predictions": datasets.Value("int64"),
            "references": datasets.Value("int64"),
        }

    def _compute(self, predictions, references, sample_weight=None):
        balanced_accuracy = compute_balanced_accuracy(
            y_true=references,
            y_pred=predictions,
            sample_weight=sample_weight,
        )

        return {"balanced_accuracy": balanced_accuracy}
