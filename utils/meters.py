from enum import Enum
import torch
import numpy as np

import datetime
import os
from collections import defaultdict, deque

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score
import logging

# Set up local logging
logger = logging.basicConfig(
    filename="local_logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        torch.dist.all_reduce(total, torch.dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
        replace_with=((-1, 0),),
        do_stats=True,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.idx_mask = np.array([False] * num_videos)
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_feats = None
        self.replace_with = replace_with
        self.do_stats = do_stats

        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = torch.zeros((num_videos, num_cls)) if multi_label else torch.zeros((num_videos)).long()

        self.meta = {}

        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_meta(self, meta):
        for key, val in meta.items():
            if key not in self.meta:
                if isinstance(val, torch.Tensor):
                    self.meta[key] = [float(v) for v in val]
                else:
                    self.meta[key] = val
            else:
                if isinstance(val, (list,)):
                    self.meta[key].extend(val)
                elif isinstance(val, torch.Tensor):
                    self.meta[key].extend([v.item() for v in val])

    def update_stats(self, preds: torch.Tensor, labels: torch.Tensor, clip_ids, feats=None, meta=None):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for k, v in self.replace_with:
            if any(labels == k):
                print(f"Replacing {k} with {v}")
                labels[[labels == k]] = v

        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.idx_mask[vid_id] = True
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[ind])
            else:
                raise NotImplementedError("Ensemble Method {} is not supported".format(self.ensemble_method))

            if feats is not None:
                if self.video_feats is None:
                    self.video_feats = torch.zeros((self.video_labels.shape[0], feats.shape[-1]))

                self.video_feats[vid_id] += feats[ind]

            self.clip_count[vid_id] += 1

        self.update_meta(meta)

    def log_iter_stats(self, cur_iter, ks=(1,)):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        if self.do_stats:
            curr_preds, curr_labs = self.video_preds[self.idx_mask], self.video_labels[self.idx_mask]

            num_topks_correct = topks_correct(curr_preds, curr_labs, ks)

            topks = [(x / np.sum(self.idx_mask)) * 100.0 for x in num_topks_correct]

            curr_preds_idxs = curr_preds.argmax(dim=1)
            bal_acc = balanced_accuracy_score(curr_labs, curr_preds_idxs) * 100

            stats = {
                "split": "test_iter",
                "cur_iter": "{}".format(cur_iter + 1),
                "cur_acc": topks[0].cpu().numpy().item(),
                "cur_bal_acc": bal_acc,
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
            }
        else:
            stats = {
                "split": "test_iter",
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
            }

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        clip_check = self.clip_count == self.num_clips
        if not all(clip_check):
            logger.warning(
                "clip count Ids={} = {} (should be {})".format(
                    np.argwhere(~clip_check),
                    self.clip_count[~clip_check],
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            mean_ap = get_map(self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy())
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        elif self.do_stats:
            num_topks_correct = topks_correct(self.video_preds, self.video_labels, ks)
            topks = [(x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct]

            assert len({len(ks), len(topks)}) == 1

            for k, topk in zip(ks, topks):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)

            preds_idxs = self.video_preds.argmax(dim=1)
            bal_acc = balanced_accuracy_score(self.video_labels, preds_idxs) * 100

            self.stats["bal_acc"] = bal_acc

        logging.log_json_stats(self.stats)

        if self.video_feats is not None:
            self.video_feats = self.video_feats / self.num_clips

        for key, val in self.meta.items():
            if isinstance(val, (list,)):
                self.meta[key] = val[:: self.num_clips]
            else:
                raise ValueError
