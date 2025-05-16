import torch
from torch.utils.data import WeightedRandomSampler
from data.feature_dataset import MultiFeatureDataset
import logging


def get_domain_weighted_sampler(dataset: MultiFeatureDataset, max_cap: float = 10.0) -> WeightedRandomSampler:
    """
    Create a domain-weighted sampler that gives equal importance to different domains.

    Args:
        dataset: Dataset with domain_ids property
        max_cap: Maximum oversampling factor (default: 10.0)

    Returns:
        WeightedRandomSampler instance
    """
    domains = dataset.domain_ids
    N = len(domains)

    # Calculate domain weights (with capping)
    domain_counts = torch.bincount(domains)
    max_count = domain_counts.max().item()
    oversample = torch.tensor(max_count) / domain_counts.float()  # ideal factor
    logging.info(f"Domain counts: {domain_counts}")
    logging.info(f"Max count: {max_count}")
    logging.info(f"Initial oversample: {oversample}")

    oversample = oversample.clamp(max=max_cap)  # cap at max_cap
    logging.info(f"Capped oversample: {oversample}")

    # Get per-sample weights
    sample_weights = oversample[domains]

    # Create sampler
    sampler = WeightedRandomSampler(sample_weights.double(), num_samples=N, replacement=True)  # 1 pseudo-epoch

    return sampler
