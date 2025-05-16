from typing import Optional
from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from torch.utils.data import DataLoader
import torch
import os
import psutil, math

from torch.utils.data import RandomSampler

# Add necessary imports for the new get_train_dataloader
from transformers.trainer_utils import seed_worker
from transformers.utils.import_utils import is_datasets_available
import torch.nn.functional as F

from data.transforms.tube_masking import TubeMasker
from data.transforms.fourier import PhaseOnly, SpectralTransform
from models.cb_loss import ClassBalancedLoss

# Import datasets if needed, handle potential absence
if is_datasets_available():
    import datasets


def default_worker_init_fn(worker_id):
    """
    Default worker_init_fn that sets thread settings if num_threads is specified.
    This function needs to be defined at the top level to be pickleable.
    """

    # ---------- autodetect + heuristics ----------
    logical_cores = psutil.cpu_count(logical=True)  # e.g. 128
    num_threads = max(2, logical_cores // 16)  # e.g. 16 workers, 4 gpus and 4 workers each in total.

    # ---------- apply settings ----------
    os.system(
        "taskset -p 0xffffffffffffffffffffffffffffffffffffff %d   > /dev/null 2>&1" % os.getpid()
    )  # Removed as it might be not portable/necessary
    # os.system("taskset -p %d" % os.getpid())  # Removed

    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(num_threads)
    # Redundant call, torch.set_num_threads is sufficient for intra-op
    # torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)  # Set inter-op parallelism

    # print(f"Worker {worker_id} initialized with {num_threads} threads.")

    seed_worker(worker_id)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    except AttributeError:
        # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
        return False


class ThreadAwareTrainer(Trainer):
    """
    Trainer subclass that initializes each DataLoader worker to set PyTorch thread settings.
    Also supports domain-weighted sampling and custom loss functions.
    """

    def __init__(self, *args, num_threads: int = None, fourier=True, train_sampler=None, 
                 loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store num_threads, but the actual init logic is now in the top-level function
        self.num_threads = num_threads
        # Set the worker_init_fn to use if num_threads is specified
        self.custom_worker_init_fn = default_worker_init_fn
        # Store custom sampler if provided
        self.train_sampler = train_sampler
        # Custom loss function if provided
        self.loss_function = loss_function

        self.fourier_trans = PhaseOnly(in_arrangement="t c h w") if fourier else None
        input_size = (8, 14, 14)  # Patches in (T, H, W)
        mask_ratio = 0.9  # Ratio of patches to mask
        self.tube_masker = TubeMasker(input_size, mask_ratio)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use the custom train_sampler if provided, otherwise will use no sampler 
        if `train_dataset` does not implement `__len__`, or a random sampler
        (adapted to distributed training if necessary).

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": self.custom_worker_init_fn,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # Use custom sampler if provided, otherwise use default sampler
            if self.train_sampler is not None:
                dataloader_params["sampler"] = self.train_sampler
            else:
                dataloader_params["sampler"] = self._get_train_sampler()
                
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            inputs.pop("labels", None)

        # If the model accepts loss kwargs, we pass them in
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        if self.fourier_trans is not None:
            inputs = self.fourier_trans(inputs)
        # inputs = self.tube_masker(inputs)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.processing_class.model_input_names[0] if self.processing_class is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)
