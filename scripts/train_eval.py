import logging
import os
import torch
import psutil, math

# ---------- autodetect + heuristics ----------
physical_cores = psutil.cpu_count(logical=False)  # e.g. 128
logical_cores = psutil.cpu_count(logical=True)  # e.g. 256
reserve = max(2, math.ceil(logical_cores * 0.04))  # 4 % head-room (≥2)
threads = max(2, (logical_cores - reserve) // (16 + 1))

# ---------- apply settings ----------
torch.set_num_threads(threads)
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)

# ---------- logging ----------
# print(
#    f"physical_cores={physical_cores} logical_cores={logical_cores} reserve={reserve} threads_per_proc={threads} when using 16 workers in total"
# )


import time
from typing import Optional

import evaluate
import hydra
import numpy as np

import torch.nn as nn
import torch.multiprocessing as mp  # Added back: Needed for setting start method
import wandb
import yaml
from accelerate import Accelerator  # Added: Import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    VideoMAEConfig,
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
    VideoMAEFeatureExtractor,
    TrainerCallback,  # Keep TrainerCallback if other callbacks are used, remove if only TorchProfilerCallback was used
)

from training.ta_trainer import ThreadAwareTrainer as Trainer

import utils.assertions as assertions
from balanced_accuracy.balanced_accuracy import BalancedAccuracy
from configs.config import Config
from data.dataset_utils import get_dataset
import sys

# print(f"CPU Count: {os.cpu_count()}")
# print(f"Num threads: {torch.get_num_threads()}")  # Removed: Let accelerate/torch manage threads
# torch.set_num_threads(os.cpu_count() / 8)  # Removed
# os.system("taskset -p 0xffffffffffffffffffffffffffffffffffffff %d   > /dev/null 2>&1" % os.getpid())  # Removed
# os.system("taskset -p %d" % os.getpid())  # Removed


# print(f"Num threads: {torch.get_num_threads()}") # Removed

# Set up local logging
logging.basicConfig(
    filename="local_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_model_variant(cfg, label2id, id2label):
    """
    Get the model variant based on the configuration.
    """
    variant = cfg.model.get("variant", "video-mae")
    model, pretrain = None, False
    match variant:
        case "video-mae-finetune" | "video-mae" | "finetune":
            logging.info("Loading finetuning model.")
            model = VideoMAEForVideoClassification.from_pretrained(
                cfg.model.name_or_path,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            pretrain = False
        case "video-mae-pretrain" | "pretrain":
            logging.info("Loading pretraining model.")
            model_config = VideoMAEConfig.from_pretrained(
                cfg.model.name_or_path,
            )
            model = VideoMAEForPreTraining(model_config)
            pretrain = True
        case "from_scratch" | "skip_pretraining":
            logging.info("Loading model with skip pretraining.")
            model_config = VideoMAEConfig.from_pretrained(
                cfg.model.name_or_path,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            model_config.label2id = label2id
            model_config.id2label = id2label
            model = VideoMAEForVideoClassification(model_config)
            pretrain = False
        case "feature_extractor":
            logging.info("Loading feature extractor model.")
            model = VideoMAEFeatureExtractor.from_pretrained(cfg.model.name_or_path)
            return model, False
        case _:
            raise ValueError(f"Model variant {cfg.model.variant} is not supported.")
    if cfg.model.get("checkpoint_path", None) is not None:
        logging.info(f"Loading checkpoint from {cfg.model.checkpoint_path}.")
        from safetensors.torch import load_file

        checkpoint = load_file(cfg.model.checkpoint_path, device="cpu")
        model.load_state_dict(checkpoint, strict=False)
    return model, pretrain


def scaled_lr(cfg, accelerator=None):
    # Calculate the scaled learning rate
    base_lr = cfg.training.optimizer.get("BASE_LR", 1e-4)

    # Get global batch size (accounting for all processes)
    per_device_batch_size = cfg.training.batch_size
    if accelerator is not None:
        # When using accelerator, use its process count
        num_processes = accelerator.num_processes
    else:
        # Fallback if accelerator not provided
        num_processes = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    global_batch_size = per_device_batch_size * num_processes
    scaled_lr = base_lr * global_batch_size / 256

    if not accelerator or (accelerator and accelerator.is_main_process):
        logging.info(f"Base learning rate: {base_lr}")
        logging.info(f"Local batch size: {per_device_batch_size}")
        logging.info(f"Global batch size: {global_batch_size}")
        if scaled_lr < base_lr:
            logging.warning(
                f"Scaled learning rate ({scaled_lr}) is less than base learning rate ({base_lr}). Using base_lr as lower bound."
            )
        else:
            logging.info(f"Scaled learning rate (base * global_batch_size / 256): {scaled_lr}")

    if scaled_lr < base_lr:
        scaled_lr = base_lr

    return scaled_lr


def get_finetune_training_args(cfg, new_model_name, accelerator=None):
    lr = scaled_lr(cfg, accelerator)

    args = TrainingArguments(
        new_model_name,
        do_train=cfg.do_train,
        do_eval=cfg.do_train,
        remove_unused_columns=False,
        eval_strategy="steps" if cfg.do_train else "no",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.evaluation.batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True if cfg.do_train else False,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="wandb",
        dataloader_num_workers=cfg.hardware.num_data_workers,  # Set explicitly based on config
        # dataloader_pin_memory=True, # Keep commented: Trainer default is usually True for GPU
        # dataloader_persistent_workers=True, # Keep commented: Trainer default is False
        # dataloader_prefetch_factor=2, # Keep commented: Trainer default is usually 2
        # fp16=True, # Keep commented: Handled by accelerate config
        # optim="adamw_torch", # Keep commented: Trainer default is adamw_torch
        ddp_find_unused_parameters=False,  # Keep commented: Handled by accelerate
        seed=42,
        lr_scheduler_type=cfg.training.scheduler.name,
        max_grad_norm=cfg.training.optimizer.get("max_grad_norm", 1.0),
        label_smoothing_factor=0.1,
    )

    return args


def get_pretrain_training_args(cfg, new_model_name, accelerator=None):
    lr = scaled_lr(cfg, accelerator)

    args = TrainingArguments(
        new_model_name,
        do_train=cfg.do_train,
        do_eval=False,
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5000,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=cfg.training.batch_size,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="wandb",
        dataloader_num_workers=cfg.hardware.num_data_workers,  # Set explicitly based on config
        dataloader_pin_memory=False,  # Keep commented: Trainer default is usually True for GPU
        dataloader_persistent_workers=True,  # Keep commented: Trainer default is False
        dataloader_prefetch_factor=2,  # Keep commented: Trainer default is usually 2
        optim="adamw_torch",  # Keep commented: Trainer default is adamw_torch
        ddp_find_unused_parameters=False,
        seed=42,
        lr_scheduler_type=cfg.training.scheduler.name,
        max_grad_norm=cfg.training.optimizer.get("max_grad_norm", 1.0),
        fp16=True,
        fourier=cfg.use_fourier,
    )

    return args


# Define collate_fn at the top level so it can be pickled for mp spawn.
def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    collated = {}
    collated["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    if "bool_masked_pos" in examples[0]:
        # If bool_masked_pos is present, we are in pretraining case and don't need labels
        collated["bool_masked_pos"] = torch.stack([example["bool_masked_pos"] for example in examples])
    else:
        # Otherwise use labels and no bool_masked_pos
        collated["labels"] = torch.tensor([example["label"] for example in examples])

    return collated


def run(cfg, debug=False):
    # Added: Instantiate Accelerator
    accelerator = Accelerator()

    # Use accelerator.is_main_process for main process checks
    if accelerator.is_main_process:
        # Initialize wandb run
        job_cfg = HydraConfig.get().job
        sweep_params = job_cfg.override_dirname
        param_str = sweep_params.replace("=", "").replace(",", "_")
        run_name = f"{wandb.util.generate_id()}-{param_str}"
        run = wandb.init(
            project=str(cfg.wandb.project),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(start_method="thread"),
        )
        wandb.log(data={"Working Directory": os.getcwd()})
    else:
        run = None  # Ensure run is None on non-main processes

    # Logging should ideally also be main-process only or handled carefully in multi-process
    # For simplicity, keeping existing logging, but be aware it might write from multiple processes
    logging.info(f"Current working directory: {os.getcwd()}")

    # Load YAML file
    if cfg.dataset.annotations.get("encoding_file", None) is not None:
        with open(cfg.dataset.annotations.encoding_file, "r") as file:
            label2id = yaml.safe_load(file)
    else:
        label2id = {label: i for i, label in enumerate(range(cfg.dataset.annotations.num_classes))}

    id2label = {i: label for label, i in label2id.items()}

    # Use accelerator.is_main_process for WandB logging
    if accelerator.is_main_process and run:
        table = wandb.Table(columns=["Class", "ID"])

        for label, id_ in label2id.items():
            table.add_data(label, id_)

        # Log multiple configurations
        wandb.config.update({"Class Encodings": id2label})

    model, pretrain = get_model_variant(cfg, label2id, id2label)

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")

    if cfg.do_train:
        train_dataset = get_dataset(cfg, "train", run)
        val_dataset = get_dataset(cfg, "val", run) if not pretrain else None
    else:
        logging.info("Skipping training.")
        train_dataset, val_dataset = (None, None)

    new_model_name = f"{cfg.model.name_or_path}-finetuned-{cfg.dataset.name}"

    if debug:
        new_model_name = f"debug-{new_model_name}"
        assertions.check_dataset_model_cfg(dataset=train_dataset, model=model, cfg=cfg)
        assertions.check_dataset_model_cfg(dataset=val_dataset, model=model, cfg=cfg)

    args = (
        get_finetune_training_args(cfg, new_model_name, accelerator)
        if not pretrain
        else get_pretrain_training_args(cfg, new_model_name, accelerator)
    )

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_id = f"{rank}_{local_rank}"

    # Create metrics with unique experiment IDs
    metric_acc = evaluate.load("accuracy", experiment_id=f"accuracy_{node_id}")
    metric_bacc = BalancedAccuracy(experiment_id=f"balanced_accuracy_{node_id}")

    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        acc = metric_acc.compute(predictions=predictions, references=eval_pred.label_ids)["accuracy"]
        bacc = metric_bacc.compute(predictions=predictions, references=eval_pred.label_ids)["balanced_accuracy"]

        # if debug:
        #     assertions.check_predictions(
        #         eval_pred.label_ids, predictions
        #     )
        return {"accuracy": acc, "balanced_accuracy": bacc}

    # Use standard Trainer, not ProfilingTrainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset if cfg.do_train else None,
        eval_dataset=val_dataset if cfg.do_train else None,
        # processing_class=image_processor, # This argument is deprecated/unused in recent Trainer versions
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        # callbacks=[TorchProfilerCallback()], # Removed profiler callback
    )

    try:
        if cfg.do_train:
            if cfg.training.resume_from_checkpoint:
                logging.info(f"Resuming training from checkpoint at {cfg.training.resume_from_checkpoint}")
                # Use accelerator.is_main_process for WandB logging
                if accelerator.is_main_process and run:
                    wandb.log(data={"Resumed from checkpoint": cfg.training.resume_from_checkpoint})
                train_results = trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
            else:
                logging.info("Starting training from scratch.")
                # Use accelerator.is_main_process for WandB logging
                if accelerator.is_main_process and run:
                    wandb.log(data={"Starting training from scratch": True})
                    wandb.log(data={"Resumed from checkpoint": False})
                train_results = trainer.train()

        if cfg.do_test:
            test_dataset = get_dataset(cfg, "test", run)
            # Evaluation should ideally run on the main process after training or be handled by Trainer's distributed evaluation
            results = trainer.evaluate(test_dataset)
            logging.info(f"Final test results: {results}")
            # Use accelerator.is_main_process for WandB logging
            if accelerator.is_main_process and run:
                wandb.log(data={"Final test results": str(results)})
    except KeyboardInterrupt:
        try:
            logging.info("Training interrupted. Saving model.")
            # Use accelerator.is_main_process for WandB logging
            if accelerator.is_main_process and run:
                wandb.log(data={"Training interrupted": True})
            # Saving should be done carefully in distributed setting, Trainer handles this
            trainer.save_model(new_model_name)
            print("Interrupted run, press Ctrl+C again to exit.")
            time.sleep(2)
            if accelerator.is_main_process and wandb.run:
                wandb.finish(11)
        except KeyboardInterrupt:
            if accelerator.is_main_process and wandb.run:
                wandb.finish(111)
            raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Check if trainer exists before saving model
        if "trainer" in locals() and trainer is not None:
            # Saving should be done carefully in distributed setting, Trainer handles this
            trainer.save_model(new_model_name)
        # Use accelerator.is_main_process for WandB logging
        if accelerator.is_main_process and run:
            wandb.log(data={"Error": str(e)})
            wandb.finish(2)
        raise e
    finally:
        # Use accelerator.is_main_process for WandB logging
        if accelerator.is_main_process and run:
            run.finish()


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config",
)
def main(cfg: Config):
    OmegaConf.resolve(cfg)
    # Logging from main process is preferred
    # logging.info(OmegaConf.to_yaml(cfg)) # Consider moving inside run() guarded by is_main_process

    debug = cfg.get("debug", False)
    run(cfg, debug=debug)


if __name__ == "__main__":
    # Set start method before Hydra/main and any CUDA/DataLoader usage
    try:
        mp.set_start_method("spawn", force=True)
        logging.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # Might have already been set by Accelerate/torchrun or in a previous run
        logging.warning(f"Could not set multiprocessing start method (might be already set): {e}")

    main()
    print("Arguments for completed run/sweep:")
    print(" ".join(sys.argv))
