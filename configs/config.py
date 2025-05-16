from dataclasses import dataclass, field
from typing import List, Optional, Union
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class AnnotationConfigSingleLabel:
    ann_type: str = MISSING
    ann_name: str = MISSING
    annf_train: str = MISSING
    annf_val: str = MISSING
    annf_test: str = MISSING
    path_format: str = MISSING
    attribute: str = MISSING
    num_classes: int = MISSING
    encoding_file: int = MISSING


@dataclass
class AnnotationConfigMultiLabel:
    ann_type: str = MISSING
    ann_name: str = MISSING
    annf_train: str = MISSING
    annf_val: str = MISSING
    annf_test: str = MISSING
    attributes: List[str] = MISSING
    num_classes: List[int] = MISSING
    encoding_file: int = MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    dataset_fps: List[float] = MISSING
    model_fps: float = MISSING
    num_frames: int = MISSING
    video_root_train: str = MISSING
    video_root_val: str = MISSING
    video_root_test: str = MISSING
    path_format_train: str = MISSING
    path_format_val: str = MISSING
    path_format_test: str = MISSING
    annotations: Union[AnnotationConfigSingleLabel, AnnotationConfigMultiLabel] = MISSING


@dataclass
class ModelConfig:
    type: str = MISSING
    model_type: str = MISSING
    name_or_path: str = MISSING
    out_features: int = MISSING
    num_frames: int = MISSING
    model_fps: float = MISSING


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    weight_decay: float = MISSING
    learning_rate: float = MISSING
    batch_size_scaling: int = MISSING
    lr_schedule: str = MISSING


@dataclass
class TrainingConfig:
    num_epochs: int = MISSING
    batch_size: int = MISSING
    optimizer: OptimizerConfig = MISSING


@dataclass
class EvaluationConfig:
    metric: str = "top1"
    clips_per_video: int = MISSING
    crops_per_clip: int = MISSING


@dataclass
class HardwareConfig:
    gpus: List[int] = None


@dataclass
class Config:
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    hardware: HardwareConfig = MISSING


# cs = ConfigStore.instance()
# cs.store(name="config", node=Config)
