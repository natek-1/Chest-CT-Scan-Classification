import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_file: str


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_filename: str
    updated_base_model_filename: str
    augmentation: bool
    image_size: List
    batch_size: int
    include_top: bool
    epochs: int
    classes: int
    weights: str
    learning_rate: float

