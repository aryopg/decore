from dataclasses import dataclass
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfigs:
    name: str = MISSING
    data_dir: str = MISSING
    num_samples: int = MISSING
    variation: Optional[str] = None


@dataclass
class DataLoaderConfigs:
    batch_size: int = MISSING
    num_workers: int = MISSING
    drop_last: bool = MISSING
    pin_memory: bool = MISSING


@dataclass
class ModelConfigs:
    name: str = MISSING
    model_type: str = MISSING
    configs: dict = MISSING


@dataclass
class DecoderConfigs:
    name: str = MISSING
    method: str = MISSING
    configs: Optional[dict] = None


@dataclass
class RunnerConfigs:
    data: DataConfigs = MISSING
    data_loader: DataLoaderConfigs = MISSING
    decoder: DecoderConfigs = MISSING
    model: ModelConfigs = MISSING
    wandb_project: str = MISSING
    wandb_entity: str = MISSING
    count_flops: bool = False
    debug: bool = False
    random_seed: int = 1234


def register_base_configs() -> None:
    configs_store = ConfigStore.instance()
    configs_store.store(name="base_config", node=RunnerConfigs)
    configs_store.store(group="data", name="base_data_config", node=DataConfigs)
    configs_store.store(
        group="data_loader", name="base_data_loader_config", node=DataLoaderConfigs
    )
    configs_store.store(group="model", name="base_model_config", node=ModelConfigs)
    configs_store.store(
        group="decoder", name="base_decoder_config", node=DecoderConfigs
    )
