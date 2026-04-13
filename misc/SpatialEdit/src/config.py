from dataclasses import dataclass, field
import json
import importlib.util
import inspect
from pathlib import Path

from src.distributed.parallel_states import print_rank0


@dataclass
class ExpConfig:
    seed: int = 42

    # DIT
    dit_ckpt: str = None
    dit_ckpt_type: str = "pt"
    dit_arch_config: str = None
    dit_precision: str = "bf16"
    is_repa: bool = False
    repa_layer: int = 20
    repa_lambda: float = 0.5
    repa_aligh: str = 'patch'

    # VAE
    vae_arch_config: str = None
    vae_precision: str = "bf16"
    enable_denormalization: bool = False

    # Text Encoder
    text_encoder_arch_config: str = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 512

    # Scheduler
    scheduler_arch_config: str = None

    # Inference
    use_lora: bool = False
    lora_rank: int = 16
    training_mode: bool = False
    use_fsdp_inference: bool = False
    hsdp_shard_dim: int = 1
    sp_size: int = 1
    enable_activation_checkpointing: bool = False

    def __post_init__(self):
        pass

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def load_config_class_from_pyfile(file_path: str):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ExpConfig) and obj is not ExpConfig:
            print_rank0(
                f"Dynamically loaded config class: '{obj.__name__}' from '{file_path}'")
            return obj

    raise ValueError(
        f"No class inheriting from 'ExpConfig' was found in '{file_path}'.")
