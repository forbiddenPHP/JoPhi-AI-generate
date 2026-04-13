from dataclasses import dataclass, field
from src.config import ExpConfig


@dataclass
class SpatialEditConfig(ExpConfig):
    seed: int = 42

    # DIT
    dit_ckpt: str = "models/SpatialEdit-16B"
    dit_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.dit.Transformer3DModel",
        "params": {
            "hidden_size": 4096,
            "in_channels": 16,
            "heads_num": 32,
            "mm_double_blocks_depth": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "rope_dim_list": [16, 56, 56],
            "text_states_dim": 4096,
            "rope_type": "rope",
            "dit_modulation_type": "wanx",
            "unpatchify_new": True,
        }
    })
    dit_precision: str = "bf16"
    is_repa: bool = False

    # VAE
    vae_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.vae.WanxVAE",
        "params": {
            "pretrained": "models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        },
    })
    vae_precision: str = "bf16"

    # Text Encoder
    text_encoder_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.text_encoder.load_text_encoder",
        "params": {
            "text_encoder_ckpt": "models/Qwen3-VL-8B-Instruct",
        },
    })
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    # Scheduler
    scheduler_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.common.diffusion.schedulers.FlowMatchDiscreteScheduler",
        "params": {
            "num_train_timesteps": 1000,
            "shift": 1.5,
        },
    })

    # Inference
    use_lora: bool = True
    lora_rank: int = 16
    hsdp_shard_dim: int = 1
