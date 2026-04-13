import os
import torch
import torch.distributed as dist

from src.utils.logging import get_logger
from src.utils.constants import PRECISION_TO_TYPE
from src.utils.utils import build_from_config


def _ensure_safetensors(pt_path: str) -> str:
    """Convert .pth to .safetensors if needed, return safetensors path."""
    from safetensors.torch import save_file
    import gc as _gc

    st_path = pt_path.rsplit(".", 1)[0] + ".safetensors"
    if os.path.exists(st_path):
        return st_path

    logger = get_logger()
    logger.info(f"Converting {pt_path} → safetensors (one-time)...")
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    save_file(state, st_path)
    del state
    _gc.collect()
    logger.info(f"Saved {st_path}")
    return st_path


def load_dit(cfg, device: torch.device, progress_cb=None) -> torch.nn.Module:
    """Load DiT model for inference."""
    logger = get_logger()
    def _p(msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    state_dict = None
    if cfg.dit_ckpt is not None:
        _p(f"Loading from: {cfg.dit_ckpt}")

        if cfg.dit_ckpt_type == "pt":
            st_path = _ensure_safetensors(cfg.dit_ckpt)
            _p("Reading safetensors directly to device...")
            from safetensors.torch import load_file
            state_dict = load_file(st_path, device=str(device))
            _p(f"State dict loaded on {device}: {len(state_dict)} keys")
        else:
            raise ValueError(
                f"Unknown dit_ckpt_type: {cfg.dit_ckpt_type}, must be 'safetensor' or 'pt'")

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    _p("Building model structure (meta device, 0 bytes)...")
    model_kwargs = {'dtype': dtype, 'device': torch.device('meta'), 'args': cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    _p("Model structure built.")

    if state_dict is not None:
        _p("Filtering state dict...")
        load_state_dict = {}
        for k, v in state_dict.items():
            if not cfg.is_repa and 'repa' in k:
                continue
            if k == "img_in.weight" and hasattr(model, 'img_in') and model.img_in.weight.shape != v.shape:
                logger.info(f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v_new = torch.zeros(model.img_in.weight.shape, dtype=v.dtype, device=v.device)
                v_new[:, :v.shape[1], :, :, :] = v
                v = v_new
            load_state_dict[k] = v
        _p(f"Assigning {len(load_state_dict)} params directly on {device}...")
        model.load_state_dict(load_state_dict, strict=True, assign=True)
        if device.type == "mps":
            torch.mps.synchronize()
        _p("Clearing state dict...")
        del state_dict, load_state_dict
        import gc; gc.collect()
        _p("Ready.")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params / 1e9:.2f}B parameters")

    return model.eval()
