# Research: Normalmap Worker Dependencies

## Installation
`pip install diffusers transformers accelerate torch safetensors Pillow`

Alles pip. `MarigoldNormalsPipeline` ist seit diffusers v0.28.0 eingebaut.

## Standalone Usage
```python
from diffusers import MarigoldNormalsPipeline
import torch

pipe = MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-v1-1",
    variant="fp16",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("mps")
pipe.enable_attention_slicing()

result = pipe("input.png")
normal_map = result.prediction[0]
```

## Modell
- HuggingFace: `prs-eth/marigold-normals-v1-1`
- fp16 Download: ~2.5GB
- Architektur: SD2-basiert (~865M UNet, ~340M Text Encoder, ~83M VAE)
- LCM-Variante (`prs-eth/marigold-normals-lcm-v0-1`) ist deprecated — v1.1 schafft 1-4 Steps mit DDIM

## RAM
- Modell fp16: ~2.5GB
- Inference Peak (768px): ~4-6GB mit Attention Slicing
- Minimum: 8GB, empfohlen: 16GB+

## Conda Env
- Python 3.11 oder 3.12
- torch 2.4.1+ (ARM64 MPS)
- diffusers 0.31.0+
- transformers, accelerate, safetensors, Pillow
