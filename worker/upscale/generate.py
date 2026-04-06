"""Image upscaling using Real-ESRGAN (standalone, no basicsr dependency).

Usage:
  python generate.py image --engine upscale --images photo.png -o upscaled.png
  python generate.py image --engine upscale --images photo.png --model 2x -o upscaled.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_MODELS_DIR.mkdir(exist_ok=True)

def _list_models():
    """Output available models as JSON and exit."""
    import json as _json
    models = [
        {"model": "4x", "notice": "default"},
        {"model": "2x", "notice": ""},
        {"model": "anime", "notice": ""},
        {"model": "ultrasharp", "notice": "4x UltraSharp"},
    ]
    print(_json.dumps(models))
    sys.exit(0)


_MODEL_INFO = {
    "4x": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "2x": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
    "ultrasharp": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        "filename": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
    },
}


# ── Standalone RRDBNet (no basicsr dependency) ────────────────────────────────

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    """Inverse of PixelShuffle: (B, C, H, W) → (B, C*s*s, H/s, W/s)."""
    b, c, h, w = x.shape
    return x.view(b, c, h // scale, scale, w // scale, scale).permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h // scale, w // scale)


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4  # pixel_unshuffle expands channels
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_up3 = None  # 4x uses two 2x upsamples
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = pixel_unshuffle(x, scale=2)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Upsample (always 2 stages: x2 model needs 4x from H/2 due to pixel_unshuffle)
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ── Upscaling logic ───────────────────────────────────────────────────────────

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _download_model(model_key):
    info = _MODEL_INFO[model_key]
    model_path = _MODELS_DIR / info["filename"]
    if model_path.exists():
        return model_path
    import urllib.request
    print(f"  Downloading {info['filename']} …", file=sys.stderr)
    urllib.request.urlretrieve(info["url"], str(model_path))
    return model_path


def _convert_old_esrgan_keys(state_dict):
    """Convert old ESRGAN key format to new RRDBNet format.

    Old format (e.g. 4x-UltraSharp):
      model.0.weight           → conv_first
      model.1.sub.N.RDB*.conv* → body.N.rdb*.conv*
      model.3.weight           → conv_body
      model.6.weight           → conv_up1
      model.8.weight           → conv_up2
      model.10.weight          → conv_last
    Note: model.2/4/5/7/9 are activations/upsamples with no weights.
    """
    import re
    # Fixed mapping for structural layers
    _FIXED = {
        "model.0.": "conv_first.",
        "model.3.": "conv_up1.",
        "model.6.": "conv_up2.",
        "model.8.": "conv_hr.",
        "model.10.": "conv_last.",
    }
    new_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model.1.sub."):
            # Check if it's the conv_body (last sub-block, only weight/bias, no RDB)
            parts = k.split(".")
            block_idx = int(parts[3])
            remainder = ".".join(parts[4:])  # e.g. "weight" or "RDB1.conv1.0.weight"
            if remainder in ("weight", "bias"):
                # Pure conv layer = conv_body
                nk = f"conv_body.{remainder}"
            else:
                # RRDB block
                nk = f"body.{block_idx}.{remainder}"
                for i in range(1, 4):
                    nk = nk.replace(f".RDB{i}.", f".rdb{i}.")
                nk = re.sub(r'(conv\d)\.0\.(weight|bias)', r'\1.\2', nk)
        else:
            for old_prefix, new_prefix in _FIXED.items():
                if nk.startswith(old_prefix):
                    nk = nk.replace(old_prefix, new_prefix)
                    break
        new_dict[nk] = v
    return new_dict


def _upscale_image(img_np, model_path, scale, num_block, device):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64,
                    num_block=num_block, num_grow_ch=32)

    loadnet = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if "params_ema" in loadnet:
        state_dict = loadnet["params_ema"]
    elif "params" in loadnet:
        state_dict = loadnet["params"]
    else:
        state_dict = loadnet

    # Detect old ESRGAN format (keys start with "model.")
    first_key = next(iter(state_dict))
    if first_key.startswith("model."):
        state_dict = _convert_old_esrgan_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model = model.to(device)

    img = img_np.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
    # Pad to even dimensions for pixel_unshuffle (x2 model)
    _, _, h, w = img.shape
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    img = img.contiguous().to(device)

    with torch.no_grad():
        output = model(img)
    # Remove padding from output
    if pad_h or pad_w:
        out_h = (h + pad_h) * scale
        out_w = (w + pad_w) * scale
        output = output[:, :, :h * scale, :w * scale]

    if device.type == "mps":
        torch.mps.synchronize()

    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output


def main():
    if "--list-models" in sys.argv:
        _list_models()

    parser = argparse.ArgumentParser(description="Real-ESRGAN image upscaling")
    parser.add_argument("--images", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("-o", "--output", default="upscaled.png", help="Output file path")
    parser.add_argument("--model", default="4x", choices=list(_MODEL_INFO.keys()),
                        help="Model: 4x (default), 2x, anime, ultrasharp")
    parser.add_argument("-s", "--outscale", type=float, default=None,
                        help="Final upsampling scale (e.g. 2, 4, 3.5). Default: model's native scale.")
    args = parser.parse_args()

    device = _device()
    model_key = args.model
    info = _MODEL_INFO[model_key]
    native_scale = info["scale"]
    outscale = args.outscale or native_scale

    print(f"Loading Real-ESRGAN ({model_key}, {native_scale}x) …", file=sys.stderr)
    model_path = _download_model(model_key)

    outputs = []
    for img_path in args.images:
        print(f"  Upscaling: {img_path}", file=sys.stderr)

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        orig_w, orig_h = img.width, img.height

        result_np = _upscale_image(img_np, model_path, native_scale, info["num_block"], device)
        result = Image.fromarray(result_np)

        # Resize to target outscale if different from native
        if outscale != native_scale:
            target_w = int(orig_w * outscale)
            target_h = int(orig_h * outscale)
            result = result.resize((target_w, target_h), Image.LANCZOS)
            print(f"  Rescaled to {target_w}x{target_h} (outscale={outscale})", file=sys.stderr)

        out_base = Path(args.output)
        if len(args.images) == 1 and not str(out_base).endswith("/") and not out_base.is_dir():
            out_path = out_base
        else:
            # Output is a directory
            out_dir = out_base
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(img_path).stem
            out_path = out_dir / f"{stem}_upscaled.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path))
        print(f"  Saved: {out_path} ({result.size[0]}x{result.size[1]})", file=sys.stderr)
        outputs.append(str(out_path))

    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
