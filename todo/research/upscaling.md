# Research: Upscaling & Sharpening

## Real-ESRGAN (bester Kandidat)
- Repo: `xinntao/Real-ESRGAN`
- **MPS nicht offiziell supportet**, aber PR #836 hat die nötigen Patches:
  - `torch.backends.mps.is_available()` check
  - `img.contiguous().to('mps')` statt `img.to('mps')`
  - `torch.mps.synchronize()` statt CUDA-Variante
- Modelle: `RealESRGAN_x4plus.pth` (~64MB), `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus_anime_6B.pth`
- RAM: ~1-2GB
- Alternative: Core ML Conversion → bis zu 78x Speedup vs CPU (Neural Engine)

## SwinIR / HAT
- SwinIR: MPS experimentell und buggy (Issue #118)
- HAT (Nachfolger): Pure PyTorch, sollte auf MPS gehen, aber kein macOS-Testing dokumentiert
- RAM: ~2-4GB

## MLX-Native Upscaler
- **Existiert nicht** (Stand März 2026). MLX-Ökosystem fokussiert auf LLMs/Image-Gen, nicht Super-Resolution.

## 2x vs 4x
| | 2x | 4x |
|--|----|----|
| Speed | ~2-4x schneller | Baseline |
| Qualität | Weniger Artefakte | Mehr halluzinierte Details |
| RAM | ~50% weniger | Voll |

## Empfehlung
1. **Real-ESRGAN Fork** mit MPS-Patches aus PR #836
2. Models sind klein (~64MB), RAM-Verbrauch minimal
3. Optional: Core ML Conversion für maximale Performance
