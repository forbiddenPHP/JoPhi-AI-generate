# Research: Stil-Transfer

## IP-Adapter (Empfehlung)
- **SD1.5**: `h94/IP-Adapter` (~100MB, 22M Params extra), ~6-8GB RAM, funktioniert auf MPS
  - Varianten: standard, light, plus, plus-face
  - Direkt in diffusers integriert via `load_ip_adapter()`
  - Strength über `scale` Parameter steuerbar (0.0-1.0)
- **FLUX**: `XLabs-AI/flux-ip-adapter-v2`, braucht 32GB+ (oder GGUF-Quantisierung ~12GB)
  - ~85s auf M4 Max, ~145s auf M2 Max (1024x1024)

## Standalone Alternativen
- **AesPA-Net** (ICCV 2023): Pure PyTorch, ~2-4GB, nicht diffusion-basiert
- **StyTr2** (CVPR 2022): Transformer-basiert, ~2-3GB
- **Classic Neural Style Transfer** (Gatys): VGG-basiert, ~1-2GB, langsam aber zuverlässig

## macOS-Probleme
- FP8 nicht auf MPS → GGUF für FLUX
- IP-Adapter Face-Modelle teils instabil mit bestimmten MPS PyTorch-Versionen

## Empfehlung
1. **Primär**: IP-Adapter + SD1.5 — bewährt, leichtgewichtig, gut auf MPS
2. **Qualität**: IP-Adapter + FLUX mit GGUF — nur 32GB+ Macs
3. **Fallback**: AesPA-Net für nicht-diffusion Style Transfer
