# Research: Scribble/Sketch-to-Image

## ControlNet Scribble SD1.5 (Empfehlung)
- Model: `lllyasviel/control_v11p_sd15_scribble` (~1.4GB)
- Zusammen mit SD1.5: ~6-8GB RAM
- Funktioniert auf MPS via `StableDiffusionControlNetPipeline`

### Preprocessors
- **HED**: Holistically-Nested Edge Detection — zuverlässig auf MPS, gute Strukturerkennung
- **Pidinet**: Sauberere Linien — ⚠️ **KAPUTT auf Apple Silicon MPS** (dokumentierter Bug)
- **Fake Scribble**: HED vereinfacht zu dicken Linien, guter Kompromiss

## T2I-Adapter Sketch (leichtere Alternative)
- Model: `TencentARC/t2iadapter_sketch_sd15v2` (~300MB vs 1.4GB ControlNet)
- Weniger präzise Kontrolle, aber schnellere Inferenz

## FLUX ControlNet Sketch
- **Shakker-Labs Union Pro 2.0**: Soft-Edge-Modus (kein dedizierter Scribble-Modus)
- **XLabs**: HED/Scribble Conditioning, getestet M1-M4
- Braucht 32GB+ RAM (oder GGUF ~12GB)
- Scribble Conditioning Scale optimal: 0.65-0.80

## Empfehlung
1. **Primär**: ControlNet v1.1 Scribble + SD1.5 + HED Preprocessor
2. **Leicht**: T2I-Adapter Sketch wenn ControlNet zu schwer
3. **Qualität**: FLUX + Shakker-Labs Soft Edge — nur 32GB+ Macs
4. **Pidinet vermeiden** auf Apple Silicon
