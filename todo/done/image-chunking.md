# TODO: Image Generation — Memory-basiertes Chunking

## Problem
Bei großen Bildern (1360x768+) kann der Speicher knapp werden, besonders mit 9B-Modellen auf Apple Silicon.

## Lösung
Vor der Berechnung freien Speicher prüfen, davon 8/9 als Budget. Basierend darauf:
- VAE-Encoding/Decoding in Tiles aufteilen
- Denoising: ggf. Attention in Chunks
- Model-Loading: CPU-Offloading wenn nötig

## Betroffene Stellen
- `worker/image/generate.py` — Hauptpipeline
- `worker/image/flux2/src/flux2/sampling.py` — Denoising-Loop
- `worker/image/flux2/src/flux2/autoencoder.py` — VAE encode/decode

## Speicher auslesen (macOS)
```python
import psutil
free_mem = psutil.virtual_memory().available
budget = free_mem * 8 // 9
```

## Gilt für ALLE Modelle, nicht nur 9B.
