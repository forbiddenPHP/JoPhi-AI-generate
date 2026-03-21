# Research: NormalMap Estimation

## Marigold-Normals v1.1 (Empfehlung)
- HuggingFace: `prs-eth/marigold-normals-v1-1`
- LCM-Variante (schnell, 1-4 Steps): `prs-eth/marigold-normals-lcm-v0-1`
- Diffusion-basiert, fine-tuned von Stable Diffusion
- **In diffusers core seit v0.28.0** — `MarigoldNormalsPipeline`
- Läuft auf MPS via `.to('mps')`, ~4-6GB RAM
- CVPR 2024 Oral, Best Paper Award Candidate
- Effektive Auflösung: ~768px (SD-Limit)
- Output: 3D Unit Vectors in Screen-Space Kamerakoordinaten

## DSINE (CVPR 2024 Oral)
- Repo: `baegwangbin/DSINE`
- Discriminatives Modell (kein Diffusion) → schnellere Inferenz (Single Forward Pass)
- ~2-4GB RAM
- ⚠️ Offiziell nur CUDA, aber Standard-PyTorch-Ops → sollte auf MPS gehen mit Device-Änderungen

## Omnidata v2
- Repo: `EPFL-VILAB/omnidata`
- DPT-Hybrid, trainiert auf 3D-Scan-Daten
- Input: 384x384, ~2-3GB RAM
- Kein offizieller MPS-Test

## Als Conditioning für Bildgenerierung
- **SD1.5**: `lllyasviel/sd-controlnet-normal` — funktioniert auf MPS (~6-8GB)
  - Farbcodierung: Blau=vorne, Rot=links, Grün=oben
- **FLUX**: ❌ Kein Normal-Map ControlNet verfügbar — Depth ist der nächste Proxy

## Empfehlung
1. **Estimation**: Marigold-Normals v1.1 — diffusers-nativ, LCM für Speed
2. **Conditioning**: SD1.5 ControlNet Normal
3. **Achtung**: Farbkonventionen zwischen Estimator und ControlNet müssen übereinstimmen (ScanNet vs OpenGL vs DirectX)
