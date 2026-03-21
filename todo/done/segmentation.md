# Research: Bildebenen trennen (Segmentierung)

## Background Removal

### RMBG 2.0 (BRIA AI) — bester Kandidat
- HuggingFace: `briaai/RMBG-2.0` (BiRefNet-Architektur)
- Pure PyTorch via `transformers` (`AutoModelForImageSegmentation`)
- Output: Alpha-Matte (8-bit grayscale) → mit Original zu transparentem PNG kombinieren
- Sollte auf MPS laufen (Standard-PyTorch-Ops)
- RAM: ~1-2GB
- ⚠️ Lizenz prüfen — BRIA hat kommerzielle Einschränkungen

### rembg (einfachste Option)
- Repo: `danielgatis/rembg`
- Models: u2net, u2netp (leicht), isnet-general-use, isnet-anime
- ONNX Runtime — CPU oder Core ML, NICHT MPS
- RAM: ~500MB-1GB
- Einfachste Integration (CLI + Library), aber langsamer ohne GPU

## Multi-Object Segmentation

### SAM 2 (Segment Anything Model 2)
- Repo: `facebookresearch/sam2` — MPS offiziell gelistet
- **Bekannte MPS-Probleme**: `pin_memory()` + `.to()` Konflikt, Placeholder storage Errors
- Fix: `pin_memory()` Calls entfernen, Build mit `SAM2_BUILD_CUDA=0`
- PR #567 hat MPS-Workarounds
- RAM: Large ~2.5GB, Base ~500MB

### SAM 2 Core ML (zuverlässigster Weg)
- Apple-offizielle Conversion: `apple/coreml-sam2-large`, `apple/coreml-sam2.1-large`
- Python Wrapper: `mikeesto/sam2-coreml-python`
- float16, optimiert für Neural Engine
- ~4s pro Bild auf M3 (small)
- Limitation: Nur Bild-Segmentierung, kein Video

### SAM 3
- ❌ **Geht NICHT auf Apple Silicon** — Triton-Dependency (kein macOS-Support)

## Transparent PNG Pipeline
1. Segmentierung → Binary Mask
2. `alpha = mask * 255`
3. RGBA: `np.concatenate([rgb, alpha], axis=2)`
4. `PIL.Image.save("output.png")`

## Empfehlung
1. **Background Removal**: RMBG 2.0 via transformers auf MPS
2. **Multi-Object**: SAM 2 Core ML (Apple-Version) für Zuverlässigkeit
3. **Quick & Simple**: rembg mit isnet (ONNX, kein GPU-Stress)
4. **Vermeiden**: SAM 3 (kein macOS)
