# LTX-2.3

- **License:** Apache License 2.0
- **Copyright:** 2025 Lightricks
- **Source:** https://github.com/Lightricks/LTX-2
- **Worker:** `worker/ltx2/`

## Description

LTX-2.3 is a 22B parameter text/image/audio-to-video generation model.
Two-stage pipeline: generates at half resolution, then upscales 2x with refinement.
Includes Gemma 3 12B text encoder.

## Components

- **ltx-core:** Core model implementation (Apache 2.0)
- **ltx-pipelines:** Pipeline implementations (Apache 2.0)
- **Gemma 3 12B:** Text encoder (Gemma License, google/gemma-3-12b-it)
