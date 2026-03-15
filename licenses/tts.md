# mlx-audio

- **License:** MIT License
- **Copyright:** 2024 Prince Canuma
- **Source:** https://github.com/Blaizzy/mlx-audio

## Models

- **Qwen3-TTS** — Apache License 2.0, Alibaba Cloud
- Downloaded from HuggingFace at runtime

## Dependencies

- **mlx** (0.31.1) — Apple ML framework, Apache 2.0
- **mlx-metal** (0.31.1) — Metal GPU backend, Apache 2.0
- **transformers** (5.3.0) — Apache 2.0

## Notes

No modifications to the upstream package. Our changes are only in `worker/tts/generate_speech.py` (our own wrapper script).
