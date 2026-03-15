# mlx-whisper

- **Version:** 0.4.3
- **License:** MIT License
- **Source:** MLX Community (Apple Silicon optimized Whisper)

## Dependencies

- **mlx** (0.31.1) — Apple ML framework, Apache 2.0
- **mlx-metal** (0.31.1) — Metal GPU backend, Apache 2.0

## Default Model

- `mlx-community/whisper-large-v3-turbo` (downloaded from HuggingFace)

## Notes

No modifications to the upstream package. Our changes are only in `worker/whisper/transcribe.py` (our own wrapper script).
