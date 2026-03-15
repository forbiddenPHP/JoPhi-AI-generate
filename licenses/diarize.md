# pyannote.audio

- **Version:** 4.0.4
- **License:** MIT License
- **Copyright:** Herve Bredin, CNRS
- **Source:** https://github.com/pyannote/pyannote-audio

## Gated Models

The following HuggingFace models require accepting terms before use:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

A HuggingFace token (`HF_TOKEN`) is required for model download.

## Notes

No modifications to the upstream package. Our changes are only in `worker/diarize/diarize.py` (our own wrapper script).
