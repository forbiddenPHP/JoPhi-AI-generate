# Unified Server API

**Priorität:** Unbestimmte Zeit — erst wenn alle Basis-Funktionalitäten fertig sind.

## Ziel
Alle Engines unter einem Port vereinen. Modelle gezielt im RAM halten, damit bei wiederkehrenden Aufgaben das Loading entfällt. Dynamisches Load/Unload je nach Bedarf und freiem RAM.

## API-Schema

```
http://localhost:$port/generate/<medium>/<engine>[/<mode>]
```

Spiegelt die CLI-ABI 1:1 wider:

```
CLI:  python generate.py <medium> --engine <engine> [--endpoint <mode>]
API:  GET/POST /generate/<medium>/<engine>[/<mode>]
```

### Beispiele

```
POST /generate/voice/ai-tts          → TTS
POST /generate/voice/rvc             → Voice Conversion (existiert bereits auf :5100)
POST /generate/voice/clone-tts       → Voice Cloning
POST /generate/audio/sfx             → Sound Effects
POST /generate/audio/enhance         → Audio Enhancement
POST /generate/audio/demucs          → Stem Separation
POST /generate/audio/ace-step        → Music Generation
POST /generate/text/whisper           → Transcription
POST /generate/text/ollama/chat       → LLM Chat
POST /generate/text/ollama/generate   → LLM Generate
POST /generate/image/flux.2          → Image Generation
POST /generate/image/sd1.5           → Stable Diffusion
```

CLI-Params → JSON-Body. Dateien als multipart/form-data.

## Voraussetzungen
- RVC-Server bekommt Wrapper unter neuem Schema (bestehende API bleibt intern)
- Dynamisches Model-Loading/-Unloading nach RAM-Verfügbarkeit
- Gesamtfortschritt über alle Steps wird damit möglich
