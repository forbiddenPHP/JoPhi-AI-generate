# Video Pipeline — TODO

## Worker: LTX-2.3

### Phase 1: Setup

- [ ] `worker/ltx2/` Verzeichnis anlegen
- [ ] Repo clonen: `git clone --depth 1 https://github.com/Lightricks/LTX-2.git` nach `worker/ltx2/`
- [ ] `.git/` entfernen
- [ ] Projektstruktur verstehen: 3 Packages (ltx-core, ltx-pipelines, ltx-trainer) — ltx-trainer brauchen wir nicht
- [ ] `install.sh` schreiben (conda env `ltx2`, Python-Version ermitteln)
- [ ] Original-Dependencies aus `pyproject.toml` extrahieren → `requirements.txt` mit exakten Versionen (`==`)
- [ ] CUDA-only Deps entfernen/ersetzen: `xformers`, `tensorrt-llm` raus
- [ ] `flash_attn` raus (nicht installierbar auf macOS)
- [ ] Build-System: Original nutzt `uv` — wir brauchen pip-kompatible Installation
- [ ] `wheels/` befüllen: `pip download -d wheels/`
- [ ] `requirements.lock` erzeugen: `pip freeze`
- [ ] Gemma 3 12B Text-Encoder: Download-Pfad klären, in models/ oder separat
- [ ] Modelle downloaden (.safetensors): dev, distilled, upscaler, LoRAs
- [ ] Model-Backup-Pfade in `backup-models.sh` eintragen
- [ ] `./licenses/ltx2.md` anlegen

### Phase 2: MPS-Patches

- [ ] `get_device()` patchen → MPS-Detection (`torch.backends.mps.is_available()`)
- [ ] `torch.cuda.synchronize()` → `torch.mps.synchronize()` oder entfernen
- [ ] `torch.cuda.empty_cache()` → `torch.mps.empty_cache()`
- [ ] `str(device).startswith("cuda")` Device-Checks → device-agnostisch
- [ ] Attention: xFormers/FA3 → Fallback auf `torch.nn.functional.scaled_dot_product_attention`
- [ ] bfloat16 auf MPS testen (ab PyTorch 2.3+ unterstützt)
- [ ] Vocoder conv1d Limit (65536 Output-Channels) — Workaround oder Frame-Count-Beschränkung dokumentieren
- [ ] PyTorch-Version testen: 2.4.1 vs neuere (MPS strided API Bug)
- [ ] Distilled-Modell auf MPS testen (Community meldet Artefakte)
- [ ] Eventuell eigenes Chunking (K*) wie bei Flux
- [ ] Eventuell Monkey-Patches / Stubs für nicht-triviale CUDA-Ops

### Phase 3: Worker Entry Point

- [ ] `worker/ltx2/generate.py` schreiben
- [ ] CLI-Args: `--prompt`, `--model`, `--output`, `--seed`, `--steps`, `--cfg-scale`, `--width`, `--height`, `--fps`, `--frames`, `--audio`, `--images`, `--negative-prompt`, `--lora`, `--enhance-prompt`
- [ ] `--list-models` → JSON-Array der verfügbaren Modelle
- [ ] Progress-Events auf stderr (kompatibel mit progress.py)
- [ ] Letzter stdout = `json.dumps([pfad1, ...])` — Array = fertig
- [ ] Pipeline-Dispatch: welche Pipeline-Klasse je nach Input (T2V, I2V, A2V, Retake, etc.)

### Phase 4: generate.py Integration

- [ ] Video-Stub ausbauen: `generate.py video ltx2.3 ...`
- [ ] Worker-Konstanten: `LTX_WORKER_DIR`, `LTX_ENV`
- [ ] `cmd_video()` → `_video_ltx2(args)` → `conda run -n ltx2 python worker/ltx2/generate.py ...`
- [ ] Bestehende Params wiederverwenden (--prompt, --model, --seed, --steps, --cfg-scale, --width, --height, --images, --audio, --lora, --negative-prompt)
- [ ] Neue Params: --fps, --frames, --enhance-prompt
- [ ] Eventuell: `generate.py audio ltx2.3` — virtueller Worker (Minimal-Video + Audio-Extraktion)
- [ ] Step in `setup.sh` hinzufügen
- [ ] README.md + USAGE aktualisieren
- [ ] Tests in `tests/suites/`

---

## Worker: Wan 2.2

### Phase 1: Setup

- [ ] `worker/wan2/` Verzeichnis anlegen
- [ ] Repo clonen: `git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git` nach `worker/wan2/`
- [ ] `.git/` entfernen
- [ ] Projektstruktur: `wan/` Library + `generate.py` Entry Point
- [ ] `install.sh` schreiben (conda env `wan2`, Python-Version ermitteln)
- [ ] Original-Dependencies aus `requirements.txt` extrahieren → exakte Versionen (`==`)
- [ ] `flash_attn` raus (CUDA-only, nicht installierbar auf macOS)
- [ ] `dashscope` optional machen (Cloud-API für Prompt Extension)
- [ ] S2V-Dependencies (`requirements_s2v.txt`) separat evaluieren — brauchen wir S2V sofort?
- [ ] Animate-Dependencies (`requirements_animate.txt`) separat evaluieren
- [ ] `numpy<2` pinnen
- [ ] `wheels/` befüllen: `pip download -d wheels/`
- [ ] `requirements.lock` erzeugen: `pip freeze`
- [ ] Modelle downloaden: TI2V-5B (.safetensors/HF-Diffusers oder .pth original), T2V-A14B, I2V-A14B
- [ ] Checkpoint-Format klären: .pth (original) vs .safetensors (ComfyUI/HF repackaged)
- [ ] Text-Encoder UMT5-XXL: Download-Pfad klären
- [ ] CLIP Vision Model (für I2V): Download-Pfad klären
- [ ] Model-Backup-Pfade in `backup-models.sh` eintragen
- [ ] `./licenses/wan2.md` anlegen

### Phase 2: MPS-Patches

- [ ] Device-Handling: `torch.device(f"cuda:{device_id}")` → `torch.device("mps")`
- [ ] `flash_attn` → `torch.nn.functional.scaled_dot_product_attention` (CPU-Fallback existiert im Code, für MPS erweitern)
- [ ] `assert q.device.type == 'cuda'` in `modules/attention.py` → entfernen/anpassen
- [ ] `torch.amp.autocast('cuda')` → `torch.amp.autocast('mps')` oder entfernen
- [ ] `@torch.amp.autocast('cuda', enabled=False)` in RoPE → device-agnostisch
- [ ] `torch.cuda.empty_cache()` → `torch.mps.empty_cache()`
- [ ] Checkpoint-Loading: `map_location='mps'` bei torch.load()
- [ ] bfloat16 auf MPS testen
- [ ] Distributed-Code (FSDP, NCCL, Ulysses) — deaktivieren/umgehen (Single-Device)
- [ ] Eventuell eigenes Chunking (K*) wie bei Flux
- [ ] Eventuell Monkey-Patches / Stubs für nicht-triviale CUDA-Ops

### Phase 3: Worker Entry Point

- [ ] `worker/wan2/generate.py` — entweder Original-`generate.py` wrappen oder eigenen schreiben
- [ ] CLI-Args: `--prompt`, `--model`, `--output`, `--seed`, `--steps`, `--cfg-scale`, `--width`, `--height`, `--fps`, `--frames`, `--audio`, `--images`, `--negative-prompt`, `--pose-video`
- [ ] `--list-models` → JSON-Array der verfügbaren Modelle
- [ ] Task-Dispatch: `--task` intern mappen (t2v, i2v, ti2v, s2v, animate)
- [ ] Progress-Events auf stderr
- [ ] Letzter stdout = `json.dumps([pfad1, ...])` — Array = fertig

### Phase 4: generate.py Integration

- [ ] Video-Stub ausbauen: `generate.py video wan2.2 ...`
- [ ] Worker-Konstanten: `WAN_WORKER_DIR`, `WAN_ENV`
- [ ] `cmd_video()` → `_video_wan2(args)` → `conda run -n wan2 python worker/wan2/generate.py ...`
- [ ] Bestehende Params wiederverwenden
- [ ] Neue Params: --fps, --frames, --pose-video
- [ ] Step in `setup.sh` hinzufügen
- [ ] README.md + USAGE aktualisieren
- [ ] Tests in `tests/suites/`

---

## Reihenfolge

1. LTX-2.3 zuerst (hat Audio-Generation, mehr Pipelines im Repo)
2. Wan 2.2 danach (TI2V-5B als leichtestes Modell für schnelle Tests)

## Offene Fragen

- PyTorch-Version: Beide Repos wollen unterschiedliche torch-Versionen (LTX: ~=2.7, Wan: >=2.4.0) — separate Envs lösen das
- Gemma 3 vs UMT5: Beide Text-Encoder sind groß, RAM-Planung nötig
- LTX-2.3 Audio-only: Minimale Video-Größe für Audio-Extraktion ermitteln (vermutlich teilbar durch 64)
- Wan S2V: Sofort implementieren oder auf Phase 2 verschieben? (14B MoE = 57 GB)
