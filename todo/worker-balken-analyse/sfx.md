# SFX Worker — Progress-Analyse

**Entry Point:** `worker/sfx/generate.py`
**Inference:** `worker/sfx/src/inference.py` (denoising loop)
**Model-Klasse:** `worker/sfx/api/ezaudio.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. T5 Download (HuggingFace, nur bei Erstlauf) | Ja (`Downloading {model_name} …`, generate.py Z.31) | Ja (tqdm über files, Z.33) | OK |
| 2. Autoencoder laden | Ja (`Loading autoencoder …`, ezaudio.py Z.76) | Nein | Simulierter tqdm-Balken |
| 3. T5 Tokenizer + Encoder laden | Ja (`Loading text encoder …`, ezaudio.py Z.84) | Ja (simulierter tqdm, Z.94-106) | OK |
| 4. UNet laden | Ja (`Loading diffusion model …`, ezaudio.py Z.121) | Ja (simulierter tqdm, Z.123-135) | OK |
| 5. Text encodieren (tokenizer + text_encoder forward) | Nein | Nein | Label + simulierter tqdm-Balken |
| 6. Denoising Loop (DDIM) | Nein (nur in generate.py: `Generating audio …`, Z.83) | Ja (tqdm in inference.py Z.70) | Label in generate.py ist OK, tqdm in inference.py läuft — prüfen ob `file=sys.stderr` |
| 7. Autoencoder decode (latent → wav) | Nein | Nein | Label + simulierter tqdm-Balken |
| 8. WAV speichern | Ja (`Generated in …`, Z.106) | Nein | Simulierter tqdm-Balken |

## Konkrete Code-Vorschläge

### ezaudio.py Z.76-79 — Autoencoder laden: Balken fehlt

```python
# Z.76-80 ersetzen:
print("  Loading autoencoder …", file=sys.stderr)
_bar_ae = tqdm(total=1, desc="Loading autoencoder", file=sys.stderr)
_bar_ae.refresh()
autoencoder = Autoencoder(ckpt_path=vae_path,
                          model_type=params['autoencoder']['name'],
                          quantization_first=params['autoencoder']['q_first']).to(device)
autoencoder.eval()
_bar_ae.update(1)
_bar_ae.close()
```

### inference.py Z.70 — Denoising Loop: Hat tqdm, aber `file=sys.stderr` fehlt!

```python
# Z.70 ersetzen:
for t in tqdm(noise_scheduler.timesteps, desc="Generating SFX", file=sys.stderr):
```

### inference.py Z.39-50 — Text Encoding: Label + Balken fehlt

```python
# VOR Z.39 einfügen:
print("  Encoding text …", file=sys.stderr)
_bar_enc = tqdm(total=1, desc="Encoding text", file=sys.stderr)
_bar_enc.refresh()
# ... bestehender Code Z.39-50 ...
# NACH Z.50 einfügen:
_bar_enc.update(1)
_bar_enc.close()
```

### inference.py Z.102-107 — Autoencoder Decode: Label + Balken fehlt

```python
# VOR Z.102 einfügen:
print("  Decoding audio …", file=sys.stderr)
_bar_dec = tqdm(total=1, desc="Decoding audio", file=sys.stderr)
_bar_dec.refresh()
# ... bestehender Code Z.102-107 ...
# NACH Z.107 (return) einfügen (vor return):
_bar_dec.update(1)
_bar_dec.close()
```

### generate.py Z.100-104 — WAV speichern: Balken fehlt

```python
# Z.100-104 ersetzen:
print("  Saving audio …", file=sys.stderr)
_bar_save = tqdm(total=1, desc="Saving audio", file=sys.stderr)
_bar_save.refresh()
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out_path), audio, sr)
_bar_save.update(1)
_bar_save.close()
```

## Import nötig

- `inference.py`: `from tqdm import tqdm` ist bereits importiert (Z.8), aber `import sys` fehlt!
- `generate.py`: `from tqdm import tqdm` muss hinzugefügt werden.
