# Image Worker (FLUX.2) — Progress-Bar Analyse

**Entry Point:** `worker/image/generate.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Autoencoder laden | Teilweise (`Loading autoencoder …` Z.189) | Nein | Balken fehlt |
| 2 | Referenz-Bilder encoden | Teilweise (`Encoding N reference image(s) …` Z.219) | Nein | Balken fehlt |
| 3 | Text-Encoder laden | Teilweise (`Loading text encoder …` Z.224) | Nein | Balken fehlt |
| 4 | Prompt encoden | Teilweise (`Encoding prompt …` Z.229) | Nein | Balken fehlt |
| 5 | Text-Encoder freigeben | Teilweise (`Freeing text encoder …` Z.239) | Nein | Balken fehlt |
| 6 | Flow-Modell laden | Teilweise (`Loading flow model …` Z.246) | Nein | Balken fehlt |
| 7 | Denoising | Teilweise (`Denoising …` Z.260) | Nein | Balken fehlt (real ueber timesteps ideal) |
| 8 | Decoding | Teilweise (`Decoding …` Z.287) | Nein | Balken fehlt |
| 9 | Bild speichern | Teilweise (`Saved: …` Z.296) | Nein | Balken fehlt |

Alle Steps haben Labels (gut!), aber keiner hat einen tqdm-Balken.

## Konkrete Code-Vorschlaege

### Am Datei-Anfang (nach Z.14) — tqdm import

```python
from tqdm import tqdm
```

### Z.189-191 — Autoencoder laden (Step 1)

```python
print("  Autoencoder laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Autoencoder laden", file=sys.stderr)
_bar.refresh()
ae = load_ae(model_name, device=device)
ae.eval()
_bar.update(1)
_bar.close()
```

### Z.219-221 — Referenz-Bilder encoden (Step 2)

```python
print(f"  {len(img_ctx)} Referenz-Bilder encoden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Referenz-Bilder encoden", file=sys.stderr)
_bar.refresh()
with torch.no_grad():
    ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
_bar.update(1)
_bar.close()
```

### Z.224-226 — Text-Encoder laden (Step 3)

```python
print("  Text-Encoder laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Text-Encoder laden", file=sys.stderr)
_bar.refresh()
text_encoder = load_text_encoder(model_name, device=device)
text_encoder.eval()
_bar.update(1)
_bar.close()
```

### Z.229-236 — Prompt encoden (Step 4)

```python
print("  Prompt encoden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Prompt encoden", file=sys.stderr)
_bar.refresh()
# ... encoding logic ...
_bar.update(1)
_bar.close()
```

### Z.239-243 — Text-Encoder freigeben (Step 5)

```python
print("  Text-Encoder freigeben …", file=sys.stderr)
_bar = tqdm(total=1, desc="Text-Encoder freigeben", file=sys.stderr)
_bar.refresh()
del text_encoder
gc.collect()
if device.type == 'mps':
    torch.mps.empty_cache()
_bar.update(1)
_bar.close()
```

### Z.246-248 — Flow-Modell laden (Step 6)

```python
print("  Flow-Modell laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Flow-Modell laden", file=sys.stderr)
_bar.refresh()
model = load_flow_model(model_name, device=device)
model.eval()
_bar.update(1)
_bar.close()
```

### Z.260-278 — Denoising (Step 7, REAL Balken moeglich)

Die Funktionen `denoise()`, `denoise_cached()`, `denoise_cfg()` in `flux2/sampling.py` iterieren ueber `timesteps`. Ein echter Balken waere ideal — entweder in `sampling.py` direkt einen `tqdm(timesteps)` einbauen, oder einen Callback. Minimalversion simuliert:

```python
print("  Denoising …", file=sys.stderr)
_bar = tqdm(total=num_steps, desc="Denoising", file=sys.stderr)
_bar.refresh()
# Idealfall: denoise() akzeptiert callback=lambda: _bar.update(1)
# Fallback simuliert:
x = denoise_fn(...)
_bar.update(num_steps)
_bar.close()
```

Besser: In `flux2/sampling.py` die `for`-Loop ueber `timesteps` mit `tqdm` wrappen.

### Z.287-294 — Decoding + Speichern (Steps 8+9)

```python
print("  Decoding …", file=sys.stderr)
_bar = tqdm(total=1, desc="Decoding", file=sys.stderr)
_bar.refresh()
x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
x = ae.decode(x).float()
_bar.update(1)
_bar.close()

print("  Bild speichern …", file=sys.stderr)
_bar = tqdm(total=1, desc="Bild speichern", file=sys.stderr)
_bar.refresh()
# ... clamp, rearrange, save ...
_bar.update(1)
_bar.close()
```
