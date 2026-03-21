# Upscale Worker — Progress-Analyse

**Entry Point:** `worker/upscale/generate.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Model-Download (nur bei Erstlauf) | Ja (`Downloading {filename} …`, Z.145) | Nein | Simulierter tqdm-Balken (oder real über urllib mit Content-Length) |
| 2. Model laden (RRDBNet + load_state_dict + to(device)) | Ja (`Loading Real-ESRGAN …`, Z.207) | Nein | Simulierter tqdm-Balken |
| 3. Bild laden (Image.open) | Ja (`Upscaling: {img_path}`, Z.212) | Nein | Simulierter tqdm-Balken |
| 4. Upscale Inference (model forward) | Nein | Nein | Label + simulierter tqdm-Balken |
| 5. Ergebnis speichern (result.save) | Ja (`Saved: {out_path}`, Z.232) | Nein | Simulierter tqdm-Balken |

## Konkrete Code-Vorschläge

### Z.207-208 — Model laden: Balken fehlt

```python
from tqdm import tqdm

# Z.207-208 ersetzen:
print(f"  Loading Real-ESRGAN ({model_key}, {info['scale']}x) …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading model", file=sys.stderr)
_bar.refresh()
model_path = _download_model(model_key)
_bar.update(1)
_bar.close()
```

### Z.150-163 — _upscale_image Funktion: Model-Init hat kein Label/Balken

Die eigentliche Model-Initialisierung (RRDBNet + load_state_dict) passiert in `_upscale_image()` (Z.150-163), nicht beim Download. Das wird pro Bild aufgerufen!

```python
# In _upscale_image(), Z.151-163 wrappen:
print("  Loading weights …", file=sys.stderr)
_bar_w = tqdm(total=1, desc="Loading weights", file=sys.stderr)
_bar_w.refresh()
model = RRDBNet(...)
loadnet = torch.load(...)
# ... state_dict laden ...
model.eval()
model = model.to(device)
_bar_w.update(1)
_bar_w.close()
```

### Z.175-176 — Inference: Label + Balken fehlt

```python
# VOR Z.175 einfügen:
print("  Upscaling …", file=sys.stderr)
_bar_inf = tqdm(total=1, desc="Upscaling", file=sys.stderr)
_bar_inf.refresh()
with torch.no_grad():
    output = model(img)
_bar_inf.update(1)
_bar_inf.close()
```

### Z.231-232 — Speichern: Balken fehlt

```python
# Z.231-232 ersetzen:
print("  Saving result …", file=sys.stderr)
_bar_save = tqdm(total=1, desc="Saving", file=sys.stderr)
_bar_save.refresh()
result.save(str(out_path))
_bar_save.update(1)
_bar_save.close()
```

### Mehrere Bilder — Übergreifender Balken

```python
# Z.211 ersetzen:
for img_path in tqdm(args.images, desc="Processing images", file=sys.stderr):
```

## Import nötig

`from tqdm import tqdm` am Anfang der Datei.

## Hinweis

`_upscale_image()` erstellt das Model **jedes Mal neu** pro Bild. Bei Batch-Verarbeitung sollte man das Model einmal laden und wiederverwenden. Aber das ist ein separates Optimierungs-Thema.
