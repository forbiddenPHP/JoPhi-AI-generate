# pose Worker — Progress-Bar Analyse

**Entry Point:** `worker/pose/generate.py`

## tqdm-Setup

Kein tqdm-Import vorhanden. Keine Balken.

---

## Step-Analyse

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| ONNX-Models finden/laden | Ja (Z.76) | Nein | Simulierter Balken fehlt |
| Pro Bild: Pose Detection | Ja (Z.82) | Nein | Simulierter Balken fehlt |
| Pro Bild: Speichern | Ja (Z.101) | Nein | Trivial, kein Balken noetig |

## Handlungsbedarf

Zwei simulierte Balken fehlen: Model laden und pro-Bild Inference.

### Vorschlag: tqdm importieren (nach Zeile 13)

```python
from tqdm import tqdm
```

### Vorschlag: Model laden (Zeile 76-77)

```python
# Zeile 76: print("Loading DWPose models ...", file=sys.stderr) — BLEIBT
_bar = tqdm(total=1, desc="Loading DWPose models", file=sys.stderr); _bar.refresh()
detector = _load_detector()
_bar.update(1); _bar.close()
```

### Vorschlag: Pro Bild Processing (Zeile 82-91)

```python
# Zeile 82: print(f"Processing: {img_path}", ...) — BLEIBT
_bar = tqdm(total=1, desc=f"Detecting pose", file=sys.stderr); _bar.refresh()
result = detector(img, ...)
_bar.update(1); _bar.close()
```

### Alternative: Batch-Balken bei mehreren Bildern

Wenn mehrere Bilder verarbeitet werden (Zeile 80 Loop), waere ein Gesamt-Balken sinnvoller:

```python
_bar = tqdm(total=len(args.images), desc="Detecting poses", file=sys.stderr); _bar.refresh()
for img_path in args.images:
    img = Image.open(img_path)
    print(f"  Processing: {img_path}", file=sys.stderr)
    result = detector(img, ...)
    # ... save ...
    _bar.update(1)
_bar.close()
```
