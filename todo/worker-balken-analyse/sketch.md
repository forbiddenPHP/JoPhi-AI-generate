# Sketch Worker — Progress-Analyse

**Entry Point:** `worker/sketch/generate.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Model-Download (nur bei Erstlauf) | Ja (`Downloading HED prototxt …` / `Downloading HED model …`, Z.57/60) | Nein | Simulierter tqdm-Balken (oder real über urllib mit Content-Length) |
| 2. Model laden (cv2.dnn.readNetFromCaffe) | Nein | Nein | Label + simulierter tqdm-Balken |
| 3. Bild laden (cv2.imread) | Ja (`Processing: {img_path}`, Z.84) | Nein | Simulierter tqdm-Balken |
| 4. HED Inference (blobFromImage + forward) | Ja (`Extracting sketch (HED) …`, Z.80) — aber Label ist VOR der Schleife, nicht pro Bild | Nein | Simulierter tqdm-Balken pro Bild |
| 5. Ergebnis speichern (cv2.imwrite) | Ja (`Saved: {out_path}`, Z.112) | Nein | Simulierter tqdm-Balken |

## Konkrete Code-Vorschläge

### Z.77-78 — Model laden: Label + Balken fehlt

```python
# Z.77-78 ersetzen:
from tqdm import tqdm

print("  Loading HED model …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading model", file=sys.stderr)
_bar.refresh()
cv2.dnn_registerLayer("Crop", _CropLayer)
net = cv2.dnn.readNetFromCaffe(str(_PROTO_PATH), str(_MODEL_PATH))
_bar.update(1)
_bar.close()
```

### Z.83-113 — Verarbeitung pro Bild: Balken fehlt

Wenn mehrere Bilder: tqdm über die Bilder-Schleife (real).
Pro Bild Inference als simulierter Balken:

```python
# Z.82-83 ersetzen:
for img_path in tqdm(args.images, desc="Processing images", file=sys.stderr):
    print(f"  Processing: {img_path}", file=sys.stderr)

    _bar_load = tqdm(total=1, desc="Loading image", file=sys.stderr)
    _bar_load.refresh()
    img = cv2.imread(img_path)
    if img is None:
        _bar_load.close()
        print(f"  ERROR: Cannot read: {img_path}", file=sys.stderr)
        sys.exit(1)
    _bar_load.update(1)
    _bar_load.close()

    _bar_inf = tqdm(total=1, desc="HED inference", file=sys.stderr)
    _bar_inf.refresh()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(...)
    net.setInput(blob)
    result = net.forward()
    _bar_inf.update(1)
    _bar_inf.close()

    # ... edges + save ...
    _bar_save = tqdm(total=1, desc="Saving", file=sys.stderr)
    _bar_save.refresh()
    cv2.imwrite(str(out_path), edges)
    _bar_save.update(1)
    _bar_save.close()
```

## Import nötig

`from tqdm import tqdm` am Anfang der Datei.
