# Depth Worker — Progress-Bar Analyse

**Entry Point:** `worker/depth/generate.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Pipeline/Modell laden | Teilweise (`Loading Depth Anything V2 …` Z.54) | Nein | Balken fehlt |
| 2 | Bild(er) laden + Inference | Teilweise (`Processing: …` Z.59) | Nein | Balken fehlt (real ueber Bilder-Loop, oder simuliert pro Bild) |
| 3 | Ergebnis speichern | Teilweise (`Saved: …` Z.73) | Nein | Balken fehlt |

## Konkrete Code-Vorschlaege

### Vor Z.36 (vor `def main()`) — tqdm import

```python
from tqdm import tqdm
```

### Z.54-55 — Modell laden (Step 1)

```python
print("  Modell laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Modell laden", file=sys.stderr)
_bar.refresh()
pipe = pipeline("depth-estimation", model=model_id, device=device)
_bar.update(1)
_bar.close()
```

### Z.57-74 — Bilder verarbeiten (Step 2+3, real ueber Bilder-Loop)

```python
_bar = tqdm(total=len(args.images), desc="Depth-Estimation", file=sys.stderr)
_bar.refresh()
for img_path in args.images:
    print(f"  Verarbeite: {img_path}", file=sys.stderr)

    img = Image.open(img_path)
    result = pipe(img)
    depth_image = result["depth"]

    if len(args.images) == 1:
        out_path = Path(args.output)
    else:
        stem = Path(img_path).stem
        out_path = Path(args.output).parent / f"{stem}_depth.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_image.save(str(out_path))
    outputs.append(str(out_path))
    _bar.update(1)
_bar.close()
```
