# Lineart Worker — Progress-Bar Analyse

**Entry Point:** `worker/lineart/generate.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Modell laden (nur TEED) | Nein (implizit in `_extract_teed`) | Nein | Label + Balken fehlt |
| 2 | Bilder verarbeiten | Teilweise (`Processing: …` Z.62) | Nein | Balken fehlt (real ueber Bilder-Loop) |
| 3 | Ergebnis speichern | Teilweise (`Saved: …` Z.75) | Nein | Balken fehlt |

Hinweis: Bei `--model canny` gibt es kein Modell-Laden (rein OpenCV). Bei `--model teed` wird das TEED-Modell in `_extract_teed()` geladen — das ist der langsame Teil.

## Konkrete Code-Vorschlaege

### Am Datei-Anfang — tqdm import

```python
from tqdm import tqdm
```

### Z.19-23 — TEED Modell-Laden extrahieren und mit Balken versehen

Aktuell wird das Modell bei jedem Bild neu geladen (in `_extract_teed`). Besser: einmal vor dem Loop laden.

```python
if args.model == "teed":
    from controlnet_aux import TEEDdetector
    print("  TEED-Modell laden …", file=sys.stderr)
    _bar = tqdm(total=1, desc="TEED-Modell laden", file=sys.stderr)
    _bar.refresh()
    teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
    _bar.update(1)
    _bar.close()
    extract_fn = lambda img: teed(img)
else:
    extract_fn = _extract_canny
```

### Z.60-76 — Bilder verarbeiten (real ueber Loop)

```python
print("  Lineart extrahieren …", file=sys.stderr)
_bar = tqdm(total=len(args.images), desc="Lineart extrahieren", file=sys.stderr)
_bar.refresh()
for img_path in args.images:
    print(f"  Verarbeite: {img_path}", file=sys.stderr)

    img = Image.open(img_path)
    result = extract_fn(img)

    if len(args.images) == 1:
        out_path = Path(args.output)
    else:
        stem = Path(img_path).stem
        out_path = Path(args.output).parent / f"{stem}_lineart.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))
    outputs.append(str(out_path))
    _bar.update(1)
_bar.close()
```
