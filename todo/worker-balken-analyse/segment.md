# Segment Worker — Progress-Analyse

**Entry Point:** `worker/segment/generate.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Bild laden (Image.open) | Ja (`Processing: {img_path}`, Z.60) | Nein | Simulierter tqdm-Balken |
| 2. Segmentierung (BiRefNet via rembg) | Ja (`Segmenting (BiRefNet) …`, Z.29) | Nein | Simulierter tqdm-Balken |
| 3. Maske invertieren / Background erstellen | Nein | Nein | Label + simulierter tqdm-Balken |
| 4. Ergebnis speichern | Ja (`Saved: {path}`, Z.73/87) | Nein | Simulierter tqdm-Balken |

## Konkrete Code-Vorschläge

### Z.29 — Segmentierung: Label korrekt, Balken fehlt

```python
# VOR Z.29 einfügen:
from tqdm import tqdm

# Z.29-31 ersetzen durch:
print("  Segmenting (BiRefNet) …", file=sys.stderr)
_bar = tqdm(total=1, desc="Segmenting", file=sys.stderr)
_bar.refresh()
session = new_session("birefnet-general")
foreground = remove(img, session=session)
_bar.update(1)
_bar.close()
```

### Z.34-42 — Maske invertieren: Label + Balken fehlt

```python
# Zwischen Z.31 (foreground = ...) und Z.34 (fg_np = ...) einfügen:
print("  Creating background mask …", file=sys.stderr)
_bar2 = tqdm(total=1, desc="Background mask", file=sys.stderr)
_bar2.refresh()
# ... bestehender Code Z.34-41 ...
_bar2.update(1)
_bar2.close()
```

### Z.60 — Bild laden: Balken fehlt

```python
# Z.60-62 ersetzen durch:
print(f"  Loading image …", file=sys.stderr)
_bar_load = tqdm(total=1, desc="Loading image", file=sys.stderr)
_bar_load.refresh()
img = Image.open(img_path).convert("RGB")
_bar_load.update(1)
_bar_load.close()
```

### Z.71/86 — Speichern: Balken fehlt

```python
# Beim Speichern (Z.71 für both-Modus, Z.86 für single-Modus):
print("  Saving result …", file=sys.stderr)
_bar_save = tqdm(total=1, desc="Saving", file=sys.stderr)
_bar_save.refresh()
# ... save-Code ...
_bar_save.update(1)
_bar_save.close()
```

## Import nötig

`from tqdm import tqdm` muss am Anfang der Datei hinzugefügt werden (oder innerhalb der Funktion).
