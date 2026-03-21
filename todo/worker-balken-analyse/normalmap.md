# normalmap Worker — Progress-Bar Analyse

**Entry Point:** `worker/normalmap/generate.py`

## tqdm-Setup

Kein tqdm-Import vorhanden. Kein globaler Override. Keine Balken.
Die Marigold-Pipeline (diffusers) nutzt intern tqdm fuer Denoising-Steps, aber ohne `file=sys.stderr` Override landen die Balken nicht zuverlaessig auf stderr (in Pipes disabled).

---

## Step-Analyse

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Model laden (Marigold-Normals) | Ja (Z.49) | Nein | Simulierter Balken fehlt |
| Attention Slicing aktivieren | Nein | Nein | Trivial, kein Balken noetig |
| Pro Bild: Processing (Inference) | Ja (Z.59) | Nein | Simulierter Balken fehlt (intern hat pipe() tqdm, aber ohne stderr-Force) |
| Pro Bild: Speichern | Ja (Z.75) | Nein | Trivial, kein Balken noetig |

## Handlungsbedarf

1. tqdm importieren und globalen Override einrichten (wie ltx2) damit diffusers-interne Balken auf stderr landen
2. Simulierte Balken fuer Model-Laden und pro-Bild Processing

### Vorschlag: tqdm-Override (nach Zeile 12, vor torch import)

```python
import tqdm as _tqdm_mod, tqdm.auto as _tqdm_auto
_OrigTqdm = _tqdm_mod.tqdm
class _ForceTqdm(_OrigTqdm):
    def __init__(self, *a, **kw):
        kw["file"] = sys.stderr
        kw["disable"] = False
        super().__init__(*a, **kw)
        self.disable = False
        self.refresh()
_tqdm_mod.tqdm = _ForceTqdm
_tqdm_auto.tqdm = _ForceTqdm
```

Mit diesem Override werden die internen diffusers-Denoising-Balken automatisch auf stderr sichtbar. Dann braucht der Inference-Step keinen separaten simulierten Balken.

### Vorschlag: Model laden (Zeile 49-55)

```python
from tqdm import tqdm
# Zeile 49: print("Loading Marigold-Normals v1.1 …", ...) — BLEIBT
_bar = tqdm(total=1, desc="Loading Marigold-Normals", file=sys.stderr); _bar.refresh()
pipe = MarigoldNormalsPipeline.from_pretrained(...)
pipe.enable_attention_slicing()
_bar.update(1); _bar.close()
```

### Vorschlag: Pro Bild Processing (Zeile 59-62)

Falls der tqdm-Override nicht reicht (Marigold hat nur 4 Steps, geht schnell):

```python
# Zeile 59: print(f"  Processing: {img_path}", ...) — BLEIBT
_bar = tqdm(total=1, desc=f"Processing {Path(img_path).name}", file=sys.stderr); _bar.refresh()
result = pipe(img, num_inference_steps=args.steps)
_bar.update(1); _bar.close()
```
