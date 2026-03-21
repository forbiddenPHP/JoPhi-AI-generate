# Langdetect Worker — Progress-Bar Analyse

**Entry Point:** `worker/langdetect/detect.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Sprache erkennen | Nein | Nein | Alles fehlt |

## Bewertung

Dieser Worker ist extrem simpel: ein einziger `detect(text)` Aufruf, der in Millisekunden laeuft. Es gibt kein Modell-Laden (langdetect nutzt vortrainierte Profile), keine Iteration, kein nennenswertes Processing.

**Empfehlung:** Fuer Konsistenz kann man einen minimalen simulierten Balken einbauen, aber der Nutzen ist gering. Die Ausfuehrung dauert <100ms.

## Konkrete Code-Vorschlaege (optional)

### Z.29-31 — Erkennung mit Balken

```python
from tqdm import tqdm
import sys

print("  Sprache erkennen …", file=sys.stderr)
_bar = tqdm(total=1, desc="Sprache erkennen", file=sys.stderr)
_bar.refresh()
try:
    code = detect(args.text)
except Exception as e:
    _bar.close()
    print(f"ERROR: Language detection failed: {e}", file=sys.stderr)
    print("en")
    return
_bar.update(1)
_bar.close()
```

Hinweis: Da dieser Worker so schnell ist, flackert der Balken nur kurz. Kann man auch weglassen — kein User wartet hier auf Progress.
