# music Worker — Progress-Bar Analyse

**Entry Points:** `worker/music/generate.py` (Musik-Generierung), `worker/music/transcribe.py` (Lyrics-Transkription)

## tqdm-Setup

Kein tqdm-Import vorhanden. Kein globaler Override. Keine Balken.

---

## generate.py — Step-Analyse

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Model laden (HeartMuLa) | Ja (Z.75) | Nein | Simulierter Balken fehlt |
| Temp-Dateien schreiben (Lyrics/Tags) | Nein | Nein | Trivial, kein Balken noetig |
| Musik generieren (Inference) | Ja (Z.97) | Nein | Simulierter Balken fehlt (Dauer unbekannt) |
| Ergebnis speichern | Nein | Nein | Intern in pipe(), kein Balken noetig |

### Vorschlag: tqdm importieren (nach Zeile 22)

```python
from tqdm import tqdm
```

### Vorschlag: Model laden (Zeile 75-88)

```python
# Zeile 75: print("Loading HeartMuLa model …", file=sys.stderr) — BLEIBT
_bar = tqdm(total=1, desc="Loading HeartMuLa model", file=sys.stderr); _bar.refresh()
# ... Zeilen 82-88: pipe = HeartMuLaGenPipeline.from_pretrained(...)
_bar.update(1); _bar.close()
```

### Vorschlag: Musik generieren (Zeile 97-113)

```python
# Zeile 97: print(...) — BLEIBT
_bar = tqdm(total=1, desc="Generating music", file=sys.stderr); _bar.refresh()
# ... Zeilen 101-113: pipe(...)
_bar.update(1); _bar.close()
```

---

## transcribe.py — Step-Analyse

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Model laden (HeartTranscriptor) | Ja (Z.54) | Nein | Simulierter Balken fehlt |
| Transkription (Inference) | Ja (Z.64) | Nein | Simulierter Balken fehlt |

### Vorschlag: tqdm importieren (nach Zeile 19)

```python
from tqdm import tqdm
```

### Vorschlag: Model laden (Zeile 54-62)

```python
# Zeile 54: print("Loading HeartTranscriptor …", ...) — BLEIBT
_bar = tqdm(total=1, desc="Loading HeartTranscriptor", file=sys.stderr); _bar.refresh()
# ... Zeilen 58-62: pipe = HeartTranscriptorPipeline.from_pretrained(...)
_bar.update(1); _bar.close()
```

### Vorschlag: Transkription (Zeile 64-80)

```python
# Zeile 64: print("Transcribing …", ...) — BLEIBT
_bar = tqdm(total=1, desc="Transcribing", file=sys.stderr); _bar.refresh()
# ... Zeilen 69-80: result = pipe(...)
_bar.update(1); _bar.close()
```
