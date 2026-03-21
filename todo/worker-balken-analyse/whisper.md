# Whisper Worker — Progress-Analyse

**Entry Point:** `worker/whisper/transcribe.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Model laden (mlx_whisper Import + erstes transcribe) | Ja (`Loading whisper model …`, Z.52) | Nein | Simulierter tqdm-Balken |
| 2. Transkription (mlx_whisper.transcribe) | Ja (`Transcribing …`, Z.202) | Nein | Simulierter tqdm-Balken |
| 3. Ergebnis aufbereiten (segments parsen) | Nein | Nein | Nicht nötig (instant) |
| 4. Output-Dateien schreiben | Ja (`Saved: {out_path}`, Z.153) | Nein | Simulierter tqdm-Balken (oder real über Formate) |

## Konkrete Code-Vorschläge

### Z.52-53 — Model laden: Balken fehlt

```python
from tqdm import tqdm

# Z.52-53 ersetzen:
print("  Loading whisper model …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading model", file=sys.stderr)
_bar.refresh()
import mlx_whisper
_bar.update(1)
_bar.close()
```

**Hinweis:** Der eigentliche Model-Download/-Load passiert erst beim ersten `mlx_whisper.transcribe()` Aufruf (Z.61). Der Import allein lädt das Model nicht. Der Balken für "Loading model" müsste also den transcribe-Aufruf umfassen, oder man splittet:

### Z.61-67 — Transkription: Balken fehlt

```python
# Z.61-67 ersetzen:
print("  Transcribing …", file=sys.stderr)
_bar_trans = tqdm(total=1, desc="Transcribing", file=sys.stderr)
_bar_trans.refresh()
result = mlx_whisper.transcribe(
    str(input_path),
    path_or_hf_repo=model,
    **kwargs,
)
_bar_trans.update(1)
_bar_trans.close()
```

### Z.145-155 — Output speichern: Balken fehlt (real möglich über Formate)

```python
# Z.149-154 ersetzen:
for fmt in tqdm(formats, desc="Saving formats", file=sys.stderr):
    writer = FORMAT_WRITERS[fmt]
    out_path = out_dir / f"{stem}.{fmt}"
    writer(entry, out_path)
```

### Z.196-246 — Mehrere Dateien: Übergreifender Balken

```python
# Z.196 ersetzen:
for input_file in tqdm(args.input, desc="Processing files", file=sys.stderr):
```

## Import nötig

`from tqdm import tqdm` am Anfang der Datei.

## Hinweis

mlx_whisper hat intern möglicherweise eigene Progress-Ausgaben. Prüfen, ob die auf stderr gehen und ob sie mit progress.py kompatibel sind. Falls ja, könnte man sich den simulierten Balken für Transkription sparen.
