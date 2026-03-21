# rvc Worker — Progress-Bar Analyse

**Entry Point:** `worker/rvc/start.sh` (startet API-Server), Logik in `generate.py` Funktion `_tts_rvc()` (Zeile 1441)

## Architektur

Der RVC Worker ist kein eigenstaendiges Python-Script mit Inference, sondern ein API-Server (`rvc-python`). Die eigentliche Nutzung laeuft ueber `generate.py:_tts_rvc()`, die HTTP-Requests an den laufenden Server schickt.

## tqdm-Setup

Kein tqdm in der Client-Logik (`_tts_rvc`). Der Server selbst ist eine Blackbox (rvc-python Paket).

---

## Step-Analyse (_tts_rvc in generate.py, Zeile 1441-1530)

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Voice Model setzen (API POST) | Nein | Nein | Label + Simulierter Balken fehlt |
| Auto-Pitch Kalibrierung | Ja (Z.1470) | Nein | Simulierter Balken fehlt |
| Pro Datei: ffmpeg Konvertierung (non-WAV) | Nein | Nein | Label + Simulierter Balken fehlt |
| Pro Datei: F0 Detection (Auto-Pitch) | Ja (Z.1509-1517) | Nein | Simulierter Balken fehlt |
| Pro Datei: RVC Conversion (API POST) | Ja (Z.1509-1517, Dateiname) | Nein | Simulierter Balken fehlt |

## Handlungsbedarf

Die _tts_rvc Funktion ist in generate.py, nicht im Worker-Verzeichnis. Balken muessten dort eingefuegt werden.

### Vorschlag: Voice Model setzen (Zeile 1456-1457)

```python
# Vor Zeile 1456:
if model_name:
    print(f"  Setting voice model: {model_name} …", file=sys.stderr)
    _bar = tqdm(total=1, desc=f"Setting voice {model_name}", file=sys.stderr); _bar.refresh()
    api_post("/models/" + model_name)
    _bar.update(1); _bar.close()
```

### Vorschlag: Pro Datei Conversion (Zeile 1521-1527)

```python
# Nach Zeile 1519:
_bar = tqdm(total=1, desc=f"Converting {input_path.name}", file=sys.stderr); _bar.refresh()
# ... Zeile 1521-1527: api_post("/convert_file", ...)
_bar.update(1); _bar.close()
```

### Vorschlag: Batch-Balken (Zeile 1484 Loop)

```python
_bar = tqdm(total=total, desc="RVC conversion", file=sys.stderr); _bar.refresh()
for i, input_path in enumerate(input_paths, 1):
    # ... conversion logic ...
    _bar.update(1)
_bar.close()
```
