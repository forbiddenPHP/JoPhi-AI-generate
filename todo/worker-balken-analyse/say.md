# say Worker — Progress-Bar Analyse

**Entry Point:** `worker/say/list_models.py` (nur Model-Listing), Logik in `generate.py` Funktion `_voice_say()` (Zeile 1533)

## Architektur

Der say Worker hat kein eigenes generate-Script. Er nutzt den nativen macOS `say`-Befehl. Die gesamte Logik steckt in `generate.py:_voice_say()`. Das `worker/say/list_models.py` listet nur die verfuegbaren macOS-Stimmen auf.

## tqdm-Setup

Kein tqdm. Keine Balken.

---

## Step-Analyse (_voice_say in generate.py, Zeile 1533-1581)

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Text vorbereiten / Dateinamen bauen | Nein | Nein | Trivial, kein Balken noetig |
| macOS say Ausfuehrung | Ja (Z.1556) | Nein | Simulierter Balken fehlt |
| Optional: RVC Post-Processing | Ja (Z.1567) | Nein | Wird an _tts_rvc delegiert (siehe rvc.md) |

## Handlungsbedarf

Minimal — nur ein simulierter Balken fuer den `say`-Aufruf sinnvoll. Die RVC-Weiterleitung erbt Balken aus der rvc-Analyse.

### Vorschlag: macOS say (Zeile 1556-1562)

```python
# Zeile 1556: _emit(f"say → {wav_path.name}" + ...) — BLEIBT
_bar = tqdm(total=1, desc="macOS say", file=sys.stderr); _bar.refresh()
r = run_worker(cmd)
_bar.update(1); _bar.close()
if r.returncode != 0:
    ...
```

### Hinweis

Da `say` normalerweise sehr schnell ist (< 1s), ist der Balken hier eher fuer Konsistenz als fuer echte Fortschrittsanzeige. Bei langen Texten kann `say` aber durchaus mehrere Sekunden dauern.
