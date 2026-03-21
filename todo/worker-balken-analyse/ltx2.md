# ltx2 Worker — Progress-Bar Analyse (REFERENZ)

**Entry Point:** `worker/ltx2/generate.py`

## tqdm-Setup

Der ltx2 Worker hat einen globalen tqdm-Override (Zeilen 24-34), der:
- `file=sys.stderr` erzwingt
- `disable=False` erzwingt (auch in Pipes)
- `.refresh()` sofort nach Erstellung aufruft

Damit fangen alle internen tqdm-Balken (HF-Downloads, Denoising-Loops in den Pipelines) automatisch auf stderr.

## Step-Analyse

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|:-:|:-:|-----------|
| Model-Dateien resolven (Download) | Ja (Z.104, 116) | Ja (HF-Download hat eigene tqdm) | -- |
| Pipeline laden (Distilled/TwoStage/A2V/Retake) | Ja (Z.424, 502, 532, 556) | Nein | Simulierter Balken fehlt |
| Prompt Enhancement (Gemma) | Nein (intern in Pipeline) | Nein | Intern, schwer abzufangen |
| Denoising / Inference | Nein (intern) | Ja (Pipeline nutzt tqdm intern, wird durch Override gefangen) | -- |
| Video Encoding | Ja (Z.601) | Nein | Simulierter Balken fehlt |
| Transcode (extend/retake) | Ja (Z.380) | Nein | Simulierter Balken fehlt |
| Trim (extend) | Ja (Z.399) | Nein | Simulierter Balken fehlt |
| Concat (extend) | Ja (Z.486) | Nein | Simulierter Balken fehlt |

## Handlungsbedarf

Der ltx2 Worker dient als Referenz und ist schon gut aufgestellt. Die internen Denoising-Loops werden durch den globalen tqdm-Override automatisch erfasst. Es fehlen simulierte Balken fuer die Lade-/Encode-/ffmpeg-Steps.

### Vorschlag: Pipeline laden (z.B. Distilled, Zeile 532)

```python
# Nach Zeile 532: print("  Loading Distilled pipeline …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading Distilled pipeline", file=sys.stderr); _bar.refresh()
# ... Zeile 535-540: pipeline = DistilledPipeline(...)
_bar.update(1); _bar.close()
```

### Vorschlag: Video Encoding (Zeile 601)

```python
# Zeile 601 ersetzen:
print("  Encoding output …", file=sys.stderr)
_bar = tqdm(total=1, desc="Encoding output", file=sys.stderr); _bar.refresh()
# ... encode_video(...)
_bar.update(1); _bar.close()
```

### Vorschlag: Transcode (Zeile 380)

```python
# Nach Zeile 380: print(f"  Transcoding ... …", file=sys.stderr)
_bar = tqdm(total=1, desc="Transcoding source", file=sys.stderr); _bar.refresh()
# ... subprocess.run([ffmpeg ...])
_bar.update(1); _bar.close()
```
