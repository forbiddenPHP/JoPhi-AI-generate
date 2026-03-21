# Diarize Worker — Progress-Bar Analyse

**Entry Point:** `worker/diarize/diarize.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | WAV-Konvertierung (wenn noetig) | Teilweise (`Converting to WAV …` Z.243) | Nein | Balken fehlt |
| 2 | Pyannote Pipeline laden | Teilweise (`Loading pyannote pipeline …` Z.266) | Nein | Balken fehlt |
| 3 | Diarization ausfuehren | Teilweise (`Running diarization …` Z.274) | Nein | Balken fehlt |
| 4 | Original-Audio lesen | Nein | Nein | Label + Balken fehlt |
| 5 | Speaker-Tracks schreiben | Teilweise (`Writing speaker tracks …` Z.312) | Nein | Balken fehlt (real ueber Speakers-Loop) |
| 6 | Diarization-JSON speichern | Nein | Nein | Label + Balken fehlt |

## Konkrete Code-Vorschlaege

### Am Datei-Anfang — tqdm import

```python
from tqdm import tqdm
```

### Z.240-248 — WAV-Konvertierung (Step 1)

```python
print("  WAV konvertieren …", file=sys.stderr)
_bar = tqdm(total=1, desc="WAV konvertieren", file=sys.stderr)
_bar.refresh()
subprocess.run(
    ["ffmpeg", "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1", tmp_wav.name],
    capture_output=True, check=True,
)
_bar.update(1)
_bar.close()
wav_path = Path(tmp_wav.name)
```

### Z.266-271 — Pipeline laden (Step 2)

```python
print("  Pipeline laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Pipeline laden", file=sys.stderr)
_bar.refresh()
os.environ["HF_TOKEN"] = token
pipeline_obj = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline_obj.to(device)
_bar.update(1)
_bar.close()
```

### Z.274-279 — Diarization (Step 3)

```python
print("  Diarization …", file=sys.stderr)
_bar = tqdm(total=1, desc="Diarization", file=sys.stderr)
_bar.refresh()
result = pipeline_obj(str(wav_path), **diarize_kwargs)
_bar.update(1)
_bar.close()
```

### Z.291-294 — Audio lesen (Step 4)

```python
print("  Audio lesen …", file=sys.stderr)
_bar = tqdm(total=1, desc="Audio lesen", file=sys.stderr)
_bar.refresh()
data, sr = sf.read(str(input_path), dtype="float32")
_bar.update(1)
_bar.close()
```

### Z.312-331 — Speaker-Tracks (Step 5, real ueber Speakers)

```python
print("  Speaker-Tracks schreiben …", file=sys.stderr)
_bar = tqdm(total=len(speakers), desc="Speaker-Tracks", file=sys.stderr)
_bar.refresh()
for speaker in speakers:
    # ... track erstellen ...
    sf.write(str(out_path), track, sr)
    output_paths.append(str(out_path))
    _bar.update(1)
_bar.close()
```

### Z.333-348 — JSON speichern (Step 6)

```python
print("  Diarization-JSON speichern …", file=sys.stderr)
_bar = tqdm(total=1, desc="JSON speichern", file=sys.stderr)
_bar.refresh()
# ... JSON write ...
_bar.update(1)
_bar.close()
```
