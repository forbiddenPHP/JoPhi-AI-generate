# Separate Worker — Progress-Analyse

**Entry Point:** `worker/separate/separate.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Model laden (get_model + to(device)) | Ja (`Loading separation model …`, Z.32) | Nein | Simulierter tqdm-Balken |
| 2. Audio laden (AudioFile.read) | Ja (`Loading audio …`, Z.49) | Nein | Simulierter tqdm-Balken |
| 3. Stems separieren (apply_model) | Ja (`Separating stems …`, Z.57) | Ja (progress=True, Z.58) — demucs hat eigene tqdm | OK, aber `file=sys.stderr` prüfen! |
| 4. Stems schreiben (sf.write) | Ja (`Writing stems …`, Z.63) | Nein | tqdm über model.sources (real, 4 Items) |

## Konkrete Code-Vorschläge

### Z.32-46 — Model laden: Balken fehlt

```python
# Z.32 ersetzen + Balken einfügen:
from tqdm import tqdm

print("  Loading separation model …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading model", file=sys.stderr)
_bar.refresh()
model = get_model(model_name)
model.to(device)
_bar.update(1)
_bar.close()
```

### Z.49-54 — Audio laden: Balken fehlt

```python
# Z.49-54 ersetzen:
print("  Loading audio …", file=sys.stderr)
_bar2 = tqdm(total=1, desc="Loading audio", file=sys.stderr)
_bar2.refresh()
wav = AudioFile(input_path).read(streams=0, samplerate=model.samplerate,
                                  channels=model.audio_channels)
ref = wav.mean(0)
wav -= ref.mean()
wav /= ref.std() + 1e-8
_bar2.update(1)
_bar2.close()
```

### Z.57-58 — Separierung: Hat bereits `progress=True`

Demucs' `apply_model` hat eine eigene tqdm-Bar. **Prüfen:** Ob die auf `sys.stderr` geht (Standard bei tqdm, sollte passen). Falls nicht, muss der `progress`-Parameter angepasst werden.

### Z.63-67 — Stems schreiben: Balken fehlt (real möglich)

```python
# Z.63-67 ersetzen:
print("  Writing stems …", file=sys.stderr)
for i, source_name in tqdm(enumerate(model.sources), total=len(model.sources),
                            desc="Writing stems", file=sys.stderr):
    out_path = output_dir / f"{stem}_{source_name}.wav"
    sf.write(str(out_path), sources[i].cpu().numpy().T, model.samplerate)
```

## Import nötig

`from tqdm import tqdm` am Anfang der Datei.
