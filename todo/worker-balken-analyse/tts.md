# TTS Worker — Progress-Analyse

**Entry Point:** `worker/tts/generate_speech.py`

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Model laden (mlx_audio load_model) | Ja (`Loading model …`, Z.204) | Nein | Simulierter tqdm-Balken |
| 2. Sprache erkennen (langdetect) | Ja (`Detected language: …`, Z.214) | Nein | Nicht nötig (instant) |
| 3. Segment-Generierung (model.generate) | Ja (`[{i}/{total}] {voice} …`, Z.249) | Nein | Simulierter tqdm-Balken pro Segment |
| 4. Audio zusammenfügen (np.concatenate) | Nein | Nein | Label + simulierter tqdm-Balken |
| 5. WAV speichern (sf.write) | Ja (`Saved: {output_path}`, Z.286) | Nein | Simulierter tqdm-Balken |

## Konkrete Code-Vorschläge

### Z.204-206 — Model laden: Balken fehlt

```python
from tqdm import tqdm

# Z.204-206 ersetzen:
print("  Loading model …", file=sys.stderr)
_bar = tqdm(total=1, desc="Loading model", file=sys.stderr)
_bar.refresh()
from mlx_audio.tts.utils import load_model
model = load_model(args.model)
_bar.update(1)
_bar.close()
```

### Z.245-279 — Segment-Generierung: Label vorhanden, Balken fehlt

Zwei Optionen:
1. **Übergreifender Balken** über alle Segmente (real):
2. **Simulierter Balken** pro Segment:

```python
# Option 1: Real-Balken über Segmente (Z.245 ersetzen):
for i, (voice, seg_instruct, seg_lang, text) in tqdm(
        enumerate(segments, 1), total=total,
        desc="Generating speech", file=sys.stderr):
    # ... bestehender Code ...
```

```python
# Option 2: Simulierter Balken pro Segment (innerhalb der Schleife, nach Z.249):
_bar_seg = tqdm(total=1, desc=f"Segment {i}/{total}", file=sys.stderr)
_bar_seg.refresh()
results = list(model.generate(**gen_kwargs))
_bar_seg.update(1)
_bar_seg.close()
```

### Z.282-285 — Zusammenfügen + Speichern: Label + Balken fehlt

```python
# VOR Z.282 einfügen:
print("  Concatenating segments …", file=sys.stderr)
_bar_cat = tqdm(total=1, desc="Concatenating", file=sys.stderr)
_bar_cat.refresh()
final_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
_bar_cat.update(1)
_bar_cat.close()

print("  Saving audio …", file=sys.stderr)
_bar_save = tqdm(total=1, desc="Saving", file=sys.stderr)
_bar_save.refresh()
sf.write(str(output_path), final_audio, native_sr)
_bar_save.update(1)
_bar_save.close()
```

## Import nötig

`from tqdm import tqdm` am Anfang der Datei.
