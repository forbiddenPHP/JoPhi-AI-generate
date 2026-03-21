# Enhance Worker — Progress-Bar Analyse

**Entry Point:** `worker/enhance/enhance.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Modell laden | Teilweise (`Loading enhance model …` Z.145) | Nein | Balken fehlt |
| 2 | MPS Conv1d Patch (wenn MPS) | Nein | Nein | Label + Balken fehlt (oder in Step 1 integrieren) |
| 3 | Audio laden (torchaudio.load) | Nein | Nein | Label + Balken fehlt |
| 4 | Enhancement/Denoise Inference | Teilweise (`Enhancing audio …` Z.161, `[i/total] mode: name` Z.162) | Nein | Balken fehlt (real ueber Dateien-Loop, oder simuliert pro Datei) |
| 5 | Audio speichern | Nein | Nein | Balken fehlt |

Hinweis: Die Enhancement-Funktion `enhance()` aus `resemble_enhance` hat intern einen ODE-Solver mit `steps` Iterationen. Ein echter Balken darueber waere ideal, erfordert aber Patch in der Library. Alternativ simulierter Balken pro Datei.

## Konkrete Code-Vorschlaege

### Am Datei-Anfang (nach Z.27) — tqdm import

```python
from tqdm import tqdm
```

### Z.145-146 — Modell laden (Step 1, simuliert)

Das eigentliche Laden passiert lazy beim ersten `enhance()`/`denoise()` Call. Trotzdem Label + Balken um den Import-Block:

```python
print("  Enhance-Modell laden …", file=sys.stderr)
_bar = tqdm(total=1, desc="Enhance-Modell laden", file=sys.stderr)
_bar.refresh()
# ... pre-load if possible ...
_bar.update(1)
_bar.close()
```

### Z.152-175 — Dateien verarbeiten (Steps 3-5, real ueber Dateien-Loop)

```python
_bar = tqdm(total=total, desc="Enhancement", file=sys.stderr)
_bar.refresh()
for i, input_file in enumerate(args.input, 1):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = output_dir / input_path.name
    print(f"  [{i}/{total}] {mode}: {input_path.name}", file=sys.stderr)

    try:
        process_file(
            input_path, out_path, device,
            denoise_only=args.denoise_only,
            enhance_only=args.enhance_only,
            strength=args.strength,
            steps=args.steps,
        )
        output_paths.append(str(out_path))
    except Exception as e:
        print(f"ERROR: {input_path.name}: {e}", file=sys.stderr)
        sys.exit(1)
    _bar.update(1)
_bar.close()
```

### Innerhalb `process_file()` — Feinere Balken (optional)

In Z.82 (Audio laden) und Z.117-119 (Audio speichern) koennte man je einen simulierten Balken einbauen. Wichtiger waere aber ein Balken ueber die `nfe` Steps im Enhancement-Solver — das erfordert jedoch einen Patch in `resemble_enhance.enhancer.inference.enhance()`.
