# ACE Worker — Progress-Bar Analyse

**Entry Point:** `worker/ace/generate.py`

## Steps und Status

| # | Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|---|------|-------------------|-------------------|-----------|
| 1 | Model-Checkpoint pruefen/downloaden | Nein | Nein | Label + simulierter Balken |
| 2 | DiT Handler initialisieren | Teilweise (`Loading ACE-Step model …` Z.78) | Nein | Balken fehlt |
| 3 | LLM Handler initialisieren | Nein | Nein | Label + simulierter Balken |
| 4 | Generierung (generate_music) | Teilweise (`Generating music …` Z.145) | Nein | Balken fehlt (real wenn moeglich, sonst simuliert) |
| 5 | Ergebnis speichern/verschieben | Nein | Nein | Label + simulierter Balken |

## Konkrete Code-Vorschlaege

### Vor Z.66 — tqdm import + ForceTqdm Patch (wie ltx2)

```python
import tqdm as _tqdm_mod, tqdm.auto as _tqdm_auto
_OrigTqdm = _tqdm_mod.tqdm
class _ForceTqdm(_OrigTqdm):
    def __init__(self, *a, **kw):
        kw["file"] = sys.stderr
        kw["disable"] = False
        super().__init__(*a, **kw)
        self.disable = False
        self.refresh()
_tqdm_mod.tqdm = _ForceTqdm
_tqdm_auto.tqdm = _ForceTqdm
```

### Z.82-87 — Checkpoint Download (Step 1)

```python
print("  Checkpoint pruefen …", file=sys.stderr)
_bar = tqdm.tqdm(total=1, desc="Checkpoint pruefen", file=sys.stderr)
_bar.refresh()
ok, msg = ensure_dit_model(config_path, checkpoints_dir)
_bar.update(1)
_bar.close()
```

### Z.89-95 — DiT Handler laden (Step 2)

```python
print("  DiT-Handler laden …", file=sys.stderr)
_bar = tqdm.tqdm(total=1, desc="DiT-Handler laden", file=sys.stderr)
_bar.refresh()
dit_handler = AceStepHandler()
dit_handler.initialize_service(
    project_root=project_root,
    config_path=config_path,
    device=device,
)
_bar.update(1)
_bar.close()
```

### Z.97-104 — LLM Handler laden (Step 3)

```python
print("  LLM-Handler laden …", file=sys.stderr)
_bar = tqdm.tqdm(total=1, desc="LLM-Handler laden", file=sys.stderr)
_bar.refresh()
llm_handler = LLMHandler()
llm_handler.initialize(
    checkpoint_dir=project_root,
    lm_model_path="acestep-5Hz-lm-0.6B",
    backend=lm_backend,
    device=device,
)
_bar.update(1)
_bar.close()
```

### Z.163-170 — Generierung (Step 4)

```python
print("  Musik generieren …", file=sys.stderr)
_bar = tqdm.tqdm(total=1, desc="Musik generieren", file=sys.stderr)
_bar.refresh()
result = generate_music(
    dit_handler=dit_handler,
    llm_handler=llm_handler,
    params=params,
    config=config,
    save_dir=save_dir,
)
_bar.update(1)
_bar.close()
```

Hinweis: Falls `generate_music` intern einen Denoising-Loop hat, waere ein echter Balken ueber die Inference-Steps besser. Dafuer muesste man in `acestep/inference.py` schauen ob dort ein iterierbarer Loop existiert.

### Z.177-189 — Ergebnis speichern (Step 5)

```python
print("  Audio speichern …", file=sys.stderr)
_bar = tqdm.tqdm(total=1, desc="Audio speichern", file=sys.stderr)
_bar.refresh()
# ... move/save logic ...
_bar.update(1)
_bar.close()
```
