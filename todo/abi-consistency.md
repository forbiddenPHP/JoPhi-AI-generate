# ABI-Konsistenz: README.md vs generate.py

## Erledigt
- [x] `-p` Konflikt gelöst: `-p`=`--prompt`, `-pf`=`--prompt-file`, `-pt`=`--port`
- [x] `--timeout` aus README entfernt (darf nicht existieren)
- [x] `--base-url`, `--api-key` in README nachgetragen
- [x] Tests geprüft — nutzen korrekte Flags
- [x] Usage-Header in generate.py geprüft — war bereits korrekt

## Offen
- [ ] USAGE-Docstring: einige Params fehlen als Beispiele (--word-timestamps, --thinking, --stream, --screen-log-format json) — nice-to-have, kein Blocker
