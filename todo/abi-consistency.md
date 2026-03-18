# ABI-Konsistenz: README.md vs generate.py

## Problem
- `--prompt` hat bei `image` den Shortcut `-p`, bei `text` nicht
- Usage-Header in generate.py möglicherweise veraltet
- README.md Beispiele müssen mit tatsächlicher CLI übereinstimmen

## Aufgaben
- [ ] Alle Flags in generate.py extrahieren (pro Subparser)
- [ ] Mit README.md Dokumentation vergleichen
- [ ] Inkonsistenzen fixen (Shortcuts, Defaults, Beschreibungen)
- [ ] Usage-Header in generate.py aktualisieren
- [ ] Tests prüfen ob sie die korrekten Flags nutzen
