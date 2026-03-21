# Worker stdout JSON-Array Audit

Regel: Jeder Worker MUSS als letzte stdout-Zeile ein JSON-Array aller Output-Pfade ausgeben.
Ausnahme: text-Worker (eigenes Event-Protokoll).

Audit-Datum: 2026-03-21

| Worker | JSON-Array? | Alle Outputs? | Nichts danach? | stderr flush? | Problem |
|--------|------------|---------------|----------------|---------------|---------|
| ace | Ja | Ja | Ja | Ja (Z200) | -- |
| depth | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| diarize | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| enhance | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| image | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| langdetect | NEIN | -- | -- | -- | **Kein JSON-Array.** Gibt nur ISO-Code als plain text aus (z.B. `de`). Kein Array-Format. Ist aber Hilfs-Worker, schreibt keine Dateien. |
| lineart | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| ltx2 | Ja | Ja | Ja | Ja (Z649) | -- |
| music | Ja | Ja | Ja | Ja (Z120) | -- |
| normalmap | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| pose | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| rvc | Ja | Ja | Ja | Nein | Worker ist API-Server (start.sh). JSON-Array kommt aus `generate.py:_tts_rvc()` (Z1530). Kein eigenes generate.py. Kein stderr flush im Dispatcher. |
| say | Ja | Ja | Ja | Nein | Worker hat kein eigenes generate.py. JSON-Array kommt aus `generate.py:_voice_say()` (Z1581). Kein stderr flush im Dispatcher. |
| sd15 | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| segment | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| separate | NEIN | -- | -- | -- | **Kein JSON-Array.** `separate()` gibt `None` zurueck (kein return). `main()` ruft nur `separate()` auf, kein `print(json.dumps(...))`. Stem-Pfade werden nie auf stdout ausgegeben. |
| sfx | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| sketch | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| tts | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| upscale | Ja | Ja | Ja | Nein | Kein `sys.stderr.flush()` vor JSON-Array |
| whisper | Ja | Ja* | Ja | Ja (Z205) | *Wenn kein `--output` angegeben: Array ist leer `[]` (keine Dateien geschrieben). Korrekt, aber leer. |

## Kritische Probleme

### 1. separate — Kein JSON-Array (FEHLT)

`worker/separate/separate.py`: Die `separate()` Funktion gibt keinen Rueckgabewert zurueck.
`main()` ruft `separate()` auf, aber gibt kein JSON-Array auf stdout aus.

**Fix:** In `main()` nach `separate()` Aufruf die Pfade sammeln und als JSON-Array ausgeben.

### 2. langdetect — Kein JSON-Array

`worker/langdetect/detect.py`: Gibt nur einen ISO-Code als plain text aus.
Ist ein Hilfs-Worker der keine Dateien erzeugt — fragwuerdiger Sonderfall.
Wenn langdetect als normaler Worker zaehlt: fehlendes Array.
Wenn reiner Hilfs-Worker: akzeptabel.

## Fehlende stderr flush (15 Worker)

Folgende Worker haben kein `sys.stderr.flush()` direkt vor dem JSON-Array:
depth, diarize, enhance, image, lineart, normalmap, pose, sd15, segment, sfx, sketch, tts, upscale, rvc (Dispatcher), say (Dispatcher)

Risiko: Bei gepuffertem stderr koennte stderr-Output nach dem JSON-Array auf stdout ankommen,
was den Parser verwirrt (stderr und stdout werden beim Lesen interleaved).

Worker MIT korrektem flush: ace, ltx2, music, whisper.
