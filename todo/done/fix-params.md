# Fix: Parameter-Inkonsistenzen vereinheitlichen

## 1. `--topk` → `--top-k`

**Problem:** Audio nutzt `--topk`, Text nutzt `--top-k`. Sollte überall `--top-k` sein (wie `--top-p`).

**Dateien:**
- `generate.py:3049` — `p_audio.add_argument("--topk", ...)` → `"--top-k"`
- `generate.py:1853` — `args.topk` → `args.top_k`
- `generate.py:1854` — Weiterleitung an ACE-Step bleibt `--lm-top-k` (ACE eigenes Flag)
- `generate.py:1960-1961` — `args.topk` → `args.top_k`, Weiterleitung an HeartMuLa bleibt `--topk` (upstream)
- `README.md:438` — `--topk` → `--top-k`

**Achtung:** Die Worker-internen Flags bleiben wie sie sind:
- `worker/ace/generate.py` nutzt `--lm-top-k` (eigenes Mapping in generate.py)
- `worker/music/heartlib/` nutzt intern `topk` (upstream-Code, nicht ändern)
- Nur die generate.py ABI nach außen wird vereinheitlicht

## 2. `--input-language` → `--language`

**Problem:** Whisper nutzt `--input-language`, alle anderen nutzen `--language`.

**Dateien:**
- `generate.py:3102` — `p_text.add_argument("--input-language", ...)` → `"--language"`
- `generate.py:36` — Docstring-Beispiel `--input-language de` → `--language de`
- `generate.py:2213-2214` — `args.input_language` → `args.language`
- `generate.py:2229-2230` — `args.input_language` → `args.language`
- `README.md:654` — `--input-language` → `--language`
- `tests/suites/test_whisper.py:58` — `"--input-language"` → `"--language"`
- `tests/suites/test_whisper.py:69` — `"--input-language"` → `"--language"`
- `tests/suites/test_whisper.py:80` — `"--input-language"` → `"--language"`

**Achtung:** Whisper-Worker (`worker/whisper/`) akzeptiert `--language` nativ, also kein Worker-Fix nötig. Nur die generate.py-Seite + Tests.

## 3. `-p` Kurzform-Konflikt

**Problem:** `-p` bedeutet:
- Bei `voice`: `--prompt-file` (Sidecar-Datei laden)
- Bei `image`: `--prompt` (Text-Prompt)

Da es verschiedene Subparser sind, crasht es technisch nicht. Aber es ist verwirrend für User.

**Optionen:**
- A) `-p` überall = `--prompt`, Voice bekommt anderes Kürzel für `--prompt-file` (z.B. `-P` oder kein Kürzel)
- B) Lassen wie es ist (verschiedene Mediums, verschiedene Kontexte)

**Dateien (falls Option A):**
- `generate.py:2988` — `voice_text_grp.add_argument("--prompt-file", "-p", ...)` → Kürzel entfernen oder ändern
- `README.md:191` — Dokumentation anpassen

**Empfehlung:** Option A — `-p` = `--prompt` überall. `--prompt-file` behält kein Kürzel (wird selten genutzt).
