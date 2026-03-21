# Text Worker — Progress-Analyse

**Entry Point:** `worker/text/inference.py`

## Besonderheit

Der Text Worker nutzt ein eigenes Event-Protokoll (`@inference:{JSON}`) statt tqdm-Balken. Progress.py parst diese Events direkt. **Kein tqdm nötig** — das Protokoll ist bewusst anders als bei Medien-Workern.

## Steps und Status

| Step | Label vorhanden? | Balken vorhanden? | Was fehlt |
|------|-------------------|-------------------|-----------|
| 1. Ollama-Client erstellen | Nein | Nein | Nicht nötig (instant) |
| 2. Config laden | Nein | Nein | Nicht nötig (instant) |
| 3. Images vorbereiten (optional) | Ja (`inference_mode: downloading image`, Z.211) | Nein | Kein Balken nötig — Event-Protokoll |
| 4. Inference starten | Ja (`inference_gotcha`, Z.306/355) | Nein | Event-basiert, kein Balken nötig |
| 5. Token-Streaming (stream mode) | Ja (`inference_token`, Z.318/371) | Nein | Event-basiert, kein Balken nötig |
| 6. Ergebnis | Ja (`inference_result`, Z.332/391) | Nein | Event-basiert, kein Balken nötig |

## Fazit

**Kein Handlungsbedarf.** Der Text Worker ist der einzige Worker mit einem eigenen Event-Protokoll (`@inference:`), das von progress.py speziell behandelt wird. tqdm-Balken wären hier kontraproduktiv — die Token-Streaming-Events sind das Äquivalent zum Fortschrittsbalken.

Falls gewünscht, könnte man für den Image-Download-Schritt (URL-Bilder) einen tqdm-Balken ergänzen, aber das ist ein Edge Case.
