# Renderer-Konzept

## Architektur

Drei getrennte Phasen:

1. **Generieren** — TTS, Video-KI, Enhance, Demucs etc. Fuellt den Mediapool. Kein Teil des Renderers.
2. **Projektdatei** — Timeline-Beschreibung mit Referenzen auf den Mediapool.
3. **Rendern** — Liest Projektdatei, baut ffmpeg-Befehle, gibt fertiges Video/Audio aus.

Der Renderer kennt keine KI. Er sieht nur Dateien und Anweisungen.

---

## Mediapool

Alle Quelldateien (Video, Audio, Bilder) liegen im Mediapool. Die Timeline referenziert nur — nichts wird kopiert oder veraendert. Eine Source kann beliebig oft referenziert werden (verschiedene Ausschnitte, verschiedene Einstellungen). Alles ist non-destruktiv.

Eine Video-Source ist implizit zwei Tracks: Video-Track + Audio-Track. Diese werden im Renderer getrennt behandelt.

---

## Timeline

### Layer-System

```
Layer -N  ...              Overlays (ueber Master)
Layer -2
Layer -1
──────────────────────────
Layer  0   MASTER          Definiert Mindestlaenge
──────────────────────────
Layer +1
Layer +2
Layer +N  ...              Underlays (unter Master)
```

- **Layer 0 (Master)**: Hauptspur. Definiert die Mindestlaenge der Timeline.
- **Negative Layer**: Overlays — ueberdecken den Master (Video per Opacity).
- **Positive Layer**: Underlays — liegen unter dem Master.
- **Gesamtlaenge**: `max(Ende aller Clips auf allen Layern)`. Clips koennen ueber den Master hinausragen.

### Clips

Jeder Clip ist eine virtuelle Referenz — kein echtes Schneiden oder Kopieren.

Pflichtfelder:
- `id` — Eindeutige, stabile ID (z.B. `interview.wav.dialogue.segment.1`)
- `type` — `video` | `audio` | `image` | `color` | `meta`
- `layer` — Auf welchem Layer (int, default 0)
- `position` — Wo auf der Timeline (Frame)

Optionale Felder:
- `in` / `out` — Ausschnitt aus Source (Frames)
- `volume` — Audio-Lautstaerke (dB)
- `mute` — Expliziter Audio-Mute (bool)
- `opacity` — Video-Sichtbarkeit (%)
- `pan` — Stereo-Position (-1.0 bis 1.0)
- `fade-in` / `fade-out` — Dauer in Sekunden
- `transform` — Video-Position und Groesse (x/y/w/h in Pixel)
- `color` — Farbe fuer type=color (#hex)
- `length` — Dauer fuer type=color/image (Frames)
- `chain[]` — Render-Chain (weitere Effekte)

Jeder `type` kann jederzeit zu jedem anderen werden. Der Type ist nur der aktuelle Zustand.

### Meta-Blocks

Meta-Blocks sind Elemente auf Layer 0 mit Produktionsanweisungen (Szenenbeschreibung, Referenzbilder, Dialog, Kameraanweisungen). Fuer den Renderer sind sie wie jeder andere Block — sie haben Laenge, Farbe, ggf. Audio/Video.

- **Laenge 0 Frames** — Im Render unsichtbar. Im Editor sichtbar (visuelle Darstellungsbreite).
- **Laenge > 0 Frames** — Rendert als Solid Color. Audio auf anderen Layern spielt trotzdem weiter.

### Sub-Clips

Aus einer Source koennen beliebig viele Sub-Clips erzeugt werden, jeder mit eigener ID und eigenen Einstellungen:

```
--clip 0:id=dialog-satz1,in=0,out=44100,position=0,volume=0
--clip 0:id=dialog-satz2,in=88200,out=132300,position=44100,mute=true
--clip 0:id=dialog-satz3,in=176400,out=220500,position=88200,volume=-3
```

ID-Schema: `{dateiname}.{kontext}.{nummer}` (z.B. `interview.wav.dialogue.segment.1`)

---

## Render-Regeln

### Video

Layer-Stack von oben (-N) nach unten (+N). Sobald ein Layer `opacity=100%` hat, wird alles darunter uebersprungen (Render-Optimierung).

Kein Master-Video an einer Stelle = schwarzer Hintergrund (oder was darunter liegt).

### Audio

Alle Layer werden zusammengemixt. Einzige Ausnahme: `mute=true`. Ein Track mit `-80dB` ist leise aber nicht stumm — wird gemixt.

Audio ist "transparent": Wenn auf einem Block kein Audio liegt, scheint Audio von anderen Layern durch. Es gibt kein "schwarzes Audio".

Ueber Meta-Blocks (Solid Color) hinweg wird Audio von anderen Layern weiter gemixt. Nur der Master selbst hat dort Stille.

### Video und Audio sind getrennt

Auch wenn Audio im Video enthalten ist — im Renderer gibt es zwei getrennte Streams. Video und Audio werden unabhaengig behandelt, unabhaengig gerendert, unabhaengig gecached.

---

## Caching

### Prinzip

Die Timeline wird in gleichmaessige Segmente zerlegt (z.B. 5 Sekunden). Jedes Segment bekommt einen Hash ueber alle Clips die in diesem Zeitfenster aktiv sind.

Stimmt der Hash → aus dem Cache nehmen. Stimmt er nicht → neu rendern.

### Getrennte Caches

- **Video-Cache**: Aendert sich nur Volume/Pan eines Clips → Video-Cache bleibt gueltig.
- **Audio-Cache**: Aendert sich nur Opacity/Transform → Audio-Cache bleibt gueltig.

### Invalidierung

- **Mediapool-Aenderung**: Hash pro Datei (Dateigroesse + mtime). Aendert sich eine Source → alle Segmente die Clips dieser Source enthalten sind invalidiert.
- **Timeline-Aenderung**: Aendert sich ein Clip (in/out, position, volume, ...) → alle Zeitabschnitte in denen dieser Clip aktiv ist werden invalidiert.
- **Laengen-Aenderung**: Aendert sich die Breite eines Master-Clips → alle nachfolgenden Positionen verschieben sich → alles ab dort wird invalidiert.

### Cache-Keys

Pro Segment: `{zeitbereich}_{audio|video}_{hash_aller_aktiven_clips}`

---

## CLI-Format

### Einfacher Fall (bestehende Engines)

```
python generate.py output --engine audio-concatenate a.wav b.wav -o out.wav
python generate.py output --engine audio-mucs a.wav b.wav c.wav -o mix.wav
```

`--clip` fuer einfache Per-Input-Optionen:

```
--clip 0:volume=0.8,fade-in=0.3
--clip 1:pan=-0.5,fade-out=1.0
```

### Volle Timeline

```
python generate.py output --project timeline.json -o final.mp4
```

### --clip Format (aufgeloest, Renderer-Ebene)

```
--clip INDEX:id=clip-id,layer=0,type=video,in=0,out=441000,position=0,volume=0,opacity=100
```

`INDEX` = Mediapool-Index (positionale Input-Dateien). Bei `type=color` kein Index noetig.

---

## ffmpeg

Der Renderer baut aus der aufgeloesten Timeline ffmpeg-Befehle:

- **Video-Compositing**: `overlay`-Filter mit Opacity, Transform (Position/Groesse)
- **Audio-Mix**: `amix` / `amerge` mit Volume, Pan, Fade
- **Solid Color**: `color=c=#hex:s=WxH:d=SECONDS` Source-Filter
- **Standbild**: `loop=1:size=FRAMES` fuer Images mit bestimmter Dauer

---

## Implementierung

### Vorhanden

- [x] `--clip` Parser (`_parse_clip_opts`) mit: fade_in, fade_out, crossfade, volume, start, end, pan
- [x] `audio-concatenate` Engine (sequenzielles Zusammenfuegen mit Crossfade)
- [x] `audio-mucs` Engine (paralleler Mix aller Inputs, flach ohne Layer)

### Zu implementieren

#### Phase 1 — Erweiterung --clip

- [ ] `pan-to` — Animierter Pan-Sweep ueber die Clip-Dauer (von `pan` nach `pan-to`, linear). Use-Case: Raumschiff fliegt von links nach rechts (`pan=-1,pan-to=1`). Offene Fragen: Reicht linearer Sweep? Braucht man Kurven (ease-in/out)? Braucht man `volume-to` analog dazu, oder reichen `fade-in`/`fade-out`? ffmpeg-Umsetzung via `aeval` mit `t`-Variable oder `sendcmd`. Genauer durchdenken wenn es so weit ist.
- [ ] `id` pro Clip (eindeutige, stabile ID)
- [ ] `layer` (int, default 0)
- [ ] `type` (video|audio|image|color|meta)
- [ ] `position` (Frame-basiert)
- [ ] `in`/`out` Frame-basiert (aktuell: `start`/`end` in Sekunden)
- [ ] `opacity` (%)
- [ ] `mute` (bool)
- [ ] `transform` (x/y/w/h)
- [ ] `color`, `length` fuer type=color

#### Phase 2 — Projektdatei

- [ ] JSON-Schema definieren (mediapool, timeline, layers, clips)
- [ ] Parser: Projektdatei → aufgeloeste Clip-Liste
- [ ] `--project timeline.json` Argument in generate.py
- [ ] Validierung (IDs eindeutig, Source-Indizes gueltig, Frames plausibel)

#### Phase 3 — Renderer

- [ ] Video-Compositing: Overlay-Stack mit Opacity, Transform
- [ ] Audio-Mix: Alle Layer mixen, Mute respektieren
- [ ] Solid Color Rendering (ffmpeg color source)
- [ ] Standbild Rendering (image mit Dauer)
- [ ] Render-Optimierung: Video-Layer skippen bei opacity=100% darueber
- [ ] Getrenntes Audio/Video Rendering

#### Phase 4 — Caching

- [ ] Timeline in gleichmaessige Segmente zerlegen (z.B. 5 Sekunden)
- [ ] Hash pro Segment ueber alle aktiven Clips
- [ ] Getrennter Audio-Cache und Video-Cache
- [ ] Mediapool-Hashing (Dateigroesse + mtime)
- [ ] Invalidierung: Aenderung → betroffene Segmente ermitteln → neu rendern
- [ ] Finaler Zusammenbau aus Cache-Segmenten
