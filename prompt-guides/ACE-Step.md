# ACE-Step 1.5 - Prompt Guide

ACE-Step uses two text inputs: a **caption** (style description) and **lyrics** (song text with structure tags).

## Caption (--tags)

The caption describes the desired musical style. ACE-Step accepts multiple description formats: **short tags** (`"disco, happy, female"`), **descriptive text** (`"upbeat electronic dance music with heavy synth bass"`), or **use-case scenarios** (`"background music for a coffee shop"`).

**Max length:** 512 characters.

### What to include
- Genre and style (e.g. "upbeat electronic dance music")
- Instruments (e.g. "with heavy synth bass and arpeggiated leads")
- Mood and energy (e.g. "energetic and euphoric")
- Vocal style (e.g. "female vocal, breathy tone")
- Tempo hints (e.g. "fast-paced" or use `--bpm`)
- BPM and key (e.g. "110 bpm, G major")
- **Singer descriptions** for multi-voice songs (see Multi-Voice section below)

### Caption Examples by Genre

**Short tags (comma-separated):**
- `"electronic, rock, pop"`
- `"funk, pop, soul, melodic"`
- `"disco"`
- `"surf music"`
- `"alternative rock, pop, rock"`
- `"country rock, folk rock, southern rock, bluegrass, pop"`
- `"electronic rap"`
- `"Cuban music, salsa, son, Afro-Cuban, traditional Cuban"`

**Detailed descriptions with instruments/mood:**
- `"dark, death rock, metal, hardcore, electric guitar, powerful, bass, drums, 110 bpm, G major"`
- `"Dark Electro, Industrial Techno, Gothic Rave"`
- `"cyberpunk, Acid jazz, electro, em, soft electric drums"`
- `"aggressive, Heavy Riffs, Blast Beats, Satanic Black Metal"`
- `"808 bass, smooth melody, deep voice, trap beat"`
- `"drum & bass, 160bpm, ethereal dark liquid, deep bassline, female vocals"`

**Use-case scenarios:**
- `"background music for parties, radio broadcasts, streaming platforms, female voice"`
- `"Nightclubs, dance parties, workout playlists, radio broadcasts"`

**Emphasis through repetition** (experimental, works for strong style lock-in):
- `"DUBSTEP, OBSCURE, DUBSTEP, DARKNESS, DUBSTEP, FEAR, DUBSTEP, TERROR, DUBSTEP"`

**Minimal prompts** (model fills in the gaps):
- `"female voice"` — just vocal gender, model decides genre
- `"saxphone, jazz"` — instrument + genre only

### Tips
- Be specific rather than generic ("bright plucked synth" > "synth")
- Describe the sound you want, not just genre labels
- Avoid conflicting descriptions ("calm aggressive" confuses the model)
- One style per generation works best
- **Text position correlates with temporal instrument appearance** — instruments mentioned earlier in the caption tend to appear earlier in the generated audio
- Include BPM for tempo-critical genres: `"phonk, 130 bpm"`, `"lo-fi hip hop, 60 bpm"`
- Include key for harmonic control: `"B Flat Major, allegro"`, `"G# min keyscale"`

## Lyrics

### Structure Tags
Use section markers for song structure. **Only standard tags are recognized** — the model will try to SING any unrecognized text in brackets.

**Recognized section tags:**
- `[Verse 1]`, `[Verse 2]`, etc.
- `[Chorus]`
- `[Pre-Chorus]`
- `[Bridge]`
- `[Intro]`, `[Outro]`
- `[Instrumental]`, `[Instrumental Break]`
- `[Guitar Solo]`, `[Synth Solo]`, etc.
- Descriptive: `[Music fades out]`, `[Beat fades out]`, `[Song ends abruptly]`

**WARNING:** Custom tags like `[Singer Tommy]`, `[UUID-xxx]`, `[choir]` will be **sung as lyrics**, not interpreted as metadata. The model has no tag filtering — everything goes directly to the tokenizer.

### Multi-Voice / Duet Format

Singer attribution goes **inside the section tag**, separated by ` - ` or `: `.

**Proven formats (from official examples):**

```
[Verse 1 - Male]
lyrics here

[Verse 2 - Female]
lyrics here

[Chorus - Duet]
lyrics here

[Chorus - Both]
lyrics here
```

Alternative colon format:
```
[Verse 1: Female Vocal]
[Verse 2: Male Vocal]
[Chorus: Duet]
```

With vocal style modifiers:
```
[Bridge - Male Spoken Word, Filtered]
[Chorus - Duet, Powerful Belting]
[Final Verse - Layered Vocals]
```

**Background vocals / second voice** use parentheses within lyrics:
```
[Chorus - Duet]
Darling! (Yes?) I've got something to say!
We rise together (together)
Into the light (into the light)
```

**Caption for multi-voice songs** should describe the vocal characters:
```
A romantic duet ballad with confident male baritone leading verse 1
and bright sassy female vocalist taking verse 2. Both singers unite
in powerful call-and-response choruses with full harmonies.
```

**What does NOT work:**
- `[male singer Tommy]` as a separate line → gets sung
- `[UUID-xxx]` markers → gets sung
- `[Singer One, female, named Elisabeth]` → gets sung
- Any custom tag format the model wasn't trained on

### Vocal Style Tags (inside section markers)
- `[Verse - raspy vocal]`
- `[Bridge - whispered]`
- `[Chorus - falsetto]`
- `[Chorus - powerful belting]`
- `[Bridge - spoken word]`
- `[Chorus - harmonies]`
- `[Verse - call and response]`
- `[Bridge - ad-lib]`

### Instrumental Mode

Three ways to generate instrumental (no vocals):

1. **`[inst]`** as the entire lyrics text — shorthand, recommended
2. **`[instrumental]`** as the entire lyrics text — long form, same effect
3. **`--instrumental`** CLI flag — forces instrumental regardless of lyrics

```bash
# Using [inst] shorthand
python generate.py audio ace-step -l "[inst]" -t "lo-fi hip hop with jazzy piano" -s 60

# Using --instrumental flag
python generate.py audio ace-step -l "[inst]" -t "cinematic orchestral" --instrumental -s 120
```

Both `[inst]` and `[instrumental]` are case-insensitive and whitespace-tolerant.

**Instrumental caption examples** (from official demos):
- `"Nightclubs, dance parties, workout playlists, radio broadcasts"`
- `"Minimal Techno"`
- `"phonk, russian dark accordion, 130 bpm, russian psaltery, russian harmonica, psychedelic, dark"`
- `"saxphone, jazz"`
- `"sonata, piano, Violin, B Flat Major, allegro"`
- `"tango finlandés, guitarra clásica"`
- `"Psychedelic trance"`
- `"volin, solo, fast tempo"`

### Language Support
19 languages officially supported. Use `--language` to set the vocal language explicitly (ISO 639-1 code). Without it, the model tries to auto-detect from lyrics — but auto-detection often fails, especially for German and other non-English languages.

```bash
# Explicit language for best results
python generate.py audio ace-step -l "..." -t "..." --language de -s 120
python generate.py audio ace-step -l "..." -t "..." --language ja -s 60
```

**Best performance (top 10):** English (`en`), Chinese (`zh`), Russian (`ru`), Spanish (`es`), Japanese (`ja`), German (`de`), French (`fr`), Portuguese (`pt`), Italian (`it`), Korean (`ko`).

Also supported: Bengali, Hindi, Arabic, Thai, Vietnamese, Indonesian, Turkish, Dutch, Polish, and more. Less common languages may underperform due to training data imbalance.

**Recommendation:** Always set `--language` for non-English songs. The auto-detection (`unknown`) uses Chain-of-Thought inference which is unreliable and can make vocals sound foreign.

### Tips
- 6-10 syllables per line for best coherence
- Consistent structure tags help the model understand song form
- Parenthetical backing vocals work: `(Oh my)`, `(Yeah)`
- Max 4096 characters
- For long captions, use a file and pass via `$(cat caption.txt)`

## Text2Samples (LoRA)

Fine-tuned variant for generating **instrument loops, sound effects, and musical elements** without vocals. Uses a LoRA trained on pure instrumental and sample data.

**Caption format:** instrument, BPM, key, loop type — very structured.

**Examples:**
- `"Acoustic Guitar, 191.0 bpm"`
- `"110.0 bpm, electric, bass, loops, G# min keyscale"`
- `"loops, fills, acoustic, 90.0 bpm, lo-fi hip hop, drums"`
- `"160.0 bpm, grooves, drums"`
- `"140.0 bpm, Electronic Drum Kit"`
- `"A# maj keyscale, electric, guitar, loops, 103.0 bpm"`
- `"melody, erhu"`
- `"G# keyscale, 80.0 bpm, brass & woodwinds, flute, algoza"`
- `"guitar, 115.0 bpm, B min keyscale, loops, melody"`
- `"clean, metallic, hand pan, organic, 120.0 bpm, A min keyscale, loops"`
- `"fx, music, koto, melody, G min keyscale, strings, 90.0 bpm, loops"`
- `"Lead Electric Guitar, loops"`
- `"layered, A# min keyscale, lo-fi hip hop, loops, chords, synth, pads, lo-fi, 60.0 bpm"`
- `"F keyscale, loops, saxphone, 125.0 bpm"`

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--language` | auto | ISO 639-1 | Vocal language (`en`, `de`, `zh`, `ja`, …) |
| `--cfg-scale` | 7.0 | 1.0-15.0 | Text adherence (higher = more faithful to caption) |
| `--temperature` | 0.85 | 0.0-2.0 | LM creativity (higher = more varied) |
| `--steps` | 8 | 1-20 (turbo) | Quality vs. speed (more = better, slower) |
| `--shift` | 3.0 | 1.0-5.0 | Timestep modification (3.0 for turbo) |
| `--seed` | random | any int | Reproducibility |

### Seed Sensitivity

ACE-Step is **highly sensitive to random seeds**. The same prompt + lyrics with different seeds can produce very different results (described as "gacha-style"). If you get a bad result, try different seeds before changing the prompt.

```bash
# Try multiple seeds to find a good generation
python generate.py audio ace-step -l "[inst]" -t "jazz piano trio" --seed 42 -s 30
python generate.py audio ace-step -l "[inst]" -t "jazz piano trio" --seed 123 -s 30
python generate.py audio ace-step -l "[inst]" -t "jazz piano trio" --seed 7777 -s 30
```

## Known Limitations

- **Seed sensitivity:** Results vary significantly with different seeds ("gacha-style")
- **Style-specific weaknesses:** Some genres underperform (e.g. Chinese rap)
- **Continuity artifacts:** Unnatural transitions in repainting/extend operations
- **Vocal quality:** Coarse vocal synthesis, lacking nuance in some styles
- **Control granularity:** Fine-grained musical parameter control is limited

## Comparison with HeartMuLa

| Aspect | HeartMuLa | ACE-Step |
|--------|-----------|----------|
| Caption | Comma-separated tags | Natural language or tags or scenarios |
| Max duration | ~5 min | 10 min (600s) |
| Speed | Moderate | Very fast (< 2s on A100) |
| Apple Silicon | MPS (partial) | MLX (native) |
| Thinking | No | Yes (LM Chain-of-Thought) |
| Extra tasks | Text-to-music only | Cover, repaint, extract, lego |
| Multi-voice | Not supported | Via section tags + caption |
| Instrumental | Not supported | `[inst]` or `--instrumental` |
| Text2Samples | No | Yes (LoRA) — loops, SFX, instruments |
