# ACE-Step 1.5 - Prompt Guide

ACE-Step uses two text inputs: a **caption** (style description) and **lyrics** (song text with structure tags).

## Caption (--tags)

The caption describes the desired musical style. Unlike HeartMuLa's tag-list format, ACE-Step accepts natural language descriptions.

**Max length:** 512 characters.

### What to include
- Genre and style (e.g. "upbeat electronic dance music")
- Instruments (e.g. "with heavy synth bass and arpeggiated leads")
- Mood and energy (e.g. "energetic and euphoric")
- Vocal style (e.g. "female vocal, breathy tone")
- Tempo hints (e.g. "fast-paced" or use `--bpm`)
- **Singer descriptions** for multi-voice songs (see Multi-Voice section below)

### Examples
- `"upbeat electronic dance music with progressive build and heavy synth bass"`
- `"acoustic folk ballad with fingerpicked guitar and soft female vocal"`
- `"aggressive trap beat with 808 bass, hi-hats, and dark atmosphere"`
- `"cinematic orchestral score with sweeping strings and brass"`
- `"lo-fi hip hop with jazzy piano chords and vinyl crackle"`

### Tips
- Be specific rather than generic ("bright plucked synth" > "synth")
- Describe the sound you want, not just genre labels
- Avoid conflicting descriptions ("calm aggressive" confuses the model)
- One style per generation works best
- **Text position correlates with temporal instrument appearance** — instruments mentioned earlier in the caption tend to appear earlier in the generated audio

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

### Special Values
- `[Instrumental]` as the entire lyrics → generates instrumental track
- Use `--instrumental` flag for guaranteed no-vocals output

### Language Support
50+ languages supported. The model auto-detects language from lyrics. Supported languages include: English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian, Bengali, Hindi, Arabic, Thai, Vietnamese, Indonesian, Turkish, Dutch, Polish, and many more.

### Tips
- 6-10 syllables per line for best coherence
- Consistent structure tags help the model understand song form
- Parenthetical backing vocals work: `(Oh my)`, `(Yeah)`
- Max 4096 characters
- For long captions, use a file and pass via `$(cat caption.txt)`

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--cfg-scale` | 7.0 | 1.0-15.0 | Text adherence (higher = more faithful to caption) |
| `--temperature` | 0.85 | 0.0-2.0 | LM creativity (higher = more varied) |
| `--steps` | 8 | 1-20 (turbo) | Quality vs. speed (more = better, slower) |
| `--shift` | 3.0 | 1.0-5.0 | Timestep modification (3.0 for turbo) |
| `--seed` | random | any int | Reproducibility |

## Comparison with HeartMuLa

| Aspect | HeartMuLa | ACE-Step |
|--------|-----------|----------|
| Caption | Comma-separated tags | Natural language description |
| Max duration | ~5 min | 10 min (600s) |
| Speed | Moderate | Very fast (< 2s on A100) |
| Apple Silicon | MPS (partial) | MLX (native) |
| Thinking | No | Yes (LM Chain-of-Thought) |
| Extra tasks | Text-to-music only | Cover, repaint, extract, lego |
| Multi-voice | Not supported | Via section tags + caption |
