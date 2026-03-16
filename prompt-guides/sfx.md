# EzAudio (SFX) - Prompt Guide

EzAudio generates sound effects from text descriptions. It uses a diffusion model (MaskDiT) with a T5 text encoder (`google/flan-t5-xl`) trained on AudioCaps-style captions. Output is mono 24kHz WAV, max 10 seconds.

## Prompt Style

EzAudio expects **AudioCaps-style captions** — plain English descriptions of what you *hear*, not what you see. Present tense, concrete action verbs, no abstract concepts.

### Core Pattern

**[Subject] [action verb] + optional [context/environment]**

```
a dog barking in the distance
light guitar music is playing
footsteps crunch on the forest floor as crickets chirp
```

### Layering Sounds

Use **"as"** or **"while"** for simultaneous sounds:

```
a duck quacks as waves crash gently on the shore
water lightly splashing as a bird chirps and wind blows into a microphone
a man talking as water splashes and gurgles and a motor engine hums in the background
```

### Sequencing Events

Use **"then"** or **"followed by"** for temporal order:

```
a truck engine running followed by a truck horn honking
a man yells, slams a door and then speaks
a vehicle engine revving then accelerating at a high rate
```

### Spatial & Distance Cues

Add where the sound comes from:

```
a horse clip-clops in a windy rain as thunder cracks in the distance
kids playing and laughing nearby
a motor engine hums in the background
```

### Material & Surface Descriptions

Specify what surfaces or materials are involved:

```
footsteps crunch on the forest floor
wood stirring in a pot followed by a wooden object falling
a metal surface is whipped followed by tires skidding
```

### Intensity Modifiers

Control how strong/soft a sound is:

```
waves crash gently on the shore
water lightly splashing
a vehicle engine accelerating at a high rate
```

## Good Prompt Examples

| Prompt | Category |
|--------|----------|
| `a dog barking in the distance` | Animals |
| `a horse clip-clops in a windy rain as thunder cracks in the distance` | Animals + Weather |
| `a duck quacks as waves crash gently on the shore` | Animals + Nature |
| `footsteps crunch on the forest floor as crickets chirp` | Footsteps + Nature |
| `light guitar music is playing` | Music |
| `a piano playing as plastic bonks` | Music + Objects |
| `a truck engine running followed by a truck horn honking` | Vehicles |
| `a vehicle engine revving then accelerating at a high rate` | Vehicles |
| `food sizzling as a woman is talking` | Kitchen + Human |
| `multiple clanging and clanking sounds` | Mechanical |
| `water lightly splashing as a bird chirps` | Nature |
| `a man yells, slams a door and then speaks` | Human activity |
| `kids playing and laughing nearby` | Human activity |

## What Works Well

- **Animals:** barking, quacking, clip-clops, chirping, howling
- **Nature/Weather:** rain, thunder, wind, waves, crickets, bird songs
- **Vehicles:** engines, horns, tires skidding, acceleration
- **Human Activity:** footsteps, talking, yelling, laughing, door slamming
- **Music Textures:** guitar playing, piano playing (as ambient texture, not composed melodies)
- **Kitchen/Domestic:** sizzling, stirring, clanging, water pouring
- **Water:** splashing, gurgling, waves crashing, pouring

## What Does NOT Work

- **Intelligible speech** — it generates "a person talking" as a sound texture, NOT specific words
- **Composed music** — it generates music-like textures, not melodies in a specific key or tempo
- **Abstract/emotional descriptions** — "scary sound" is worse than "door creaking slowly in an empty hallway"
- **Visual descriptions** — "a car on a highway" is worse than "engine humming with wind noise"
- **Very rare sounds** not represented in AudioCaps training data
- **Precise timing** — "barking at exactly 3 seconds" is not supported

## Prompt Writing Rules

1. **Describe what you HEAR, not what you SEE.** "engine revving" not "a car on a highway"
2. **Use present tense.** "a dog is barking" or "a dog barking"
3. **Layer with "as" / "while".** `footsteps crunch on the forest floor as crickets chirp`
4. **Sequence with "then" / "followed by".** `engine running followed by a horn honking`
5. **Add spatial cues.** "in the distance", "nearby", "in the background"
6. **Add material info.** "on wooden floor", "on metal surface"
7. **Keep it under ~25 words.** T5 encoder max is 100 tokens, but training captions average 8–15 words
8. **Use concrete action verbs.** clip-clops, crunches, splashes, sizzles, clangs, buzzes, hums
9. **Avoid abstract concepts.** "peaceful" → "birds chirping with gentle water flowing"

## Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--seconds` | 10 | 1–10 | Duration in seconds (max 10) |
| `--cfg-scale` | 5.0 | 1.0–10.0 | Text adherence (higher = closer to prompt, risk of artifacts) |
| `--steps` | 100 | 25–200 | Quality vs. speed (more = better, slower) |
| `--seed` | random | any int | Reproducibility (same seed + prompt = same output) |
| `--model` | s3_xl | s3_xl, s3_l | Model size (xl = better quality) |

### Parameter Tips

- **Default (100 steps, CFG 5.0)** is the paper's benchmark setting — best quality
- **Draft/preview:** 50 steps, CFG 3.0 — faster, still decent
- **High CFG (>7)** can cause over-saturation artifacts. The model uses guidance rescale (0.75) internally to mitigate this, but extreme values still degrade quality
- **Steps beyond 100** have diminishing returns
- **25 steps** is the minimum for useful output

## Usage Examples

```bash
# Simple sound effect
python generate.py audio --engine sfx --text "a dog barking in the distance" -o dog.wav

# Layered nature scene, 8 seconds
python generate.py audio --engine sfx --text "rain falling on leaves as thunder rumbles in the distance" --seconds 8 -o rain.wav

# Fast draft with lower quality
python generate.py audio --engine sfx --text "a car horn honking" --steps 50 --cfg-scale 3.0 -o horn.wav

# Reproducible output
python generate.py audio --engine sfx --text "waves crashing on a rocky shore" --seed 42 -o waves.wav
```
