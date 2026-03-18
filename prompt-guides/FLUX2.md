# FLUX.2 — Prompt Guide

Prompt engineering guide for FLUX.2 Klein image generation and editing.

---

## Basics

**Write prose, not tags.** FLUX.2 works best with natural language descriptions, not comma-separated keywords.

```
Good: "A weathered leather journal lies open on an oak desk, morning light
      revealing handwritten entries in faded ink"

Bad:  "journal, leather, oak desk, morning light, handwriting"
```

**Front-load the subject.** The model prioritizes what appears first. Don't bury your main subject at the end.

**No quality tags needed.** Unlike older models, FLUX.2 does not benefit from "masterpiece, best quality, 4k, ultra detailed." These are ignored.

**Describe what you want, not what to avoid.** Instead of "no blur" say "sharp focus throughout." Instead of "no people" say "empty scene."

---

## Prompt Structure

Follow this hierarchy: **Subject > Environment > Style > Technical Details**

```
[Subject description]. [Setting/environment]. [Lighting].
[Style/mood]. [Camera/composition details].
```

### Length

| Length | Words | Use case |
|--------|-------|----------|
| Short | 10-30 | Quick concepts, exploration |
| Medium | 30-80 | Most production work |
| Long | 80-150 | Complex editorial, detailed scenes |

Keep it under ~150 words. Every word should serve the image.

---

## Lighting (Most Important Element)

Lighting has the highest impact on output quality. Specify source, quality, direction, and temperature.

| Type | Prompt fragment |
|------|-----------------|
| Soft natural | `soft diffused light from a large window camera-left, gentle shadows` |
| Golden hour | `low golden hour sun backlighting the subject, warm rim light on hair` |
| Studio dramatic | `single spotlight from above, deep shadows, Rembrandt triangle lighting` |
| Overcast | `even overcast daylight, no harsh shadows, neutral color temperature` |
| Neon/urban | `neon signs reflecting in wet pavement, pink and blue color cast` |

---

## Camera & Lens

FLUX.2 understands real photography terminology:

| Effect | Prompt fragment |
|--------|-----------------|
| Portrait bokeh | `85mm lens at f/1.8, shallow depth of field, creamy bokeh` |
| Wide establishing | `24mm wide-angle, deep focus f/11, dramatic perspective` |
| Macro detail | `100mm macro lens, extreme close-up, razor-thin focal plane` |
| Telephoto | `200mm telephoto, compressed background, subject isolation` |
| Film stock | `shot on Kodak Portra 400, slight grain, warm skin tones` |

### Camera Angles

`eye level` | `low angle` (hero shot) | `high angle` (vulnerability) | `bird's-eye view` | `over-the-shoulder` | `dutch angle`

### Composition

- `rule of thirds, subject positioned on left vertical`
- `centered symmetrical composition`
- `leading lines drawing eye to the subject`
- `negative space on the right for text overlay`

---

## Emphasis (No Weight Syntax)

FLUX.2 Klein does not use `(word:1.5)` weight syntax. Use natural language emphasis instead:

- "prominently featuring..."
- "with particular attention to..."
- "especially detailed..."
- "the focal point is..."

---

## Styles

### Photorealistic

Specify camera gear and shooting conditions:

```
Professional headshot of a male architect in his 40s,
salt-and-pepper beard, black-rimmed glasses, charcoal blazer.
Modern office background softly blurred. Natural window light
from left creating gentle shadows. Shot on Canon 5D Mark IV,
85mm lens at f/2.0, shallow depth of field.
```

### Artistic / Illustration

Lead with the art style:

- `watercolor painting of...`
- `oil painting with thick impasto brushstrokes of...`
- `pencil sketch, cross-hatching technique...`
- `flat vector illustration, bold geometric shapes, limited palette of teal and coral`
- `anime-style, soft cel shading, vibrant color palette`

**Do not mix styles.** "Photorealistic portrait, watercolor style" confuses the model. Commit to one aesthetic.

### Style + Mood Tags

End prompts with explicit anchors for consistency:

```
[Scene description]. Style: cinematic noir photography.
Mood: tense, mysterious, atmospheric.
```

---

## Image Editing (Reference Images)

### Single Reference

Upload one image, describe the transformation:

| Action | Example prompt |
|--------|---------------|
| Change background | `the same person, now standing in a Japanese garden with cherry blossoms` |
| Add element | `add a black cat sitting on the windowsill` |
| Remove element | `the same scene without any people, just the empty bench` |
| Swap object | `replace the red car with a vintage blue pickup truck` |
| Change color | `change the jacket color to deep burgundy` |
| Style transfer | `transform this into Studio Ghibli animation style` |
| Change lighting | `same scene but at golden hour with warm sunset light` |

### Multi-Reference (up to 10 images)

Reference specific images by number:

- `the person from image 1 is petting the cat from image 2`
- `change the color of the gloves to the color shown in image 2`
- `place the product from image 1 on the table from image 2 with lighting from image 3`

**Color matching tip:** Include a solid color square as one of your reference images for exact color matching.

---

## Text in Images

FLUX.2 can render text. Always specify:

```
A white coffee mug with the text 'GOOD MORNING' in bold
sans-serif black letters, centered on the mug surface.
```

Include: exact text in quotes, font style, color, placement, and capitalization.

---

## Prompt Templates

### Product Photography

```
High-end product photography of [PRODUCT] on [SURFACE].
[LIGHTING] creates [REFLECTIONS/SHADOWS]. [BACKGROUND].
Commercial photography, shallow depth of field, sharp focus on [FOCAL_POINT].
```

### Environmental Portrait

```
Portrait of [PERSON] in [LOCATION], [ACTION].
[LIGHT_SOURCE] illuminates [DETAILS]. Documentary photography,
environmental portrait, [COLOR_PALETTE], shallow focus on [DETAIL].
```

### Architecture / Interior

```
[ROOM_TYPE] interior, [TIME_OF_DAY] through [WINDOW_TYPE].
[MATERIALS]. Architectural photography, wide-angle composition,
natural color grading emphasizing [QUALITY].
```

---

## Model Selection

| Model | Best for |
|-------|----------|
| `4b-distilled` | Fast exploration, rapid prototyping, real-time |
| `4b` | Fine-tuning, quality control, configurable steps |
| `9b-distilled` | Best quality, final renders |
| `9b` | Research, maximum flexibility |

**Workflow:** Use `4b-distilled` for exploration, lock the seed when you find a good composition, then render final with `9b-distilled`.

---

## Iteration Tips

1. Start with a short prompt (~30 words) to nail the concept
2. Lock the seed once you find a good composition (`--seed 42`)
3. Add one detail at a time: lighting, then style, then technical specs
4. Use `4b-distilled` for all exploratory iterations
5. Switch to `9b-distilled` for the final version

---

## Common Mistakes

- Burying the subject at the end of the prompt
- Mixing conflicting styles ("photorealistic watercolor")
- Omitting lighting description
- Using SD/SDXL quality tags ("masterpiece, best quality")
- Writing prompts over 150 words without purpose
- Vague style references ("make it look good")
- Missing composition guidance (where should elements be?)
