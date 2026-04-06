# LTX-2 — Prompt Guide

Prompt engineering guide for LTX-2.3 video and audio generation.

---

## Basics

**Write a single flowing paragraph.** LTX-2 expects 4-8 descriptive sentences in present tense — not bullet points, not comma-separated tags.

```
Good: "A close-up of a weathered fisherman mending nets on a wooden dock.
      Golden hour light catches the frayed rope fibers. His calloused hands
      work steadily, pulling thread through mesh. The camera slowly dollies
      in. Waves lap gently against the pilings, seagulls cry in the distance."

Bad:  "fisherman, dock, golden hour, nets, close-up, cinematic"
```

**Present tense.** Always describe what is happening, not what happened or will happen.

**Match detail to duration.** Short clips (2-4s) need short prompts. Longer videos (8-20s) need substantially more detail — a 10-word prompt for a 10-second video leaves the model making arbitrary choices.

**No quality tags.** "masterpiece, best quality, 4k" do nothing. Describe the scene instead.

**Iterate freely.** LTX-2 is designed for fast experimentation. Refining prompts is the workflow.

---

## Prompt Structure (6 Elements)

Build your paragraph in this order:

### 1. Establish the Shot

Use cinematography terms matching your genre. Specify shot scale and style.

| Scale | Use |
|-------|-----|
| `extreme close-up` | Texture, detail, emotion |
| `close-up` | Face, object, intimate moment |
| `medium shot` | Upper body, conversation |
| `wide shot` | Full body in environment |
| `extreme wide shot` | Landscape, establishing |

### 2. Set the Scene

Lighting, color palette, textures, atmosphere. This has the biggest impact on output quality.

**Lighting:**

| Type | Prompt fragment |
|------|-----------------|
| Golden hour | `warm golden hour light spills across the scene` |
| Noir | `harsh overhead light cuts through cigarette smoke, deep shadows` |
| Overcast | `soft overcast daylight, muted colors, no harsh shadows` |
| Neon | `neon signs reflect off wet asphalt, pink and blue color cast` |
| Candlelight | `warm flickering candlelight, deep amber tones, soft shadows dance` |
| Natural sunlight | `bright natural sunlight, crisp shadows on the ground` |
| Dramatic shadows | `dramatic side lighting, half the face in deep shadow` |

**Color Palette:** vibrant, muted, monochromatic, high contrast, warm tones, cool blues

**Textures:** rough stone, smooth metal, worn fabric, glossy surfaces, weathered wood

**Atmosphere:** fog, rain, dust, smoke, particles, mist, snow

### 3. Describe the Action

Write action as a natural sequence flowing from beginning to end.

```
She reaches for the coffee cup, pauses, then lifts it slowly to her lips.
Steam rises and curls in the morning light.
```

Match detail to shot scale — close-ups need more precision than wide shots.

### 4. Define the Character(s)

Age, hair, clothing, distinguishing features. **Show emotion through physical cues** — don't write "sad", write "her shoulders drop, she looks away."

```
A woman in her 30s with dark curly hair pulled back, wearing a faded
denim jacket. She tilts her head slightly, a faint smile forming.
```

Never use abstract emotion labels without visual description.

### 5. Identify Camera Movement

Specify how and when the camera moves. Describe what appears after the movement completes.

| Movement | Prompt fragment |
|----------|-----------------|
| Static | `the camera holds steady` / `static frame` |
| Pan | `the camera pans slowly to the right` |
| Track | `the camera tracks alongside the subject` |
| Dolly in | `the camera pushes in gradually` |
| Pull back | `the camera pulls back revealing the full scene` |
| Handheld | `slight handheld shake, documentary feel` |
| Circle | `the camera circles around the subject` |
| Tilt | `the camera tilts upward` |
| Overhead | `overhead view looking straight down` |
| Over-the-shoulder | `over-the-shoulder framing` |
| Follow | `the camera follows from behind` |

### 6. Describe the Audio

LTX-2.3 generates synchronized audio. Describe ambient sounds, music, speech, or singing explicitly.

**Ambient settings:** coffeeshop noise, wind and rain, forest ambience with birds, traffic hum, waves crashing

**Dialogue:** Place spoken words in quotation marks. Specify language and accent if needed.

**Dialogue style:** energetic announcer, resonant voice with gravitas, distorted radio-style, robotic monotone, childlike curiosity

**Volume:** whisper, mutter, normal, shout, scream

```
Wind rustles through dry leaves. In the distance, a church bell tolls twice.
He whispers: "It's time." (pause, steady) "Let's go."
```

---

## Dialogue in Action

Dialogue must be **embedded in the action flow**, not separated. Describe what the character does while speaking, in which language, and how they deliver the line.

```
Good: "He picks up the phone, glances at the screen, and says in German:
      'Ja, ich bin unterwegs.' He grabs his jacket and heads for the door."

Bad:  "Dialogue: 'Ja, ich bin unterwegs.'"
```

**Pattern:**
```
...and [action] as [he/she] [says|shouts|whispers|mutters] in [language]:
"[dialogue]" [continuing action].
```

**Break long dialogue into segments** with acting directions between them:
```
"I remember after you kids came along..." He pauses and looks to the side,
then continues, "your mom..." His eyes widen momentarily. He finishes with
a cracking voice, "said something I never quite understood."
```

**Performance cues** in brackets guide delivery:
- `(quietly)` — tone
- `(with urgency)` — energy
- `(steady, composed)` — control
- `(cracking voice)` — emotion

**Examples:**
```
She leans into the microphone and announces in English: "Ladies and
gentlemen, welcome aboard." She straightens up, adjusting her uniform.

He slams the table and shouts in Japanese: "Mou takusan da!" then storms
out, slamming the door behind him.

The old man chuckles softly, stirring his tea, and murmurs in French:
"C'est la vie." He takes a slow sip, gazing out the window.
```

The language tag determines which language the model speaks in the generated audio:
- `says in English: "Hello world"` → English audio
- `says in French: "Bonjour le monde"` → French audio
- `says in German: "Hallo Welt"` → German audio

Write the dialogue text in the target language. The model generates matching speech audio.

- Characters can talk and sing in multiple languages
- Describe delivery style: whisper, shout, mutter, sing, deadpan

---

## Audio-Guided Video (Audio-to-Video)

When generating video from an existing audio track, the audio anchors the temporal structure. Your prompt describes the **visual interpretation** of that audio.

- Describe what scenes, subjects, and camera work should accompany the soundtrack
- Match visual energy to audio energy — quiet audio = calm visuals, crescendo = dramatic reveal
- The prompt maps audio rhythm to motion: beats to cuts, melody to camera movement
- Describe how the visuals respond to the audio, not the audio itself

```
# Audio input: orchestral piece building to crescendo
"A vast empty cathedral, soft morning light filtering through stained glass.
The camera glides slowly down the center aisle. Dust particles float in
shafts of colored light. As the music swells, the camera accelerates,
tilting upward toward the vaulted ceiling. Light intensifies, flooding
the frame in gold."
```

---

## Image Conditioning (Reference Images)

When using `--image` to condition specific frames, your prompt must describe the action **in the exact order of the reference images**. The model interpolates between them — your text is the roadmap.

```
# Two reference images: frame 0 = woman at desk, frame 60 = woman at window
--image desk.png 0 1.0 --image window.png 60 1.0

"A woman sits at a cluttered desk reviewing documents under warm lamp light.
She pauses, pushes back her chair, and stands. She walks across the room
toward the window. She reaches the window and looks out at the city below,
resting one hand on the glass. Traffic hums faintly from the street."
```

**Rules for image-conditioned prompts:**
- Describe what happens **between** the reference frames, in sequence
- The prompt is the bridge — without it, transitions feel random
- More reference images = more structured prompt, matching each transition
- `--image-first` / `--image-last` shortcuts still need action described from start to end

### Character Order Matches Image Layout

When a reference image contains multiple characters, **describe them left to right** as they appear in the image. The model maps text to visual regions sequentially — if the prompt order doesn't match the spatial order, characters get swapped or confused.

```
# Keyframe: Johannes (left), Claude (right) sitting at a desk
Good: "Johannes raises his hand for a high five, laughing.
      Claude claps into Johannes' hand and laughs."

Bad:  "Claude and Johannes high-five each other."
      → Model can't resolve who is who, or swaps them
```

**Rules:**
- One action per character, described separately — no collective terms ("both", "they", "the two")
- Spatial order in the image = mention order in the prompt
- Each character's action is its own sentence or clause
- This applies to all image-conditioned generation (`--image`, `--image-first`, `--image-last`)

### Image-to-Video vs. Text-to-Video

**Image-to-Video:** Focus on motion and action — the visual starting point is already defined. Describe what happens next: how the subject moves, how the camera follows, what sounds emerge. Avoid re-describing static elements already visible in the image.

**Text-to-Video:** More exploratory. The prompt defines everything — subject, environment, motion, audio. Best for building new moments from scratch and testing composition.

---

## IC-LoRA Control (Video-to-Video)

IC-LoRAs condition video generation on a **reference video** — a control signal that is positionally aligned to the output. The reference video can be at half resolution (downscale factor 2) for efficiency. None, one, or both IC-LoRAs can be active simultaneously.

### Union Control (Canny + Depth + Pose)

Uses structural control signals extracted from a reference video: edge maps (Canny), depth maps, or pose skeletons. The model follows the spatial structure while generating new content.

```
--lora models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0
--video-conditioning reference_canny.mp4 1.0
```

**Prompt approach:** Describe the scene, mood, and style — not the structure. The reference video already defines spatial layout, poses, and edges. Your prompt fills in appearance, lighting, texture, and audio.

```
"A woman in a red dress walks through a sunlit garden. Warm golden light,
shallow depth of field. Birds chirp softly, leaves rustle in a gentle breeze."
```

The Canny/Depth/Pose reference handles *where* things are. The prompt handles *what they look like*.

### Motion Track Control (Spline Trajectories)

Guides object or region motion using colored spline overlays drawn on a reference video. Trajectories can be extracted from existing videos (e.g., SpatialTrackerV2) or drawn manually.

```
--lora models/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors 1.0
--video-conditioning reference_splines.mp4 1.0
```

**Prompt approach:** Describe the scene and subjects, but let the splines handle motion direction and timing. Avoid contradicting the trajectories — don't say "walks left" if the spline goes right.

```
"A red ball rolls across a wooden floor in a sunlit room. The camera holds
steady. Soft ambient light, warm wood tones."
```

### Combining Both

Both LoRAs can be active simultaneously for structural + motion control:

```
--lora models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0
--lora models/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors 1.0
--video-conditioning reference.mp4 1.0
```

### Reference Input

The `--video-conditioning` parameter expects a **video file** as control signal. Additionally, `--image` can be used for single-frame references at specific positions — both can be combined.

- **Video reference:** Full temporal control (edges, depth, poses, or splines across all frames)
- **Image reference:** Anchor specific frames (e.g., start/end keyframes)

### Optional Attention Mask

A grayscale mask video controls per-region conditioning strength (0.0 = ignore, 1.0 = full):

```
--conditioning-attention-mask mask.mp4 0.5
```

Useful for applying control only to specific areas while letting the model freely generate the rest.

---

## Lensing

Specifying lens details improves spatial coherence, especially at higher resolutions.

| Effect | Prompt fragment |
|--------|-----------------|
| Portrait | `85mm lens, shallow depth of field, soft bokeh` |
| Wide angle | `24mm wide-angle, deep focus, dramatic perspective` |
| Macro | `macro lens, extreme detail, razor-thin focal plane` |
| Anamorphic | `anamorphic lens, horizontal flares, cinematic aspect ratio` |
| Telephoto | `200mm telephoto, compressed background, subject isolation` |
| Film stock | `shot on 16mm film, visible grain, warm color grading` |
| Specific aperture | `50mm f/2.8, moderate depth of field` |

---

## Emphasis

No `(word:1.5)` weight syntax. Use natural language:

- "prominently featuring..."
- "the focal point is..."
- "with particular attention to the texture of..."

---

## Styles

LTX-2 handles stylized aesthetics well. Lead with the style, commit to one.

### Animation
- `Stop-motion animation style.`
- `2D hand-drawn animation, fluid lines.`
- `3D animated, Pixar-like rendering.`
- `Claymation, visible fingerprints in the clay.`

### Stylized
- `Comic book panel, bold outlines, halftone shading.`
- `Cyberpunk, rain-soaked streets, holographic signage.`
- `Pixel art, 8-bit aesthetic, bright saturated colors.`
- `Surreal, Dali-esque melting forms.`
- `Minimalist, clean geometric shapes, white space.`
- `Painterly oil painting style, visible brushstrokes.`

### Cinematic
- `Period drama, 1920s costumes, sepia undertones.`
- `Film noir, high contrast black and white.`
- `Epic space opera, vast starfields, lens flares.`
- `Thriller, tight framing, shallow depth of field.`
- `Documentary, handheld, natural lighting.`
- `Arthouse, long takes, muted palette.`
- `Experimental film, abstract shapes, unconventional cuts.`

**Don't mix styles.** "Photorealistic watercolor" confuses the model. Commit to one aesthetic.

---

## Negative Prompt

The official recommended negative prompt:

```
shaky, glitchy, low quality, worst quality, deformed, distorted,
disfigured, motion smear, motion artifacts, fused fingers,
bad anatomy, weird hand, ugly, transition, static.
```

---

## Long Videos (10-20 seconds)

Longer shots enable dramatic reveals, panoramic storytelling, emotional exchanges, and moments of stillness. Eight-second clips are great for rhythm — twenty seconds lets a scene breathe.

### Structure

Treat the prompt like a mini scene script:

1. **Scene header** — place and time
2. **Atmosphere** — tone and mood in one sentence
3. **Blocking** — how subjects move, in sequence with the camera
4. **Dialogue** — quotes for speech, bracketed cues for performance

**Order actions and make sure the movement fits the duration.** Too much detail = rushed motion.

### Realism and Consistency

- Start with a close-up, pull out gradually — grounds the scene, preserves facial detail
- Wider shots soften likeness; have characters turn away or maintain consistent distance
- Avoid abrupt reframing or rapid zooms — pursue smooth, natural motion
- Add **soft closing actions** (character reactions, camera drift) to carry motion to the end
- Let dialogue end before the video does — fill remaining time with transitional actions

### Multi-Character Scenes

Let the camera linger on one speaker before moving to the next. Avoid abrupt cuts — flow naturally through reactions and pauses.

```
WOMAN (quietly): "Funny how quiet it gets." She takes a breath, glances
toward the horizon. A small, knowing smile crosses her face. She nods
once, almost to herself, then walks on.

The camera follows her for a few steps, then slows as she moves away.
A man sits on a tractor nearby. He glances in her direction, a subtle
smile. MAN (quietly, to himself): "Still is." The camera holds as the
tractor rolls on.
```

### Blocking (Subject + Camera Choreography)

```
Extreme close-up: sunlight glinting in her silver hair. She turns slowly,
scanning the empty square. The camera pulls back as she takes a step
forward. She pauses at the fountain, rests one hand on the stone rim.
The camera continues pulling back, revealing the silent town around her.
```

---

## What Works Well

- Cinematic compositions with thoughtful lighting and shallow depth of field
- Emotive human moments with subtle gestures and facial nuance
- Atmospheric effects (fog, mist, rain, golden-hour glow, reflections)
- Precise camera language with explicit movement instructions
- Stylized aesthetics (painterly, noir, analog film, pixel art, comic book)
- Lighting and mood control (backlighting, color palettes, rim light, flickering lamps)
- Characters talking and singing in multiple languages
- Clear blocking for multi-character scenes

## What to Avoid

- **Internal emotions** — Don't write "sad." Show it: "her gaze drops, she turns away"
- **Text and logos** — Readable text generation is unreliable
- **Complex physics** — Fast twisting, chaotic motion, jumping. Dancing works better
- **Overloaded scenes** — Too many characters or layered actions reduce coherence
- **Conflicting lighting** — Moonlight + noon sun confuses the model (unless clearly motivated)
- **Over-constrained numbers** — "3 birds at 45 degrees" doesn't work. Use natural language
- **Contradictory directions** — "still peaceful lake with crashing waves" confuses the model
- **Vague prompts** — "a nice video of nature" gives the model unlimited bad options
- **Overcomplicated prompts** — Each added instruction increases the chance some won't appear. Start simple, layer up

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Too vague | "A nice video of nature" — arbitrary results | Be specific about subject, action, lighting |
| Over-constrained | "3 birds flying left at 45 degrees" — ignored | Use natural language descriptions |
| Mismatched duration | 10 words for 10 seconds — not enough direction | Match prompt length to video length |
| Contradictory | "still lake with dramatic waves" — confused output | Maintain internal consistency |
| Mixed styles | "photorealistic watercolor" — incoherent aesthetic | Commit to one style |
| Emotion labels | "she feels happy" — no visual anchor | Show through physical cues |

---

## Model Selection

| Model | Best for |
|-------|----------|
| `distilled` (default) | Fast generation, most use cases, audio+video |
| `dev` | Maximum quality, two-stage pipeline, more control |

**Workflow:** Use `distilled` for exploration, lock the seed when you find a good composition, optionally re-render with `dev`.

---

## Guidance Scale

| Model | Guidance |
|-------|----------|
| `dev` (non-distilled) | `4.0` |
| `distilled` | `1.0` |

---

## Terminology Reference

### Camera Language
follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame, dolly-in

### Film Characteristics
film grain, lens flares, pixelated edges, jittery stop-motion, anamorphic squeeze

### Scale Indicators
expansive, epic, intimate, claustrophobic

### Pacing & Temporal Effects
slow motion, time-lapse, rapid cuts, lingering shot, continuous shot, freeze-frame, fade-in, fade-out, seamless transition, dynamic movement, sudden stop

### Visual Effects
particle systems, motion blur, depth of field

---

## Example Prompts

### News Broadcast (from official guide)

```
The shot opens on a news reporter standing before cordoned-off cars with
yellow caution tape fluttering behind. Warm early sunlight reflects off
the camera lens. Faint chatter and distant drilling fill the air. The
composed but visibly excited reporter looks directly into the camera,
microphone in hand. He says: "Thank you, Sylvia. And yes -- this is a
sentence I never thought I'd say on live television -- but this morning,
here in the quiet town of New Castle, Vermont... black gold has been
found!" He gestures toward the field behind him. The camera pans right
slowly, revealing a construction site surrounded by hard-hatted workers.
With a sudden roar, a geyser of oil erupts from the ground, blasting
upward in a violent plume. Workers cheer and scramble as the black stream
glistens in morning light. The camera shakes slightly. Reporter shouts
off-screen: "There it is, folks -- the moment New Castle will never
forget!" The camera pulls back, revealing the entire scene silhouetted
against the wild fountain of oil.
```

### Animated / Stylized (from official guide)

```
The camera opens in a calm, sunlit frog yoga studio. Warm morning light
washes over the wooden floor as incense smoke drifts lazily. The senior
frog instructor sits cross-legged at center, eyes closed, voice deep and
calm: "We are one with the pond." All frogs answer softly: "Ommm..." "We
are one with the mud." "Ommm..." He smiles faintly. "We are one with the
flies." A pause. The camera pans to one frog whose eyes dart. Its tongue
snaps out, catching a fly mid-air. The master exhales slowly, still
serene. "But we do not chase the flies... not during class." The guilty
frog lowers its head in shame, folding its hands back into meditation
pose. Other frogs resume: "Ommm..." The camera holds on the embarrassed
frog, eyes closed too tightly.
```

### Cinematic Portrait

```
A close-up of a young woman sitting by a rain-streaked window in a dimly
lit cafe. Soft amber light from a nearby lamp illuminates one side of her
face, leaving the other in shadow. She traces a finger along the glass,
watching droplets merge and slide. The camera holds steady, then slowly
pushes in. Rain patters against the window, a muffled jazz piano plays
from inside the cafe.
```

### Action / Outdoor

```
A wide shot of a lone surfer paddling into a massive turquoise wave at
sunrise. The camera tracks from the shore, following the surfer as the
wave begins to curl overhead. Morning light refracts through the water,
casting prismatic patterns. The surfer pops up, carving a clean line
across the face of the wave. Ocean roar fills the audio, wind rushing
past.
```

### Stylized / Pixel Art

```
Pixel art style. A tiny knight walks through a dark forest, sword drawn,
lantern swinging from a belt loop. Fireflies blink in and out around
twisted trees. The camera scrolls slowly to the right, revealing a
glowing cave entrance ahead. Chiptune music plays softly, footsteps
crunch on leaves.
```

### Dialogue Scene

```
Medium shot of two old friends reuniting at a train station platform.
Warm afternoon light, steam drifting from a departing train. The taller
man in a wool coat opens his arms. He says: "I almost didn't recognize
you." (laughing) The shorter man grips his hand firmly: "It's been too
long." (pause, quietly) "Way too long." The camera slowly orbits around
them. Train whistle in the background, indistinct crowd murmur.
```

---

## Iteration Tips

1. Start with 2-3 sentences to nail the core concept
2. Lock the seed once you find a good composition (`--seed N`)
3. Layer detail one element at a time: lighting, then action, then audio
4. Keep it under 8 sentences for standard clips
5. For 20-second shots, write a full scene paragraph
6. Use `distilled` for all exploratory iterations
