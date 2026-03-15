# AI-TTS (Qwen3-TTS) Prompt Guide

## Bracket Syntax

Control speech via `[...]` tags in the text. Fields are split on `:` `/` `-` separators and classified automatically:

- Known voice name → Voice
- Known language → Language
- Everything else → Instruct (speaking style)

Order doesn't matter. Case-insensitive.

Separators: `:` `/` `-` (with any amount of whitespace).

```
[Dylan - excited - english] Hello!
[english - excited - Dylan] Hello!     # identical
[Dylan] Hallo.                          # voice only
[english] Hello!                        # language switch, voice inherited from previous segment
[excited] Wow!                          # instruct only, voice inherited from previous segment
```

### Inheritance

- **Voice:** inherited from previous segment
- **Instruct:** NOT inherited (belongs to the moment)
- **Language:** falls back to `--language` flag, then autodetect


## Voices

| Name      | Description                                            |
|-----------|--------------------------------------------------------|
| Aiden     | Sunny American male, clear midrange                    |
| Dylan     | Youthful Beijing male, clear natural timbre            |
| Eric      | Lively Chengdu male, slightly husky brightness         |
| Ryan      | Dynamic male, strong rhythmic drive                    |
| Uncle_Fu  | Seasoned male, low mellow timbre                       |
| Vivian    | Bright, slightly edgy young female                     |
| Serena    | Warm, gentle young female                              |
| Ono_Anna  | Playful Japanese female, light nimble timbre            |
| Sohee     | Warm Korean female, rich emotion                       |


## Languages

| Language   | Model Name  | ISO |
|------------|-------------|-----|
| German     | german      | de  |
| English    | english     | en  |
| French     | french      | fr  |
| Japanese   | japanese    | ja  |
| Korean     | korean      | ko  |
| Chinese    | chinese     | zh  |
| Russian    | russian     | ru  |
| Portuguese | portuguese  | pt  |
| Spanish    | spanish     | es  |
| Italian    | italian     | it  |

Both work: `[german]` or `[de]`.


## Instruct — Speaking Style

The instruct field accepts natural language (English works best).
Multiple terms can be combined with commas.

### Emotion

```
calm, gentle, soothing
excited, enthusiastic, energetic
angry, furious, aggressive
sad, melancholic, somber
happy, cheerful, joyful
nervous, anxious, hesitant
confident, assertive, firm
```

### Delivery

```
whispered, soft, quiet
loud, shouting, yelling
fast, fast-paced, rapid
slow, slow-paced, deliberate
monotone, flat
dramatic, theatrical
sarcastic, ironic, dry
```

### Combining Attributes

Combine multiple attributes for better results:

```
[Dylan - calm, slow, deep voice] ...
[Vivian - excited, fast, high pitch] ...
[Uncle_Fu - angry, loud, commanding] ...
[Serena - whispered, gentle, intimate] ...
```

### Tips

- **Be specific:** "excited, fast-paced" instead of "nice"
- **Multiple dimensions:** combine emotion + tempo + volume
- **English preferred:** instruct tags work most reliably in English
- **Less is more:** 2-3 precise attributes > 10 vague ones
- **1.7B > 0.6B:** the larger model responds better to instructs


## Examples

### Simple Dialog
```
[Dylan] Willkommen zur Sendung.
[Vivian] Danke für die Einladung!
```

### Dialog with Mood
```
[Dylan - enthusiastic, warm] Willkommen zur Sendung!
[Vivian - cheerful, bright] Danke für die Einladung!
[Eric - skeptical, dry] Mal sehen, ob es sich lohnt.
```

### Multilingual Dialog
```
[Dylan - calm - german] Und Peter sagte:
[english - energetic] "Hey there, how are you?"
[Uncle_Fu - excited - german] und wir freuten uns so sehr, dass wir
[english] "Oh my god, what the quack!"
[german] riefen und in großes Gelächter ausbrachen.
```
