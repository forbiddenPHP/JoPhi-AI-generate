"""Test: AI-TTS — Qwen3-TTS neural speech (6 variants)."""

import sys

TALKSHOW_TEXT = """[Dylan] Willkommen zur Sendung. Heute reden wir über die Zukunft der Stadt. Ono Anna, was denkst du?
[Ono_Anna] Ich finde, wir brauchen mehr Grünflächen. Die Luft in der Innenstadt ist unerträglich.
[Eric] Das stimmt, aber wer soll das bezahlen? Die Stadt hat kein Geld.
[Sohee] Vielleicht sollten wir kreativ werden. In Seoul haben sie alte Bahngleise in Parks verwandelt.
[Uncle_Fu] Gute Idee. Aber hier fehlt der politische Wille dafür.
[Vivian] Ich glaube, das Problem ist ein anderes. Die Leute engagieren sich nicht mehr.
[Dylan] Interessant. Eric, was würdest du vorschlagen?
[Eric] Bürgerinitiativen. Wenn genug Druck da ist, bewegt sich auch die Politik.
[Ono_Anna] Genau. Und Social Media hilft dabei enorm.
[Sohee] Stimmt. Eine virale Kampagne kann mehr bewirken als jede Petition.
[Uncle_Fu] Solange es nicht nur beim Klicken bleibt. Man muss auch rausgehen.
[Vivian] Da bin ich ganz bei dir. Am Ende zählt, was auf der Straße passiert."""


def register(suite):
    out = suite.out_dir

    suite.add(
        name="AI-TTS Aiden EN basic",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "-v", "Aiden",
            "--text", "Hello world, this is a test of Qwen three TTS.",
            "-o", str(out / "aiden_en_basic.wav"),
        ],
        output=out / "aiden_en_basic.wav",
    )

    suite.add(
        name="AI-TTS Serena whispering",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "-v", "Serena",
            "-t", "whispering, slow",
            "--text", "And then — silence. The kind that swallows you whole.",
            "-o", str(out / "serena_whispering.wav"),
        ],
        output=out / "serena_whispering.wav",
    )

    suite.add(
        name="AI-TTS Dylan DE",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "-v", "Dylan", "--language", "de",
            "--text", "Die Sonne ging langsam unter und tauchte die Stadt in ein warmes, goldenes Licht.",
            "-o", str(out / "dylan_de.wav"),
        ],
        output=out / "dylan_de.wav",
    )

    suite.add(
        name="AI-TTS Uncle_Fu EN",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "-v", "Uncle_Fu", "--language", "en",
            "--text", "Listen carefully. What I am about to tell you will change everything you thought you knew.",
            "-o", str(out / "uncle_fu_en.wav"),
        ],
        output=out / "uncle_fu_en.wav",
    )

    suite.add(
        name="AI-TTS Dialog 2 voices",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "--text", "[Aiden: excited] Hey, have you heard the news? It's incredible! [Serena: calm, slow] Yes, I have. Let's discuss it calmly.",
            "-o", str(out / "dialog_2_voices.wav"),
        ],
        output=out / "dialog_2_voices.wav",
    )

    suite.add(
        name="AI-TTS Talkshow 6 voices DE",
        cmd=[
            sys.executable, "generate.py", "voice", "ai-tts",
            "--language", "de",
            "--text", TALKSHOW_TEXT,
            "-o", str(out / "talkshow_6_voices_de.wav"),
        ],
        output=out / "talkshow_6_voices_de.wav",
    )
