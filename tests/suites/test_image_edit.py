"""Test: Image Editing — FLUX.2 Klein prompt-based image editing.

Demonstrates FLUX.2 Klein's native editing capabilities.
No masks needed — the model understands edit instructions directly.
For iterative workflows: output of step 1 becomes input for step 2.
"""

import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
JOHANNES = ASSETS / "johannes.png"
LIVINGROOM = ASSETS / "livingroom.png"


def register(suite):
    out = suite.out_dir

    # ── Johannes edits ────────────────────────────────────────────────────

    suite.add(
        name="Edit: remove glasses",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(JOHANNES),
            "-p", "remove the glasses from the man's face",
            "-W", "512", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_no_glasses.png"),
        ],
        output=out / "edit_no_glasses.png",
    )

    suite.add(
        name="Edit: smooth skin",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(JOHANNES),
            "-p", "make the man's skin perfectly smooth and youthful, remove all wrinkles",
            "-W", "512", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_smooth_skin.png"),
        ],
        output=out / "edit_smooth_skin.png",
    )

    suite.add(
        name="Edit: remove person",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(JOHANNES),
            "-p", "remove the person from the photo, show only the pool and hotel background",
            "-W", "512", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_no_person.png"),
        ],
        output=out / "edit_no_person.png",
    )

    for age in [10, 20, 30, 40, 50, 60, 70, 80]:
        suite.add(
            name=f"Edit: turn to age {age}",
            cmd=[
                sys.executable, "generate.py", "image",
                "flux.2", "--model", "4b-distilled",
                "--images", str(JOHANNES),
                "-p", f"turn this 47 year old man to the age of {age} years",
                "-W", "512", "-H", "512", "--seed", "42",
                "-o", str(out / f"edit_age{age}.png"),
            ],
            output=out / f"edit_age{age}.png",
        )

    # ── Livingroom edits ──────────────────────────────────────────────────

    suite.add(
        name="Edit: add painting on wall",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(LIVINGROOM),
            "-p", "add a framed oil painting of a sunset over the ocean on the wall above the sofa",
            "-W", "768", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_wall_painting.png"),
        ],
        output=out / "edit_wall_painting.png",
    )

    suite.add(
        name="Edit: remove books from table",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(LIVINGROOM),
            "-p", "remove the books from the table",
            "-W", "768", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_no_books.png"),
        ],
        output=out / "edit_no_books.png",
    )

    suite.add(
        name="Edit: replace mug with flowers",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(LIVINGROOM),
            "-p", "replace the mug on the table with a small bouquet of colorful flowers",
            "-W", "768", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_flowers.png"),
        ],
        output=out / "edit_flowers.png",
    )

    suite.add(
        name="Edit: replace lamp with bookshelf",
        cmd=[
            sys.executable, "generate.py", "image",
            "flux.2", "--model", "4b-distilled",
            "--images", str(LIVINGROOM),
            "-p", "replace the lamp in the background with a slim bookshelf",
            "-W", "768", "-H", "512", "--seed", "42",
            "-o", str(out / "edit_bookshelf.png"),
        ],
        output=out / "edit_bookshelf.png",
    )
