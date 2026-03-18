"""Test: OpenPose — DWPose pose estimation + pose transfer with FLUX.2."""

import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
REF_IMAGE = ASSETS_DIR / "johannes.png"


def register(suite):
    out = suite.out_dir

    # ── Pose extraction ──────────────────────────────────────────────────────

    suite.add(
        name="OpenPose: wholebody (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "openpose",
            "--images", str(REF_IMAGE),
            "-o", str(out / "pose_wholebody.png"),
        ],
        output=out / "pose_wholebody.png",
    )

    suite.add(
        name="OpenPose: body only (johannes)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "openpose",
            "--images", str(REF_IMAGE),
            "--pose-mode", "body",
            "-o", str(out / "pose_body.png"),
        ],
        output=out / "pose_body.png",
    )

    # ── Generate base portraits (no reference) ───────────────────────────────

    suite.add(
        name="OpenPose: generate woman portrait",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "portrait photo of a young woman with dark hair, looking directly at the camera, neutral background, studio lighting",
            "-W", "504", "-H", "504",
            "--seed", "42",
            "-o", str(out / "portrait_woman.png"),
        ],
        output=out / "portrait_woman.png",
    )

    suite.add(
        name="OpenPose: generate man portrait",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "portrait photo of a middle-aged man with a beard, looking directly at the camera, neutral background, studio lighting",
            "-W", "504", "-H", "504",
            "--seed", "73",
            "-o", str(out / "portrait_man.png"),
        ],
        output=out / "portrait_man.png",
    )

    suite.add(
        name="OpenPose: generate monster portrait",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "portrait of a friendly green monster with big eyes and horns, looking directly at the camera, dark moody background",
            "-W", "504", "-H", "504",
            "--seed", "99",
            "-o", str(out / "portrait_monster.png"),
        ],
        output=out / "portrait_monster.png",
    )

    # ── Pose transfer: wholebody skeleton ────────────────────────────────────

    suite.add(
        name="OpenPose: pose transfer woman (wholebody)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_wholebody.png"), str(out / "portrait_woman.png"),
            "-W", "504", "-H", "504",
            "--seed", "42",
            "-o", str(out / "pose_transfer_woman_wholebody.png"),
        ],
        output=out / "pose_transfer_woman_wholebody.png",
    )

    suite.add(
        name="OpenPose: pose transfer man (wholebody)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_wholebody.png"), str(out / "portrait_man.png"),
            "-W", "504", "-H", "504",
            "--seed", "73",
            "-o", str(out / "pose_transfer_man_wholebody.png"),
        ],
        output=out / "pose_transfer_man_wholebody.png",
    )

    suite.add(
        name="OpenPose: pose transfer monster (wholebody)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_wholebody.png"), str(out / "portrait_monster.png"),
            "-W", "504", "-H", "504",
            "--seed", "99",
            "-o", str(out / "pose_transfer_monster_wholebody.png"),
        ],
        output=out / "pose_transfer_monster_wholebody.png",
    )

    # ── Pose transfer: body-only skeleton ─────────────────────────────────────

    suite.add(
        name="OpenPose: pose transfer woman (body)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_body.png"), str(out / "portrait_woman.png"),
            "-W", "504", "-H", "504",
            "--seed", "42",
            "-o", str(out / "pose_transfer_woman_body.png"),
        ],
        output=out / "pose_transfer_woman_body.png",
    )

    suite.add(
        name="OpenPose: pose transfer man (body)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_body.png"), str(out / "portrait_man.png"),
            "-W", "504", "-H", "504",
            "--seed", "73",
            "-o", str(out / "pose_transfer_man_body.png"),
        ],
        output=out / "pose_transfer_man_body.png",
    )

    suite.add(
        name="OpenPose: pose transfer monster (body)",
        cmd=[
            sys.executable, "generate.py", "image",
            "--engine", "flux.2",
            "--model", "4b-distilled",
            "-p", "change the pose and camera angle of the subject in image 2 to the pose and camera angle in image 1",
            "--images", str(out / "pose_body.png"), str(out / "portrait_monster.png"),
            "-W", "504", "-H", "504",
            "--seed", "99",
            "-o", str(out / "pose_transfer_monster_body.png"),
        ],
        output=out / "pose_transfer_monster_body.png",
    )
