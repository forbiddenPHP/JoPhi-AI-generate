"""Create minimal deepspeed stubs so resemble-enhance can import on macOS.

deepspeed requires CUDA and doesn't compile on Apple Silicon.
Only the training code uses it — inference works fine with these stubs.
"""

import site
import pathlib

sp = site.getsitepackages()[0]
ds = pathlib.Path(sp) / "deepspeed"
ds.mkdir(exist_ok=True)

# deepspeed/__init__.py
(ds / "__init__.py").write_text(
    "class DeepSpeedConfig:\n"
    "    def __init__(self, *a, **kw): pass\n"
    "\n"
    "def initialize(*a, **kw): pass\n"
)

# deepspeed/accelerator/
acc = ds / "accelerator"
acc.mkdir(exist_ok=True)
(acc / "__init__.py").write_text(
    "class _Acc:\n"
    '    def device_name(self): return "mps"\n'
    '    def device(self, i=0): return "mps"\n'
    "\n"
    "def get_accelerator(): return _Acc()\n"
)

# deepspeed/runtime/
rt = ds / "runtime"
rt.mkdir(exist_ok=True)
(rt / "__init__.py").write_text("")
(rt / "engine.py").write_text("class DeepSpeedEngine: pass\n")
(rt / "utils.py").write_text("def clip_grad_norm_(*a, **kw): pass\n")

# pip-compatible metadata so pip sees deepspeed==0.12.4 as installed
dist = pathlib.Path(sp) / "deepspeed-0.12.4.dist-info"
dist.mkdir(exist_ok=True)
(dist / "METADATA").write_text(
    "Metadata-Version: 2.1\n"
    "Name: deepspeed\n"
    "Version: 0.12.4\n"
    "Summary: Stub for macOS (inference only)\n"
)
(dist / "INSTALLER").write_text("pip\n")
(dist / "RECORD").write_text("")
(dist / "top_level.txt").write_text("deepspeed\n")

print("  \033[0;32m✓\033[0m deepspeed stubs installed")

# ── Patch download.py to skip git clone when model_repo was restored ──────
dl = pathlib.Path(sp) / "resemble_enhance" / "enhancer" / "download.py"
if dl.exists():
    code = dl.read_text()
    marker = "restored from backup"
    if marker not in code:
        old = 'def download():\n    logger.info("Downloading the model...")\n\n    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():'
        new = (
            'def download():\n'
            '    logger.info("Downloading the model...")\n'
            '\n'
            '    run_dir = REPO_DIR / "enhancer_stage2"\n'
            '\n'
            '    # If model_repo was restored from backup (no .git), skip download\n'
            '    if REPO_DIR.exists() and not (REPO_DIR / ".git").exists() and run_dir.exists():\n'
            '        logger.info("Model repo already present (restored from backup), skipping download.")\n'
            '        return run_dir\n'
            '\n'
            '    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():'
        )
        # Also remove the duplicate run_dir assignment at the end
        code = code.replace(old, new)
        code = code.replace('\n    run_dir = REPO_DIR / "enhancer_stage2"\n\n    return run_dir', '\n    return run_dir')
        dl.write_text(code)
        print("  \033[0;32m✓\033[0m download.py patched (skip clone for restored models)")
    else:
        print("  \033[0;32m✓\033[0m download.py already patched")
