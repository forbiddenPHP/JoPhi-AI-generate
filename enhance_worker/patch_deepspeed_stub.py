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

print("  \033[0;32m✓\033[0m deepspeed stubs installed")
