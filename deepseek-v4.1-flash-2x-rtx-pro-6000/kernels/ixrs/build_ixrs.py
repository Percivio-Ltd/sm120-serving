#!/usr/bin/env python3
"""Build ixrs.so (ixrs.cu for sm_120a) with the flags of the recipe's rowtopk build (recipe/kernels/build_rowtopk.py) plus
-Xptxas -v, which only prints register and shared-memory use. Run it inside the serving image (same torch and CUDA), with
or without a GPU:

    IXTK_BUILD_DIR=/some/empty/dir python3 build_ixrs.py

Output: $IXTK_BUILD_DIR/ixrs.so (library and C++ namespace ixrs, loaded beside the recipe's rtx8 op). Build options for
tests and benchmarks only: IXTK_MUTANT=1 builds the negative control ixrsm.so (keeps the last threshold ties instead of the
first, so a byte comparison must catch it); IXRS_FORCE_C=8|4|2 forces a cluster size (ixrs<C>.so)."""
import glob, os, sys, time
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")   # no GPU needed: never query the device
from torch.utils.cpp_extension import load
K = os.path.dirname(os.path.abspath(__file__))
bd = os.environ.get("IXTK_BUILD_DIR") or sys.exit("IXTK_BUILD_DIR is required (a fresh, empty directory)")
mut = os.environ.get("IXTK_MUTANT", "0") == "1"
fc = os.environ.get("IXRS_FORCE_C", "")
if fc not in ("", "8", "4", "2") or (mut and fc):
    sys.exit("IXRS_FORCE_C must be 8, 4 or 2, and is not combined with IXTK_MUTANT=1")
FLAGS = ["-O3", "--use_fast_math", "-lineinfo", "-gencode=arch=compute_120a,code=sm_120a", "-Xptxas", "-v"]
if mut:
    name, extra = "ixrsm", ["-DIXTK_MUTANT_LAST_TIES", "-DIXTK_NS=ixrsm"]
elif fc:
    name, extra = f"ixrs{fc}", [f"-DIXRS_FORCE_C={fc}", f"-DIXTK_NS=ixrs{fc}"]
else:
    name, extra = "ixrs", []
os.makedirs(bd, exist_ok=True)
t = time.time()
print(f"building {name} (ptxas -v lines follow)", flush=True)
load(name=name, sources=[f"{K}/ixrs.cu"], build_directory=bd, is_python_module=False, verbose=True,
     extra_cuda_cflags=FLAGS + extra)
print(f"built {name} in {time.time() - t:.0f}s -> {glob.glob(os.path.join(bd, name + '*.so'))}", flush=True)
