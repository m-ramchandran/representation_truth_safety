"""CPU fallback for rs_shared_cv.py.

Activation caches come from the configured Hugging Face dataset. GPU-only
libraries imported by the classifier package are stubbed because the
random-slopes path does not use them, and fit_torch is switched to the CPU.
The saved merged prediction files must already be in the local results tree.

    python paper_final_figures/scripts/local_rs_shared_cv.py gemma_tf_a:29 qwen_tf_a:24
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
os.environ["REPO_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import importlib.abc                                      # noqa: E402
import importlib.machinery                                # noqa: E402


class _GpuStubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Any cupy / cupyx / cuml module resolves to a MagicMock."""
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("cupy", "cupyx", "cuml"):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None
    def create_module(self, spec):
        m = MagicMock(name=spec.name)
        m.__path__ = []          # a package, so submodules resolve here too
        m.__spec__ = spec
        return m
    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _GpuStubs())

import torch                                              # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
import classifiers.data as data                           # noqa: E402
from classifiers.scripts import fit_torch as ft           # noqa: E402
import psc_rs_shared_cv as job                            # noqa: E402

ft.DEVICE = "cpu"
_load = job.load_datasets
# the fold assignment comes from the stored npz, so the prompt reads that
# --group-folds triggers are not needed here
job.load_datasets = lambda layer, raw, roles, name, groups=True: _load(
    layer, raw, roles, name, groups=False)


def main() -> None:
    for spec in sys.argv[1:]:
        base, layer = spec.split(":")
        t0 = time.time()
        print(f"=== {base} layer {layer} on {ft.DEVICE}, "
              f"{torch.get_num_threads()} threads", flush=True)
        sys.argv = ["psc_rs_shared_cv", base, layer]
        job.main()
        print(f"=== {base} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
