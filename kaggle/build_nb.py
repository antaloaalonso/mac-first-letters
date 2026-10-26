"""Assemble vesuvius_band.ipynb from src/*.py (the notebook is self-contained: each module is
embedded with %%writefile). Run after editing anything in src/:  python build_nb.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ["umbilicus_est.py", "fetch.py", "render_gpu.py", "montage.py", "ink.py", "pipeline.py"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


INTRO = """
# Vesuvius First Letters: one z band → CUDA spiral fit → every winding → ink_9um

Self-contained and resumable. **Settings:** Accelerator *GPU T4 x2* (or RTX Pro 6000; not L4), **Internet ON**, Persistence off.
Edit the `RUN` cell only. Everything that must survive goes to `/kaggle/working/<tag>/` (montages, per-winding ink TIFs,
fit checkpoint + meshes, logs). To continue an unfinished run in a new session, attach the previous version's output as an
input and set `resume_from` to its `/kaggle/input/...` path.

Modes:
- `control`: PHerc1667 w018 (2.399 µm mesh, known text; it is in ink_9um's training labels, so this checks the stack, not generalisation), rendered at level 2 (9.6 µm).
  No fit. ~15–25 min. Letters must show in **one** direction only.
- `band`: stage official tracks (only the band's byte ranges) + crossings + lasagna rows + umbilicus → `fit_spiral.py`
  on CUDA → render every fitted winding from the 9 µm scan at level 0 → ink_9um seed42+43, both directions → void-masked montages.
"""

RUN = r'''
import time; T0 = time.time()
RUN = dict(
    mode="control",            # "control" | "band"
    scroll="PHerc1667",        # e.g. "PHerc0826"
    volume="20251217075048",   # volume id prefix (S3 <scroll>/volumes/<id>-...zarr)
    level=2,                   # pyramid level to render: 2 for 2.4 um scans (-> 9.6 um), 0 for ~9 um scans
    z0=None, z1=None,          # band in level-0 slices; None/"auto" = densest 1000-slice band of published tracks
    steps=3000,                # fit iterations (autosaves every 1000; resumable)
    sense="CW",                # spiral_outward_sense (catalog has no handedness keys; CW worked for PHerc0826/0846A)
    windings=[0, 9999],        # winding index range to render
    ink_mode=None,             # None = "full" for control, "screen" for band. screen: seed42 only, 25% overlap (~4.5x
                               # cheaper, both directions). full: seed42+43 ensemble, 50% overlap (confirm candidates)
    winding_list=None,         # e.g. ["w052","w053"]: only these windings (use with ink_mode="full" + resume_from)
    ink_batch=16,
    hecate=False,              # also run scrollprize/hecate 9.6 um (about 10x slower than ink_9um)
    max_hours=2.8,             # wall budget for this session, incl. setup
    reserve_hours_for_ink=1.0, # stop the fit early (resumable) to leave this much for render+ink
    tag="w018",                # output folder name under /kaggle/working
    resume_from=None,          # e.g. "/kaggle/input/vesuvius-band-pherc0826" (previous output attached)
    float32=True,              # FIT_SPIRAL_MAX_PRECISION_FLOAT=32 (T4 / RTX fp64 is slow)
    triton=True,               # falls back to the eager kernels automatically on a Triton/ptxas failure
    cache_gb=25,
)
'''

SETUP = r'''
import os, subprocess, sys, json, glob
W = "/kaggle/working"; TMP = "/tmp/vb"; os.makedirs(TMP, exist_ok=True)
VILLA = f"{TMP}/villa"; VFIT = f"{TMP}/venv_fit"; VINK = f"{TMP}/venv_ink"
VILLA_REF = "322cb0899ac9efe2588d5a3395dce6dc2dea3c89"   # commit the local Mac pipeline was validated on
def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode: raise RuntimeError(f"failed: {cmd}")
    return r
sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv")
sh("df -h /kaggle/working /tmp | cat; nproc; free -g | head -2")
sh("pip install -q uv")
if not os.path.exists(f"{VILLA}/.git"):
    sh(f"git init -q {VILLA} && cd {VILLA} && git remote add origin https://github.com/ScrollPrize/villa.git "
       f"&& git sparse-checkout set spiral-fitting vc_object_store vesuvius lasagna "
       f"&& (git fetch -q --depth 1 origin {VILLA_REF} && git checkout -q FETCH_HEAD "
       f"|| (git fetch -q --depth 1 origin main && git checkout -q FETCH_HEAD))")
sh(f"cd {VILLA} && git log -1 --format='%H %cd'")
# ink + render env. What ink_9um actually imports (measured on the validated local venv, model built):
# torch, torchvision, timm, einops, huggingface_hub, zarr>=3, numcodecs, pynrrd, imagecodecs, tifffile, numba,
# opencv, scipy, requests/pyyaml/aiohttp. Fast path: reuse Kaggle's preinstalled torch/timm/torchvision
# (venv with system site-packages) and add only the small missing ones; fall back to a full py3.12 venv.
INK_LIGHT = "'zarr>=3,<4' 'numcodecs>=0.15.1,<0.17' pynrrd imagecodecs tifffile einops timm huggingface-hub tqdm requests pyyaml aiohttp fsspec"
INK_CHECK = (f"cd {TMP} && PYTHONPATH={VILLA}/vesuvius/src {VINK}/bin/python -c \"import torch, torchvision, timm, zarr, cv2, numba; "
             f"import vesuvius.ink_detection.inference.infer; assert torch.cuda.is_available() and int(zarr.__version__[0]) >= 3; "
             f"print('ink env ok', torch.__version__, torch.cuda.device_count(), zarr.__version__)\"")
if not os.path.exists(f"{VINK}/bin/python"):
    sh(f"uv venv -q --system-site-packages --python {sys.executable} {VINK}")
    sh(f"uv pip install -q --python {VINK}/bin/python {INK_LIGHT}")
    if sh(INK_CHECK, check=False).returncode:
        print("fast ink env failed its check -> full py3.12 venv", flush=True)
        sh(f"rm -rf {VINK} && uv venv -q --python 3.12 {VINK}")
        sh(f"uv pip install -q --python {VINK}/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        sh(f"uv pip install -q --python {VINK}/bin/python {INK_LIGHT} numpy pillow scipy opencv-python-headless numba")
        if sh(INK_CHECK, check=False).returncode:  # driver older than the cu128 runtime: cu126 wheels
            sh(f"uv pip install -q --python {VINK}/bin/python --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126")
            sh(INK_CHECK)
# spiral-fitting env (requires python >= 3.14; builds the vc_spiral C++ extension)
if RUN["mode"] == "band" and not os.path.exists(f"{VFIT}/bin/python"):
    sh(f"uv python install -q 3.14 && uv venv -q --python 3.14 {VFIT}")
    sh(f"uv pip install -q --python {VFIT}/bin/python 'torch>=2.11,<2.13' --index-url https://download.pytorch.org/whl/cu128")
    # editable, like the validated local install: a plain install leaves spiral-fitting/vc_spiral/__init__.py (no .so)
    # shadowing the built package when fit_spiral.py runs from its own folder -> no native packed-track loader
    r = sh(f"cd {VILLA}/spiral-fitting && uv pip install -q --python {VFIT}/bin/python -e .", check=False)
    if r.returncode:  # old system g++ without enough C++23: get a newer one and retry
        sh("apt-get -qq update && (apt-get -qq install -y g++-13 || apt-get -qq install -y g++-12)", check=False)
        cxx = "g++-13" if os.path.exists("/usr/bin/g++-13") else "g++-12"
        sh(f"cd {VILLA}/spiral-fitting && CXX={cxx} CC={cxx.replace('++', 'cc')} uv pip install -q --python {VFIT}/bin/python -e .")
CHECK_NATIVE = ("import tracks, vc_spiral.track_store as t, vc_spiral.track_crossings, torch; "
                "assert tracks._load_native_track_store() is not None; print('native ok', t.__file__, torch.__version__, torch.cuda.is_available())")
if RUN["mode"] == "band":
    r = sh(f"cd {VILLA}/spiral-fitting && {VFIT}/bin/python -c \"{CHECK_NATIVE}\"", check=False)
    if r.returncode:  # fall back: put the built extensions next to the source package the fitter's cwd resolves
        so = glob.glob(f"{VFIT}/lib/python3.14/site-packages/vc_spiral/*.so") + glob.glob(f"{VILLA}/spiral-fitting/build/**/vc_spiral/*.so", recursive=True)
        for f in so: sh(f"cp -n {f} {VILLA}/spiral-fitting/vc_spiral/")
        sh(f"cd {VILLA}/spiral-fitting && {VFIT}/bin/python -c \"{CHECK_NATIVE}\"")
# checkpoints
CK = f"{TMP}/ck"
dl = (f"from huggingface_hub import hf_hub_download as h\n"
      f"for f in ['hybrid_3d2d-seed42/step-075000.pth','hybrid_3d2d-seed43/step-075000.pth']: print(h('scrollprize/ink_9um', f, local_dir='{CK}/ink_9um'))\n"
      + (f"for f in ['hecate.py','hecate_9.6um.pth']: print(h('scrollprize/hecate', f, local_dir='{CK}/hecate'))\n" if RUN["hecate"] else ""))
open(f"{TMP}/dl.py", "w").write(dl); sh(f"{VINK}/bin/python {TMP}/dl.py")
smoke = (f"import sys, numpy as np; sys.path.insert(0, '{TMP}/src'); from ink import Ink; from render_gpu import write_zarr\n"
         f"if __name__ == '__main__':\n"
         f" write_zarr('{TMP}/smoke.zarr', np.random.default_rng(0).integers(60, 200, (31, 256, 256), dtype=np.uint8), {{}})\n"
         f" ink = Ink('{CK}/ink_9um/hybrid_3d2d-seed42/step-075000.pth', ['{CK}/ink_9um/hybrid_3d2d-seed43/step-075000.pth'], amp='fp16', batch=4)\n"
         f" print('ink smoke', ink('{TMP}/smoke.zarr', '{TMP}/smoke.tif'))\n")
SMOKE = smoke
print(f"setup done in {(time.time() - T0) / 60:.1f} min")
'''

GO = r'''
open(f"{TMP}/smoke.py", "w").write(SMOKE)
sh(f"cd {TMP} && PYTHONPATH={VILLA}/vesuvius/src {VINK}/bin/python {TMP}/smoke.py")
P = dict(RUN, out=f"{W}/{RUN['tag']}", tmp=f"{TMP}/{RUN['tag']}", villa=VILLA, fit_python=f"{VFIT}/bin/python", t_start=T0,
         ink_checkpoints=[f"{TMP}/ck/ink_9um/hybrid_3d2d-seed42/step-075000.pth", f"{TMP}/ck/ink_9um/hybrid_3d2d-seed43/step-075000.pth"],
         hecate_py=f"{TMP}/ck/hecate/hecate.py", hecate_ck=f"{TMP}/ck/hecate/hecate_9.6um.pth")
json.dump(P, open(f"{TMP}/params.json", "w"))
env = dict(os.environ, PYTHONPATH=f"{VILLA}/vesuvius/src", CHUNK_CACHE=f"{TMP}/chunks", OMP_NUM_THREADS="4")
p = subprocess.Popen([f"{VINK}/bin/python", "-u", f"{TMP}/src/pipeline.py", f"{TMP}/params.json"], env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if "\r" not in line: print(line, end="")
print("exit", p.wait(), f"total {(time.time() - T0) / 60:.1f} min")
'''

SHOW = r'''
from IPython.display import Image, display
for f in sorted(glob.glob(f"{W}/{RUN['tag']}/montage/*.jpg"))[:6] + sorted(glob.glob(f"{W}/{RUN['tag']}/*_masked.jpg")):
    print(f); display(Image(f, width=1200))
sh(f"du -sh {W}/{RUN['tag']}; ls {W}/{RUN['tag']}")
'''

cells = [md(INTRO), code(RUN), code(SETUP)]
for f in SRC:
    cells.append(code(f"%%writefile /tmp/vb/src/{f}\n" + open(os.path.join(HERE, "src", f)).read()))
cells.insert(3, code("os.makedirs('/tmp/vb/src', exist_ok=True)"))
cells += [code(GO), code(SHOW)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
      "isGpuEnabled": True}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "vesuvius_band.ipynb"), "w"), indent=1)
print("wrote vesuvius_band.ipynb,", len(cells), "cells")
