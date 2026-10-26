"""In-process ink_9um inference (model loaded once, many windings): same entry points and flags as
`python -m vesuvius.ink_detection.inference.infer <zarr> <ck> <out> --overlap 0.5 --blend-mode hann
--direction both --no-compile --ensemble-checkpoint <ck2>` in ../run_segments.sh."""
import os, subprocess, sys
from pathlib import Path
import numpy as np, tifffile


class Ink:
    def __init__(self, ck, ensemble=(), gpus="", amp="auto", batch=16, workers=4, overlap=0.5):
        from vesuvius.ink_detection.inference import infer as I
        from vesuvius.ink_detection.inference.inference_runtime import prepare_model_for_inference
        argv = ["in.zarr", str(ck), "out.tif", "--overlap", str(overlap), "--blend-mode", "hann", "--batch-size", str(batch),
                "--direction", "both", "--no-compile", "--amp-dtype", amp, "--num-workers", str(workers)]
        for e in ensemble:
            argv += ["--ensemble-checkpoint", str(e)]
        if gpus:
            argv += ["--gpus", gpus]
        self.I = I
        self.args = I.normalize_inference_paths(I.parse_args(argv))
        cm = I.configure_model(self.args)
        model, self.device = prepare_model_for_inference(cm.model, gpu_ids=self.args.gpu_ids, compile_model=False,
                                                         compile_mode=self.args.compile_mode)
        from dataclasses import replace
        self.cm = replace(cm, model=model)

    def __call__(self, zarr_path, out_tif):
        """Writes out_tif (forward) and <stem>_reverse.tif, zlib-compressed."""
        out_tif = Path(out_tif)
        outs = []
        for d in ("forward", "reverse"):
            o = out_tif if d == "forward" else out_tif.with_name(f"{out_tif.stem}_reverse{out_tif.suffix}")
            tmp = o.with_name(o.stem + ".raw.tif")
            self.args.input_zarr = Path(zarr_path)
            self.I.infer_single_zarr(args=self.args, input_zarr=Path(zarr_path), configured_model=self.cm,
                                     device=self.device, output_tiff=tmp, layer_direction=d)
            tifffile.imwrite(o, tifffile.imread(tmp), compression="zlib")
            tmp.unlink()
            outs.append(o)
        return outs


def hecate(hecate_py, ck, zarr_path, out_png, spacing_um, reverse=False, device="cuda"):
    cmd = [sys.executable, str(hecate_py), "--checkpoint", str(ck), "--input", str(zarr_path), "--output", str(out_png),
           "--spacing-um", str(spacing_um), "--device", device, "--batch-size", "8"]
    if reverse:
        cmd.append("--reverse")
    return subprocess.run(cmd, capture_output=True, text=True)
