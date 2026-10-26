"""Resumable band pipeline: stage inputs -> CUDA spiral fit -> render every winding -> ink (ink_9um
seed42+43, both directions, optional Hecate) -> void-masked montages. All persistent state lives in
OUT=/kaggle/working/<tag>; re-running (same session, or a new one with the previous output attached
as an input and P["resume_from"] pointing at it) skips finished work.

    python pipeline.py params.json
"""
import glob, json, os, queue, re, shutil, signal, subprocess, sys, threading, time
import numpy as np

P = json.load(open(sys.argv[1]))
T_START = float(P.get("t_start", time.time()))
BUDGET = float(P.get("max_hours", 2.8)) * 3600
OUT = P["out"]; TMP = P["tmp"]; VILLA = P["villa"]; PY = P.get("fit_python", sys.executable)
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
os.environ.setdefault("CHUNK_CACHE", os.path.join(TMP, "chunks"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch, render_gpu, montage  # noqa: E402

LOG = open(os.path.join(OUT, "log.txt"), "a")
# ink passes: "screen" = seed42 alone at 25% overlap (~4.5x cheaper, both directions kept: the fitted
# meshes' recto side is not known in advance); "full" = seed42+43 ensemble at 50% overlap (the validated
# local recipe), for the control and for confirming candidate windings found by a screen.
INK_MODE = P.get("ink_mode") or ("full" if P["mode"] == "control" else "screen")
PRED = os.path.join(OUT, "pred" if INK_MODE == "full" else "pred_screen")
DONE = os.path.join(OUT, "windings.json" if INK_MODE == "full" else "windings_screen.json")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def left():
    return BUDGET - (time.time() - T_START)


def state(**kw):
    p = os.path.join(OUT, "state.json")
    s = json.load(open(p)) if os.path.exists(p) else {}
    s.update(kw)
    json.dump(s, open(p, "w"), indent=1)
    return s


def resume_copy():
    src = P.get("resume_from")
    if not src:
        return
    cand = [d for d in glob.glob(os.path.join(src, "**", "state.json"), recursive=True)
            if os.path.basename(os.path.dirname(d)) == os.path.basename(OUT)]
    if not cand:
        log("resume_from given but no matching state.json under", src); return
    d = os.path.dirname(cand[0])
    fitted = bool(find_meshes(os.path.join(d, "runs"))[1])
    for root, _, files in os.walk(d):
        for f in files:
            if fitted and f.endswith(".ckpt"):
                continue  # meshes exist: the fit is not resumed, so its checkpoint is not needed
            s = os.path.join(root, f); t = os.path.join(OUT, os.path.relpath(s, d))
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(s, t)
    log("resumed state from", d)


def find_meshes(root):
    """(run_dir, [winding dirs]) of the newest fit under root holding plain wNNN tifxyz windings.
    The exporter (spiral_helpers.save_mesh) writes <run>/meshes/<name>[_<tag>]/wNNN[_spliced][_<tag>];
    keep the plain (non-spliced) windings in any such layout."""
    best = None
    for dp, dns, fns in os.walk(root):
        ws = sorted(os.path.join(dp, d) for d in dns
                    if re.match(r"w\d{3}(_|$)", d) and "_spliced" not in d and os.path.exists(os.path.join(dp, d, "x.tif")))
        if ws:
            run = dp.split(os.sep + "meshes" + os.sep)[0] if os.sep + "meshes" + os.sep in dp + os.sep else dp
            key = (os.path.getmtime(dp), len(ws))
            if best is None or key > best[0]:
                best = (key, run, ws)
            dns[:] = []
    return (best[1], best[2]) if best else (None, [])


# ---------------------------------------------------------------- inputs + fit
def stage_inputs(vol_name, voxel):
    scroll, ts = P["scroll"], vol_name.split("-")[0]
    ds = os.path.join(TMP, "dataset")
    os.makedirs(ds, exist_ok=True)
    log("tracks index + crossings ...")
    store, dbm = fetch.stage_tracks(scroll, ts, ds)
    z0, z1 = P.get("z0"), P.get("z1")
    if z0 in (None, "auto"):
        z0, z1, npts = fetch.auto_band(store, int(P.get("band_height", 1000)))
        log(f"auto band [{z0},{z1}) ({npts / 1e6:.1f}M track points)")
    z0, z1 = int(z0), int(z1)
    state(z0=z0, z1=z1)
    n = fetch.stage_band_coords(store, f"{fetch.DL}/{scroll}/{ts}/tracks/{dbm}", z0, z1)
    log(f"band tracks: {n}")
    las = fetch.stage_lasagna(scroll, ts, ds, z0, z1) if P.get("use_lasagna", True) else None
    log("lasagna:", "none" if not las else f"group {las[0]} x{las[1]}")
    umb = fetch.published_umbilicus(scroll, ts)
    src = "published"
    if umb is None and las:
        umb = {"control_points": fetch.lasagna_umbilicus(las[2], las[0], las[1], z0, z1)}; src = "lasagna-estimate"
    if umb is None or not umb.get("control_points"):
        umb = {"control_points": fetch.ct_umbilicus(f"{fetch.S3}/{scroll}/volumes/{vol_name}", z0, z1)}; src = "ct-centroid"
    json.dump(umb, open(os.path.join(ds, "umbilicus.json"), "w"), indent=1)
    shutil.copy(os.path.join(ds, "umbilicus.json"), os.path.join(OUT, f"umbilicus_{src}.json"))
    pts = [p for p in umb["control_points"] if z0 - 500 <= p["z"] <= z1 + 500]
    log(f"umbilicus ({src}): {len(umb['control_points'])} pts, band pts:",
        [(round(p['z']), round(p['x']), round(p['y'])) for p in pts[:6]])
    spec = fetch.write_spec(ds, scroll, voxel, P.get("sense", "CW"), dbm, las)
    shutil.copy(os.path.join(ds, "spiral-scroll.json"), os.path.join(OUT, "spiral-scroll.json"))
    if las:
        log("packing lasagna resident pools ...")
        r = subprocess.run([PY, "pack_resident_pools.py", os.path.join(ds, "lasagna_inputs"), "--what", "normals,grad_mag",
                            "--normal-group", las[0], "--io-threads", "8", "--force"],
                           cwd=os.path.join(VILLA, "spiral-fitting"), capture_output=True, text=True)
        log(r.stdout[-1500:], r.stderr[-1500:])
        if r.returncode:
            raise SystemExit("pack_resident_pools failed")
    return ds, z0, z1, bool(las)


def fit(ds, z0, z1, has_las):
    runs = os.path.join(OUT, "runs")
    tag = P.get("tag", "band")
    ov = {"z_begin": z0, "z_end": z1, "input_use_verified_patches": False, "input_use_tracks": True,
          "input_use_normals": has_las, "input_use_gradient_magnitude": has_las,
          "input_use_winding_inference": False, "input_use_outer_shell": False, "input_use_fibers": False,
          "input_use_pcl_absolute": False, "input_use_pcl_relative": False, "input_use_pcl_same_winding": False,
          "input_use_pcl_drawn_control_points": False, "dense_spacing_mode": "grad_mag",
          "loss_weight_shell_outer": 0, "optimizer_num_training_steps": int(P.get("steps", 3000))}
    if not has_las:
        ov["loss_weight_dense_spacing"] = 0
    ov.update(P.get("fit_overrides", {}))
    env = dict(os.environ, FIT_SPIRAL_DEVICE="cuda", FIT_SPIRAL_CACHE_DIR=os.path.join(TMP, "cache_spiral"),
               FIT_SPIRAL_OUT_DIR=runs, FIT_SPIRAL_RUN_TAG=tag, FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(ov),
               WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    if P.get("float32", True):
        env["FIT_SPIRAL_MAX_PRECISION_FLOAT"] = "32"
    # preflight: the fit must load the band from the packed .vctracks store; the .dbm is only a stub
    dbm = json.load(open(os.path.join(ds, "spiral-scroll.json")))["paths"]["tracks_dbm"]
    chk = (f"import tracks; c = tracks._load_packed_track_collection({os.path.join(ds, dbm)!r}, {z0}, {z1}); "
           f"assert c is not None and len(c) > 0, 'packed track store not usable'; print('packed store ok:', len(c), 'tracks')")
    r = subprocess.run([PY, "-c", chk], cwd=os.path.join(VILLA, "spiral-fitting"), capture_output=True, text=True)
    log(r.stdout[-800:], r.stderr[-2500:])
    if r.returncode:
        raise SystemExit("packed track store preflight failed (see above)")
    triton = P.get("triton", True)
    for attempt in range(2):
        prev = sorted(d for d in glob.glob(os.path.join(runs, f"*_{tag}")) if os.path.exists(os.path.join(d, "checkpoint_fitted.ckpt")))
        e = dict(env)
        if prev:
            e.update(FIT_SPIRAL_RUN_DIR=prev[-1], FIT_SPIRAL_RESUME_PATH=os.path.join(prev[-1], "checkpoint_fitted.ckpt"))
            log("resuming fit from", prev[-1])
        if not triton:
            e["FIT_SPIRAL_TRITON"] = "0"
        logf = os.path.join(OUT, f"fit_{tag}.log")
        log(f"fit [{z0},{z1}) steps={ov['optimizer_num_training_steps']} triton={triton} -> {logf}")
        with open(logf, "a") as lf:
            pr = subprocess.Popen([PY, "fit_spiral.py", "--dataset", ds], cwd=os.path.join(VILLA, "spiral-fitting"),
                                  env=e, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            reserve = float(P.get("reserve_hours_for_ink", 1.0)) * 3600
            while pr.poll() is None:
                time.sleep(20)
                if left() < reserve:
                    log("fit out of budget: stopping (resume next session from the 1000-step autosave)")
                    os.killpg(pr.pid, signal.SIGINT); time.sleep(60)
                    if pr.poll() is None:
                        os.killpg(pr.pid, signal.SIGKILL)
                    state(fit="incomplete"); return None
        tail = open(logf, errors="replace").read()[-6000:]
        run_dir, meshes = find_meshes(runs)
        if pr.returncode == 0 and meshes:
            state(fit="done", run_dir=os.path.relpath(run_dir, OUT)); log(f"fit done: {len(meshes)} windings in {run_dir}")
            return run_dir
        if pr.returncode == 0:
            log("fit exited 0 but no wNNN meshes found; runs tree:\n" + "\n".join(
                os.path.relpath(dp, runs) for dp, _, _ in os.walk(runs) if dp.count(os.sep) - runs.count(os.sep) <= 3))
        log(f"fit exited {pr.returncode}; tail:\n{tail[-2500:]}")
        if triton and re.search(r"triton|ptxas|CompilationError|PTX", tail, re.I):
            triton = False; state(triton=False); continue
        raise SystemExit("fit failed")
    raise SystemExit("fit failed twice")


# ---------------------------------------------------------------- render + ink
def winding_loop(items, vol_url, level, frame_level, voxel_um):
    """items: [(name, tifxyz_dir)]. Renders in a background thread while the GPU runs ink."""
    import torch
    from ink import Ink, hecate
    pred = PRED; rend = os.path.join(OUT, "renders"); os.makedirs(pred, exist_ok=True); os.makedirs(rend, exist_ok=True)
    done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
    todo = [(n, d) for n, d in items if n not in done]
    log(f"windings: {len(items)} total, {len(todo)} to do")
    if not todo:
        return
    ngpu = torch.cuda.device_count()
    rdev = f"cuda:{ngpu - 1}" if ngpu else "cpu"
    cc = torch.cuda.get_device_capability(0) if ngpu else (0, 0)
    amp = P.get("amp") or ("fp16" if cc < (8, 0) else "auto")
    ck = P["ink_checkpoints"] if INK_MODE == "full" else P["ink_checkpoints"][:1]
    overlap = float(P.get("ink_overlap") or (0.5 if INK_MODE == "full" else 0.25))
    ink = Ink(ck[0], ck[1:], gpus=",".join(map(str, range(ngpu))) if ngpu > 1 else "", amp=amp,
              batch=int(P.get("ink_batch", 16)), overlap=overlap)
    log(f"ink ready ({INK_MODE}): {len(ck)} checkpoint(s), overlap {overlap}, gpus={ngpu}, amp={amp}, render on {rdev}")
    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    per = []

    def producer():
        for n, d in todo:
            if stop.is_set():
                break
            t0 = time.time()
            z = os.path.join(TMP, "renders", f"{n}.zarr")
            try:
                cv, vm = render_gpu.render(d, vol_url, z, layers=int(P.get("layers", 31)), level=level,
                                           flip=bool(P.get("flip", True)), frame_level=frame_level, device=rdev, log=log)
            except Exception as e:
                log(f"render fail {n}: {e!r}"); q.put((n, None, 0, 0, 0)); continue
            L = cv.shape[0]
            mid = cv[L // 2 - 6:L // 2 + 7].max(0)
            import cv2
            cv2.imwrite(os.path.join(rend, f"{n}_max.png"), mid)
            cover = float(((cv[L // 2] > 0) & vm).sum() / max(1, vm.sum()))
            area = float(vm.sum()) * (voxel_um * 2 ** level * 1e-4) ** 2
            del cv
            q.put((n, z, cover, area, time.time() - t0))
            render_gpu.prune_cache(os.environ["CHUNK_CACHE"], float(P.get("cache_gb", 25)))
        q.put(None)
    th = threading.Thread(target=producer, daemon=True); th.start()
    summary_p = os.path.join(OUT, f"windings_summary{'' if INK_MODE == 'full' else '_screen'}.json")
    summary = json.load(open(summary_p)) if os.path.exists(summary_p) else {}
    while True:
        it = q.get()
        if it is None:
            break
        n, z, cover, area = it[:4]
        if z is None:
            continue
        t0 = time.time()
        if cover < float(P.get("min_cover", 0.15)):
            log(f" {n}: papyrus cover {cover:.2f} < min -> skip ink")
        else:
            ink(z, os.path.join(pred, f"{n}.tif"))
            if P.get("hecate"):
                for rev in (False, True):
                    r = hecate(P["hecate_py"], P["hecate_ck"], z, os.path.join(pred, f"{n}_hecate{'_reverse' if rev else ''}.png"),
                               voxel_um * 2 ** level, reverse=rev, device="cuda:0")
                    if r.returncode:
                        log(" hecate fail", r.stderr[-800:])
        shutil.rmtree(z, ignore_errors=True)
        dt = time.time() - t0 + it[4]
        per.append(dt)
        done.add(n); json.dump(sorted(done), open(DONE, "w"))
        summary[n] = {"cover": round(cover, 3), "area_cm2": round(area, 2), "sec": round(dt)}
        json.dump(summary, open(summary_p, "w"), indent=1)
        log(f" {n}: cover {cover:.2f}, {area:.1f} cm2, {dt:.0f}s; {len(done)}/{len(items)} done, {left() / 60:.0f} min left")
        if left() < 1.5 * np.median(per[-5:]) + 600:
            log("time budget reached: stopping windings (rerun with resume to continue)")
            stop.set()
            while q.get() is not None:
                pass
            break
    th.join(timeout=5)


def montages(name):
    pred = PRED; rend = os.path.join(OUT, "renders"); mdir = os.path.join(OUT, "montage")
    name = f"{name}_{INK_MODE}"
    os.makedirs(mdir, exist_ok=True)
    for d in ("forward", "reverse"):
        outs = montage.montage(pred, rend, d, os.path.join(mdir, f"{name}_{d}"), width=int(P.get("montage_width", 3000)))
        log(f"montage {d}: {len(outs)} sheets")


def main():
    resume_copy()
    vol_name, voxel = fetch.resolve_volume(P["scroll"], P["volume"])
    vol_url = f"{fetch.S3}/{P['scroll']}/volumes/{vol_name}"
    level = int(P.get("level", 0))
    log(f"== {P['mode']} {P['scroll']} {vol_name} ({voxel} um, render level {level} = {voxel * 2 ** level:.2f} um); "
        f"budget {BUDGET / 3600:.1f} h")
    state(mode=P["mode"], scroll=P["scroll"], volume=vol_name)
    if P["mode"] == "control":
        seg = P.get("segment", "20240304144031-w018_20240304144031_flatboi")
        mesh = P.get("mesh", "20240304144031-on-20251217075048-2.399um.tifxyz")
        d = os.path.join(TMP, "control", "w018")
        os.makedirs(d, exist_ok=True)
        for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
            fetch.download(f"{fetch.S3}/{P['scroll']}/segments/{seg}/mesh/{mesh}/{f}", os.path.join(d, f), threads=8, part=16 << 20)
        ref = [k for k in fetch.s3_list(f"{P['scroll']}/segments/{seg}/ink-detection/downsampled/")[1] if "2.399um" in k]
        if ref:
            fetch.download(f"{fetch.S3}/{ref[0]}", os.path.join(OUT, "official_2.4um_ink_ds8.jpg"))
        winding_loop([("w018", d)], vol_url, level, int(P.get("frame_level", 0)), voxel)
        montages("control")
        for suf in ("", "_reverse"):
            f = os.path.join(PRED, f"w018{suf}.tif")
            if os.path.exists(f):
                montage.masked_fullres(f, os.path.join(OUT, "renders", "w018_max.png"), os.path.join(OUT, f"w018{suf}_{INK_MODE}_masked.jpg"))
        log("control done"); return
    run, ws = find_meshes(os.path.join(OUT, "runs"))
    if ws:
        log(f"fit already done: {len(ws)} windings in {run} -> skipping inputs + fit")
    else:
        ds, z0, z1, has_las = stage_inputs(vol_name, voxel)
        log(f"inputs staged, {left() / 60:.0f} min left")
        run = fit(ds, z0, z1, has_las)
        if run is None:
            return
        run, ws = find_meshes(os.path.join(OUT, "runs"))
    lo, hi = P.get("windings", [0, 9999])
    items = [(os.path.basename(w)[:4], w) for w in ws if lo <= int(os.path.basename(w)[1:4]) <= hi]
    if P.get("winding_list"):  # e.g. ["w052", "w053"]: confirm only these (typically with ink_mode="full")
        items = [t for t in items if t[0] in set(P["winding_list"])]
    # outer windings first (largest area per GPU-minute; inner turns are often crushed)
    items.sort(key=lambda t: -int(t[0][1:]))
    winding_loop(items, vol_url, level, 0, voxel)
    montages(P["scroll"])
    state(last_finished=time.strftime("%Y-%m-%d %H:%M"))
    log("band done for this session")


if __name__ == "__main__":
    main()
