"""Coverage atlas: every scroll surface we have run ink detection on, with a negative result.

    python atlas/atlas.py <root> atlas/sources.json atlas/

<root> is the working directory the meshes and predictions live under. Each source in
sources.json is one batch of surfaces (published segments, GrowPatch patches or spiral
windings) on one scan. A surface counts only if both ink directions were predicted for it.
For every surface the atlas measures, from its tifxyz mesh:
  - true surface area (sum of quad cross products, scan voxels -> cm^2)
  - z range (um), and radius from the umbilicus when one is given (mm)
and writes atlas.json, ATLAS.md and one coverage map per scroll
(top-down XY footprint and radius-vs-height).
"""
import glob, io, json, os, re, sys, urllib.request
import numpy as np, tifffile

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def load_mesh(path, cache):
    """tifxyz from a local dir or an S3 prefix (x/y/z.tif + meta.json), as (H, W, 3) zyx-agnostic xyz."""
    if path.startswith("s3:"):
        key = path[3:].rstrip("/")
        local = os.path.join(cache, key.replace("/", "__"))
        os.makedirs(local, exist_ok=True)
        for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
            p = os.path.join(local, f)
            if not os.path.exists(p):
                open(p, "wb").write(urllib.request.urlopen(f"{S3}/{key}/{f}", timeout=120).read())
        path = local
    xyz = np.stack([tifffile.imread(os.path.join(path, f"{a}.tif")).astype(np.float64) for a in "xyz"], -1)
    valid = (xyz[..., 2] > 0) & np.isfinite(xyz).all(-1) & (xyz > -1).all(-1)
    return xyz, valid


def surface_area_vx2(xyz, valid):
    du = xyz[1:, :-1] - xyz[:-1, :-1]
    dv = xyz[:-1, 1:] - xyz[:-1, :-1]
    ok = valid[1:, :-1] & valid[:-1, 1:] & valid[:-1, :-1]
    return float(np.linalg.norm(np.cross(du, dv), axis=-1)[ok].sum())


def umbilicus_xy(path):
    pts = sorted(json.load(open(path))["control_points"], key=lambda p: p["z"])
    z = np.array([p["z"] for p in pts]); x = np.array([p["x"] for p in pts]); y = np.array([p["y"] for p in pts])
    return lambda zz: (np.interp(zz, z, x), np.interp(zz, z, y))


def main(root, sources_path, out_dir):
    cfg = json.load(open(sources_path))
    cache = os.path.join(out_dir, ".mesh_cache")
    surfaces, samples = [], {}
    for src in cfg["sources"]:
        vox = float(src["voxel_um"]); cm_per_vx = vox * 1e-4
        umb = umbilicus_xy(os.path.join(root, src["umbilicus"])) if src.get("umbilicus") else None
        preds = sorted(f for f in glob.glob(os.path.join(root, src["pred_glob"])) if not f.endswith("_reverse.tif"))
        preds = [f for f in preds if os.path.exists(f.replace(".tif", "_reverse.tif"))]
        for f in preds:
            name = os.path.basename(f)[:-4]
            mesh = src.get("mesh_map", {}).get(name) or src["mesh"].format(name=name)
            if not mesh:
                print(f"  skip {name}: no mesh mapped (re-render or test of another surface)")
                continue
            if not mesh.startswith("s3:"):
                mesh = os.path.join(root, mesh)
            xyz, valid = load_mesh(mesh, cache)
            p = xyz[valid]
            rec = {"scroll": src["scroll"], "scan": src["scan"], "source": src["id"], "kind": src["kind"],
                   "surface": name, "area_cm2": surface_area_vx2(xyz, valid) * cm_per_vx ** 2,
                   "z_um": [float(p[:, 2].min() * vox), float(p[:, 2].max() * vox)]}
            sub = p[:: max(1, len(p) // 4000)]
            if umb:
                ux, uy = umb(sub[:, 2])
                r = np.hypot(sub[:, 0] - ux, sub[:, 1] - uy) * vox / 1000
                rec["radius_mm"] = [float(np.percentile(r, 5)), float(np.median(r)), float(np.percentile(r, 95))]
                sub_r = r
            else:
                sub_r = None
            surfaces.append(rec)
            samples.setdefault(src["scroll"], []).append((src["id"], sub * vox / 1000, sub_r))
        print(src["id"], sum(1 for x in surfaces if x["source"] == src["id"]), "surfaces", flush=True)

    by = {}
    for s in surfaces:
        k = (s["scroll"], s["scan"], s["source"], s["kind"])
        b = by.setdefault(k, {"n": 0, "area": 0.0, "z0": 1e18, "z1": -1e18, "r0": 1e18, "r1": -1e18})
        b["n"] += 1; b["area"] += s["area_cm2"]
        b["z0"] = min(b["z0"], s["z_um"][0]); b["z1"] = max(b["z1"], s["z_um"][1])
        if "radius_mm" in s:
            b["r0"] = min(b["r0"], s["radius_mm"][0]); b["r1"] = max(b["r1"], s["radius_mm"][2])
    notes = {s["id"]: s.get("result", "") for s in cfg["sources"]}
    L = ["# Coverage atlas: surfaces scanned for ink, with negative results", "",
         "Every surface below was rendered and run through `ink_9um` (seed42+43 ensemble unless noted) in both "
         "depth directions. None showed line-structured text. Areas are true mesh surface areas. "
         "Heights are scan z in µm; radii are distances from the umbilicus (where one was available), "
         "5th–95th percentile.", "",
         '**How strong these negatives are:** on ~9 µm scans (the 8.64 and 9.362 µm rows) `ink_9um` loses sensitivity, and the sweep\'s tile ranking puts real held-out 9 µm text below artifacts ([`bench/SCORE_NOTE.md`](../bench/SCORE_NOTE.md)). Read those rows as "screened", not "shown empty". The 2.4 µm rows are stronger negatives (held-out AUROC 0.90 vs. 0.85 at 9.36 µm, `bench/RESULTS9.md`).', "",
         "| scroll | scan | batch | kind | surfaces | area cm² | z range (mm) | radius (mm) | result |",
         "|---|---|---|---|---|---|---|---|---|"]
    tot = {}
    for (sc, scan, sid, kind), b in by.items():
        rr = f"{b['r0']:.1f}–{b['r1']:.1f}" if b["r1"] > 0 else "—"
        L.append(f"| {sc} | {scan} | {sid} | {kind} | {b['n']} | {b['area']:.0f} | "
                 f"{b['z0'] / 1000:.1f}–{b['z1'] / 1000:.1f} | {rr} | {notes.get(sid, '')} |")
        t = tot.setdefault(sc, [0, 0.0]); t[0] += b["n"]; t[1] += b["area"]
    L += ["", "| scroll | surfaces | area cm² |", "|---|---|---|"]
    L += [f"| {sc} | {n} | {a:.0f} |" for sc, (n, a) in tot.items()]
    L += [f"| **total** | **{sum(n for n, _ in tot.values())}** | **{sum(a for _, a in tot.values()):.0f}** |", "",
          "Per-surface rows (area, z, radius) are in `atlas.json`. Coverage maps: "
          + ", ".join(f"`coverage_{sc}.png`" for sc in tot) + "."]
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"surfaces": surfaces}, open(os.path.join(out_dir, "atlas.json"), "w"), indent=1)
    open(os.path.join(out_dir, "ATLAS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for sc, items in samples.items():
        has_r = any(r is not None for _, _, r in items)
        fig, ax = plt.subplots(1, 2 if has_r else 1, figsize=(12 if has_r else 6, 5.5), squeeze=False)
        ids = sorted({i for i, _, _ in items}); cmap = plt.get_cmap("tab10")
        for i, p, r in items:
            c = cmap(ids.index(i) % 10)
            ax[0, 0].scatter(p[:, 0], p[:, 1], s=0.3, color=c, alpha=0.4)
            if has_r and r is not None:
                ax[0, 1].scatter(r, p[:, 2], s=0.3, color=c, alpha=0.4)
        for i in ids:
            ax[0, 0].scatter([], [], color=cmap(ids.index(i) % 10), label=i, s=20)
        ax[0, 0].set_aspect("equal"); ax[0, 0].invert_yaxis()
        ax[0, 0].set_xlabel("x (mm)"); ax[0, 0].set_ylabel("y (mm)"); ax[0, 0].set_title(f"{sc}: top-down footprint")
        ax[0, 0].legend(fontsize=7, markerscale=1, loc="best")
        if has_r:
            ax[0, 1].set_xlabel("radius from umbilicus (mm)"); ax[0, 1].set_ylabel("z (mm)")
            ax[0, 1].set_title(f"{sc}: radius vs height")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"coverage_{sc}.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:4])
