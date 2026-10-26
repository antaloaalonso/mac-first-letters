"""Estimate a scroll's umbilicus from lasagna normals: per z slice, the point closest (weighted
least squares) to all lines through sheet pixels along their in-plane normal. Sheet normals of a
rolled scroll point (roughly) at the scroll axis, so these lines meet near the umbilicus.

    estimate(read_slice, zs, scale) -> [{"x","y","z"}...] in level-0 voxels
read_slice(z_lo_res) -> (nx_u8, ny_u8, grad_mag_u8) 2D arrays of one lasagna slice (group grid)."""
import numpy as np


def slice_centre(nx, ny, gm, prev=None, iters=6):
    n = np.stack([(nx.astype(np.float32) - 128) / 127, (ny.astype(np.float32) - 128) / 127], -1)
    mag = np.linalg.norm(n, axis=-1)
    w = gm.astype(np.float32) * (mag > 0.3) * ((nx > 0) | (ny > 0))
    ys, xs = np.nonzero(w > np.percentile(w[w > 0], 50) if (w > 0).any() else w > 0)
    if len(ys) < 200:
        return None, 0.0
    if len(ys) > 200000:
        sel = np.random.default_rng(0).choice(len(ys), 200000, replace=False); ys, xs = ys[sel], xs[sel]
    u = n[ys, xs] / mag[ys, xs, None]
    p = np.stack([xs, ys], -1).astype(np.float64)
    wt = w[ys, xs].astype(np.float64)
    c = p.mean(0) if prev is None else np.asarray(prev, np.float64)
    for it in range(iters):
        P = np.eye(2)[None] - u[:, :, None] * u[:, None, :]          # projector off the normal
        r = np.einsum('nij,nj->ni', P, c[None] - p)
        d = np.linalg.norm(r, axis=-1)
        if it:  # Cauchy-reweight: ignore lines that pass far from the current centre (folds, cracks)
            s = np.median(d) + 1e-6
            ww = wt / (1 + (d / s) ** 2)
        else:
            ww = wt
        A = (ww[:, None, None] * P).sum(0); b = (ww[:, None, None] * P @ p[:, :, None])[:, :, 0].sum(0)
        c = np.linalg.solve(A + 1e-6 * np.eye(2), b)
    ev = np.linalg.eigvalsh(A)
    return c, float(ev[0] / (ev[1] + 1e-9))  # conditioning: ~0 when all normals are parallel (crushed)


def estimate(read_slice, zs_lo, scale, smooth=5):
    pts, prev = [], None
    for z in zs_lo:
        nx, ny, gm = read_slice(z)
        c, cond = slice_centre(nx, ny, gm, prev)
        if c is None:
            continue
        pts.append((z, c[0], c[1], cond)); prev = c
    a = np.array(pts)
    if len(a) >= smooth:  # running median over z
        k = smooth // 2
        a[:, 1] = [np.median(a[max(0, i - k):i + k + 1, 1]) for i in range(len(a))]
        a[:, 2] = [np.median(a[max(0, i - k):i + k + 1, 2]) for i in range(len(a))]
    return [{"x": float(x * scale), "y": float(y * scale), "z": float(z * scale), "cond": round(c, 3)}
            for z, x, y, c in a]
