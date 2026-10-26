"""Fine-tune ink_9um (hybrid_3d2d) for native 9.362 um input.

    python ft9.py <data_dir> <out.pth> [--base ...seed42/step-075000.pth] [--iters 2000]

Trains on every <data_dir>/<case>/{vol.npy,labels.npz} written by fetch_native9.py: the
published 9.362 um surface volumes of PHerc0139 w035/w039/w040/w041/w044 with the ink_9um native9
labels, inside their supervision masks. The RESULTS9 held-out regions (pherc0139-w016,
pherc0814-46527) are never read. Targets keep ink_9um's 0.5 label smoothing (ink 0.75,
background 0.25) so outputs stay on the original scale. The number of iterations is fixed in
advance; there is no checkpoint selection on held-out data.
"""
import argparse, copy, glob, os, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from vesuvius.ink_detection.inference.infer import configure_single_model, normalize_flat_patch
from vesuvius.ink_detection.models.checkpoint import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--base", default="checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth")
ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smooth", type=float, default=0.5)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

cm = configure_single_model(Path(a.base), amp_dtype="default")
model, D, P, prep = cm.model.to(dev), cm.input_depth, cm.patch_size, cm.preprocessing
segs = []
for d in sorted(glob.glob(os.path.join(a.data, "*", "vol.npy"))):
    vol = np.load(d, mmap_mode="r"); lab = np.load(d.replace("vol.npy", "labels.npz"))
    sup = lab["supervision_mask"] > 0; ink = lab["inklabels"] > 0
    L, H, W = vol.shape
    ok = sup & (np.asarray(vol[L // 2]) > 0)
    yx = np.argwhere(ok)
    yx = yx[(yx[:, 0] >= P // 2) & (yx[:, 0] < H - P // 2) & (yx[:, 1] >= P // 2) & (yx[:, 1] < W - P // 2)]
    ink_yx = yx[ink[yx[:, 0], yx[:, 1]]]
    segs.append((os.path.basename(os.path.dirname(d)), vol, ink, ok, yx, ink_yx))
    print(f"{segs[-1][0]}: vol {vol.shape}, supervised px {len(yx)}, ink {len(ink_yx)}", flush=True)
weights = np.array([len(s[4]) for s in segs], float); weights /= weights.sum()


def sample():
    xs, ys, ms = [], [], []
    for b in range(a.bs):
        name, vol, ink, ok, yx, ink_yx = segs[rng.choice(len(segs), p=weights)]
        pool = ink_yx if (b % 2 == 0 and len(ink_yx)) else yx
        cy, cx = pool[rng.integers(len(pool))]
        y0, x0 = cy - P // 2, cx - P // 2
        L = vol.shape[0]; z0 = (L - D) // 2 + int(rng.integers(-2, 3))  # centred window, +-2 jitter
        z0 = min(max(z0, 0), L - D)
        img = np.asarray(vol[z0:z0 + D, y0:y0 + P, x0:x0 + P])
        lab = ink[y0:y0 + P, x0:x0 + P].astype(np.float32); m = ok[y0:y0 + P, x0:x0 + P]
        k = int(rng.integers(4)); fl = rng.random() < 0.5
        img = np.rot90(img, k, (1, 2)); lab = np.rot90(lab, k); m = np.rot90(m, k)
        if fl: img, lab, m = img[:, :, ::-1], lab[:, ::-1], m[:, ::-1]
        img = normalize_flat_patch(np.ascontiguousarray(img), prep)
        img = img * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05)
        xs.append(img); ys.append(lab.copy()); ms.append(m.astype(np.float32))
    f = lambda v: torch.from_numpy(np.stack(v)).float().to(dev)
    return f(xs)[:, None], f(ys)[:, None], f(ms)[:, None]


def save(it):
    payload = load_checkpoint(a.base)
    sd = model.state_dict(); new = {}
    for k in payload["model"]:  # map the wrapped keys back onto the original names
        cands = [s for s in sd if s.endswith(k)]
        new[k] = sd[min(cands, key=len)].detach().cpu()
    payload["model"] = new; payload["step"] = int(payload.get("step", 0)) + it
    payload["config"] = copy.deepcopy(payload["config"])
    payload["finetune"] = {"note": "native 9.362 um fine-tune on ink_9um native9 labels (PHerc0139 w035/w039/w040/w041/w044)",
                           "args": vars(a), "iter": it}
    out = a.out if it == a.iters else a.out.replace(".pth", f"-it{it}.pth")
    torch.save(payload, out); print("saved", out, flush=True)


opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
model.train(); t0 = time.time()
for it in range(1, a.iters + 1):
    x, y, m = sample()
    target = y * (1 - a.smooth) + a.smooth / 2
    logit = model(x)
    bce = (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * m).sum() / m.sum().clamp(min=1)
    p = logit.sigmoid() * m
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
    loss = bce + dice
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if it % 50 == 0:
        print(f"it {it} loss {loss.item():.4f} bce {bce.item():.4f} dice {dice.item():.4f} {(time.time() - t0) / it:.2f}s/it", flush=True)
    if it % 1000 == 0 or it == a.iters:
        save(it)
