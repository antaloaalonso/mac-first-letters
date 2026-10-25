"""Prefetch level-L CT chunks (128^3) that contain predicted surface within [z0, z1) into the render cache.
usage: prefetch_band.py <surface.zarr> <volume_url> <level> <z0> <z1> [workers]"""
import sys, os, numpy as np, zarr
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from render import RemoteVolume
surf, url, level, z0, z1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
vol = RemoteVolume(url, level)
s = zarr.open(surf + '/2', mode='r')  # 4x-downsampled surface prediction (surface frame == CT level 2)
f = 4 * 2 ** (2 - level)  # surface-level-2 voxels per CT chunk edge ratio helper
occ = np.asarray(s[z0 // 4:(z1 + 3) // 4]) > 0
cs = 128 // 4 * 2 ** (level - 2) if level >= 2 else 32 // 2 ** (2 - level)  # CT chunk edge in surf-L2 voxels
chunks = set()
for zi in range(0, occ.shape[0], cs):
    for yi in range(0, occ.shape[1], cs):
        for xi in range(0, occ.shape[2], cs):
            if occ[zi:zi + cs, yi:yi + cs, xi:xi + cs].any():
                chunks.add(((z0 // 4 + zi) // cs, yi // cs, xi // cs))
print(len(chunks), 'chunks to fetch', flush=True)
with ThreadPoolExecutor(int(sys.argv[6]) if len(sys.argv) > 6 else 48) as ex:
    for i, _ in enumerate(ex.map(lambda c: vol.chunk(*c), sorted(chunks))):
        if i % 1000 == 0: print(i, flush=True)
print('done')
