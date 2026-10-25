"""Delete least-recently-used chunk files until the cache is under the budget (GB)."""
import os, sys
root, budget = sys.argv[1], float(sys.argv[2]) * 1e9
files = []
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f); st = os.stat(p); files.append((st.st_atime, st.st_size, p))
total = sum(s for _, s, _ in files)
for _, s, p in sorted(files):
    if total <= budget: break
    os.remove(p); total -= s
print(f"cache {total/1e9:.1f} GB")
