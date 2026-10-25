# Mirror an S3 prefix of the open-data bucket to a local dir (parallel, resumable).
import sys, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from s3ls import ls, B
prefix, dest = sys.argv[1], sys.argv[2]
files = [e for e in ls(prefix, "") if e[0] == "F"]
def get(e):
    _, k, size = e
    p = os.path.join(dest, k[len(prefix):])
    if os.path.exists(p) and os.path.getsize(p) == size: return 0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for _ in range(5):
        try:
            urllib.request.urlretrieve(B + urllib.parse.quote(k), p + ".part"); os.replace(p + ".part", p); return size
        except Exception as ex: err = ex
    print("FAIL", k, err); return 0
with ThreadPoolExecutor(32) as ex: n = sum(ex.map(get, files))
print(f"{len(files)} files, {n/1e6:.1f} MB fetched -> {dest}")
