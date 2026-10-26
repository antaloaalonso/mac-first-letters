"""Inventory the published inputs of every First Letters scroll (network only, light).
usage: python probe.py > ../inventory.json"""
import re, json, sys, urllib.request, concurrent.futures as cf
DL = "https://dl.ash2txt.org/datasets/spiral_datasets/"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCROLLS = ["PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
           "PHerc0306B", "PHerc0358", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc0800",
           "PHerc0813", "PHerc0826", "PHerc0846A", "PHerc0846B", "PHerc1203", "PHerc1218", "PHerc1545", "PHerc1667"]

def get(u):
    try: return urllib.request.urlopen(u, timeout=60).read().decode("utf8", "replace")
    except Exception as e: return ""

def ls_dl(u):
    """nginx fancyindex rows -> [(name, size)]"""
    h = get(u); out = []
    for m in re.finditer(r'<a href="([^"?]+)"[^>]*>[^<]*</a></td><td[^>]*>([^<]*)</td>', h):
        if m.group(1) not in ("../", "/"): out.append((m.group(1), m.group(2).strip()))
    if not out:
        out = [(n, "") for n in re.findall(r'href="([^"?]+)"', h) if not n.startswith(("..", "/", "http"))]
    return out

def ls_s3(prefix):
    h = get(f"{S3}?list-type=2&delimiter=/&prefix={prefix}")
    return re.findall(r"<Prefix>([^<]+)</Prefix>", h)[1:], re.findall(r"<Key>([^<]+)</Key>", h)

def probe(s):
    r = {"scroll": s, "spiral": {}, "volumes": [], "representations": {}}
    for v, _ in ls_dl(DL + s + "/"):
        r["spiral"][v.strip("/")] = {sub.strip("/"): ls_dl(DL + s + "/" + v + sub) for sub, _ in ls_dl(DL + s + "/" + v)}
    vp, _ = ls_s3(f"{s}/volumes/"); r["volumes"] = [p.split("/")[-2] for p in vp]
    rp, _ = ls_s3(f"{s}/representations/")
    for p in rp:
        sub, keys = ls_s3(p)
        r["representations"][p.split("/")[-2]] = [x.split("/")[-2] for x in sub] + [k.split("/")[-1] for k in keys]
    for kind in ("predictions",):
        if kind in r["representations"]:
            for name in r["representations"][kind]:
                sub, keys = ls_s3(f"{s}/representations/{kind}/{name}/")
                r["representations"][f"{kind}/{name}"] = [x.split("/")[-2] for x in sub] + [k.split("/")[-1] for k in keys]
    return r

with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(probe, SCROLLS))
json.dump(res, sys.stdout, indent=1)
