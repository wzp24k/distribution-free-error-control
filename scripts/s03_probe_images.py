"""Characterise the manual-label TIFF crops and their segmentation masks.

Goal: decide how to compute per-cell marker intensities for Breast / Melanoma,
which do not ship a precomputed CellTable.
"""

import os
import glob
import numpy as np
import tifffile
import pandas as pd

DATA = r"D:\TT paper\0926Distribution-free error\data"
DEPOT = os.path.join(DATA, "depot", "CellTuneDepot")
OUT = r"D:\TT paper\0926Distribution-free error\work\s03_images.txt"
_fh = open(OUT, "w", encoding="utf-8")


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


for ds in ["Breast_Jun0524", "Melanoma_Oct2522", "CRC_Feb2623", "GVHD_Feb1323"]:
    base = os.path.join(DEPOT, "ManualLabels", ds, "Images")
    imgs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    log("=" * 95)
    log(f"##### {ds}: {len(imgs)} image folders")
    log("=" * 95)
    for name in imgs[:2]:
        d = os.path.join(base, name)
        files = sorted(os.listdir(d))
        log(f"\n  --- {name}: {len(files)} files")
        seg = [f for f in files if "segmentation" in f.lower()]
        log(f"      segmentation file(s): {seg}")
        for f in files[:3] + seg:
            p = os.path.join(d, f)
            try:
                a = tifffile.imread(p)
                log(f"      {f:32s} shape={a.shape} dtype={a.dtype} "
                    f"min={a.min()} max={a.max()} mean={a.mean():.2f}")
            except Exception as e:
                log(f"      {f:32s} ERROR {e}")
        if seg:
            s = tifffile.imread(os.path.join(d, seg[0]))
            u = np.unique(s)
            log(f"      seg unique count={len(u)} first20={u[:20]} last5={u[-5:]}")
            lab = pd.read_csv(os.path.join(DEPOT, "ManualLabels", ds, "Tables", "Labels.csv"))
            ids = set(lab[lab["image"] == name]["cellID"].tolist())
            segids = set(int(x) for x in u if x != 0)
            log(f"      label-file cellIDs for this image: n={len(ids)} "
                f"min={min(ids) if ids else None} max={max(ids) if ids else None}")
            log(f"      mask ids: n={len(segids)} min={min(segids) if segids else None} "
                f"max={max(segids) if segids else None}")
            log(f"      mask ids present in label file: "
                f"{len(segids & ids)}/{len(segids)}; "
                f"labelled ids present in mask: {len(ids & segids)}/{len(ids)}")
    sz = {}
    for name in imgs:
        files = os.listdir(os.path.join(base, name))
        sz[name] = len(files)
    log(f"\n  file-count per image folder: min={min(sz.values())} max={max(sz.values())}")

_fh.close()
