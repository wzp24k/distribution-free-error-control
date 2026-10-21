"""Robust inventory of CellTuneDepot + demo data.

Answers the questions that decide the whole experimental design:
  A. what columns do the HighQualityLabels tables carry (features? coordinates?)
  B. can manual expert labels be joined to precomputed HQ features?
  C. how much do the marker panels overlap across datasets?
  D. what does the demo parquet contain?
"""

import os
import json

import pandas as pd

DATA = r"D:\TT paper\0926Distribution-free error\data"
OUT = r"D:\TT paper\0926Distribution-free error\work\s01_inventory.txt"
DEPOT = os.path.join(DATA, "depot", "CellTuneDepot")
DEMO = os.path.join(DATA, "demo", "CellTune_Store", "CellTune_Data", "Demo_Data")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
_fh = open(OUT, "w", encoding="utf-8")


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


def cols_of(path):
    df = pd.read_csv(path, nrows=2)
    return list(df.columns), df


log("#" * 95)
log("# A. HighQualityLabels tables")
log("#" * 95)
hq_info = {}
for ds in ["CRC_Feb2623", "GVHD_Feb1323"]:
    t = os.path.join(DEPOT, "HighQualityLabels", ds, "Tables")
    log(f"\n===== {ds}")
    ct_cols, ct_head = cols_of(os.path.join(t, "CellTable.csv"))
    log(f"  CellTable.csv  n_cols={len(ct_cols)}")
    log(f"    {ct_cols}")
    pop_cols, pop_head = cols_of(os.path.join(t, "Populations.csv"))
    log(f"  Populations.csv n_cols={len(pop_cols)} -> {pop_cols}")
    log(f"    head:\n{pop_head.to_string()}")
    hq_info[ds] = {
        "celltable_cols": ct_cols,
        "populations_cols": pop_cols,
    }

log("\n" + "#" * 95)
log("# B. Manual expert labels  ->  joinable to HQ features?")
log("#" * 95)

MANUAL = ["Breast_Jun0524", "CRC_Feb2623", "GVHD_Feb1323", "Melanoma_Oct2522"]
manual = {}
for ds in MANUAL:
    p = os.path.join(DEPOT, "ManualLabels", ds, "Tables", "Labels.csv")
    df = pd.read_csv(p)
    manual[ds] = df
    log(f"\n===== {ds}: {df.shape} cols={list(df.columns)}")
    log(f"  n_images={df['image'].nunique()}  n_classes={df['class'].nunique()}")
    log(f"  cellID min/max = {df['cellID'].min()}/{df['cellID'].max()}")
    log(f"  class counts: {dict(df['class'].value_counts())}")

for ds in ["CRC_Feb2623", "GVHD_Feb1323"]:
    t = os.path.join(DEPOT, "HighQualityLabels", ds, "Tables")
    pop = pd.read_csv(os.path.join(t, "Populations.csv"))
    lab = manual[ds]
    log(f"\n--- join test {ds}: HQ Populations {pop.shape} vs manual {lab.shape}")
    log(f"  HQ populations columns: {list(pop.columns)}")
    idcol = [c for c in pop.columns if c.lower() in ("cellid", "cell_id", "cell")] or [pop.columns[1]]
    imgcol = [c for c in pop.columns if c.lower() in ("image", "fov", "image_name")] or [pop.columns[0]]
    hq_ids = set(zip(pop[imgcol[0]].astype(str), pop[idcol[0]].astype(int)))
    man_ids = set(zip(lab["image"].astype(str), lab["cellID"].astype(int)))
    inter = man_ids & hq_ids
    log(f"  HQ id space: {len(hq_ids)}; manual: {len(man_ids)}; overlap = {len(inter)} "
        f"({100.0*len(inter)/max(len(man_ids),1):.1f}% of manual)")
    shared_img = sorted(set(lab['image'].astype(str)) & set(pop[imgcol[0]].astype(str)))
    log(f"  shared images: {len(shared_img)} of {lab['image'].nunique()} manual / "
        f"{pop[imgcol[0]].nunique()} HQ")
    if shared_img:
        s = shared_img[0]
        a = lab[lab['image'].astype(str) == s]['cellID'].astype(int)
        b = pop[pop[imgcol[0]].astype(str) == s][idcol[0]].astype(int)
        log(f"  example image '{s}': manual IDs {sorted(a)[:8]} ... (n={len(a)}); "
            f"HQ IDs {sorted(b)[:8]} ... (n={len(b)}); overlap={len(set(a)&set(b))}")

log("\n" + "#" * 95)
log("# C. Marker panel overlap")
log("#" * 95)
panels = {}
for ds in MANUAL:
    mk = pd.read_csv(os.path.join(DEPOT, "ManualLabels", ds, "Tables", "MarkerTable.csv"))
    key = "Channel" if "Channel" in mk.columns else "Marker"
    m = [str(x) for x in mk[key].tolist() if str(x).lower() != "segmentation_labels"]
    panels["ML:" + ds] = set(m)
for ds in ["CRC_Feb2623", "GVHD_Feb1323"]:
    mk = pd.read_csv(os.path.join(DEPOT, "HighQualityLabels", ds, "Tables", "MarkerTable.csv"))
    key = "Channel" if "Channel" in mk.columns else "Marker"
    m = [str(x) for x in mk[key].tolist() if str(x).lower() != "segmentation_labels"]
    panels["HQ:" + ds] = set(m)
demo_mk = pd.read_csv(os.path.join(DEMO, "Tables", "MarkerTable.csv"))
key = "Channel" if "Channel" in demo_mk.columns else "Marker"
panels["DEMO"] = set(str(x) for x in demo_mk[key].tolist())

names = list(panels)
log(f"  panel sizes: { {k: len(v) for k, v in panels.items()} }")
log("\n  pairwise Jaccard of marker panels (excluding segmentation):")
log("        " + "  ".join(f"{n[:14]:>14s}" for n in names))
for a in names:
    row = []
    for b in names:
        u = len(panels[a] | panels[b])
        row.append(f"{len(panels[a] & panels[b]) / max(u, 1):14.3f}")
    log(f"  {a[:14]:>14s} " + "  ".join(row))
log("\n  markers shared by ALL panels: " + str(sorted(set.intersection(*panels.values()))))

log("\n" + "#" * 95)
log("# D. Demo data")
log("#" * 95)
dl = pd.read_csv(os.path.join(DEMO, "Labels", "Labels.csv"))
log(f"  Labels.csv {dl.shape} cols={list(dl.columns)}")
log(f"    n_images={dl['image'].nunique()} n_classes={dl['class'].nunique()}")
log(f"    classes: {dict(dl['class'].value_counts())}")
ct = pd.read_csv(os.path.join(DEMO, "Tables", "CellTypeTable.csv"))
log(f"  CellTypeTable.csv {ct.shape}: cols={list(ct.columns)}")
log(ct.to_string())

parq = os.path.join(DEMO, "Tables", "cellTable_features_calculated.parquet")
if os.path.exists(parq):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parq)
    log(f"\n  parquet: rows={pf.metadata.num_rows} row_groups={pf.metadata.num_row_groups} "
        f"size={os.path.getsize(parq)/1e6:.0f} MB")
    log(f"  columns({len(pf.schema_arrow.names)}): {list(pf.schema_arrow.names)}")
    t = pf.read_row_group(0).to_pandas()
    log(f"  row_group0 shape={t.shape}")
    with pd.option_context("display.width", 250, "display.max_columns", 100):
        log(t.head(3).to_string())

json.dump(hq_info, open(r"D:\TT paper\0926Distribution-free error\work\hq_info.json", "w"),
          indent=2)
_fh.close()
