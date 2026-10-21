"""Probe the demo parquet (how CellTune builds features) and verify that every
expert-labelled cell has a matching precomputed feature row in the HQ tables."""

import os
import pandas as pd
import pyarrow.parquet as pq

DATA = r"D:\TT paper\0926Distribution-free error\data"
DEPOT = os.path.join(DATA, "depot", "CellTuneDepot")
DEMO = os.path.join(DATA, "demo", "CellTune_Store", "CellTune_Data", "Demo_Data")
OUT = r"D:\TT paper\0926Distribution-free error\work\s02_probe.txt"

_fh = open(OUT, "w", encoding="utf-8")


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


log("=" * 95)
log("DEMO data")
log("=" * 95)
dl = pd.read_csv(os.path.join(DEMO, "Labels", "Labels.csv"))
log(f"Labels.csv {dl.shape} cols={list(dl.columns)}")
log(f"  n_fov={dl['fov'].nunique()} n_classes={dl['class'].nunique()}")
log(f"  fov: {sorted(dl['fov'].unique())}")
log(f"  classes: {dict(dl['class'].value_counts())}")
log(dl.head(4).to_string())

ct = pd.read_csv(os.path.join(DEMO, "Tables", "CellTypeTable.csv"))
log(f"\nCellTypeTable.csv {ct.shape} cols={list(ct.columns)}")
log(ct.to_string())
mk = pd.read_csv(os.path.join(DEMO, "Tables", "MarkerTable.csv"))
log(f"\nMarkerTable.csv {mk.shape} cols={list(mk.columns)}")
log(mk.to_string())

parq = os.path.join(DEMO, "Tables", "cellTable_features_calculated.parquet")
if os.path.exists(parq):
    pf = pq.ParquetFile(parq)
    names = list(pf.schema_arrow.names)
    log(f"\nparquet rows={pf.metadata.num_rows} row_groups={pf.metadata.num_row_groups} "
        f"size={os.path.getsize(parq)/1e6:.0f} MB  n_cols={len(names)}")
    log(f"columns:\n{names}")
    t = pf.read_row_group(0).to_pandas()
    log(f"row_group0 shape={t.shape}")
    with pd.option_context("display.width", 250, "display.max_columns", 100,
                           "display.max_rows", 200):
        log(t.head(3).T.to_string())
    key = [c for c in names if c.lower() in ("image", "fov")] + \
          [c for c in names if c.lower() in ("cellid",)]
    log(f"\nkey columns: {key}")
    imgs = t[key[0]].unique()[:10]
    log(f"  first images in parquet: {list(imgs)}")

log("\n" + "=" * 95)
log("HQ CellTable coverage of expert-labelled cells")
log("=" * 95)
for ds in ["CRC_Feb2623", "GVHD_Feb1323"]:
    t = os.path.join(DEPOT, "HighQualityLabels", ds, "Tables")
    ctf = os.path.join(t, "CellTable.csv")
    cell = pd.read_csv(ctf, usecols=["image", "cellID", "Area", "centroid_x",
                                     "centroid_y", "class"])
    lab = pd.read_csv(os.path.join(DEPOT, "ManualLabels", ds, "Tables", "Labels.csv"))
    m = lab.merge(cell, on=["image", "cellID"], how="left", suffixes=("_manual", "_hq"))
    n_missing = m["class_hq"].isna().sum()
    log(f"\n{ds}: CellTable rows={len(cell)} images={cell['image'].nunique()} "
        f"classes={cell['class'].nunique()}")
    log(f"  manual cells={len(lab)}  missing features in CellTable = {n_missing}")
    log(f"  HQ class distribution: {dict(cell['class'].value_counts())}")
    if n_missing == 0:
        agree = (m["class_manual"].astype(str) == m["class_hq"].astype(str)).mean()
        log(f"  manual-vs-HQ label agreement (all manual cells) = {agree:.3f}")
        sub = m[~m["class_manual"].isin(["Unidentified"]) & ~m["class_hq"].isin(["Unidentified"])]
        if len(sub):
            log(f"  agreement excluding Unidentified (n={len(sub)}) = "
                f"{(sub['class_manual'].astype(str) == sub['class_hq'].astype(str)).mean():.3f}")
    log(f"  feature columns used: Area, centroid_x, centroid_y present: "
        f"{[c for c in ['Area','centroid_x','centroid_y'] if c in cell.columns]}")

_fh.close()
