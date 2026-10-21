"""Inventory the CellTuneDepot / CellTune demo data before building any pipeline.

Writes a plain-text report so findings can be cited in the paper's data-audit section.
"""

import os
import sys
import json
import glob

import pandas as pd

DATA = r"D:\TT paper\0926Distribution-free error\data"
OUT = r"D:\TT paper\0926Distribution-free error\work\s00_data_report.txt"


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


log = Tee(OUT)


def show_csv(path, nrows=5, full=False):
    if not os.path.exists(path):
        log(f"MISSING: {path}")
        return None
    size = os.path.getsize(path) / 1e6
    df = pd.read_csv(path)
    log(f"--- {path}  ({size:.2f} MB)  shape={df.shape}")
    log(f"    columns: {list(df.columns)}")
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        log(df.head(nrows).to_string())
    return df


def value_counts(df, col, top=40):
    if df is None or col not in df.columns:
        return
    vc = df[col].value_counts(dropna=False)
    log(f"    {col}: {len(vc)} unique -> {dict(vc.head(top))}")


log("=" * 100)
log("PART 1 -- CellTuneDepot / ManualLabels (expert gold standard)")
log("=" * 100)

MANUAL = ["Breast_Jun0524", "CRC_Feb2623", "GVHD_Feb1323", "Melanoma_Oct2522"]
manual_summary = {}

for ds in MANUAL:
    base = os.path.join(DATA, "depot", "CellTuneDepot", "ManualLabels", ds)
    tab = os.path.join(base, "Tables")
    log("\n" + "#" * 90)
    log(f"##### {ds}")
    log("#" * 90)

    labels = show_csv(os.path.join(tab, "Labels.csv"), nrows=5)
    ct = show_csv(os.path.join(tab, "CellTypes.csv"), nrows=60)
    mk = show_csv(os.path.join(tab, "MarkerTable.csv"), nrows=80)

    if labels is not None:
        value_counts(labels, "image", top=200)
        value_counts(labels, "class", top=60)
        log(f"    n_cells = {len(labels)};  n_images = {labels['image'].nunique()}")
        log(f"    cellID range: {labels['cellID'].min()} .. {labels['cellID'].max()}")

    img_dir = os.path.join(base, "Images")
    subs = sorted(os.listdir(img_dir))
    log(f"    Images: {len(subs)} subdirs, first 10 = {subs[:10]}")
    if subs:
        f0 = sorted(os.listdir(os.path.join(img_dir, subs[0])))
        log(f"    [{subs[0]}] {len(f0)} files:")
        log(f"        {f0}")
    manual_summary[ds] = {
        "n_labels": 0 if labels is None else int(len(labels)),
        "n_images": 0 if labels is None else int(labels["image"].nunique()),
        "n_celltypes": 0 if ct is None else int(len(ct)),
        "n_markers": 0 if mk is None else int(len(mk)),
    }

log("\n" + "=" * 100)
log("PART 2 -- CellTuneDepot / HighQualityLabels (semi-supervised, model generated)")
log("=" * 100)

hq_headers = {}
for ds in ["CRC_Feb2623", "GVHD_Feb1323"]:
    base = os.path.join(DATA, "depot", "CellTuneDepot", "HighQualityLabels", ds)
    tab = os.path.join(base, "Tables")
    log("\n" + "#" * 90)
    log(f"##### {ds}")
    log("#" * 90)

    head = pd.read_csv(os.path.join(tab, "CellTable.csv"), nrows=5)
    hq_headers[ds] = list(head.columns)
    log(f"--- CellTable.csv columns ({len(head.columns)}): {list(head.columns)}")
    log(head.to_string()[:2000])

    pop = show_csv(os.path.join(tab, "Populations.csv"), nrows=5)
    if pop is not None:
        value_counts(pop, "class", top=60)
        value_counts(pop, "image", top=100)
        log(f"    n_cells = {len(pop)}; n_images = {pop['image'].nunique()}")
    show_csv(os.path.join(tab, "CellTypes.csv"), nrows=60)
    show_csv(os.path.join(tab, "MarkerTable.csv"), nrows=80)

log("\n" + "=" * 100)
log("PART 3 -- Demo data (GVHD, with precomputed features)")
log("=" * 100)
demo = os.path.join(DATA, "demo", "CellTune_Store", "CellTune_Data", "Demo_Data")
dl = show_csv(os.path.join(demo, "Labels", "Labels.csv"), nrows=5)
if dl is not None:
    value_counts(dl, "image", top=200)
    value_counts(dl, "class", top=60)
    log(f"    n_cells = {len(dl)}; n_images = {dl['image'].nunique()}")
show_csv(os.path.join(demo, "Tables", "CellTypeTable.csv"), nrows=60)
show_csv(os.path.join(demo, "Tables", "MarkerTable.csv"), nrows=80)

parq = os.path.join(demo, "Tables", "cellTable_features_calculated.parquet")
if os.path.exists(parq):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parq)
    log(f"--- {parq} ({os.path.getsize(parq)/1e6:.1f} MB)")
    log(f"    rows={pf.metadata.num_rows}  row_groups={pf.metadata.num_row_groups}")
    log(f"    columns({len(pf.schema_arrow.names)}): {list(pf.schema_arrow.names)}")
    t = pf.read_row_group(0).to_pandas().head(3)
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        log(t.to_string())

dsub = sorted(os.listdir(os.path.join(demo, "Images")))
log(f"    Demo Images: {len(dsub)} subdirs -> {dsub}")
if dsub:
    f0 = sorted(os.listdir(os.path.join(demo, "Images", dsub[0])))
    log(f"    [{dsub[0]}] {len(f0)} files -> {f0}")

log("\n" + "=" * 100)
log("PART 4 -- SUMMARY")
log("=" * 100)
log(json.dumps(manual_summary, indent=2))
log(f"HQ CellTable columns: {json.dumps({k: len(v) for k, v in hq_headers.items()})}")

log.close()
