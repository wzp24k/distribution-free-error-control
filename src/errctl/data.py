"""Assemble per-cell tables (features + expert labels) for all four cohorts.

Two very different provenances are bridged here:

* CRC and GVHD ship a HighQualityLabels ``CellTable.csv`` with mean marker
  intensity for every segmented cell, plus centroid coordinates. Expert manual
  labels for a subset of those cells join onto it by (image, cellID) with 100%
  coverage, so no image processing is needed.
* Breast and Melanoma ship only 500x500 / 1024x1024 crops with a
  ``segmentation_labels`` mask. Features are computed here from the TIFFs.

Label vocabularies differ per cohort, so every table also carries a ``coarse``
column produced by :mod:`errctl.labels`.
"""

import os

import numpy as np
import pandas as pd
import tifffile

from . import paths
from .labels import to_coarse

SEGMENTATION = "segmentation_labels"


def _marker_column(frame):
    for c in ("Channel", "Marker"):
        if c in frame.columns:
            return c
    raise KeyError("no channel/marker column in %s" % list(frame.columns))


def marker_table(ds, kind="ManualLabels"):
    """Return (marker names in panel order, full table)."""
    base = paths.manual_dir(ds) if kind == "ManualLabels" else paths.hq_dir(ds)
    t = pd.read_csv(os.path.join(base, "Tables", "MarkerTable.csv"))
    col = _marker_column(t)
    names = [str(x) for x in t[col].tolist()]
    keep = [n for n in names if SEGMENTATION not in n.lower()]
    return keep, t


def celltype_table(ds, kind="ManualLabels"):
    base = paths.manual_dir(ds) if kind == "ManualLabels" else paths.hq_dir(ds)
    return pd.read_csv(os.path.join(base, "Tables", "CellTypes.csv"))


def load_manual_labels(ds):
    p = os.path.join(paths.manual_dir(ds), "Tables", "Labels.csv")
    df = pd.read_csv(p)
    df = df.rename(columns={"fov": "image"})
    df["cellID"] = df["cellID"].astype(int)
    df["image"] = df["image"].astype(str)
    return df[["image", "cellID", "class"]]


def load_hq_celltable(ds):
    p = os.path.join(paths.hq_dir(ds), "Tables", "CellTable.csv")
    return pd.read_csv(p)


def extract_crop_features(ds, use_cache=True):
    """Mean marker intensity / area / centroid per cell, from crop TIFFs.

    Returns a long table with one row per (image, cellID) and one column per
    marker. Cached as parquet because TIFF decoding is the slow step.
    """
    cache = os.path.join(paths.CACHE, "cropfeat_%s.parquet" % ds)
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)

    markers, _ = marker_table(ds)
    img_root = os.path.join(paths.manual_dir(ds), "Images")
    rows = []
    for image in sorted(os.listdir(img_root)):
        d = os.path.join(img_root, image)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        seg_file = next((f for f in files if SEGMENTATION in f.lower()), None)
        if seg_file is None:
            continue
        seg = tifffile.imread(os.path.join(d, seg_file))
        flat = seg.ravel()
        counts = np.bincount(flat)
        ids = np.nonzero(counts)[0]
        ids = ids[ids != 0]
        ys, xs = np.divmod(np.arange(flat.size), seg.shape[1])
        csum_y = np.bincount(flat, weights=ys)
        csum_x = np.bincount(flat, weights=xs)

        table = {"image": image, "cellID": ids.astype(int)}
        for m in markers:
            f = next((x for x in files if os.path.splitext(x)[0].lower() == m.lower()), None)
            if f is None:
                table[m] = np.full(ids.size, np.nan, dtype=np.float32)
                continue
            arr = tifffile.imread(os.path.join(d, f)).ravel()
            if arr.size != flat.size:
                table[m] = np.full(ids.size, np.nan, dtype=np.float32)
                continue
            tot = np.bincount(flat, weights=arr.astype(np.float64))
            table[m] = (tot[ids] / counts[ids]).astype(np.float32)
        table["Area"] = counts[ids].astype(np.float32)
        table["centroid_x"] = (csum_x[ids] / counts[ids]).astype(np.float32)
        table["centroid_y"] = (csum_y[ids] / counts[ids]).astype(np.float32)
        rows.append(pd.DataFrame(table))

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def build_master(ds):
    """One table per cohort: features for every segmented cell + both labels.

    Columns: dataset, image, cellID, class_hq, class_expert, markers, Area,
    centroid_x, centroid_y, coarse_hq, coarse_expert, is_expert.
    """
    if ds in paths.HQ_DATASETS:
        cell = load_hq_celltable(ds)
        markers, _ = marker_table(ds, kind="HighQualityLabels")
        markers = [m for m in markers if m in cell.columns]
        cell = cell.rename(columns={"class": "class_hq"})
        expert = load_manual_labels(ds).rename(columns={"class": "class_expert"})
        cell = cell.merge(expert, on=["image", "cellID"], how="left")
    else:
        cell = extract_crop_features(ds)
        markers, _ = marker_table(ds)
        markers = [m for m in markers if m in cell.columns]
        expert = load_manual_labels(ds).rename(columns={"class": "class_expert"})
        cell = cell.merge(expert, on=["image", "cellID"], how="left")
        cell["class_hq"] = np.nan

    if "centroid_x" not in cell.columns:
        cell["centroid_x"] = np.nan
        cell["centroid_y"] = np.nan
    if "Area" not in cell.columns:
        cell["Area"] = np.nan

    cell.insert(0, "dataset", ds)
    cell["is_expert"] = cell["class_expert"].notna()
    cell["coarse_expert"] = np.where(cell["is_expert"],
                                     to_coarse(cell["class_expert"].fillna("Unidentified")),
                                     np.nan)
    cell["coarse_hq"] = np.where(cell["class_hq"].notna(),
                                 to_coarse(cell["class_hq"].fillna("Unidentified")),
                                 np.nan)
    cols = (["dataset", "image", "cellID", "class_hq", "class_expert",
             "coarse_hq", "coarse_expert", "is_expert", "Area",
             "centroid_x", "centroid_y"] + markers)
    return cell[cols]


def gold(ds, drop_unidentified=True):
    """Expert-labelled cells only, with the marker columns and Area."""
    cell = build_master(ds)
    g = cell[cell["is_expert"]].copy()
    if drop_unidentified:
        g = g[g["coarse_expert"] != "Unidentified"]
    markers, _ = marker_table(ds, kind="HighQualityLabels" if ds in paths.HQ_DATASETS
                              else "ManualLabels")
    markers = [m for m in markers if m in g.columns]
    return g, markers


def feature_matrix(frame, markers, extra=("Area",)):
    cols = [m for m in markers if m in frame.columns]
    if extra:
        cols = cols + [c for c in extra if c in frame.columns]
    X = frame[cols].to_numpy(dtype=np.float64)
    return X, cols
