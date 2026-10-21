"""r44 -- the GVHD label/mask delivery defect, as a one-step reproducible table.

What this measures
------------------
The depot ships, for every manually labelled cohort, a per-FOV segmentation
mask (``segmentation_labels.tiff``, or ``.tif`` for Melanoma) whose non-zero
values are the segment ids, and a table
(``ManualLabels/<cohort>/Tables/Labels.csv``) whose ``cellID`` column names the
cells an expert labelled.  The two are supposed to agree: every labelled cell
should be a segment in the shipped mask.

They agree for Breast, CRC and Melanoma.  For GVHD they do not: only 1,794 of
the 10,283 labelled cells name a segment that exists in the shipped mask
(17.4%), and per field of view the two id sets sit in almost disjoint ranges.

This is a defect of the delivered data, not of the pipeline: the analysis never
reads the GVHD mask (the pipeline takes CRC and GVHD features from the depot's
``HighQualityLabels`` ``CellTable.csv``, see :func:`errctl.data.build_master`),
so a mask that does not cover the labels is invisible downstream.  That is why
the audit reports it as a delivery defect rather than as a modelling result.

How to reproduce it in one step
-------------------------------
    python scripts/r44_gvhd_delivery.py            # write the tables, print them
    python scripts/r44_gvhd_delivery.py --check     # exit non-zero if the numbers moved

The join is the one the paper's own loader uses: the segment ids are the unique
non-zero values of the shipped mask (:func:`errctl.data.extract_crop_features`),
and the labelled ids are ``(image, cellID)`` of ``Labels.csv``
(:func:`errctl.data.load_manual_labels`).  No image registration, no
thresholding, no parameter: the comparison is set membership.

Outputs
-------
``experiments/r44_gvhd_delivery.csv``          one row per cohort x field of view
``experiments/r44_gvhd_delivery_summary.csv``  one row per cohort
``work/r44_delivery_report.txt``               the printed report
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from errctl import paths
from errctl.data import SEGMENTATION, load_manual_labels

REPORT = os.path.join(paths.WORK, "r44_delivery_report.txt")

# Re-computed 2026-09-27 and asserted in --check mode.
EXPECTED = {
    "Breast_Jun0524": (7610, 7610),
    "CRC_Feb2623": (2505, 2505),
    "GVHD_Feb1323": (10283, 1794),
    "Melanoma_Oct2522": (21204, 21204),
}


def mask_ids(directory):
    """The unique non-zero values of the shipped segmentation mask, or None."""
    files = os.listdir(directory)
    seg_file = next((f for f in files if SEGMENTATION in f.lower()), None)
    if seg_file is None:
        return None
    seg = tifffile.imread(os.path.join(directory, seg_file))
    ids = np.unique(seg)
    return ids[ids != 0].astype(np.int64)


def scan_cohort(ds):
    """One row per field of view, with the id ranges on both sides."""
    labels = load_manual_labels(ds)
    by_image = {str(k): np.sort(v["cellID"].to_numpy(dtype=np.int64))
                for k, v in labels.groupby("image")}
    img_root = os.path.join(paths.manual_dir(ds), "Images")
    fovs = sorted({f for f in os.listdir(img_root)
                   if os.path.isdir(os.path.join(img_root, f))} | set(by_image))
    rows = []
    for fov in fovs:
        lab = by_image.get(fov, np.array([], dtype=np.int64))
        m = mask_ids(os.path.join(img_root, fov))
        mask = np.array([], dtype=np.int64) if m is None else m
        both = np.intersect1d(lab, mask)
        rows.append({
            "cohort": ds,
            "fov": fov,
            "n_labelled": int(lab.size),
            "label_id_min": int(lab.min()) if lab.size else -1,
            "label_id_max": int(lab.max()) if lab.size else -1,
            "n_mask_segments": int(mask.size),
            "mask_id_min": int(mask.min()) if mask.size else -1,
            "mask_id_max": int(mask.max()) if mask.size else -1,
            "n_matched": int(both.size),
            "n_label_without_mask": int(lab.size - both.size),
            "n_mask_without_label": int(mask.size - both.size),
            "match_frac": (float(both.size) / lab.size) if lab.size else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="assert the recomputed headline; exit non-zero if it moved")
    args = ap.parse_args()

    rows = []
    for ds in paths.MANUAL_DATASETS:
        rows.extend(scan_cohort(ds))
    detail = pd.DataFrame(rows)

    summ = (detail.groupby("cohort", as_index=False)
            .agg(n_fovs=("fov", "size"),
                 n_labelled=("n_labelled", "sum"),
                 n_matched=("n_matched", "sum"),
                 n_mask_segments=("n_mask_segments", "sum"),
                 n_label_without_mask=("n_label_without_mask", "sum"),
                 n_mask_without_label=("n_mask_without_label", "sum")))
    summ["match_frac"] = summ["n_matched"] / summ["n_labelled"]

    out1 = os.path.join(paths.RESULTS, "r44_gvhd_delivery.csv")
    out2 = os.path.join(paths.RESULTS, "r44_gvhd_delivery_summary.csv")
    detail.to_csv(out1, index=False, lineterminator="\n")
    summ.to_csv(out2, index=False, lineterminator="\n")

    lines = []
    lines.append("r44 -- do the shipped segmentation masks cover the shipped labels?")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Per cohort (labelled cells, and how many of them name a segment that")
    lines.append("exists in the shipped mask):")
    lines.append("")
    lines.append("%-18s %6s %10s %10s %8s %10s" %
                 ("cohort", "fovs", "labelled", "matched", "frac", "mask_segs"))
    for _, r in summ.iterrows():
        lines.append("%-18s %6d %10d %10d %7.4f %10d" %
                     (r["cohort"], r["n_fovs"], r["n_labelled"], r["n_matched"],
                      r["match_frac"], r["n_mask_segments"]))
    lines.append("")
    bad = summ[summ["match_frac"] < 1.0]["cohort"].tolist()
    lines.append("Cohorts with any labelled cell missing from the mask: %s"
                 % (", ".join(bad) if bad else "none"))
    lines.append("")
    for coh in bad:
        d = detail[(detail["cohort"] == coh) & (detail["n_labelled"] > 0)]
        lines.append("-- %s, per field of view (first 12 of %d with labels) --" % (coh, len(d)))
        lines.append("%-22s %9s %11s %11s %9s %11s %11s %9s" %
                     ("fov", "labelled", "lab_min", "lab_max",
                      "matched", "mask_min", "mask_max", "mask_n"))
        for _, r in d.head(12).iterrows():
            lines.append("%-22s %9d %11d %11d %9d %11d %11d %9d" %
                         (r["fov"], r["n_labelled"], r["label_id_min"], r["label_id_max"],
                          r["n_matched"], r["mask_id_min"], r["mask_id_max"],
                          r["n_mask_segments"]))
        lines.append("")
    lines.append("Expected-value check (--check):")
    ok = True
    for ds in paths.MANUAL_DATASETS:
        r = summ[summ["cohort"] == ds]
        got = (int(r["n_labelled"].iloc[0]), int(r["n_matched"].iloc[0]))
        want = EXPECTED.get(ds)
        flag = "OK " if got == want else "MOVED"
        if got != want:
            ok = False
        lines.append("  %-18s got %s  expected %s  %s" % (ds, got, want, flag))
    text = "\n".join(lines) + "\n"
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(text)
    print(text)
    print("wrote %s\nwrote %s\nwrote %s" % (out1, out2, REPORT))
    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
