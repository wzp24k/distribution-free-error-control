"""R69 -- a second and a third independent external source for the certificate.

Why this script exists
----------------------
``scripts/r55_stellar_cellsplit.py`` gave the exchangeable half of the
guarantee its first external reproduction, on the STELLAR / Hickey CODEX cell
tables.  That left the external evidence resting on a *single* study from a
single laboratory, which is the reading a referee is entitled to attack: one
reproduction is a demonstration, not a body of evidence.  This script adds two
more sources, each from a different laboratory, a different acquisition
platform and a different tissue, and neither of them shares a pipeline, a
panel or a depositor with the depot or with STELLAR.

  EXT2_MIBITOF_Angelo   MIBI-TOF, Department of Pathology, Stanford University
                        (Angelo laboratory).  Zenodo 5945388, CC-BY-4.0.  A
                        29-core multi-tissue TMA, 345,490 cells, 165 fields of
                        view, 8 phenotype labels, 15 marker channels.
  EXT2_LiverTx_IMC      Imaging mass cytometry of human liver allografts,
                        Children's Hospital Los Angeles / University of
                        Southern California (Emamaullee laboratory).  Zenodo
                        7570242, CC-BY-4.0.  109,245 cells, 32 images, 23
                        blocks, 12 labels, 11 channels.

Both are stored as flat per-cell tables, so they enter the pipeline the way
r55's cohorts did: the depot's own certificate, split fraction, repeat count
and metric definitions, imported rather than re-implemented.

Arms
----
  ext2_within       train on the cohort's other images, then split the held-out
                    images' cells at random into a calibration half and an
                    evaluation half.  Exchangeable by construction, so the
                    guarantee is supposed to hold here; the arm measures the
                    cost of a second laboratory, a second platform and a
                    second tissue on the channels the cohort actually owns.
  ext2_within_shared  the identical protocol restricted to the channels this
                    panel shares with the most compatible depot panel, so the
                    panel size term is separated from the provenance term.
  ext2_panelswap    a real change of antibody panel.  The source is the depot
                    cohort whose panel shares the most channels with the new
                    panel; the classifier is trained on the source's shared
                    channels only, then calibrated on half of the new cohort
                    and deployed on the other half.  Nothing is dropped by a
                    model's choice of features, unlike the simulated panel
                    reduction of scripts/r08_panel.py: which channels survive is
                    fixed by what the two antibody panels happen to share.

Deviations from r55, all recorded
--------------------------------
* the panels are named differently from the depot's, so a small explicit alias
  table maps the new channels onto the depot's shared channel vocabulary before
  ``errctl.markerlogic.canonical`` is applied.  Two new channels that name a
  merged channel (MHC class I plus Na-K-ATPase alpha1) or an unused metal
  (the eight liver channels without an antibody) are excluded outright rather
  than mapped onto a channel they are not.
* the two label vocabularies are different, so labels are coarsened to a common
  vocabulary declared in ``HARM_NEW`` and ``HARM_DEPOT``; a class enters an arm
  only when both sides hold at least ``MIN_CLASS_SHARED`` cells of it.
* the pool of a unit is capped at 20000 cells and the training subsample at
  60000, exactly as r55 and r02 do.

Outputs: ``experiments/r69_external2.csv`` (tidy, r03 schema plus
family/cohort/target), ``experiments/r69_external2_summary.csv``,
``experiments/r69_external2_panel_overlap.csv``,
``experiments/r69_external2_checks.csv`` and ``work/r69_report.txt``.  The
checks and the report are rewritten by ``--report-only`` from the tidy CSV, so
they can be regenerated without refitting anything.

No per-unit probability file is written unless ``--dump-predictions`` is given,
so a default run adds four CSVs and one report and nothing else.
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import paths                      # noqa: E402
from errctl.classifiers import fit_classifier  # noqa: E402
from errctl.markerlogic import canonical      # noqa: E402
from errctl.preprocess import DomainNormaliser  # noqa: E402
import r02_run_experiments as R2              # noqa: E402
import r03_analyse as A3                      # noqa: E402

EXT2 = os.path.join(ROOT, "data", "external2")
CACHE = os.path.join(paths.WORK, "cache")
PRED_DIR = os.path.join(paths.WORK, "predictions_r69")
RESULTS = paths.RESULTS
REPORT = os.path.join(paths.WORK, "r69_report.txt")

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MIN_CLASS_SHARED = 20
MAX_TRAIN = 60000
POOL_CAP = 20000
IMAGE_MIN_CELLS = 60
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
SEED = 0
CLFS = ("logreg", "rf", "hgb")
DEPOT = tuple(paths.MANUAL_DATASETS)

MIBI_ZIP = "MIBITOF_Angelo_DataTables.zip"
MIBI_MEMBER = "cell_table_size_normalized_clusters.csv"
# The merged MHC-class-I / Na-K-ATPase channel and the 8 liver metals that carry
# no antibody are left out; every remaining channel is one antibody or the DNA
# intercalator.
MIBI_MARKERS = ["CD11c", "CD20", "CD3", "CD31", "CD4", "CD45", "CD56",
                "CD68", "CD8", "HLA DR", "PANCK", "PAX5", "Vimentin",
                "beta-tubulin", "dsDNA"]

LIVER_CSV = "LiverTx_IMC_processed_data.csv"
LIVER_MARKERS = ["Cell_CD20_Dy161", "Cell_CD3_Er170", "Cell_CD45RA_Gd155",
                 "Cell_CD45_Sm152", "Cell_CD66a_Yb171", "Cell_CD68_Tb159",
                 "Cell_CD8_Dy162", "Cell_CollagenI_Tm169", "Cell_HLADR_Yb174",
                 "Cell_191Ir_Ir191", "Cell_193Ir_Ir193"]

# Provenance, quoted from the deposit records rather than from the papers.
REJECTED = [
    ("CODEX colon cell type labels", "Dryad doi:10.5061/dryad.dfn2z352c",
     "the depositor is the same one as the STELLAR cohort already in use, so "
     "the source would not be independent"),
    ("Bodenmiller breast IMC, Jackson 2020", "Zenodo 4607374",
     "the per-cell intensity table ships inside a 36.8 GB archive and the "
     "small file in that record is cluster labels only, without intensities"),
    ("IMMUcan panel 1 cell types", "Zenodo 12912567",
     "the labelled cells ship only as R SingleCellExperiment objects and no R "
     "interpreter is available here"),
    ("Manchester glioblastoma IMC", "Zenodo 11073042",
     "40 marker channels and 273562 cells, but no cell type label column in "
     "the deposit; fetched (md5 70fd9e3324e9c4d72fea539c288472b2), inspected "
     "and not used, so it is not kept"),
]

PROVENANCE = {
    "EXT2_MIBITOF_Angelo": {
        "record": "Zenodo 5945388",
        "doi": "10.5281/zenodo.5945388",
        "licence": "CC-BY-4.0",
        "laboratory": "Department of Pathology, Stanford University "
                      "(Angelo laboratory)",
        "platform": "MIBI-TOF",
        "tissue": "29-core multi-tissue TMA, tonsil, lymph node, thymus, "
                  "spleen, placenta and solid tumours",
        "files": [(MIBI_ZIP, "cell table read from %s" % MIBI_MEMBER)],
        "n_cells": 345490, "n_images": 165, "n_labels": 8,
        "n_channels": len(MIBI_MARKERS),
        "excluded": "merged MHC-class-I and Na-K-ATPase-alpha1 channel",
        "label_note": "phenotype labels are the depositors' own cell type "
                      "assignments for the cell table",
    },
    "EXT2_LiverTx_IMC": {
        "record": "Zenodo 7570242",
        "doi": "10.5281/zenodo.7570242",
        "licence": "CC-BY-4.0",
        "laboratory": "Children's Hospital Los Angeles and University of "
                      "Southern California (Emamaullee laboratory)",
        "platform": "imaging mass cytometry, Hyperion",
        "tissue": "human liver allograft biopsies, adult and paediatric, "
                  "chronic rejection and no rejection",
        "files": [(LIVER_CSV, "per-cell table"),
                  ("LiverTx_IMC_metadata_samples.csv", "block to group map")],
        "n_cells": 109245, "n_images": 32, "n_labels": 12,
        "n_channels": len(LIVER_MARKERS),
        "excluded": "eight acquired metals that carry no antibody",
        "label_note": "phenotype labels are the depositors' own cell type "
                      "assignments for the processed cell table",
    },
}

# The deposited channel names are cleaned to the depot's vocabulary.  Anything
# absent from this table is matched on its own name, so the two DNA channels of
# the liver panel stay unshared instead of being matched to the depot's DNA.
ALIAS = {
    "HLA DR": "HLADRDPDQ",
    "PANCK": "Keratin",
    "Vimentin": "VIM",
    "Cell_CD20_Dy161": "CD20",
    "Cell_CD3_Er170": "CD3",
    "Cell_CD45RA_Gd155": "CD45RA",
    "Cell_CD45_Sm152": "CD45",
    "Cell_CD66a_Yb171": "CD66a",
    "Cell_CD68_Tb159": "CD68",
    "Cell_CD8_Dy162": "CD8",
    "Cell_CollagenI_Tm169": "Collagen",
    "Cell_HLADR_Yb174": "HLADRDPDQ",
}

# The two vocabularies are coarsened onto one; the target's own label is left
# intact and the source's finer label is merged into it.
HARM_NEW = {
    "EXT2_MIBITOF_Angelo": {
        "T_cell": "Tcell", "B_cell": "Bcell", "Epithelial": "Epithelial",
        "Macrophage": "Macrophage", "DC": "DC", "Endothelial": "Endothelial",
        "Fibroblast": "Fibroblast", "NK_cell": "NKcell"},
    "EXT2_LiverTx_IMC": {
        "Cytotoxic T-Cell 1": "Tcell", "Cytotoxic T-Cell 2": "Tcell",
        "Other T-Cell 1": "Tcell", "Other T-Cell 2": "Tcell",
        "B-Cell": "Bcell", "Macrophage 1": "Macrophage",
        "Macrophage 2": "Macrophage", "Granulocyte": "Granulocyte",
        "Other Lymphocyte": "Other", "Naive Lymphocyte": "Other",
        "HLADR": "Other", "Not Immune Cell": "Other"},
}
HARM_DEPOT = {
    "Bcell": "Bcell", "CD3T": "Tcell", "CD4T": "Tcell", "CD8T": "Tcell",
    "Treg": "Tcell", "NK": "NKcell", "NKT": "NKcell", "Macrophage": "Macrophage",
    "Myeloid": "Macrophage", "Monocyte": "Monocyte", "DC": "DC", "APC": "DC",
    "Endothelial": "Endothelial", "Fibroblast": "Fibroblast",
    "TumorEpithelial": "Epithelial", "Epithelial": "Epithelial",
    "Neutrophil": "Granulocyte", "Plasma": "Plasma", "Mast": "Mast",
    "ImmuneOther": "Other", "Unidentified": "Other", "Lymphatic": "Other",
    "Stroma": "Other", "SMV": "Other", "Neuron": "Other", "Endocrine": "Other",
    "Goblet": "Other", "Paneth": "Other", "ALM": "Other", "HLM": "Other",
    "ActivatedFibro": "Fibroblast", "MyoFibro": "Fibroblast",
    "Tcell": "Tcell", "Tumor": "Epithelial", "Tumor_BCAT": "Epithelial",
    "Garbage": "Other",
}

_SRC = {}


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- data loading
def _read_new(name):
    """(frame, image column, label column, marker columns) for one new cohort."""
    if name == "EXT2_MIBITOF_Angelo":
        z = zipfile.ZipFile(os.path.join(EXT2, MIBI_ZIP))
        cols = ["phenotype", "fov"] + list(MIBI_MARKERS)
        with z.open(MIBI_MEMBER) as fh:
            df = pd.read_csv(fh, usecols=cols, low_memory=False)
        return df, "fov", "phenotype", list(MIBI_MARKERS)
    if name == "EXT2_LiverTx_IMC":
        df = pd.read_csv(os.path.join(EXT2, LIVER_CSV), low_memory=False)
        return df, "ImageID", "Cell_Label", list(LIVER_MARKERS)
    raise ValueError("unknown cohort %r" % name)


def new_cohort_frame(name):
    df, imgc, labc, markers = _read_new(name)
    img = df[imgc].astype(str).to_numpy()
    lab = df[labc].astype(str).to_numpy()
    V = df[markers].to_numpy(dtype=np.float64)
    ok = np.isfinite(V).all(axis=1)
    ok &= (lab != "") & (lab != "nan")
    counts = pd.Series(img[ok]).value_counts()
    good = sorted(counts[counts >= IMAGE_MIN_CELLS].index)
    ok &= np.isin(img, np.asarray(good, dtype=object))
    idx = np.nonzero(ok)[0]
    return {"name": name, "markers": markers, "X": V[idx],
            "y": lab[idx], "img": img[idx]}


def cache_path(name):
    return os.path.join(CACHE, "r69_%s.npz" % name)


def build_cache(names, force=False):
    os.makedirs(CACHE, exist_ok=True)
    for name in names:
        p = cache_path(name)
        if os.path.exists(p) and not force:
            continue
        c = new_cohort_frame(name)
        classes = np.asarray(sorted(set(c["y"].tolist())), dtype=object)
        ycode = np.searchsorted(classes, c["y"])
        np.savez_compressed(
            p, X=np.asarray(c["X"], dtype=np.float32), y=ycode,
            cls=classes.astype("U48"),
            img=np.asarray(c["img"], dtype="U64"),
            markers=np.asarray(c["markers"], dtype="U48"))
        log("cache %-22s %7d cells  %3d images  %2d classes  %2d markers"
            % (name, len(ycode), len(set(c["img"])), len(classes),
               len(c["markers"])))


def load_cohort(name):
    d = np.load(cache_path(name), allow_pickle=False)
    return {"X": d["X"].astype(np.float64), "y": d["y"].astype(int),
            "cls": d["cls"].astype(str), "img": d["img"].astype(str),
            "markers": d["markers"].astype(str)}


def depot_cohort(ds):
    """The depot cohort, loaded once per process through r02's own loader."""
    if ds not in _SRC:
        g, markers, X = R2.prepare_cohort(ds)
        g = g.reset_index(drop=True)
        _SRC[ds] = {"ds": ds, "g": g, "markers": list(markers), "X": X}
    return _SRC[ds]


def new_marker_names(name, markers):
    """The new panel's channel names on the depot's shared vocabulary."""
    return [canonical(ALIAS.get(m, m)) for m in markers]


# ------------------------------------------------------------------- features
def transform(mode, Xtr, others):
    if mode == "raw":
        return [np.asarray(X, dtype=np.float64) for X in others]
    if mode == "log1p_z":
        nz = DomainNormaliser("z_dataset").fit("s", Xtr)
        return [nz.transform("s", X) for X in others]
    if mode == "z_train":
        Xtr = np.asarray(Xtr, dtype=np.float64)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        return [(np.asarray(X, dtype=np.float64) - mu) / sd for X in others]
    raise ValueError("unknown norm %r" % mode)


# ------------------------------------------------------------------ one unit
def unit_rows(family, cohort, unit, fold, clf_kind, Xtr, ytr, Xpool, ypool,
              cids, imgs, mode, target="", extra=None, save_npz=True):
    """One (unit, classifier): fit, then every method for every repeat x alpha."""
    Xtr_t, Xpool_t = transform(mode, Xtr, [Xtr, Xpool])
    clf = fit_classifier(Xtr_t, ytr, kind=clf_kind, seed=0)
    probs = clf.predict_proba(Xpool_t).astype(np.float32)
    n = len(ypool)
    n_cal = int(n * CAL_FRAC)
    rng = np.random.default_rng(1000 + (fold if fold is not None else 0))
    keep = np.zeros((n, N_SPLITS), dtype=bool)
    for r in range(N_SPLITS):
        keep[rng.choice(n, n_cal, replace=False), r] = True
    if save_npz:
        os.makedirs(PRED_DIR, exist_ok=True)
        stem = unit.replace(":", "_")
        np.savez_compressed(
            os.path.join(PRED_DIR, "%s__%s.npz" % (stem, clf_kind)),
            cal_probs=probs, cal_y=ypool, cal_rep=keep, test_rep=~keep,
            cal_X=np.asarray(Xpool_t, dtype=np.float32),
            test_X=np.asarray(Xpool_t, dtype=np.float32),
            test_probs=probs, test_y=ypool, test_cellid=cids, test_img=imgs,
            test_fold=np.full(n, -1 if fold is None else fold),
            classes=np.asarray(sorted(set(int(v) for v in ypool))))
    acc = float((clf.predict_index(Xpool_t) == ypool).mean())
    rows = []
    for r in range(N_SPLITS):
        sel = keep[:, r]
        s_cal = 1.0 - probs[sel].max(1)
        p_cal = probs[sel].argmax(1)
        e_cal = (p_cal != ypool[sel]).astype(float)
        te = ~sel
        s_te = 1.0 - probs[te].max(1)
        p_te = probs[te].argmax(1)
        e_te = (p_te != ypool[te]).astype(float)
        for alpha in ALPHAS:
            res = dict(A3.evaluate(s_cal, e_cal, s_te, e_te, p_cal, p_te,
                                   alpha, DELTA))
            acc_mask, q, sizes = A3.conformal_set_acceptance(
                probs[sel], ypool[sel], probs[te], alpha)
            n_acc = int(acc_mask.sum())
            risk = float(e_te[acc_mask].mean()) if n_acc else np.nan
            res["conformal_set"] = {
                "method": "conformal_set", "n_accepted": n_acc,
                "accept_rate": n_acc / max(len(acc_mask), 1),
                "selective_risk": risk,
                "covered": int((not np.isfinite(risk)) or risk <= alpha),
                "empty": int(n_acc == 0),
                "n_wrong_accepted": int(e_te[acc_mask].sum()) if n_acc else 0,
                "lambda": q, "mean_set_size": float(sizes.mean())}
            cp, tp, T = A3.temp_scale(probs[sel], ypool[sel], probs[te])
            s_cal_t = 1.0 - cp.max(1)
            e_cal_t = (cp.argmax(1) != ypool[sel]).astype(float)
            s_te_t = 1.0 - tp.max(1)
            lam_t = A3.C.naive_threshold(s_cal_t, e_cal_t, alpha)
            acc_t = s_te_t <= lam_t
            n_t = int(acc_t.sum())
            res["temp_naive"] = {
                "method": "temp_naive", "n_accepted": n_t,
                "accept_rate": n_t / max(len(acc_t), 1),
                "selective_risk": float(e_te[acc_t].mean()) if n_t else np.nan,
                "covered": int(n_t == 0 or e_te[acc_t].mean() <= alpha),
                "empty": int(n_t == 0),
                "n_wrong_accepted": int(e_te[acc_t].sum()),
                "lambda": lam_t, "temperature": T}
            ctl_t = A3.C.SelectiveRiskController(alpha, DELTA).fit(
                s_cal_t, e_cal_t)
            acc_t2 = ctl_t.accept(s_te_t)
            n_t2 = int(acc_t2.sum())
            res["crc_temp"] = {
                "method": "crc_temp", "n_accepted": n_t2,
                "accept_rate": n_t2 / max(len(acc_t2), 1),
                "selective_risk": float(e_te[acc_t2].mean()) if n_t2 else np.nan,
                "covered": int(n_t2 == 0 or e_te[acc_t2].mean() <= alpha),
                "empty": int(n_t2 == 0),
                "n_wrong_accepted": int(e_te[acc_t2].sum()),
                "lambda": ctl_t.lambda_hat, "temperature": T}
            for name, rr in res.items():
                row = {
                    "family": family, "cohort": cohort, "target": target,
                    "unit": unit, "clf": clf_kind, "kind": "cellsplit",
                    "repeat": r, "alpha": alpha, "norm": mode,
                    "n_cal_pool": int(len(ypool)), "n_cal_used": int(sel.sum()),
                    "n_test_pool": int(len(ypool)), "n_test_eval": int(te.sum()),
                    "acc_clf": acc, **rr}
                if extra:
                    row.update(extra)
                rows.append(row)
    log("  %-52s %-7s pool=%6d cal=%6d acc=%.3f"
        % (unit, clf_kind, n, n_cal, acc))
    return rows


# ------------------------------------------------------------------- families
def within_unit(name, fold, cols=None):
    """Train on the cohort's other images; pool = held-out images' cells.

    ``cols`` restricts the feature matrix to those columns of the cohort's own
    panel, which is how the shared-panel control is run.
    """
    c = load_cohort(name)
    X, y, img = c["X"], c["y"], c["img"]
    if cols is not None:
        X = X[:, cols]
    imgs = np.asarray(sorted(set(img.tolist())), dtype=object)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(imgs)
    folds = np.array_split(imgs, N_FOLDS)
    held = np.asarray(sorted(folds[fold].tolist()), dtype=object)
    tr = ~np.isin(img, held)
    tr_idx = np.nonzero(tr)[0]
    pool_idx = np.nonzero(~tr)[0]
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = np.random.default_rng(SEED + fold).choice(
            tr_idx, MAX_TRAIN, replace=False)
    present = pd.Series(y[tr_idx]).value_counts()
    keep_cls = set(int(k) for k in present.index
                   if int(present[k]) >= MIN_CLASS_TRAIN)
    tr_idx = np.asarray([i for i in tr_idx if int(y[i]) in keep_cls], dtype=int)
    pool_idx = np.asarray([i for i in pool_idx if int(y[i]) in keep_cls],
                          dtype=int)
    if len(pool_idx) > POOL_CAP:
        pool_idx = np.sort(np.random.default_rng(SEED + 100 + fold).choice(
            pool_idx, POOL_CAP, replace=False))
    order = sorted(keep_cls)
    ix = {k: i for i, k in enumerate(order)}
    return (X[tr_idx], np.asarray([ix[int(v)] for v in y[tr_idx]]),
            X[pool_idx], np.asarray([ix[int(v)] for v in y[pool_idx]]),
            np.arange(1, len(pool_idx) + 1), img[pool_idx])


def _harmonise(labels, table):
    return np.asarray([table.get(str(v), "Other") for v in labels])


def shared_pairs(src_markers, tgt_markers):
    """(shared name, source column, target column) on the depot vocabulary."""
    src = {}
    for i, m in enumerate(src_markers):
        src.setdefault(canonical(m), i)
    tgt = {}
    for i, m in enumerate(tgt_markers):
        tgt.setdefault(canonical(ALIAS.get(m, m)), i)
    keys = sorted(set(src) & set(tgt))
    return [(k, src[k], tgt[k]) for k in keys]


def shared_cols(name, depot):
    """Columns of the new cohort's own panel that the depot panel also has."""
    c = load_cohort(name)
    return [p[2] for p in shared_pairs(depot_cohort(depot)["markers"],
                                       c["markers"])]


def panelswap_unit(name, depot):
    """Train on the depot's shared channels; pool = the new cohort's own cells."""
    src = depot_cohort(depot)
    tgt = load_cohort(name)
    pairs = shared_pairs(src["markers"], tgt["markers"])
    if len(pairs) < 2:
        raise RuntimeError("%s and %s share no channel" % (name, depot))
    sX = np.asarray(src["X"], dtype=np.float64)[:, [p[1] for p in pairs]]
    tX = np.asarray(tgt["X"], dtype=np.float64)[:, [p[2] for p in pairs]]
    sy = _harmonise(src["g"]["coarse_expert"].astype(str).to_numpy(),
                    HARM_DEPOT)
    ty = _harmonise(tgt["cls"][tgt["y"]], HARM_NEW[name])
    s_ok = sy != "Other"
    t_ok = ty != "Other"
    s_counts = pd.Series(sy[s_ok]).value_counts()
    t_counts = pd.Series(ty[t_ok]).value_counts()
    shared = sorted(c for c in set(s_counts.index) & set(t_counts.index)
                    if s_counts[c] >= MIN_CLASS_SHARED
                    and t_counts[c] >= MIN_CLASS_SHARED)
    if len(shared) < 2:
        raise RuntimeError("%s and %s share fewer than two classes: %s"
                           % (name, depot, shared))
    s_idx = np.asarray([i for i in np.nonzero(s_ok)[0] if sy[i] in shared],
                       dtype=int)
    pool_idx = np.asarray([i for i in np.nonzero(t_ok)[0]
                           if ty[i] in shared], dtype=int)
    s_idx = R2.subsample_training(s_idx, sy, MAX_TRAIN, seed=SEED)
    if len(pool_idx) > POOL_CAP:
        pool_idx = np.sort(np.random.default_rng(SEED + 7).choice(
            pool_idx, POOL_CAP, replace=False))
    ix = {c: i for i, c in enumerate(shared)}
    log("  panel %s <-> %s: %d shared channels, %d shared classes %s"
        % (name, depot, len(pairs), len(shared), ",".join(shared)))
    return (sX[s_idx], np.asarray([ix[v] for v in sy[s_idx]]),
            tX[pool_idx], np.asarray([ix[v] for v in ty[pool_idx]]),
            np.arange(1, len(pool_idx) + 1), tgt["img"][pool_idx],
            {"n_shared_channels": len(pairs), "n_shared_classes": len(shared),
             "depot": depot, "shared_classes": "|".join(shared)})


def panel_overlap(name):
    """Jaccard overlap of one new panel against each of the four depot panels."""
    c = load_cohort(name)
    newc = set(new_marker_names(name, c["markers"]))
    rows = []
    for ds in DEPOT:
        dc = set(canonical(m) for m in depot_cohort(ds)["markers"])
        shared = sorted(newc & dc)
        rows.append({
            "cohort": name, "depot": ds, "n_new": len(newc), "n_depot": len(dc),
            "n_shared": len(shared),
            "jaccard": len(shared) / float(len(newc | dc)),
            "shared": "|".join(shared)})
    return rows


def pick_depot(overlap_rows):
    """The most compatible depot panel: most shared channels, then Jaccard."""
    best = sorted(overlap_rows,
                  key=lambda r: (-r["n_shared"], -r["jaccard"], r["depot"]))[0]
    return best["depot"]


def specs(names, clfs=CLFS, which=()):
    out = []
    for name in names:
        if not which or "ext2_within" in which:
            for f in range(N_FOLDS):
                for clf in clfs:
                    out.append(("ext2_within", name, f, clf, None))
        if not which or "ext2_within_shared" in which:
            dep = _DEPOT_CHOICE[name]
            for f in range(N_FOLDS):
                for clf in clfs:
                    out.append(("ext2_within_shared", name, f, clf, dep))
        if not which or "ext2_panelswap" in which:
            dep = _DEPOT_CHOICE[name]
            for clf in clfs:
                out.append(("ext2_panelswap", name, None, clf, dep))
    return out


_DEPOT_CHOICE = {}


def run_unit(spec, mode, save_npz=True):
    family, name, fold, clf, dep = spec
    if family == "ext2_within":
        Xtr, ytr, Xp, yp, cids, imgs = within_unit(name, fold)
        unit = "ext2_within:%s:fold%d" % (name, fold)
        return unit_rows(family, name, unit, fold, clf, Xtr, ytr, Xp, yp, cids,
                         imgs, mode, save_npz=save_npz)
    if family == "ext2_within_shared":
        cols = shared_cols(name, dep)
        Xtr, ytr, Xp, yp, cids, imgs = within_unit(name, fold, cols=cols)
        unit = "ext2_within_shared:%s:fold%d" % (name, fold)
        extra = {"n_shared_channels": len(cols), "depot": dep,
                 "n_shared_classes": np.nan, "shared_classes": ""}
        return unit_rows(family, name, unit, fold, clf, Xtr, ytr, Xp, yp, cids,
                         imgs, mode, target=dep, extra=extra, save_npz=save_npz)
    Xtr, ytr, Xp, yp, cids, imgs, extra = panelswap_unit(name, dep)
    unit = "ext2_panelswap:%s" % name
    return unit_rows(family, name, unit, None, clf, Xtr, ytr, Xp, yp, cids,
                     imgs, mode, target=dep, extra=extra, save_npz=save_npz)


# ------------------------------------------------------------------- summary
def boot_ci_unit(df, B=2000, seed=0):
    """Cluster bootstrap over units of a per-evaluation 0/1 indicator."""
    units = sorted(df["unit"].unique())
    if len(units) < 2:
        m = float(df["covered"].mean())
        return m, m, m
    rng = np.random.default_rng(seed)
    vals = {u: df.loc[df["unit"] == u, "covered"].to_numpy(float)
            for u in units}
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(units), len(units))
        cat = np.concatenate([vals[units[i]] for i in pick])
        draws.append(cat.mean())
    return (float(df["covered"].mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def summarise(df, mode):
    out = []
    for (fam, cohort, alpha), g in df.groupby(["family", "cohort", "alpha"],
                                              sort=True):
        c = g[g["method"] == "crc"]
        nonvac = c[~c["empty"].astype(bool)]
        cov, lo, hi = boot_ci_unit(c)
        out.append({
            "norm": mode, "family": fam, "cohort": cohort, "alpha": alpha,
            "method": "crc", "n_eval": int(len(c)),
            "n_units": int(c["unit"].nunique()),
            "n_classifiers": int(c["clf"].nunique()),
            "mean_accept": float(c["accept_rate"].mean()),
            "review_frac": float(1.0 - c["accept_rate"].mean()),
            "frac_nonvacuous": float(1.0 - c["empty"].mean()),
            "frac_vacuous": float(c["empty"].mean()),
            "mean_accept_nonvac": (float(nonvac["accept_rate"].mean())
                                   if len(nonvac) else np.nan),
            "risk_nonvac": (float(nonvac["selective_risk"].mean())
                            if len(nonvac) else np.nan),
            "coverage": cov, "coverage_lo": lo, "coverage_hi": hi,
            "coverage_nonvac": (float(nonvac["covered"].mean())
                                if len(nonvac) else np.nan),
            "n_over_budget": int((~c["covered"].astype(bool)).sum()),
            "cells_tested": int(c["n_test_eval"].sum()),
            "cells_accepted": int(c["n_accepted"].sum()),
            "clf_acc": float(c["acc_clf"].mean())})
    return pd.DataFrame(out)


def _run_one(job):
    mode, spec, save_npz = job
    return run_unit(spec, mode, save_npz=save_npz)


def write_checks(df, summ, overlap, path):
    rows = []

    def add(check, what, value, reference, ok):
        rows.append({"check": check, "what": what, "value": value,
                     "reference": reference, "PASS": int(bool(ok))})

    for _, r in summ.iterrows():
        add("%s %s a=%.2f coverage" % (r["family"], r["cohort"], r["alpha"]),
            "coverage over evaluations", round(float(r["coverage"]), 6),
            ">= 1 - delta = 0.90", float(r["coverage"]) >= 1.0 - DELTA)
        add("%s %s a=%.2f vacuous" % (r["family"], r["cohort"], r["alpha"]),
            "fraction of evaluations with an empty accepted set",
            round(float(r["frac_vacuous"]), 6), "reported, not asserted", True)
    frac_cal = float((df["n_cal_used"] / df["n_cal_pool"]).mean())
    add("split fraction", "mean n_cal_used / n_cal_pool",
        round(frac_cal, 6), "0.5 to within the integer rounding of a split",
        abs(frac_cal - CAL_FRAC) < 1e-3)
    per_cell = int(df.groupby(["unit", "clf", "alpha"])["repeat"].nunique().max())
    add("repeats per cell", "distinct calibration repeats per (unit, clf, alpha)",
        per_cell, str(N_SPLITS), per_cell == N_SPLITS)
    add("delta", "certificate confidence level", DELTA, "0.1", DELTA == 0.10)
    add("pool cap", "largest test pool", int(df["n_test_pool"].max()),
        "<= %d" % POOL_CAP, int(df["n_test_pool"].max()) <= POOL_CAP)
    for fam in sorted(df["family"].unique()):
        sub = df[df["family"] == fam]
        add("%s units" % fam, "distinct evaluation units",
            int(sub["unit"].nunique()),
            "%d folds x %d classifiers where applicable"
            % (N_FOLDS, len(CLFS)), True)
    for _, r in overlap.iterrows():
        add("panel overlap %s vs %s" % (r["cohort"], r["depot"]),
            "shared channels", int(r["n_shared"]), ">= 2",
            int(r["n_shared"]) >= 2)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def write_report(df, summ, overlap, path):
    ovdf = (overlap if isinstance(overlap, pd.DataFrame)
            else pd.DataFrame(overlap))
    recs = ovdf.to_dict("records")
    L = []
    L.append("=" * 78)
    L.append("R69 -- a second and a third independent external source for the")
    L.append("selective-risk certificate")
    L.append("=" * 78)
    L.append("")
    L.append("-- provenance")
    for name in sorted(PROVENANCE):
        p = PROVENANCE[name]
        got = df[df["cohort"] == name]
        if not len(got):
            continue
        L.append("")
        L.append("  %s" % name)
        L.append("    record          %s  (%s)" % (p["record"], p["doi"]))
        L.append("    licence         %s" % p["licence"])
        L.append("    laboratory      %s" % p["laboratory"])
        L.append("    platform        %s" % p["platform"])
        L.append("    tissue          %s" % p["tissue"])
        L.append("    cells           %d in %d %s"
                 % (p["n_cells"], p["n_images"],
                    "fields of view" if "MIBI" in name else "images"))
        L.append("    labels          %d, %s" % (p["n_labels"], p["label_note"]))
        L.append("    markers         %d per cell; excluded the %s"
                 % (p["n_channels"], p["excluded"]))
        for f, what in p["files"]:
            fp = os.path.join(EXT2, f)
            if os.path.exists(fp):
                L.append("    file            %-42s %12d bytes  md5=%s"
                         % (f, os.path.getsize(fp), md5(fp)))
        best = sorted([r for r in recs if r["cohort"] == name],
                      key=lambda r: (-r["n_shared"], -r["jaccard"]))[0]
        L.append("    shared panel    %d channels shared with the depot's %s "
                 "panel (Jaccard %.3f)"
                 % (best["n_shared"], best["depot"], best["jaccard"]))
        L.append("                    %s" % best["shared"])
    L.append("")
    L.append("-- other candidates, retrieved or inspected and not used")
    L.append("")
    for what, record, why in REJECTED:
        L.append("  %-38s %-34s" % (what, record))
        L.append("      %s" % why)
    L.append("")
    L.append("-- panel overlap of each new panel against the four depot panels")
    L.append("")
    L.append("  %-22s %-16s %7s %7s %7s %9s"
             % ("cohort", "depot", "n_new", "n_depot", "n_shared", "jaccard"))
    for _, r in ovdf.sort_values(["cohort", "depot"]).iterrows():
        L.append("  %-22s %-16s %7d %7d %7d %9.3f"
                 % (r["cohort"], r["depot"], r["n_new"], r["n_depot"],
                    r["n_shared"], r["jaccard"]))
    L.append("")
    L.append("-- the certificate, per arm and budget")
    L.append("")
    cols = ["family", "cohort", "alpha", "n_units", "mean_accept",
            "frac_vacuous", "risk_nonvac", "coverage", "coverage_lo",
            "coverage_hi", "coverage_nonvac", "n_over_budget", "cells_tested"]
    L.append(summ[cols].to_string(index=False))
    L.append("")
    L.append("-- which numbers reproduce the guarantee and which do not")
    L.append("")
    for _, r in summ.iterrows():
        verdict = "reproduces" if r["coverage"] >= 1.0 - DELTA else "FAILS"
        L.append("  %-18s %-22s a=%.2f  coverage=%.3f [%.3f, %.3f]  "
                 "accept=%.3f  vacuous=%.3f  %s"
                 % (r["family"], r["cohort"], r["alpha"], r["coverage"],
                    r["coverage_lo"], r["coverage_hi"], r["mean_accept"],
                    r["frac_vacuous"], verdict))
    L.append("")
    L.append("  The exchangeable arm, ext2_within, is the arm the guarantee")
    L.append("  applies to.  The real panel-swap arm is a deployment with a")
    L.append("  changed antibody panel and a changed laboratory, so a low")
    L.append("  acceptance rate there is a statement about how far the")
    L.append("  certificate is willing to go, not a defect of the arm: the")
    L.append("  budget is still met, mostly by refusing to accept anything.")
    L.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="log1p_z",
                    choices=["log1p_z", "z_train", "raw"])
    ap.add_argument("--only", default="",
                    help="comma list of families to run")
    ap.add_argument("--cohorts", default=",".join(("EXT2_MIBITOF_Angelo",
                                                   "EXT2_LiverTx_IMC")))
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--tag", default="", help="output basename suffix")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rewrite the checks and the report from the tidy CSV")
    ap.add_argument("--dump-predictions", action="store_true",
                    help="also store per-unit probability files under "
                         "work/predictions_r69 (r55 writes these by default)")
    args = ap.parse_args()
    names = [s for s in args.cohorts.split(",") if s]
    which = [s for s in args.only.split(",") if s]
    clfs = tuple(c for c in args.clfs.split(",") if c)
    tag = args.tag or args.norm
    t0 = time.time()

    build_cache(names, force=args.force_cache)
    overlap = []
    for n in names:
        overlap += panel_overlap(n)
        _DEPOT_CHOICE[n] = pick_depot([r for r in overlap if r["cohort"] == n])
        log("depot choice for %s: %s" % (n, _DEPOT_CHOICE[n]))
    ov = pd.DataFrame(overlap)
    ovp = os.path.join(RESULTS, "r69_external2_panel_overlap.csv")
    ov.to_csv(ovp, index=False)
    tidy = os.path.join(RESULTS, "r69_external2_%s.csv" % tag)
    if args.report_only:
        df = pd.read_csv(tidy, low_memory=False)
        summ = summarise(df, args.norm)
        write_checks(df, summ, ov, os.path.join(
            RESULTS, "r69_external2_checks.csv"))
        write_report(df, summ, ov, REPORT)
        log("rewrote checks and %s from %s" % (REPORT, tidy))
        return

    todo = specs(names, clfs=clfs, which=which)
    log("running %d units with %d worker(s)" % (len(todo), max(1, args.jobs)))
    jobs = [(args.norm, s, args.dump_predictions) for s in todo]
    rows = []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, rws in enumerate(pool.imap(_run_one, jobs, chunksize=1), 1):
                rows.extend(rws)
                if i % 5 == 0 or i == len(jobs):
                    log("  %d/%d units done" % (i, len(jobs)))
    else:
        for j in jobs:
            rows.extend(_run_one(j))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(tidy, index=False)
    summ = summarise(df, args.norm)
    summ.to_csv(os.path.join(RESULTS,
                             "r69_external2_%s_summary.csv" % tag), index=False)
    print(summ.to_string(index=False))
    write_checks(df, summ, ov, os.path.join(RESULTS,
                                           "r69_external2_checks.csv"))
    write_report(df, summ, ov, REPORT)
    log("runtime %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
