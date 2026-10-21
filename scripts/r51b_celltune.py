
"""R51b -- wrap the published CellTune classifier in the layer.

Why
---
``scripts/r51_cellsighter.py`` shows the layer survives a convolutional
classifier that never sees a marker table.  CellTune (Bussi et al., Nature
Methods 2026) is the other published tool named in the Introduction and it is
built the other way round: a boosted tree ensemble over per-cell spatial
features, driven from an interactive annotation environment.

What is reproduced
------------------
CellTune ships as a desktop application, so the tool is not importable.  This
script reproduces the classifier its documentation specifies, on the input the
depot provides:

* the documented feature family -- mean, maximum, variance and non-zero pixel
  count per channel, over four compartments: the cell itself, an eroded cell
  centre, the ring of immediate surroundings, and the broader environment;
* the documented arcsinh cofactor for fluorescence images (0.1), applied to per
  field of view, per channel intensities scaled by their 99.9th percentile,
  because the depot ships unscaled 16-bit planes where the application expects
  display-scaled input;
* the documented default model pair, XGBoost and CatBoost, averaged into
  ``Pred_AVG`` exactly as the application averages its two models.

Deviations, all recorded
------------------------
* features come from the depot masks rather than the application's own
  extractor, so the environment compartment is the padded crop window, two cell
  radii wide, rather than a micron radius;
* the models are fitted on the source cohort alone, as for the CellSighter wrap,
  with the application's default hyperparameters rather than interactive ones;
* the tool is a reimplementation of its documentation, not the shipped binary.

Arms and readouts are the paper's own, imported from the CellSighter wrap:
``build_units``, ``analyse`` and ``summarise`` over the same three arms,
budgets, repeat counts and metric definitions.

Outputs: ``work/celltune/``, ``experiments/r51b_celltune.csv``,
``experiments/r51b_celltune_summary.csv``, ``work/r51b_report.txt``.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl import paths                   # noqa: E402
from errctl import data as D               # noqa: E402
from errctl.markerlogic import canonical   # noqa: E402

CACHE = os.path.join(paths.WORK, "celltune")
CS_ROOT = os.path.join(paths.WORK, "cellsighter")

COFACTOR = 0.1
ERODE_IT = 1
DILATE_IT = 2
N_EST = 300
DEPTH = 6
LR = 0.1
SEED = 0


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R51 = _load("r51_cellsighter")


def _stats(crop, mask):
    """mean, max, variance and non-zero pixel count per channel over a mask."""
    n_ch = crop.shape[-1]
    v = crop[mask]
    if v.shape[0] == 0:
        return np.zeros(4 * n_ch, dtype=np.float32)
    return np.concatenate([v.mean(0), v.max(0), v.var(0),
                           (v > 0).sum(0)]).astype(np.float32)


def fov_features(img, seg, c2l):
    """CellTune-style features for every labelled cell of one field of view."""
    scale = np.percentile(img, 99.9, axis=(0, 1)).astype(np.float32)
    scale[scale <= 0] = 1.0
    x = np.clip(img.astype(np.float32) / scale, 0.0, 1.5)
    t = np.arcsinh(x / COFACTOR).astype(np.float32)
    seg = seg.astype(np.int32)
    boxes = ndimage.find_objects(seg)
    height, width = seg.shape
    ids, feats = [], []
    for cid in np.nonzero(c2l >= 0)[0]:
        if cid < 1 or cid > len(boxes):
            continue
        sl = boxes[cid - 1]
        if sl is None:
            continue
        ys, xs = sl
        radius = max(ys.stop - ys.start, xs.stop - xs.start) / 2.0
        pad = int(round(2 * radius)) + 2
        y0, y1 = max(0, ys.start - pad), min(height, ys.stop + pad)
        x0, x1 = max(0, xs.start - pad), min(width, xs.stop + pad)
        m = seg[y0:y1, x0:x1] == cid
        if not m.any():
            continue
        crop = t[y0:y1, x0:x1]
        er = ndimage.binary_erosion(m, iterations=ERODE_IT)
        if not er.any():
            er = m
        di = ndimage.binary_dilation(m, iterations=DILATE_IT)
        ring = di & ~m
        env = ~di
        f = np.concatenate([_stats(crop, m), _stats(crop, er),
                            _stats(crop, ring), _stats(crop, env),
                            np.asarray([m.sum()], dtype=np.float32)])
        ids.append(int(cid))
        feats.append(f)
    if not ids:
        return (np.zeros((0, 16 * img.shape[-1] + 1), dtype=np.float32),
                np.zeros(0, dtype=np.int64))
    return np.asarray(feats, dtype=np.float32), np.asarray(ids, dtype=np.int64)



def table_data(ds, classes, panel, norm="raw"):
    """CellTune's own exported per-cell table as the feature space.

    Used for the colorectal to GVHD pair, where the colorectal cohort ships no
    images, so the compartment features cannot be recomputed: the features are
    the per-cell marker means and the area of the table CellTune itself exports,
    with the expert labels of the depot joined by image and identifier.
    """
    g, markers = D.gold(ds, drop_unidentified=True)
    stem = {canonical(m): m for m in markers}
    cols = [stem[c] for c in panel if c in stem]
    if len(cols) != len(panel):
        raise RuntimeError("%s is missing shared channels" % ds)
    code = {c: i for i, c in enumerate(classes)}
    y = g["coarse_expert"].map(code)
    keep = y.notna().to_numpy()
    block = [g[c].to_numpy(np.float32) for c in cols]
    if norm == "log1p_z":
        # the benchmark's published representation: log1p then a per-domain
        # z-score on the channel block, as errctl.data.DomainNormaliser does.
        block = [np.log1p(np.clip(b, 0, None)) for b in block]
        block = [(b - b.mean()) / (b.std() if b.std() > 0 else 1.0) for b in block]
    block.append(g["Area"].to_numpy(np.float32))
    X = np.column_stack([b[keep] for b in block]).astype(np.float32)
    return X, y.to_numpy()[keep].astype(np.int64), g["image"].to_numpy(str)[keep]



def cohort_cache(ds, force=False):
    out = os.path.join(CACHE, "%s_features.npz" % ds)
    if os.path.exists(out) and not force:
        z = np.load(out, allow_pickle=False)
        return z["X"], z["y"], z["fov"]
    root = os.path.join(CS_ROOT, ds, "CellTypes")
    dirs = (os.path.join(root, "data", "images"), os.path.join(root, "cells"),
            os.path.join(root, "cells2labels"))
    if not os.path.isdir(dirs[0]):
        raise RuntimeError("no cached cohort at %s -- run scripts/r51_cellsighter.py "
                           "--stage prep first" % dirs[0])
    Xs, ys, fs = [], [], []
    t0 = time.time()
    for fn in sorted(os.listdir(dirs[0])):
        img = np.load(os.path.join(dirs[0], fn))["data"]
        seg = np.load(os.path.join(dirs[1], fn))["data"]
        c2l = np.load(os.path.join(dirs[2], fn))["data"]
        if c2l.max() < 0:
            continue
        F, ids = fov_features(img, seg, c2l)
        if len(ids) == 0:
            continue
        Xs.append(F)
        ys.append(c2l[ids].astype(np.int64))
        fs.extend([os.path.splitext(fn)[0]] * len(ids))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    fov = np.asarray(fs)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fov=fov)
    log("features %-16s %5d cells x %d features from %d fields in %.1f s"
        % (ds, X.shape[0], X.shape[1], len(set(fs)), time.time() - t0))
    return X, y, fov


def fit_predict(Xtr, ytr, tests, n_classes, threads):
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    log("fitting XGBoost and CatBoost on %d cells, %d classes, %d threads"
        % (Xtr.shape[0], n_classes, threads))
    xgb = XGBClassifier(n_estimators=N_EST, max_depth=DEPTH, learning_rate=LR,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        objective="multi:softprob", num_class=n_classes,
                        n_jobs=threads, random_state=SEED, verbosity=0)
    t0 = time.time()
    xgb.fit(Xtr, ytr)
    log("  xgboost fitted in %.1f s" % (time.time() - t0))
    cat = CatBoostClassifier(iterations=N_EST, depth=DEPTH, learning_rate=LR,
                             loss_function="MultiClass", verbose=0,
                             thread_count=threads, random_seed=SEED,
                             allow_writing_files=False)
    t0 = time.time()
    cat.fit(Xtr, ytr)
    log("  catboost fitted in %.1f s" % (time.time() - t0))
    if list(xgb.classes_) != list(cat.classes_):
        raise RuntimeError("model class orders differ")
    order = list(xgb.classes_)
    if order != list(range(n_classes)):
        raise RuntimeError("unexpected class codes %r" % order)
    out = []
    for X in tests:
        p = (xgb.predict_proba(X) + cat.predict_proba(X)) / 2.0
        out.append(np.asarray(p, dtype=np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("features", "fit", "all"))
    ap.add_argument("--hold-every", type=int, default=2)
    ap.add_argument("--cal-repeats", type=int, default=40)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--space", default="compartment",
                    choices=("compartment", "table"))
    ap.add_argument("--norm", default="raw", choices=("raw", "log1p_z"),
                    help="table space only: raw values or log1p plus z-score")
    ap.add_argument("--src", default="Breast_Jun0524")
    ap.add_argument("--tgt", default="CRC_Feb2623")
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    gs, ms = D.gold(args.src, drop_unidentified=True)
    gt, mt = D.gold(args.tgt, drop_unidentified=True)
    panel = R51.shared_panel(ms, mt)
    classes = R51.shared_classes(gs, gt)
    log("panel %d shared markers; %d shared coarse classes %s"
        % (len(panel), len(classes), classes))

    if args.space == "table":
        Xs, ys, fs = table_data(args.src, classes, panel, args.norm)
        Xt, yt, ft = table_data(args.tgt, classes, panel, args.norm)
        fovs = sorted(set(fs))
    else:
        Xs, ys, fs = cohort_cache(args.src, force=args.force_features)
        Xt, yt, ft = cohort_cache(args.tgt, force=args.force_features)
        if args.stage == "features":
            return
        fovs = sorted(os.listdir(os.path.join(paths.manual_dir(args.src), "Images")))
    hold = [f for f in fovs[::args.hold_every] if f in set(fs)]
    train = [f for f in fovs if f not in set(hold)]

    tr = np.isin(fs, train)
    ho = np.isin(fs, hold)
    log("train %d cells on %d fields; held-out %d cells on %d fields; target %d cells"
        % (tr.sum(), len(train), ho.sum(), len(hold), len(yt)))
    if len(np.unique(ys[tr])) != len(classes):
        raise RuntimeError("training split is missing classes")

    (sp, tp) = fit_predict(Xs[tr], ys[tr], [Xs[ho], Xt], len(classes), args.threads)
    np.savez_compressed(os.path.join(CACHE, "celltune_scores%s.npz" % args.suffix),
                        src_probs=sp, src_y=ys[ho], src_img=fs[ho],
                        tgt_probs=tp, tgt_y=yt, tgt_img=ft,
                        classes=np.asarray(classes), panel=np.asarray(panel),
                        hold=np.asarray(hold))

    R51.N_CAL_REPEATS = args.cal_repeats
    df = R51.analyse(sp, ys[ho], fs[ho], hold, tp, yt)
    df.to_csv(os.path.join(paths.RESULTS, "r51b_celltune%s.csv" % args.suffix),
              index=False)
    summ = R51.summarise(df)
    summ.to_csv(os.path.join(paths.RESULTS,
                             "r51b_celltune_summary%s.csv" % args.suffix), index=False)

    acc = float((np.argmax(sp, 1) == ys[ho]).mean())
    report = os.path.join(paths.WORK, "r51b_report%s.txt" % args.suffix)
    with open(report, "w") as fh:
        fh.write("R51b -- the CellTune classifier wrapped by the layer\n")
        fh.write("feature space: %s, normalisation %s\n" % (args.space, args.norm))
        fh.write("source %s -> target %s\n" % (args.src, args.tgt))
        if args.space == "table":
            fh.write("features: CellTune's own exported cell table, the per-cell "
                     "marker means and the area, expert labels joined by image and "
                     "identifier\n")
        else:
            fh.write("reimplemented from the CellTune documentation: mean, max, "
                     "variance and non-zero pixel count per channel over the cell, an "
                     "eroded centre, the surrounding ring and the environment; arcsinh "
                     "cofactor %.1f on per-field 99.9th-percentile scaled intensities; "
                     "XGBoost and CatBoost averaged (Pred_AVG)\n" % COFACTOR)
        fh.write("shared panel (%d): %s\n" % (len(panel), ", ".join(panel)))
        fh.write("shared coarse classes (%d): %s\n" % (len(classes), ", ".join(classes)))
        fh.write("xgboost+catboost, %d trees, depth %d, lr %.2f, cpu, threads %d, "
                 "calibration repeats %d, hold_every %d\n"
                 % (N_EST, DEPTH, LR, args.threads, args.cal_repeats, args.hold_every))
        fh.write("held-out fields (%d of %d): %s\n"
                 % (len(hold), len(fovs), ", ".join(hold)))
        fh.write("source classifier accuracy on the held-out cells: %.4f\n" % acc)
        fh.write("\n%s\n" % summ.to_string(index=False))
    log("wrote r51b_celltune%s.csv (%d rows) and r51b_report%s.txt"
        % (args.suffix, len(df), args.suffix))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
