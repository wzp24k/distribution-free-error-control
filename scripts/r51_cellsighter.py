"""R51 -- wrap a published, non-tabular classifier (CellSighter) in the layer.

Why
---
Every model wrapped so far (``logreg``, ``rf``, ``hgb``) is an off-the-shelf
tabular scorer fitted on per-cell marker means, so "wraps any cell-type
classifier" is an untested generalisation.  This script tests it on CellSighter
(Amitay et al., Nature Communications 2023), the supervised spatial-proteomics
classifier named in the Introduction: a ResNet-50 over multiplexed image crops.

What is reused verbatim
-----------------------
``tools/CellSighter`` is their published repository.  We import their
``model.Model``, ``data.utils.load_crops``, ``data.data.CellCropsDataset`` and
``data.transform.train_transform`` / ``val_transform``; we drive training with
their own ``train.train_epoch`` and ``train.define_sampler`` and evaluation with
their own ``eval.val_epoch``.  The layer is the paper's own
``errctl.conformal.SelectiveRiskController`` and the readouts come from
``scripts/r03_analyse.evaluate``, so neither the model nor the analysis drifts.

Deviations, all forced by a CPU-only host and all recorded
---------------------------------------------------------
* ``device="cpu"``; their code hardcodes ``"cuda"``.
* ``num_workers=0``; on Windows their dataset would pickle whole FOV arrays out
  to each worker.
* one cohort pair and a reduced epoch count instead of their five-cohort
  50-epoch ensemble.  ``--epochs`` records what was used.
* one network rather than their ensemble; the score is that network's softmax.

Arms
----
* ``exchangeable`` -- random split of the held-out cells into calibration/test.
* ``crossfov``     -- leave-one-held-out-field-out calibration.
* ``transfer``     -- the source network applied to the target cohort on the
  shared marker panel and the shared coarse classes, exactly as
  ``scripts/r02_run_experiments.py`` builds its transfer units.

Outputs: ``work/cellsighter/``, ``experiments/r51_cellsighter.csv``,
``experiments/r51_cellsighter_summary.csv`` and ``work/r51_report.txt``.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
CS_ROOT = os.path.join(ROOT, "tools", "CellSighter")
sys.path.insert(0, CS_ROOT)

from errctl import paths                    # noqa: E402
from errctl import data as D                # noqa: E402
from errctl.markerlogic import canonical    # noqa: E402
from errctl.labels import to_coarse         # noqa: E402

import train as cs_train                    # noqa: E402
import eval as cs_eval                      # noqa: E402
from model import Model                     # noqa: E402
from data.utils import load_crops           # noqa: E402
from data.data import CellCropsDataset      # noqa: E402
from data.transform import train_transform, val_transform  # noqa: E402

CS_DATA = os.path.join(paths.WORK, "cellsighter")
WEIGHTS = os.path.join(CS_DATA, "weights")
PRED = os.path.join(paths.WORK, "predictions_cellsighter")

SRC = "Melanoma_Oct2522"
TGT = "GVHD_Feb1323"
CROP_SIZE = 128
CROP_INPUT = 60
BATCH = 16
LR = 1e-3
SHIFT = 5
HOLD_EVERY = 3
N_CAL_REPEATS = 6
CAL_SUBSAMPLE = 2500
MIN_SRC_CLASS = 30
MIN_TGT_CLASS = 20
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
SEED = 0
PAD = True
SIZE_DATA = 400
N_THREADS = 8


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manual_labels(ds):
    """Labels.csv with the BOM stripped (CRC ships one, the others do not)."""
    p = os.path.join(paths.manual_dir(ds), "Tables", "Labels.csv")
    df = pd.read_csv(p, encoding="utf-8-sig")
    df = df.rename(columns={"fov": "image"})
    df["cellID"] = df["cellID"].astype(int)
    df["image"] = df["image"].astype(str)
    return df[["image", "cellID", "class"]]


def shared_panel(src_markers, tgt_markers):
    src = {canonical(m): m for m in src_markers}
    tgt = {canonical(m): m for m in tgt_markers}
    return sorted(set(src) & set(tgt))


def shared_classes(gs, gt):
    cs = gs["coarse_expert"].value_counts()
    ct = gt["coarse_expert"].value_counts()
    return sorted(c for c in ct.index
                  if ct[c] >= MIN_TGT_CLASS and cs.get(c, 0) >= MIN_SRC_CLASS)


def dataset_root(ds):
    return os.path.join(CS_DATA, ds)


def prep_cohort(ds, classes, panel):
    """Write the CellSighter-format dataset (their README's Data Preparation)."""
    root = dataset_root(ds)
    img_dir = os.path.join(root, "CellTypes", "data", "images")
    cell_dir = os.path.join(root, "CellTypes", "cells")
    lbl_dir = os.path.join(root, "CellTypes", "cells2labels")
    for d in (img_dir, cell_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)

    markers, _ = D.marker_table(ds)
    stem = {canonical(m): m for m in markers}
    missing = [c for c in panel if c not in stem]
    if missing:
        raise RuntimeError("%s is missing panel entries %r" % (ds, missing))
    cols = [stem[c] for c in panel]
    with open(os.path.join(root, "channels.txt"), "w") as fh:
        fh.write("\n".join(cols) + "\n")

    code = {c: i for i, c in enumerate(classes)}
    labels = manual_labels(ds)
    uq = sorted(set(labels["class"].astype(str)))
    cmap = dict(zip(uq, to_coarse(pd.Series(uq)).tolist()))
    lab_by_img = {img: g for img, g in labels.groupby("image")}

    fov_root = os.path.join(paths.manual_dir(ds), "Images")
    fovs = sorted(d for d in os.listdir(fov_root)
                  if os.path.isdir(os.path.join(fov_root, d)))
    keep, written = 0, 0
    for fov in fovs:
        d = os.path.join(fov_root, fov)
        files = os.listdir(d)
        seg_file = next((f for f in files if "segmentation" in f.lower()), None)
        if seg_file is None:
            continue
        planes, ok = [], True
        for c in cols:
            f = next((x for x in files
                      if os.path.splitext(x)[0].lower() == c.lower()), None)
            if f is None:
                ok = False
                break
            planes.append(tifffile.imread(os.path.join(d, f)))
        if not ok:
            continue
        seg = tifffile.imread(os.path.join(d, seg_file)).astype(np.int64)
        image = np.stack(planes, axis=-1)

        g = lab_by_img.get(fov)
        top = int(seg.max())
        if g is not None and len(g):
            top = max(top, int(g["cellID"].max()))
        c2l = np.full(top + 1, -1, dtype=np.int32)
        if g is not None:
            for cid, cls in zip(g["cellID"].to_numpy(),
                                g["class"].astype(str).to_numpy()):
                lbl = code.get(cmap.get(cls, "Unidentified"))
                if lbl is not None:
                    c2l[int(cid)] = lbl

        np.savez_compressed(os.path.join(img_dir, fov + ".npz"), data=image)
        np.savez_compressed(os.path.join(cell_dir, fov + ".npz"), data=seg)
        np.savez_compressed(os.path.join(lbl_dir, fov + ".npz"), data=c2l)
        keep += int((c2l > -1).sum())
        written += 1
    log("prep %-16s %2d/%2d FOVs written, %5d labelled cells, %d channels, %d classes"
        % (ds, written, len(fovs), keep, len(cols), len(classes)))
    return root


def train_model(ds, train_fovs, classes, panel, epochs, tag, size_data, threads):
    root = dataset_root(ds)
    channels_path = os.path.join(root, "channels.txt")
    out = os.path.join(WEIGHTS, tag)
    os.makedirs(out, exist_ok=True)
    cfg = {
        "crop_input_size": CROP_INPUT, "crop_size": CROP_SIZE, "root_dir": root,
        "train_set": list(train_fovs), "val_set": [], "num_classes": len(classes),
        "epoch_max": epochs, "lr": LR, "blacklist": [], "batch_size": BATCH,
        "num_workers": 0, "channels_path": channels_path, "weight_to_eval": "",
        "sample_batch": True, "to_pad": PAD, "aug": True,
        "size_data": SIZE_DATA,
        "hierarchy_match": {str(i): str(i) for i in range(len(classes))},
    }
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)

    train_crops, _ = load_crops(root, channels_path, CROP_SIZE, list(train_fovs),
                                [], PAD, blacklist_channels=[])
    train_crops = np.array([c for c in train_crops if c._label >= 0])
    if size_data:
        train_crops = np.array(cs_train.subsample_const_size(train_crops, size_data))
    sampler = cs_train.define_sampler(train_crops, cfg["hierarchy_match"])
    train_ds = CellCropsDataset(train_crops,
                                transform=train_transform(CROP_INPUT, SHIFT),
                                mask=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, num_workers=0,
                                         sampler=sampler, shuffle=False)

    torch.set_num_threads(threads)
    model = Model(len(panel) + 2, len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=out)
    log("train %s: %d crops, %d classes, %d epochs, batch %d, size_data %s"
        % (tag, len(train_crops), len(classes), epochs, BATCH, size_data))
    for i in range(epochs):
        t0 = time.time()
        cs_train.train_epoch(model, loader, opt, criterion, i, writer, device="cpu")
        torch.save(model.state_dict(), os.path.join(out, "weights_%d_count.pth" % i))
        log("  epoch %d/%d done in %.1f min" % (i + 1, epochs, (time.time() - t0) / 60.0))
    writer.close()
    return out


def load_model(ds, classes, panel, tag, epochs):
    out = os.path.join(WEIGHTS, tag)
    model = Model(len(panel) + 2, len(classes))
    weights = os.path.join(out, "weights_%d_count.pth" % (epochs - 1))
    if not os.path.exists(weights):
        raise RuntimeError("no weights at %s" % weights)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model


def score_fovs(model, ds, fovs):
    root = dataset_root(ds)
    channels_path = os.path.join(root, "channels.txt")
    _, val_crops = load_crops(root, channels_path, CROP_SIZE, [], list(fovs),
                              PAD, blacklist_channels=[])
    val_crops = np.array([c for c in val_crops if c._label >= 0])
    val_ds = CellCropsDataset(val_crops, transform=val_transform(CROP_INPUT),
                              mask=True)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=32, num_workers=0,
                                         shuffle=False)
    _, probs = cs_eval.val_epoch(model, loader, device="cpu")
    y = np.array([int(c._label) for c in val_crops])
    img = np.array([str(c._image_id) for c in val_crops])
    assert len(probs) == len(y), (len(probs), len(y))
    return probs.astype(np.float32), y, img


def build_units(src_y, src_img, hold_fovs, tgt_n, rng):
    """(arm, unit, repeat, calibration index, test index, side) per evaluation."""
    out = []
    idx = np.arange(len(src_y))
    rng.shuffle(idx)
    test_idx, cal_pool = idx[: len(idx) // 2], idx[len(idx) // 2:]
    for r in range(N_CAL_REPEATS):
        pick = rng.choice(cal_pool, size=min(CAL_SUBSAMPLE, len(cal_pool)),
                          replace=False)
        out.append(("exchangeable", "heldout", r, pick, test_idx, "source"))
    for fov in hold_fovs:
        te = np.nonzero(src_img == fov)[0]
        pool = np.nonzero(src_img != fov)[0]
        if len(te) < 50 or len(pool) < 50:
            continue
        for r in range(N_CAL_REPEATS):
            pick = rng.choice(pool, size=min(CAL_SUBSAMPLE, len(pool)), replace=False)
            out.append(("crossfov", fov, r, pick, te, "source"))
    for r in range(N_CAL_REPEATS):
        pick = rng.choice(np.arange(len(src_y)),
                          size=min(CAL_SUBSAMPLE, len(src_y)), replace=False)
        out.append(("transfer", "%s->%s" % (SRC, TGT), r, pick, None, "target"))
    return out


def scores(probs):
    return 1.0 - probs.max(1), probs.argmax(1)


def analyse(src_probs, src_y, src_img, hold_fovs, tgt_probs, tgt_y):
    from scipy.stats import ks_2samp
    r03 = _load("r03_analyse")
    s_src, p_src = scores(src_probs)
    s_tgt, p_tgt = scores(tgt_probs)
    e_src = (p_src != src_y).astype(float)
    e_tgt = (p_tgt != tgt_y).astype(float)
    rng = np.random.default_rng(SEED)
    units = build_units(src_y, src_img, hold_fovs, len(tgt_y), rng)
    rows = []
    for arm, unit, rep, ca, te, side in units:
        s_cal, e_cal, p_cal = s_src[ca], e_src[ca], p_src[ca]
        if side == "source":
            s_te, e_te, p_te = s_src[te], e_src[te], p_src[te]
        else:
            s_te, e_te, p_te = s_tgt, e_tgt, p_tgt
        for alpha in ALPHAS:
            res = r03.evaluate(s_cal, e_cal, s_te, e_te, p_cal, p_te,
                               alpha, DELTA)
            crc, orc = res["crc"], res["oracle"]
            acc_o = orc["accept_rate"]
            lam = crc["lambda"]
            acc_te = float((s_te <= lam).mean())
            acc_ca = float((s_cal <= lam).mean())
            rows.append({
                "arm": arm, "unit": unit, "repeat": rep, "alpha": alpha,
                "kind": side,
                "n_cal": int(len(ca)), "n_test": int(len(s_te)),
                "accept_cert": crc["accept_rate"],
                "accept_oracle": acc_o,
                "ratio": (crc["accept_rate"] / acc_o) if acc_o > 0 else np.nan,
                "risk_cert": crc["selective_risk"],
                "covered": crc["covered"],
                "empty": crc["empty"],
                "covered_nonvac": (np.nan if crc["empty"] else crc["covered"]),
                "lambda": lam,
                "accept_gap_lf": acc_te - acc_ca,
                "ks_scores": float(ks_2samp(s_cal, s_te).statistic),
                "acc_clf": float(1.0 - e_te.mean()),
            })
    return pd.DataFrame(rows)


def summarise(df):
    keep = df[df["kind"] == "source"]
    g = (keep.groupby(["arm", "alpha"])
         .agg(n_eval=("covered", "size"),
              coverage=("covered", "mean"),
              coverage_nonvac=("covered_nonvac", "mean"),
              accept_cert=("accept_cert", "mean"),
              accept_oracle=("accept_oracle", "mean"),
              ratio=("ratio", "mean"),
              accept_gap_lf=("accept_gap_lf", "mean"),
              ks_scores=("ks_scores", "mean"),
              acc_clf=("acc_clf", "mean"))
         .reset_index())
    tr = df[(df["kind"] == "target")]
    if len(tr):
        gt = (tr.groupby(["arm", "alpha"])
              .agg(n_eval=("covered", "size"),
                   coverage=("covered", "mean"),
                   coverage_nonvac=("covered_nonvac", "mean"),
                   accept_cert=("accept_cert", "mean"),
                   accept_oracle=("accept_oracle", "mean"),
                   ratio=("ratio", "mean"),
                   accept_gap_lf=("accept_gap_lf", "mean"),
                   ks_scores=("ks_scores", "mean"),
                   acc_clf=("acc_clf", "mean"))
              .reset_index())
        gt["arm"] = gt["arm"] + " (target)"
        g = pd.concat([g, gt], ignore_index=True)
    return g


def main():
    global SRC, TGT, N_CAL_REPEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("prep", "train", "score", "analyse", "all"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hold-every", type=int, default=HOLD_EVERY)
    ap.add_argument("--size-data", type=int, default=SIZE_DATA)
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--tgt", default=TGT)
    ap.add_argument("--cal-repeats", type=int, default=N_CAL_REPEATS)
    args = ap.parse_args()
    SRC, TGT, N_CAL_REPEATS = args.src, args.tgt, args.cal_repeats

    os.makedirs(PRED, exist_ok=True)
    gs, ms = D.gold(SRC, drop_unidentified=True)
    gt, mt = D.gold(TGT, drop_unidentified=True)
    panel = shared_panel(ms, mt)
    classes = shared_classes(gs, gt)
    log("panel %d shared markers; %d shared coarse classes %s"
        % (len(panel), len(classes), classes))
    tag = "%s_%dch%s" % (SRC, len(panel), args.suffix)

    if args.stage in ("prep", "all"):
        prep_cohort(SRC, classes, panel)
        prep_cohort(TGT, classes, panel)

    fov_root = os.path.join(paths.manual_dir(SRC), "Images")
    fovs = sorted(os.listdir(fov_root))
    hold = fovs[::args.hold_every]
    train = [f for f in fovs if f not in hold]

    if args.stage in ("train", "all"):
        train_model(SRC, train, classes, panel, args.epochs, tag,
                    args.size_data, args.threads)

    if args.stage in ("score", "analyse", "all"):
        torch.set_num_threads(args.threads)
        model = load_model(SRC, classes, panel, tag, args.epochs)
        log("scoring %d held-out fields and the target cohort" % len(hold))
        sp, sy, si = score_fovs(model, SRC, hold)
        tp, ty, ti = score_fovs(model, TGT, sorted(os.listdir(
            os.path.join(paths.manual_dir(TGT), "Images"))))
        np.savez_compressed(os.path.join(PRED,
                            "cellsighter_scores%s.npz" % args.suffix),
                            src_probs=sp, src_y=sy, src_img=si,
                            tgt_probs=tp, tgt_y=ty, tgt_img=ti,
                            classes=np.asarray(classes), panel=np.asarray(panel),
                            hold=np.asarray(hold), epochs=args.epochs)
        log("source held-out %d cells, target %d cells" % (len(sy), len(ty)))
        df = analyse(sp, sy, si, hold, tp, ty)
        df.to_csv(os.path.join(paths.RESULTS,
                               "r51_cellsighter%s.csv" % args.suffix), index=False)
        summ = summarise(df)
        summ.to_csv(os.path.join(paths.RESULTS,
                                 "r51_cellsighter_summary%s.csv" % args.suffix),
                    index=False)
        report = os.path.join(paths.WORK, "r51_report%s.txt" % args.suffix)
        with open(report, "w") as fh:
            fh.write("R51 -- CellSighter wrapped by the layer\n")
            fh.write("source %s -> target %s\n" % (SRC, TGT))
            fh.write("shared panel (%d): %s\n" % (len(panel), ", ".join(panel)))
            fh.write("shared coarse classes (%d): %s\n" % (len(classes), ", ".join(classes)))
            fh.write("epochs %d, batch %d, crop %d->%d, cpu, threads %d, "
                     "size_data %d, hold_every %d\n"
                     % (args.epochs, BATCH, CROP_SIZE, CROP_INPUT,
                        args.threads, args.size_data, args.hold_every))
            fh.write("held-out fields (%d of %d): %s\n" % (len(hold), len(fovs), ", ".join(hold)))
            fh.write("\n%s\n" % summ.to_string(index=False))
        log("wrote r51_cellsighter%s.csv (%d rows) and r51_report%s.txt"
            % (args.suffix, len(df), args.suffix))
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
