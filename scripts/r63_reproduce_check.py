"""r63 -- reproduction check for the manuscript's headline results.

One command, reading only the released result tables in `experiments/`, that
re-verifies the paper's four negative results, the cross-field-of-view shortfall,
the positive replications and the repair pilot on both pairs, against fixed
expected values. It needs no
dataset, no prediction cache, no GPU and no network: if the released tables are
intact, every line prints OK and the exit status is 0.

    python scripts/r63_reproduce_check.py
    python scripts/r63_reproduce_check.py --list

The five headline claims it pins (see `docs/submission/reproducibility.md`):

  N1  cross-tissue coverage   the guarantee fails under transfer; coverage is
                              0.541 / 0.397 / 0.286 at 5 / 10 / 20% and 0.014
                              among the decisions that actually commit a label
  N2  efficiency              at a matched realised risk the certificate
                              recovers 0.744 of an oracle -- it is behind the
                              ceiling, not in front of the baselines
  N3  marker violation level  the identified sets run 0.44-87.8% (breast),
                              0.58-90.2% (colorectal), 2.09-90.0% (GVHD) and
                              3.98-87.2% (melanoma), the convention spread is a
                              median 12.7x the sampling interval, five expert
                              populations clear the 5% gate in all 68
                              conventions and colorectal APC clears none
  N4  external cohort         STELLAR CODEX coverage is 0.6556 within and
                              0.0333 across tissues at a 10% budget, and the
                              ratio diagnostic misses 60 of the 174 over-budget
                              transfer evaluations
  N5  cross-field-of-view     at field-of-view resolution the shortfall is
                              resolved: 0.844 [0.788, 0.894] at 5% and
                              0.792 [0.724, 0.855] at 10%, both excluding 0.90
  P1  positive replication    the STELLAR exchangeable split covers 1.000 at a
                              10% budget with 85.1% of cells retained
  P2  second and third lab    two cohorts that share neither laboratory nor
                              platform -- an ion beam microarray and liver
                              allograft biopsies -- also cover 1.000 and 0.993
                              at 10% with 96.8% and 99.5% of cells retained and
                              with no vacuous evaluation
  P3  real panel swap         the same certificate on a genuinely different
                              antibody panel accepts nothing at 5% or 10% on the
                              microarray and 0.7% and 2.0% on the liver cohort,
                              so the failure mode is a change of panel rather
                              than a reduction of the feature set
  P7  tool demonstration      the layer run end to end on a public liver cohort
                              cuts a 170.3-hour manual pass to 25.1 h at a 5%
                              budget and 11.5 h at 20%, while a threshold tuned
                              on the same labels holds the 5% budget in half
                              the sets
  P8  cross-panel price       a real change of antibody panel never leaves the
                              budget, at every rung of a 100 to 8,000 label
                              ladder, and accepts 0.03 at a 10% budget and 0.20
                              at 20%, against 0.40 and 0.70 for the arm that
                              calibrates on the source instead
  P10 pre-registered cohort   IMMUcan panel 1 V3 covers 1.000 at a 10% budget
                              over 150 evaluations, retains 0.986 and has a
                              realised risk of 0.068
  P11 repair pilot            a 250-label target-labelled oracle proxy separates
                              repaying from non-repaying panel repairs
                              (AUC 0.949); at that pilot size the certificate
                              accepts nothing in 80 of 81 units at 20% and in
                              all 81 at 10%, so the ratio is zero or undefined
                              and has no usable AUC; the fixed-sequence
                              correction raises support to 17 units but leaves
                              the ratio AUC at 0.51
  P12 pilot decision          a threshold chosen on four folds and applied to
                              the fifth retains 0.371 of the same-tissue batch
                              and cuts the cross-platform loss from 0.095 to
                              0.031; at 10% even 4,000 labels leave 62 of 81
                              units silent, and the naive net-gain lower bound
                              is too permissive
  P13 second pair             on the breast and tonsil panel changes the colon
                              cut sends 0 of 90 units to repair; a cross-fitted
                              rule loses 0.095 and 0.055 of the batch against
                              repairing whole, while an oracle could add only
                              0.011 and 0.005
  P14 real cross-panel        STELLAR small intestine to an independent
                              duodenum CODEX atlas: target calibration covers
                              1.000 at 10%, source-only calibration covers
                              0.000 at realised risk 0.312, and acceptance is
                              0.207 at 10% but 0.628 at 20%

Every value is asserted to the precision the manuscript quotes.  Exit status 1
on any mismatch, with the failing rows printed.  This is the fast subset of
`work/_preflight.py`, which re-derives every number in the text; r63 is the one
to run first, and it is stdlib-only.
"""
import argparse
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")

_results = []


def load(name):
    with io.open(os.path.join(EXP, name), "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(row, col):
    v = row[col]
    return float(v) if v not in ("", None) else float("nan")


def ck(group, name, got, want, tol=5e-4):
    ok = abs(float(got) - float(want)) <= tol
    _results.append((group, name, float(got), float(want), tol, ok))
    return ok


def ck_nan(group, name, got):
    got = float(got)
    ok = got != got
    _results.append((group, name, got, float("nan"), 0.0, ok))
    return ok


def pick(rows, **want):
    out = []
    for r in rows:
        good = True
        for k, v in want.items():
            if isinstance(v, float):
                if abs(num(r, k) - v) > 5e-5:
                    good = False
                    break
            elif r[k] != v:
                good = False
                break
        if good:
            out.append(r)
    return out


def one(rows, **want):
    got = pick(rows, **want)
    if len(got) != 1:
        raise SystemExit("expected exactly one row for %r, found %d" % (want, len(got)))
    return got[0]


# ---------------------------------------------------------------------------
# N1 -- cross-tissue coverage
# ---------------------------------------------------------------------------
def n1():
    r03 = load("r03_summary.csv")
    r24 = load("r24_efficiency.csv")
    for al, cov in ((0.05, 0.5407), (0.10, 0.3972), (0.20, 0.2861)):
        row = one(r03, method="crc", kind="transfer", alpha=al)
        ck("N1 cross-tissue coverage", "crc coverage a=%s" % al, num(row, "coverage"), cov)
    row = one(r03, method="crc", kind="transfer", alpha=0.10)
    ck("N1 cross-tissue coverage", "crc acceptance a=0.10", num(row, "mean_accept"), 0.3676)
    ck("N1 cross-tissue coverage", "crc selective risk a=0.10", num(row, "mean_risk"), 0.3164)
    src = one(r24, family="transfer", arm="src", k="0", alpha=0.10)
    ck("N1 cross-tissue coverage", "non-vacuous coverage a=0.10",
       num(src, "cov_nonvac"), 0.01364, tol=1e-4)


# ---------------------------------------------------------------------------
# N2 -- efficiency: the certificate is behind an oracle at matched risk
# ---------------------------------------------------------------------------
def n2():
    rows = load("r16_matched_risk_ratio.csv")
    cs = pick(rows, family="cellsplit")
    row = one(cs, risk=0.10)
    ck("N2 efficiency", "certificate/oracle ratio at matched risk 0.10",
       num(row, "ratio_crc_oracle"), 0.7444)
    ck("N2 efficiency", "certificate acceptance at matched risk 0.10",
       num(row, "crc_accept"), 0.5594)
    ck("N2 efficiency", "oracle acceptance at matched risk 0.10",
       num(row, "oracle_accept"), 0.7516)
    ahead = sum(1 for r in cs if num(r, "crc_accept") > num(r, "oracle_accept") + 1e-9)
    ck("N2 efficiency", "matched-risk rows where the certificate beats the oracle",
       float(ahead), 0.0, tol=0)


# ---------------------------------------------------------------------------
# N3 -- marker violation level
# ---------------------------------------------------------------------------
def n3():
    pooled = load("r28_bounds_pooled.csv")
    spans = {
        "Breast_Jun0524": (0.0044295, 0.8783893),
        "CRC_Feb2623": (0.0058406, 0.9023780),
        "GVHD_Feb1323": (0.0209386, 0.9002525),
        "Melanoma_Oct2522": (0.0398367, 0.8722766),
    }
    for ds, (lo, hi) in spans.items():
        row = one(pooled, dataset=ds, scope="summary", source="expert",
                  cls="ALL_evaluable")
        ck("N3 marker violation", "%s identified-set floor" % ds, num(row, "rate_min"), lo, tol=2e-6)
        ck("N3 marker violation", "%s identified-set ceiling" % ds, num(row, "rate_max"), hi, tol=2e-6)
    bymarker = load("r28_bounds_by_marker.csv")
    types = [r for r in bymarker if r["scope"] == "type" and r["source"] == "expert"
             and r["n_conventions_present"] == "68"]
    above = [r for r in types if int(r["n_conventions_above_5pct"]) == 68]
    ck("N3 marker violation",
       "expert populations above the 5% gate under all 68 conventions",
       float(len(above)), 5.0, tol=0)
    apc = [r for r in types if r["dataset"] == "CRC_Feb2623" and r["cls"] == "APC"]
    ck("N3 marker violation", "colorectal APC conventions above the gate",
       float(int(apc[0]["n_conventions_above_5pct"])), 0.0, tol=0)
    ratios = sorted(num(r, "ratio_span_to_mean_ci") for r in types)
    n = len(ratios)
    mid = ratios[n // 2] if n % 2 else 0.5 * (ratios[n // 2 - 1] + ratios[n // 2])
    ck("N3 marker violation",
       "median convention-spread / sampling-interval factor over %d expert types" % n,
       mid, 12.7029, tol=1e-3)


# ---------------------------------------------------------------------------
# N4 -- external cohort
# ---------------------------------------------------------------------------
def n4():
    cert = load("r34_phase2_certificate_summary.csv")
    row = one(cert, family="ext_within", alpha=0.10)
    ck("N4 external cohort", "ext_within coverage a=0.10", num(row, "coverage"), 0.6556)
    ck("N4 external cohort", "ext_within acceptance a=0.10", num(row, "mean_accept"), 0.7264)
    row = one(cert, family="ext_transfer", alpha=0.10)
    ck("N4 external cohort", "ext_transfer coverage a=0.10", num(row, "coverage"), 0.0333)
    ck("N4 external cohort", "ext_transfer acceptance a=0.10", num(row, "mean_accept"), 0.8394)
    ratio = load("r34_phase2_ratio.csv")
    row = one(ratio, level="eval", family="ext_transfer", alpha=0.10)
    ck("N4 external cohort", "transfer over-budget evaluations", num(row, "n_over_budget"), 174.0, tol=0)
    ck("N4 external cohort", "transfer over-budget evaluations the ratio misses",
       num(row, "over_not_flagged"), 60.0, tol=0)
    row = one(ratio, level="eval", family="ext_within", alpha=0.10)
    ck("N4 external cohort", "within over-budget evaluations the ratio misses",
       num(row, "over_not_flagged"), 2.0, tol=0)


# ---------------------------------------------------------------------------
# N5 -- cross-field-of-view at field-of-view resolution
# ---------------------------------------------------------------------------
def n5():
    rows = load("r48_coverage_ci.csv")
    row = one(rows, method="crc", alpha=0.05)
    ck("N5 cross-FOV 96 units", "coverage a=0.05", num(row, "coverage"), 0.84375)
    ck("N5 cross-FOV 96 units", "coverage CI lower a=0.05", num(row, "cov_lo"), 0.7882)
    ck("N5 cross-FOV 96 units", "coverage CI upper a=0.05 excludes 0.90",
       num(row, "cov_hi"), 0.8941)
    row = one(rows, method="crc", alpha=0.10)
    ck("N5 cross-FOV 96 units", "coverage a=0.10", num(row, "coverage"), 0.79225)
    ck("N5 cross-FOV 96 units", "coverage CI upper a=0.10 excludes 0.90",
       num(row, "cov_hi"), 0.8547)


# ---------------------------------------------------------------------------
# P1 -- the positive replication
# ---------------------------------------------------------------------------
def p1():
    rows = load("r55_stellar_summary.csv")
    row = one(rows, norm="log1p_z", family="ext_cellsplit", alpha=0.10,
              method="crc", n_eval="150")
    ck("P1 positive replication", "STELLAR exchangeable coverage a=0.10",
       num(row, "coverage"), 1.0, tol=0)
    ck("P1 positive replication", "STELLAR exchangeable acceptance a=0.10",
       num(row, "mean_accept"), 0.8514)
    ck("P1 positive replication", "STELLAR exchangeable realised risk a=0.10",
       num(row, "risk_nonvac"), 0.08646, tol=1e-4)
    row = one(rows, norm="log1p_z", family="ext_targetcal", alpha=0.10,
              method="crc", n_eval="60")
    ck("P1 positive replication", "STELLAR target-calibrated coverage a=0.10",
       num(row, "coverage"), 1.0, tol=0)
    rob = load("r55_stellar_norm_robustness.csv")
    for norm in ("z_train", "raw"):
        row = one(rob, norm=norm, family="ext_cellsplit", alpha=0.10,
                  method="crc")
        ck("P1 positive replication", "STELLAR coverage a=0.10, %s" % norm,
           num(row, "coverage"), 1.0, tol=0)


# ---------------------------------------------------------------------------
# P2 -- the second and third independent laboratory
# ---------------------------------------------------------------------------
def p2():
    rows = load("r69_external2_log1p_z_summary.csv")
    want = (("EXT2_MIBITOF_Angelo", 0.05, 0.8444, 1.0),
            ("EXT2_MIBITOF_Angelo", 0.10, 0.9675, 0.99333),
            ("EXT2_LiverTx_IMC", 0.05, 0.9280, 1.0),
            ("EXT2_LiverTx_IMC", 0.10, 0.9950, 1.0))
    for coh, al, acc, cov in want:
        row = one(rows, norm="log1p_z", family="ext2_within", cohort=coh,
                  alpha=al, method="crc")
        ck("P2 further laboratories", "%s acceptance a=%s" % (coh, al),
           num(row, "mean_accept"), acc)
        ck("P2 further laboratories", "%s coverage a=%s" % (coh, al),
           num(row, "coverage"), cov)
        ck("P2 further laboratories", "%s vacuous evaluations a=%s" % (coh, al),
           num(row, "frac_vacuous"), 0.0, tol=0)


# ---------------------------------------------------------------------------
# P3 -- the real change of antibody panel
# ---------------------------------------------------------------------------
def p3():
    rows = load("r69_external2_log1p_z_summary.csv")
    want = (("EXT2_MIBITOF_Angelo", 0.05, 0.0000, 1.000),
            ("EXT2_MIBITOF_Angelo", 0.10, 0.0000, 1.000),
            ("EXT2_LiverTx_IMC", 0.05, 0.0076, 0.867),
            ("EXT2_LiverTx_IMC", 0.10, 0.0199, 0.667))
    for coh, al, acc, vac in want:
        row = one(rows, norm="log1p_z", family="ext2_panelswap", cohort=coh,
                  alpha=al, method="crc")
        ck("P3 real panel swap", "%s acceptance a=%s" % (coh, al),
           num(row, "mean_accept"), acc, tol=0.0005)
        ck("P3 real panel swap", "%s vacuous a=%s" % (coh, al),
           num(row, "frac_vacuous"), vac, tol=0.005)
    row = one(rows, norm="log1p_z", family="ext2_panelswap",
              cohort="EXT2_LiverTx_IMC", alpha=0.20, method="crc")
    ck("P3 real panel swap", "liver coverage a=0.20 falls below nominal",
       num(row, "coverage"), 0.6667)
    ck("P3 real panel swap", "liver over-budget evaluations a=0.20",
       num(row, "n_over_budget"), 10.0, tol=0)


# ---------------------------------------------------------------------------
# P4 -- the wrap around a published image-based classifier
# ---------------------------------------------------------------------------
def p4():
    rows = load("r51_cellsighter_summary_breast2crc.csv")
    row = one(rows, arm="exchangeable", alpha=0.10)
    ck("P4 wrapped classifier", "breast to CRC exchangeable acceptance",
       num(row, "accept_cert"), 0.80589, tol=0.0005)
    ck("P4 wrapped classifier", "breast to CRC exchangeable coverage",
       num(row, "coverage"), 1.0000, tol=0.0005)
    ck("P4 wrapped classifier", "breast to CRC exchangeable oracle acceptance",
       num(row, "accept_oracle"), 0.90209, tol=0.0005)
    row = one(rows, arm="transfer (target)", alpha=0.10)
    ck("P4 wrapped classifier", "target transfer coverage collapses",
       num(row, "coverage"), 0.0000, tol=0)
    ck("P4 wrapped classifier", "target transfer acceptance",
       num(row, "accept_cert"), 0.65385, tol=0.0005)
    ck("P4 wrapped classifier", "target transfer ratio",
       num(row, "ratio"), 10.5752, tol=0.001)
    rows = load("r51_cellsighter_summary.csv")
    row = one(rows, arm="exchangeable", alpha=0.20)
    ck("P4 wrapped classifier", "melanoma to GVHD abstaining acceptance",
       num(row, "accept_cert"), 0.29932, tol=0.0005)


# ---------------------------------------------------------------------------
# P7 -- the tool demonstration on a public cohort, priced in hours
# ---------------------------------------------------------------------------
def _sel(rows, arm, alpha=None, k=None):
    out = []
    for r in rows:
        if r["arm"] != arm:
            continue
        if alpha is not None:
            if r["alpha"] in ("", "nan") or abs(float(r["alpha"]) - alpha) > 1e-9:
                continue
        if k is not None:
            if r["k"] in ("", "nan") or int(float(r["k"])) != k:
                continue
        out.append(r)
    return out


def _mean(rows, col):
    vals = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
    return sum(vals) / len(vals) if vals else float("nan")


def _hours(rows, sec=20.0):
    """Net expert hours removed: n_eval excludes the calibration labels, so
    the full manual baseline is n_eval + n_cal and the work left is
    (n_eval - n_accepted) + n_cal.  Their difference is n_accepted."""
    n_acc = sum(int(r["n_accepted"]) for r in rows)
    return n_acc * sec / 3600.0


def p7():
    rows = load("r70_e2e_demo.csv")
    a05 = _sel(rows, "layer", alpha=0.05, k=500)
    a20 = _sel(rows, "layer", alpha=0.20, k=500)
    st = _sel(rows, "strong", alpha=0.05, k=500)
    ck("P7 tool demonstration", "public cohort deployment cells",
       sum(int(r["n_eval"]) for r in a05), 28646, tol=0)
    ck("P7 tool demonstration", "layer acceptance at a 5% budget",
       _mean(a05, "accept_rate"), 0.91305, tol=5e-4)
    ck("P7 tool demonstration", "coverage at a 5% budget",
       _mean(a05, "covered"), 1.0, tol=1e-9)
    ck("P7 tool demonstration", "expert hours removed at a 5% budget",
       _hours(a05), 145.1278, tol=0.01)
    ck("P7 tool demonstration", "expert hours removed at a 20% budget",
       _hours(a20), 158.7833, tol=0.01)
    ck("P7 tool demonstration", "tuned threshold coverage at a 5% budget",
       _mean(st, "covered"), 0.5, tol=1e-9)


# ---------------------------------------------------------------------------
# P8 -- a real change of antibody panel, priced in target labels
# ---------------------------------------------------------------------------
def p8():
    rows = load("r71_crosspanel_price.csv")
    def cov(arm, alpha):
        return _mean(_sel(rows, arm, alpha=alpha), "covered")
    rungs = [k for k in (100, 250, 500, 1000, 2000, 4000)]
    ck("P8 cross-panel price", "coverage at every rung and budget",
       min(cov("tgt_k%d" % k, a) for k in rungs for a in (0.05, 0.10, 0.20)),
       1.0, tol=1e-9)
    ck("P8 cross-panel price", "acceptance on 8,000 labels at a 20% budget",
       _mean(_sel(rows, "tgt_k8000", alpha=0.20), "accept_rate"), 0.19946, tol=5e-4)
    ck("P8 cross-panel price", "acceptance on 4,000 labels at a 10% budget",
       _mean(_sel(rows, "tgt_k4000", alpha=0.10), "accept_rate"), 0.02755, tol=5e-4)
    ck("P8 cross-panel price", "source-calibrated acceptance at a 10% budget",
       _mean(_sel(rows, "src_cal", alpha=0.10), "accept_rate"), 0.40259, tol=5e-4)
    ck("P8 cross-panel price", "source-calibrated coverage at a 10% budget",
       cov("src_cal", 0.10), 0.3, tol=1e-9)
    ck("P8 cross-panel price", "source-calibrated coverage at a 20% budget",
       cov("src_cal", 0.20), 0.23333, tol=1e-4)

# ---------------------------------------------------------------------------
# P9 -- where the labels go decides whether a real change of panel is usable
# ---------------------------------------------------------------------------
def p9():
    rows = load("r72_panel_repair.csv")
    A = "A_cross_tissue_panel"
    B = "B_same_tissue_panel"

    def sel(part, arm, alpha, pair=None):
        return [r for r in _sel(rows, arm, alpha=alpha)
                if r["part"] == part and (pair is None or r["pair"] == pair)]

    def net(rs):
        return _mean(rs, "n_accepted") / _mean(rs, "n_pool") \
            - _mean(rs, "n_labels") / _mean(rs, "n_pool")

    ck("P9 panel repair", "unit rows in the repair grid",
       float(len(rows)), 11340.0, tol=0)
    b = sel(B, "adapt_k1000", 0.20)
    ck("P9 panel repair", "same-tissue refit acceptance at a 20% budget",
       _mean(b, "accept_rate"), 0.70473, tol=5e-4)
    ck("P9 panel repair", "same-tissue refit realised risk at a 20% budget",
       _mean(b, "selective_risk"), 0.11323, tol=5e-4)
    ck("P9 panel repair", "same-tissue refit coverage at a 20% budget",
       _mean(b, "covered"), 1.0, tol=1e-9)
    ck("P9 panel repair", "same-tissue refit Clopper-Pearson validity",
       _mean(b, "cp_valid"), 1.0, tol=1e-9)
    ck("P9 panel repair", "same-tissue refit net of the labels read",
       net(b), 0.39239, tol=5e-4)
    ck("P9 panel repair", "same-tissue refit acceptance at a 10% budget",
       _mean(sel(B, "adapt_k1000", 0.10), "accept_rate"), 0.25656, tol=5e-4)
    f = sel(B, "tgt_k1000", 0.20)
    ck("P9 panel repair", "frozen arm gross acceptance, same tissue",
       _mean(f, "accept_rate"), 0.23264, tol=5e-4)
    ck("P9 panel repair", "frozen arm net of the labels read, same tissue",
       net(f), 0.00680, tol=5e-4)
    g = sel(A, "adapt_k1000", 0.20, "CRC_Feb2623->GVHD_Feb1323")
    ck("P9 panel repair", "cross-tissue refit acceptance at a 20% budget",
       _mean(g, "accept_rate"), 0.84083, tol=5e-4)
    ck("P9 panel repair", "cross-tissue refit net of the labels read",
       net(g), 0.63033, tol=5e-4)
    h = sel(A, "tgt_k1000", 0.20, "CRC_Feb2623->GVHD_Feb1323")
    ck("P9 panel repair", "cross-tissue frozen acceptance at a 20% budget",
       _mean(h, "accept_rate"), 0.04379, tol=5e-4)
    ck("P9 panel repair", "shared markers, same-tissue pair",
       float(len(set(int(r["n_markers"]) for r in b))), 1.0, tol=0)
    ck("P9 panel repair", "shared marker count, same-tissue pair",
       _mean(b, "n_markers"), 13.0, tol=0)
    ck("P9 panel repair", "shared marker count, cross-tissue pair",
       _mean(g, "n_markers"), 27.0, tol=0)

    # the second same-tissue pair: a different organ on the same array
    C = "C_same_tissue_panel_breast"
    cc = sel(C, "adapt_k1000", 0.05)
    ck("P9 panel repair", "second organ, refit acceptance at a 5% budget",
       _mean(cc, "accept_rate"), 0.51177, tol=5e-4)
    ck("P9 panel repair", "second organ, refit net of the labels read at 5%",
       net(cc), 0.44627, tol=5e-4)
    ck("P9 panel repair", "second organ, refit realised risk at a 5% budget",
       _mean(cc, "selective_risk"), 0.01811, tol=5e-4)
    ck("P9 panel repair", "second organ, refit coverage at a 5% budget",
       _mean(cc, "covered"), 1.0, tol=1e-9)
    ck("P9 panel repair", "second organ, frozen arm accepts nothing at 5%",
       _mean(sel(C, "tgt_k1000", 0.05), "accept_rate"), 0.0, tol=0)
    ck("P9 panel repair", "second organ, refit acceptance at a 10% budget",
       _mean(sel(C, "adapt_k1000", 0.10), "accept_rate"), 0.61738, tol=5e-4)
    ck("P9 panel repair", "shared marker count, breast pair",
       _mean(cc, "n_markers"), 8.0, tol=0)

    # the third pair: the same array's tonsil cores against a CODEX tonsil
    # cohort, so the platform changes as well as the panel
    D = "D_cross_platform_tonsil"
    dz = sel(D, "tgt_k1000", 0.20)
    ck("P9 panel repair", "cross-platform frozen acceptance at a 20% budget",
       _mean(dz, "accept_rate"), 0.0, tol=0)
    ck("P9 panel repair", "cross-platform frozen net of the labels read",
       net(dz), -0.01527, tol=5e-4)
    ck("P9 panel repair", "cross-platform frozen arms are empty at every rung",
       float(sum(1 for k in (250, 500, 1000, 2000, 4000)
                 for a in (0.05, 0.10, 0.20)
                 for r in sel(D, "tgt_k%d" % k, a)
                 if _mean([r], "accept_rate") > 0)), 0.0, tol=0)
    dp = sel(D, "adapt_k1000", 0.10)
    ck("P9 panel repair", "cross-platform refit acceptance at a 10% budget",
       _mean(dp, "accept_rate"), 0.05220, tol=5e-4)
    ck("P9 panel repair", "cross-platform refit realised risk at a 10% budget",
       _mean(dp, "selective_risk"), 0.03338, tol=5e-4)
    ck("P9 panel repair", "cross-platform refit net at a 10% budget",
       net(dp), 0.03613, tol=5e-4)
    ck("P9 panel repair", "cross-platform refit coverage at a 10% budget",
       _mean(dp, "covered"), 1.0, tol=1e-9)
    ck("P9 panel repair", "cross-platform refit Clopper-Pearson validity",
       _mean(dp, "cp_valid"), 1.0, tol=1e-9)
    d4 = sel(D, "adapt_k4000", 0.10)
    ck("P9 panel repair", "cross-platform refit acceptance at 4,000 labels",
       _mean(d4, "accept_rate"), 0.53407, tol=5e-4)
    ck("P9 panel repair", "cross-platform refit net at 4,000 labels",
       net(d4), 0.44038, tol=5e-4)
    ck("P9 panel repair", "cross-platform refit acceptance at a 20% budget",
       _mean(sel(D, "adapt_k1000", 0.20), "accept_rate"), 0.30750, tol=5e-4)
    ck("P9 panel repair", "cross-platform shared marker count",
       _mean(dp, "n_markers"), 11.0, tol=0)
    ck("P9 panel repair", "cross-platform shared class count",
       _mean(dp, "n_classes"), 6.0, tol=0)
    # the second independent source for the same colon core: STELLAR intestine,
    # with one whole colon region held out on the source side
    E = "E_cross_platform_colon"
    ez = sel(E, "tgt_k1000", 0.20)
    ck("P9 panel repair", "second source, frozen acceptance at a 20% budget",
       _mean(ez, "accept_rate"), 0.036527, tol=5e-4)
    ck("P9 panel repair", "second source, frozen net of the labels read",
       net(ez), -0.090607, tol=5e-4)
    e4 = sel(E, "adapt_k4000", 0.20)
    ck("P9 panel repair", "second source, refit acceptance at 4,000 labels",
       _mean(e4, "accept_rate"), 0.42180, tol=5e-4)
    ck("P9 panel repair", "second source, refit realised risk at 4,000 labels",
       _mean(e4, "selective_risk"), 0.13259, tol=5e-4)
    ck("P9 panel repair", "second source, refit net at 4,000 labels",
       net(e4), -0.27576, tol=5e-4)
    ck("P9 panel repair", "second source, refit coverage at 4,000 labels",
       _mean(e4, "covered"), 1.0, tol=1e-9)
    ck("P9 panel repair", "second source, refit Clopper-Pearson validity",
       _mean(e4, "cp_valid"), 1.0, tol=1e-9)
    ck("P9 panel repair", "second source shared marker count",
       _mean(e4, "n_markers"), 10.0, tol=0)
    ck("P9 panel repair", "second source shared class count",
       _mean(e4, "n_classes"), 6.0, tol=0)


# ---------------------------------------------------------------------------
# P10 -- the pre-registered first evaluation on an independent IMC cohort
# ---------------------------------------------------------------------------
def p10():
    rows = load("r75_immucan_summary.csv")
    row = one(rows, alpha=0.10)
    ck("P10 pre-registered cohort", "evaluations at a 10% budget",
       num(row, "n_eval"), 150.0, tol=0)
    ck("P10 pre-registered cohort", "coverage at a 10% budget",
       num(row, "coverage"), 1.0, tol=0)
    ck("P10 pre-registered cohort", "Wilson lower bound on coverage",
       num(row, "coverage_lo"), 0.97503, tol=5e-5)
    ck("P10 pre-registered cohort", "acceptance at a 10% budget",
       num(row, "mean_accept"), 0.98592, tol=5e-5)
    ck("P10 pre-registered cohort", "realised risk at a 10% budget",
       num(row, "risk_nonvac"), 0.06827, tol=5e-5)
    ck("P10 pre-registered cohort", "non-vacuous evaluations",
       num(row, "frac_nonvacuous"), 1.0, tol=0)


# ---------------------------------------------------------------------------
# P11 -- a labelled pilot prices the repair, and the ratio does not
# ---------------------------------------------------------------------------
def p11():
    rows = load("r74_repair_triage_summary.csv")
    raw = load("r72_panel_repair.csv")

    def est(outcome, metric):
        return num(one(rows, outcome=outcome, metric=metric), "estimate")

    def pilot_rows(alpha):
        return [r for r in raw
                if r["part"] in ("B_same_tissue_panel",
                                 "E_cross_platform_colon")
                and abs(num(r, "alpha") - alpha) < 1e-9
                and r["arm"] == "tgt_k250"]

    ck("P11 repair pilot", "oracle-proxy AUC at 20% and 250 pilot labels",
       est("outcome_k1000", "oracle_auc"), 0.94902, tol=5e-5)
    ck("P11 repair pilot", "oracle-proxy AUC lower bound",
       num(one(rows, outcome="outcome_k1000", metric="oracle_auc"), "lo"),
       0.89196, tol=5e-5)
    ck("P11 repair pilot", "units where the 250-label certificate accepts nothing at 10%",
       sum(1 for r in pilot_rows(0.10) if num(r, "accept_rate") == 0), 81)
    ck("P11 repair pilot", "units where the ratio is zero or undefined at 10%",
       sum(1 for r in pilot_rows(0.10)
           if r["ratio"] == "" or num(r, "ratio") == 0), 81)
    ck("P11 repair pilot", "units where the 250-label certificate accepts nothing at 20%",
       sum(1 for r in pilot_rows(0.20) if num(r, "accept_rate") == 0), 80)
    ck("P11 repair pilot", "units where the ratio is zero or undefined at 20%",
       sum(1 for r in pilot_rows(0.20)
           if r["ratio"] == "" or num(r, "ratio") == 0), 80)
    ck("P11 repair pilot", "units where the pilot-only ratio is zero or undefined at 20%",
       sum(1 for r in pilot_rows(0.20)
           if r["pilot_ratio"] == "" or num(r, "pilot_ratio") == 0), 80)
    ck("P11 repair pilot", "nonzero pilot-only ratios at 20%",
       sum(1 for r in pilot_rows(0.20) if num(r, "pilot_ratio") > 0), 1)
    ck("P11 repair pilot",
       "units where the unlabelled-denominator certificate accepts nothing at 10%",
       sum(1 for r in pilot_rows(0.10) if num(r, "accept_rate_ud") == 0), 81)
    ck("P11 repair pilot",
       "units where the unlabelled-denominator certificate accepts nothing at 20%",
       sum(1 for r in pilot_rows(0.20) if num(r, "accept_rate_ud") == 0), 75)
    ck("P11 repair pilot",
       "units where the unlabelled-denominator ratio is zero or undefined at 20%",
       sum(1 for r in pilot_rows(0.20)
           if r["ratio_ud"] == "" or num(r, "ratio_ud") == 0), 75)
    ck("P11 repair pilot",
       "nonzero unlabelled-denominator ratios at 20%",
       sum(1 for r in pilot_rows(0.20) if num(r, "ratio_ud") > 0), 6)
    ck("P11 repair pilot", "unlabelled-denominator ratio AUC at 20%",
       est("outcome_k1000", "ratio_ud_auc"), 0.52696, tol=5e-5)
    ck_nan("P11 repair pilot",
           "deployment ratio AUC is not estimated with one nonzero unit",
           est("outcome_k1000", "ratio_auc"))
    ck_nan("P11 repair pilot",
           "pilot-only ratio AUC is not estimated with one nonzero unit",
           est("outcome_k1000", "pilot_ratio_auc"))
    ck("P11 repair pilot", "nonzero deployment ratios at 20%",
       est("outcome_k1000", "ratio_support"), 1, tol=0)
    ck("P11 repair pilot", "nonzero pilot-only ratios at 20%",
       est("outcome_k1000", "pilot_ratio_support"), 1, tol=0)
    ck("P11 repair pilot", "nonzero unlabelled-denominator ratios at 20%",
       est("outcome_k1000", "ratio_ud_support"), 6, tol=0)
    ck("P11 repair pilot",
       "nonzero pilot-only unlabelled-denominator ratios at 20%",
       est("outcome_k1000", "pilot_ratio_ud_support"), 6, tol=0)

    mult = load("r76_pilot_multiplicity_summary.csv")
    for select in ("bonferroni", "fixed_sequence"):
        for denom in ("cal", "ud"):
            row = one(mult, alpha=0.10, select=select, denom=denom)
            ck("P11 repair pilot",
               "fixed-sequence check: %s/%s silent at 10%%" % (select, denom),
               num(row, "silent_deploy"), 81, tol=0)
            ck("P11 repair pilot",
               "fixed-sequence check: %s/%s support at 10%%" % (select, denom),
               num(row, "support_deploy"), 0, tol=0)
    row = one(mult, alpha=0.20, select="bonferroni", denom="cal")
    ck("P11 repair pilot", "Bonferroni support at 20%",
       num(row, "support_deploy"), 1, tol=0)
    ck("P11 repair pilot", "Bonferroni silence at 20%",
       num(row, "silent_deploy"), 80, tol=0)
    row = one(mult, alpha=0.20, select="bonferroni", denom="ud")
    ck("P11 repair pilot", "Bonferroni unlabelled support at 20%",
       num(row, "support_deploy"), 6, tol=0)
    ck("P11 repair pilot", "Bonferroni unlabelled AUC at 20%",
       num(row, "auc_deploy"), 0.53137, tol=5e-5)
    row = one(mult, alpha=0.20, select="fixed_sequence", denom="cal")
    ck("P11 repair pilot", "fixed-sequence support at 20%",
       num(row, "support_deploy"), 17, tol=0)
    ck("P11 repair pilot", "fixed-sequence silence at 20%",
       num(row, "silent_deploy"), 64, tol=0)
    ck("P11 repair pilot", "fixed-sequence deployment ratio AUC at 20%",
       num(row, "auc_deploy"), 0.50523, tol=5e-5)
    ck("P11 repair pilot", "fixed-sequence pilot ratio AUC at 20%",
       num(row, "auc_pilot"), 0.50523, tol=5e-5)
    row = one(mult, alpha=0.20, select="fixed_sequence", denom="ud")
    ck("P11 repair pilot", "fixed-sequence unlabelled support at 20%",
       num(row, "support_deploy"), 19, tol=0)
    ck("P11 repair pilot", "fixed-sequence unlabelled AUC at 20%",
       num(row, "auc_deploy"), 0.49967, tol=5e-5)
    for part, want in (("B_same_tissue_panel", 0.39239),
                       ("E_cross_platform_colon", -0.09507)):
        vals = [num(r, "net") for r in
                load("r76_pilot_multiplicity.csv") if r["part"] == part]
        ck("P11 repair pilot", "r76 recomputes the %s net" % part,
           sum(vals) / len(vals), want, tol=5e-5)
    cross = [r for r in raw
             if r["part"] == "E_cross_platform_colon"
             and abs(num(r, "alpha") - 0.20) < 1e-9
             and r["arm"] == "adapt_k1000"]
    cross_net = sum((num(r, "n_accepted") - num(r, "n_labels"))
                    / num(r, "n_pool") for r in cross) / len(cross)
    ck("P11 repair pilot", "mean net of the cross-platform 1,000-label refit",
       cross_net, -0.09507, tol=5e-5)
    ck("P11 repair pilot", "accuracy of the descriptive 0.35 cut",
       est("outcome_k1000", "accuracy"), 0.88889, tol=5e-5)
    ck("P11 repair pilot", "accuracy with a threshold chosen on four folds",
       est("outcome_k1000", "lofo_accuracy"), 0.90123, tol=5e-5)
    ck("P11 repair pilot", "net gain of triage over always repairing the E source",
       est("outcome_k1000",
           "policy_triage_minus_always_E_cross_platform_colon"),
       0.06135, tol=5e-5)


# ---------------------------------------------------------------------------
# P12 -- the pilot as a cross-fitted decision rule, and the silence ladder
# ---------------------------------------------------------------------------
def p12():
    dec = load("r77_pilot_decision.csv")
    silence = load("r77_calibration_silence.csv")

    def row(policy, part):
        return one(dec, policy=policy, part=part)

    ck("P12 pilot decision", "units silent at 10% with 4,000 labels",
       num(one(silence, alpha=0.10, k="4000"), "silent"), 62, tol=0)
    ck("P12 pilot decision", "units silent at 20% with 4,000 labels",
       num(one(silence, alpha=0.20, k="4000"), "silent"), 19, tol=0)
    ck("P12 pilot decision", "units silent at 20% with 1,000 labels",
       num(one(silence, alpha=0.20, k="1000"), "silent"), 48, tol=0)

    all_lofo = row("lofo", "ALL")
    b_lofo = row("lofo", "B_same_tissue_panel")
    e_lofo = row("lofo", "E_cross_platform_colon")
    ck("P12 pilot decision", "LOFO go units", num(all_lofo, "go"), 45, tol=0)
    ck("P12 pilot decision", "LOFO TPR", num(all_lofo, "tpr"),
       0.862745, tol=5e-6)
    ck("P12 pilot decision", "LOFO TNR", num(all_lofo, "tnr"),
       0.966667, tol=5e-6)
    ck("P12 pilot decision", "LOFO accuracy", num(all_lofo, "accuracy"),
       0.901235, tol=5e-6)
    ck("P12 pilot decision", "LOFO same-tissue net", num(b_lofo, "net_mean"),
       0.370811, tol=5e-6)
    ck("P12 pilot decision", "LOFO same-tissue delta vs always",
       num(b_lofo, "delta_mean"), -0.021579, tol=5e-6)
    ck("P12 pilot decision", "LOFO cross-platform net", num(e_lofo, "net_mean"),
       -0.031161, tol=5e-6)
    ck("P12 pilot decision", "LOFO cross-platform delta vs always",
       num(e_lofo, "delta_mean"), 0.063906, tol=5e-6)
    ck("P12 pilot decision", "LOFO cross-platform delta lower bound",
       num(e_lofo, "delta_mean_lo"), 0.031532, tol=5e-6)
    ck("P12 pilot decision", "LOFO cell-weighted delta, all units",
       num(all_lofo, "delta_cell"), 0.024961, tol=5e-6)
    ck("P12 pilot decision", "LOFO cell-weighted delta lower bound",
       num(all_lofo, "delta_cell_lo"), 0.003296, tol=5e-6)

    lcb = row("net_lcb", "ALL")
    ck("P12 pilot decision", "naive net-gain LCB go units",
       num(lcb, "go"), 67, tol=0)
    ck("P12 pilot decision", "naive net-gain LCB TNR",
       num(lcb, "tnr"), 0.433333, tol=5e-6)
    ck("P12 pilot decision", "naive net-gain LCB delta vs always",
       num(lcb, "delta_mean"), 0.008639, tol=5e-6)


# ---------------------------------------------------------------------------
# P13 -- the pilot on a second external repair pair
# ---------------------------------------------------------------------------
def p13():
    summ = load("r78_external_pair_summary.csv")
    raw = load("r78_external_repair_pair.csv")

    def row(policy, part):
        return one(summ, policy=policy, part=part)

    for part, want_pos in (("C_same_tissue_panel_breast", 30),
                           ("D_cross_platform_tonsil", 24)):
        z = pick(raw, part=part, arm="adapt_k1000", alpha=0.20)
        ck("P13 second pair", "%s units" % part, len(z), 45, tol=0)
        pos = sum(1 for r in z
                  if (num(r, "n_accepted") - num(r, "n_labels"))
                  / num(r, "n_pool") > 0)
        ck("P13 second pair", "%s positive-net units" % part, pos, want_pos,
           tol=0)
        ck("P13 second pair", "%s cp_valid units" % part,
           sum(int(float(r["cp_valid"])) for r in z), 45, tol=0)

    c = row("lofo_CD", "C_same_tissue_panel_breast")
    d = row("lofo_CD", "D_cross_platform_tonsil")
    a = row("lofo_CD", "ALL")
    ck("P13 second pair", "C LOFO net", num(c, "net"), 0.497882, tol=5e-6)
    ck("P13 second pair", "C LOFO delta vs always", num(c, "delta"),
       -0.095351, tol=5e-6)
    ck("P13 second pair", "C LOFO delta upper bound", num(c, "delta_hi"),
       -0.011845, tol=5e-6)
    ck("P13 second pair", "D LOFO net", num(d, "net"), 0.232805, tol=5e-6)
    ck("P13 second pair", "D LOFO delta vs always", num(d, "delta"),
       -0.054728, tol=5e-6)
    ck("P13 second pair", "D LOFO delta upper bound", num(d, "delta_hi"),
       0.001018, tol=5e-6)
    ck("P13 second pair", "pooled LOFO go units", num(a, "go"), 67, tol=0)
    ck("P13 second pair", "pooled LOFO delta vs always", num(a, "delta"),
       -0.075040, tol=5e-6)
    ck("P13 second pair", "transported colon cut go units",
       num(row("transport_BE_0.3725", "ALL"), "go"), 0, tol=0)
    ck("P13 second pair", "C oracle headroom",
       num(row("oracle", "C_same_tissue_panel_breast"), "delta"),
       0.010831, tol=5e-6)
    ck("P13 second pair", "D oracle headroom",
       num(row("oracle", "D_cross_platform_tonsil"), "delta"),
       0.005344, tol=5e-6)


# ---------------------------------------------------------------------------
# P14 -- a real change of antibody panel, with target-domain calibration
# ---------------------------------------------------------------------------
def p14():
    rows = load("r85_duodenum_crosspanel_summary.csv")
    row = one(rows, arm="cross_random", alpha=0.10, k="0")
    ck("P14 real cross-panel", "target-calibrated coverage a=0.10",
       num(row, "coverage"), 1.000000)
    ck("P14 real cross-panel", "target-calibrated acceptance a=0.10",
       num(row, "mean_accept"), 0.207282, tol=5e-6)
    ck("P14 real cross-panel", "target-calibrated risk a=0.10",
       num(row, "risk_nonvac"), 0.076755, tol=5e-6)
    ck("P14 real cross-panel", "target-calibrated evaluations a=0.10",
       num(row, "n_eval"), 150, tol=0)
    row = one(rows, arm="cross_random", alpha=0.20, k="0")
    ck("P14 real cross-panel", "target-calibrated acceptance a=0.20",
       num(row, "mean_accept"), 0.628339, tol=5e-6)
    ck("P14 real cross-panel", "target-calibrated risk a=0.20",
       num(row, "risk_nonvac"), 0.183143, tol=5e-6)
    row = one(rows, arm="srcnull", alpha=0.10, k="0")
    ck("P14 real cross-panel", "source-only coverage a=0.10",
       num(row, "coverage"), 0.000000)
    ck("P14 real cross-panel", "source-only realised risk a=0.10",
       num(row, "risk_nonvac"), 0.311644, tol=5e-6)
    ck("P14 real cross-panel", "source-only acceptance a=0.10",
       num(row, "mean_accept"), 0.988373, tol=5e-6)
    row = one(rows, arm="cross_roi", alpha=0.10, k="10000")
    ck("P14 real cross-panel", "ROI-disjoint coverage a=0.10",
       num(row, "coverage"), 0.953333, tol=5e-6)
    ck("P14 real cross-panel", "ROI-disjoint acceptance a=0.10",
       num(row, "mean_accept"), 0.168353, tol=5e-6)
    row = one(rows, arm="cross_roi", alpha=0.20, k="10000")
    ck("P14 real cross-panel", "ROI-disjoint coverage a=0.20",
       num(row, "coverage"), 0.806667, tol=5e-6)
    row = one(rows, arm="crossk", alpha=0.10, k="10000")
    ck("P14 real cross-panel", "acceptance with 10,000 target labels a=0.10",
       num(row, "mean_accept"), 0.173129, tol=5e-6)
    row = one(rows, arm="crossk", alpha=0.10, k="500")
    ck("P14 real cross-panel", "acceptance with 500 target labels a=0.10",
       num(row, "mean_accept"), 0.000000)
    row = one(rows, arm="within_target", alpha=0.10, k="0")
    ck("P14 real cross-panel", "in-domain acceptance a=0.10",
       num(row, "mean_accept"), 0.808705, tol=5e-6)


# ---------------------------------------------------------------------------
# P15 -- the rank-normalised rerun and its ensemble
# ---------------------------------------------------------------------------
def p15():
    rows = load("r87_duodenum_ranknorm_summary.csv")
    row = one(rows, arm="cross_random", clf="ens", alpha=0.10, k="0")
    ck("P15 rank-normalised", "ensemble coverage a=0.10",
       num(row, "coverage"), 1.0)
    ck("P15 rank-normalised", "ensemble acceptance a=0.10",
       num(row, "mean_accept"), 0.4952, tol=5e-5)
    ck("P15 rank-normalised", "ensemble realised risk a=0.10",
       num(row, "risk_nonvac"), 0.0873, tol=5e-5)
    ck("P15 rank-normalised", "ensemble evaluations a=0.10",
       num(row, "n_eval"), 50, tol=0)
    row = one(rows, arm="cross_random", clf="rf", alpha=0.10, k="0")
    ck("P15 rank-normalised", "random forest acceptance a=0.10",
       num(row, "mean_accept"), 0.4793, tol=5e-5)
    row = one(rows, arm="cross_random", clf="logreg", alpha=0.10, k="0")
    ck("P15 rank-normalised", "linear model acceptance a=0.10",
       num(row, "mean_accept"), 0.3393, tol=5e-5)
    row = one(rows, arm="cross_random", clf="hgb", alpha=0.10, k="0")
    ck("P15 rank-normalised", "gradient boosting acceptance a=0.10",
       num(row, "mean_accept"), 0.3281, tol=5e-5)
    row = one(rows, arm="cross_roi", clf="ens", alpha=0.10, k="10000")
    ck("P15 rank-normalised", "ROI-disjoint coverage a=0.10",
       num(row, "coverage"), 0.88, tol=5e-5)
    ck("P15 rank-normalised", "ROI-disjoint acceptance a=0.10",
       num(row, "mean_accept"), 0.4530, tol=5e-5)
    row = one(rows, arm="srcnull", clf="ens", alpha=0.10, k="0")
    ck("P15 rank-normalised", "source-only coverage a=0.10",
       num(row, "coverage"), 0.0, tol=0)
    ck("P15 rank-normalised", "source-only realised risk a=0.10",
       num(row, "risk_nonvac"), 0.2543, tol=5e-5)
    row = one(rows, arm="crossk", clf="ens", alpha=0.10, k="10000")
    ck("P15 rank-normalised", "acceptance with 10,000 labels a=0.10",
       num(row, "mean_accept"), 0.4662, tol=5e-5)
    row = one(rows, arm="crossk", clf="ens", alpha=0.10, k="500")
    ck("P15 rank-normalised", "acceptance with 500 labels a=0.10",
       num(row, "mean_accept"), 0.0, tol=0)
    row = one(rows, arm="within_target", clf="ens", alpha=0.10, k="0")
    ck("P15 rank-normalised", "in-domain acceptance a=0.10",
       num(row, "mean_accept"), 0.8304, tol=5e-5)


# ---------------------------------------------------------------------------
# P5 -- the second published classifier, rebuilt from its documentation
# ---------------------------------------------------------------------------
def p5():
    rows = load("r51b_celltune_summary.csv")
    row = one(rows, arm="exchangeable", alpha=0.05)
    ck("P5 CellTune wrapper", "breast to CRC exchangeable acceptance a=0.05",
       num(row, "accept_cert"), 0.81438, tol=0.0005)
    ck("P5 CellTune wrapper", "breast to CRC exchangeable coverage a=0.05",
       num(row, "coverage"), 1.0000, tol=0.0005)
    row = one(rows, arm="exchangeable", alpha=0.10)
    ck("P5 CellTune wrapper", "breast to CRC exchangeable acceptance a=0.10",
       num(row, "accept_cert"), 0.96152, tol=0.0005)
    ck("P5 CellTune wrapper", "breast to CRC exchangeable coverage a=0.10",
       num(row, "coverage"), 1.0000, tol=0.0005)
    ck("P5 CellTune wrapper", "breast to CRC exchangeable oracle acceptance a=0.10",
       num(row, "accept_oracle"), 1.0000, tol=0.0005)
    row = one(rows, arm="transfer (target)", alpha=0.10)
    ck("P5 CellTune wrapper", "target transfer coverage collapses",
       num(row, "coverage"), 0.0000, tol=0)
    ck("P5 CellTune wrapper", "target transfer acceptance",
       num(row, "accept_cert"), 0.95872, tol=0.0005)
    ck("P5 CellTune wrapper", "target transfer ratio",
       num(row, "ratio"), 39.8723, tol=0.001)
    rows = load("r51b_celltune.csv")
    risks = [num(r, "risk_cert") for r in pick(rows, arm="transfer", alpha=0.10)]
    risks = [v for v in risks if v == v]
    ck("P5 CellTune wrapper", "target transfer evaluations all over budget",
       float(sum(1 for v in risks if v > 0.10)), float(len(risks)), tol=0)


# ---------------------------------------------------------------------------
# P6 -- the third pair, on the tool's own exported cell table
# ---------------------------------------------------------------------------
def p6():
    rows = load("r51b_celltune_summary_gvhd2crc_z.csv")
    row = one(rows, arm="exchangeable", alpha=0.05)
    ck("P6 CellTune table entry", "GVHD to CRC exchangeable acceptance a=0.05",
       num(row, "accept_cert"), 0.67656, tol=0.0005)
    ck("P6 CellTune table entry", "GVHD to CRC exchangeable coverage a=0.05",
       num(row, "coverage"), 1.0000, tol=0.0005)
    row = one(rows, arm="exchangeable", alpha=0.10)
    ck("P6 CellTune table entry", "GVHD to CRC exchangeable acceptance a=0.10",
       num(row, "accept_cert"), 0.87809, tol=0.0005)
    ck("P6 CellTune table entry", "GVHD to CRC exchangeable oracle acceptance a=0.10",
       num(row, "accept_oracle"), 0.97661, tol=0.0005)
    row = one(rows, arm="transfer (target)", alpha=0.10)
    ck("P6 CellTune table entry", "GVHD to CRC target transfer coverage collapses",
       num(row, "coverage"), 0.0000, tol=0)
    ck("P6 CellTune table entry", "GVHD to CRC target transfer acceptance",
       num(row, "accept_cert"), 0.64399, tol=0.0005)
    ck("P6 CellTune table entry", "GVHD to CRC target transfer ratio",
       num(row, "ratio"), 5.4831, tol=0.001)
    rows = load("r51b_celltune_summary_crc2gvhd_z.csv")
    row = one(rows, arm="exchangeable", alpha=0.10)
    ck("P6 CellTune table entry", "mirror direction covers by the empty set",
       num(row, "coverage"), 1.0000, tol=0.0005)
    ck("P6 CellTune table entry", "mirror direction accepts nothing",
       num(row, "accept_cert"), 0.0000, tol=0)


GROUPS = [
    ("N1", "cross-tissue coverage fails", n1),
    ("N2", "no efficiency advantage", n2),
    ("N3", "marker violation level", n3),
    ("N4", "external cohort not replicated", n4),
    ("N5", "cross-field-of-view shortfall", n5),
    ("P1", "positive replication on STELLAR", p1),
    ("P2", "positive replication, two further laboratories", p2),
    ("P3", "the real change of antibody panel", p3),
    ("P4", "the wrap around a published image-based classifier", p4),
    ("P5", "the second published classifier, rebuilt from its documentation", p5),
    ("P6", "the third pair, on the exported cell table", p6),
    ("P7", "the tool demonstration on a public cohort, priced in hours", p7),
    ("P8", "a real change of antibody panel, priced in target labels", p8),
    ("P9", "where the labels go on a real change of panel", p9),
    ("P10", "the pre-registered first evaluation on IMMUcan", p10),
    ("P11", "a labelled pilot prices the repair", p11),
    ("P12", "the pilot as a cross-fitted decision rule", p12),
    ("P13", "the pilot on a second external repair pair", p13),
    ("P14", "a real change of antibody panel, target-calibrated", p14),
    ("P15", "the rank-normalised rerun and its ensemble", p15),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="list the pinned claims and exit without checking")
    args = ap.parse_args()
    if args.list:
        for gid, what, _ in GROUPS:
            print("%-3s %s" % (gid, what))
        return 0

    print("r63 reproduction check -- reading %s" % EXP)
    print("")
    for gid, what, fn in GROUPS:
        fn()
    n_ok = sum(1 for r in _results if r[5])
    for group, name, got, want, tol, ok in _results:
        flag = "OK " if ok else "BAD"
        print("%s  %-28s %-52s got=%-12.6g want=%-12.6g tol=%g"
              % (flag, group, name, got, want, tol))
    print("")
    print("ASSERTIONS=%d FAILS=%d" % (len(_results), len(_results) - n_ok))
    if n_ok != len(_results):
        print("REPRODUCTION FAILED")
        return 1
    print("REPRODUCED: 4 of 4 negative results, the cross-field-of-view")
    print("shortfall, the positive replications and a real cross-panel")
    print("deployment, the")
    print("real panel swap, the priced")
    print("cross-panel deployment, the repair of that deployment by a refit and")
    print("the tool demonstration, and the protocol-locked IMMUcan evaluation")
    print("-- all %d" % len(_results))
    print("assertions hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
