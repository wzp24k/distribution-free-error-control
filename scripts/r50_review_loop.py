"""R50 -- the review budget and the label price of the repair, in the same units.

The certificate is a review budget: at a nominal budget alpha it routes a
fraction of the deployment pool to a human and commits to the rest.  Making it
valid on a deployment whose distribution has moved costs target-domain labels
(Table 5).  This script puts the two numbers side by side in the units a
laboratory plans in -- cells per unit of the deployment pool -- so that the
review the certificate already generates can be compared with the labelling the
repair requires, and so that the case in which the repair is free in throughput
can be separated from the case in which it is not.

Every number is read from a released table and none is recomputed:

  experiments/r16_review_budget.csv   the certificate's acceptance, its review
                                      fraction and the size of the evaluation
                                      pool, by split family and budget
                                      (source: scripts/r16_review_budget.py)
  experiments/r24_efficiency.csv      the target-label ladder by split family
                                      (source: scripts/r24_efficiency.py), the
                                      table behind Table 5 and Figure 8

The repair price is the smallest target-label count at which the ladder restores
nominal coverage (coverage over all evaluations, vacuous ones included, at or
above 0.90) *without* losing the source-only acceptance.  That definition is the
one a practitioner cares about: it is the count at which the repair is free in
throughput rather than merely valid.  Where no count in the ladder meets it, the
family is reported as unrepaired by calibration and the best acceptance the
ladder reaches is reported instead.  The unit count is carried through, because
the ladder drops units whose pool is too small to supply k labels, so a large-k
row rests on fewer units than the k = 0 row.

Outputs
  experiments/r50_review_loop.csv
  work/r50_report.txt
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl import paths

NOMINAL = 0.90
ALPHAS = (0.05, 0.10, 0.20)
FAMILIES = ("cellsplit", "within", "transfer")
LABEL = {
    "cellsplit": "random split",
    "within": "cross-field-of-view",
    "transfer": "cross-tissue transfer",
}
TAG = "r50_review_loop"


def load_certificate():
    p = os.path.join(paths.RESULTS, "r16_review_budget.csv")
    d = pd.read_csv(p)
    d = d[d["method"] == "crc"].copy()
    d["alpha"] = d["alpha"].astype(float).round(6)
    d = d[d["alpha"].isin(ALPHAS)]
    assert len(d) == len(FAMILIES) * len(ALPHAS), (len(d), "certificate rows")
    # cells per evaluation is the unit's deployment pool: each unit's pool is
    # reused across classifiers and calibration repeats, so the quotient is the
    # pool a unit is scored on.  For the random split it is 19,669 / 19 exactly.
    d["pool_cells"] = d["eval_cells_tested"] / d["n_eval"]
    return d.rename(columns={
        "mean_accept": "accept_src",
        "coverage": "cov_all_src",
        "coverage_nonvac": "cov_nonvac_src",
        "frac_nonvacuous": "nvf_src",
        "n_units": "n_units_src",
    })


def load_ladder():
    p = os.path.join(paths.RESULTS, "r24_efficiency.csv")
    d = pd.read_csv(p)
    d = d[d["arm"] == "tgt_k"].copy()
    d["alpha"] = d["alpha"].astype(float).round(6)
    d["k"] = d["k"].astype(int)
    d = d[d["alpha"].isin(ALPHAS)]
    dup = d.duplicated(subset=["family", "alpha", "k"])
    assert not dup.any(), ("ladder key is not unique", d[dup])
    return d.sort_values(["family", "alpha", "k"])


def repair_price(ladder, fam, alpha, accept_src):
    g = ladder[(ladder["family"] == fam) & np.isclose(ladder["alpha"], alpha)]
    g = g.sort_values("k")
    assert len(g), (fam, alpha)
    ok = g[(g["cov_all"] >= NOMINAL) & (g["accept_mean"] >= accept_src)]
    if len(ok):
        r = ok.iloc[0]
        return dict(k_repair=int(r["k"]), accept_repair=float(r["accept_mean"]),
                    cov_all_repair=float(r["cov_all"]),
                    cov_nonvac_repair=float(r["cov_nonvac"]),
                    nvf_repair=float(r["nonvac_frac"]),
                    n_units_repair=int(r["n_units"]), throughput_neutral=True,
                    best_k=int(g["k"].iloc[int(np.argmax(g["accept_mean"].values))]),
                    best_accept=float(g["accept_mean"].max()))
    i = int(np.argmax(g["accept_mean"].values))
    return dict(k_repair=-1, accept_repair=float("nan"),
                cov_all_repair=float(g["cov_all"].iloc[i]),
                cov_nonvac_repair=float(g["cov_nonvac"].iloc[i]),
                nvf_repair=float(g["nonvac_frac"].iloc[i]),
                n_units_repair=int(g["n_units"].iloc[i]), throughput_neutral=False,
                best_k=int(g["k"].iloc[i]), best_accept=float(g["accept_mean"].iloc[i]))


def build(cert, ladder):
    rows = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            c = cert[(cert["family"] == fam) & np.isclose(cert["alpha"], alpha)]
            assert len(c) == 1, (fam, alpha, len(c))
            c = c.iloc[0]
            if float(c["cov_all_src"]) >= NOMINAL:
                # the source-only certificate is already nominal here, so there
                # is nothing to buy and the price is zero rather than missing.
                rep = dict(k_repair=0, accept_repair=float(c["accept_src"]),
                           cov_all_repair=float(c["cov_all_src"]),
                           cov_nonvac_repair=float(c["cov_nonvac_src"]),
                           nvf_repair=float(c["nvf_src"]),
                           n_units_repair=int(c["n_units_src"]),
                           throughput_neutral=True, best_k=0,
                           best_accept=float(c["accept_src"]))
            else:
                rep = repair_price(ladder, fam, alpha, float(c["accept_src"]))
            pool = float(c["pool_cells"])
            review_cells = float(c["review_frac"]) * pool
            k = rep["k_repair"]
            rows.append(dict(
                family=fam, family_label=LABEL[fam], alpha=alpha,
                n_units_src=int(c["n_units_src"]), pool_cells=pool,
                review_frac=float(c["review_frac"]), review_cells=review_cells,
                accept_src=float(c["accept_src"]), cov_all_src=float(c["cov_all_src"]),
                cov_nonvac_src=float(c["cov_nonvac_src"]), nvf_src=float(c["nvf_src"]),
                k_repair=k,
                k_repair_share_of_pool=(float(k) / pool if k >= 0 else float("nan")),
                accept_repair=rep["accept_repair"], cov_all_repair=rep["cov_all_repair"],
                cov_nonvac_repair=rep["cov_nonvac_repair"], nvf_repair=rep["nvf_repair"],
                n_units_repair=rep["n_units_repair"],
                throughput_neutral=rep["throughput_neutral"],
                throughput_change=(rep["accept_repair"] - float(c["accept_src"])
                                   if k >= 0 else float("nan")),
                best_k_without_neutrality=rep["best_k"],
                best_accept_without_neutrality=rep["best_accept"],
                review_over_label_price=(review_cells / float(k) if k > 0 else float("nan")),
            ))
    return pd.DataFrame(rows)


def main():
    cert, ladder = load_certificate(), load_ladder()
    out = build(cert, ladder)

    results = os.path.join(paths.RESULTS, "%s.csv" % TAG)
    out.to_csv(results, index=False)

    lines = []

    def w(s=""):
        lines.append(s)
        print(s, flush=True)

    w("R50 -- review budget versus repair price, same units")
    w("=" * 78)
    w("certificate: r16_review_budget.csv (method crc) ; ladder: r24_efficiency.csv (arm tgt_k)")
    w("repair price = smallest k with coverage(all) >= %.2f *and* acceptance back to the" % NOMINAL)
    w("source-only value, i.e. validity restored at no cost in throughput")
    w("")
    for _, r in out.iterrows():
        w("%s, alpha = %.2f  (%d units, pool %.0f cells/unit)"
          % (r["family_label"], r["alpha"], r["n_units_src"], r["pool_cells"]))
        w("   certificate   review %.3f (%.0f cells/unit) ; acceptance %.4f ; coverage %.4f [non-vac %.4f]"
          % (r["review_frac"], r["review_cells"], r["accept_src"], r["cov_all_src"], r["cov_nonvac_src"]))
        if r["k_repair"] > 0:
            w("   repair        k = %d labels (%.3f of the pool) -> acceptance %.4f (%+.4f) ; "
              "coverage %.4f [non-vac %.4f] on %d units"
              % (r["k_repair"], r["k_repair_share_of_pool"], r["accept_repair"],
                 r["throughput_change"], r["cov_all_repair"], r["cov_nonvac_repair"],
                 r["n_units_repair"]))
            w("   in the same units: review %.0f cells/unit vs %d labels/unit -> review is %.2fx "
              "the label price" % (r["review_cells"], r["k_repair"], r["review_over_label_price"]))
        elif r["k_repair"] == 0:
            w("   repair        none needed: the certificate is already nominal at k = 0")
        else:
            w("   repair        calibration never restores throughput: best is acceptance %.4f at "
              "k = %d (source-only %.4f)"
              % (r["best_accept_without_neutrality"], r["best_k_without_neutrality"],
                 r["accept_src"]))
        w("")

    p = os.path.join(paths.WORK, "r50_report.txt")
    with open(p, "w", encoding="utf-8", newline=chr(10)) as fh:
        fh.write(chr(10).join(lines) + chr(10))
    print("wrote %s" % results)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
