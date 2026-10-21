"""Turn the tidy result tables into the tables the manuscript needs.

Reads experiments/r03_metrics.csv and its companions and writes

    experiments/r04_t1_coverage.csv      guarantee coverage by family and method
    experiments/r04_t2_efficiency.csv    what a certified budget buys, vs the oracle
    experiments/r04_t3_audit.csv         marker-consistency audit (from r06)
    experiments/r04_t4_ablations.csv     knob sweeps (from r07)
    experiments/r04_t5_simulation.csv    Monte-Carlo validation (from r09)
    experiments/r04_t6_panel.csv         panel reduction (from r08, if present)
    experiments/r04_report.txt           all of the above, readable

and prints the same content to stdout.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from errctl import paths

RES = paths.RESULTS
OUT = os.path.join(RES, "r04_report.txt")
FAMILIES = ("cellsplit", "within", "transfer")
_fh = open(OUT, "w", encoding="utf-8")


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


def save(df, name):
    p = os.path.join(RES, name)
    df.to_csv(p)
    log("   -> %s (%d rows)" % (name, len(df)))


def cohort_of(unit):
    """The cohort a unit belongs to: the target for transfer units."""
    u = unit.replace("transfer_", "")
    if "_to_" in u:
        u = u.split("_to_")[1]
    return "_".join(u.split("_")[:2])


def main():
    M = pd.read_csv(os.path.join(RES, "r03_metrics.csv"))
    M = M.rename(columns={"kind": "family"})
    M["cohort"] = M["unit"].map(cohort_of)
    ECE = pd.read_csv(os.path.join(RES, "r03_ece.csv")).rename(columns={"kind": "family"})
    ECE["cohort"] = ECE["unit"].map(cohort_of)

    log("=" * 96)
    log("1. GUARANTEE COVERAGE  (delta = 0.10, so coverage must reach 0.90)")
    log("=" * 96)
    for alpha in (0.05, 0.10, 0.20):
        sub = M[np.isclose(M["alpha"], alpha)]
        t = (sub.groupby(["family", "method"])
                .agg(accept=("accept_rate", "mean"),
                     risk=("selective_risk", "mean"),
                     coverage=("covered", "mean"),
                     abstain=("empty", "mean"),
                     n=("covered", "size"))
                .round(3))
        log("\n[alpha = %.2f]  family x method" % alpha)
        log(t.to_string())
    t1 = (M.groupby(["family", "method", "alpha"])
            .agg(accept=("accept_rate", "mean"), risk=("selective_risk", "mean"),
                 coverage=("covered", "mean"), abstain=("empty", "mean"),
                 n_eval=("covered", "size"))
            .reset_index().round(4))
    save(t1, "r04_t1_coverage.csv")

    log("\n" + "=" * 96)
    log("2. WHAT A CERTIFIED BUDGET BUYS  (alpha = 0.10)")
    log("=" * 96)
    sub = M[np.isclose(M["alpha"], 0.10)]

    def deferred_risk(s):
        """Risk among the cells the controller sends for review.

        The overall risk decomposes as accept*crc_risk + (1-accept)*deferred, so
        the deferred part follows from quantities the table already stores.
        """
        g = (s.groupby(["unit", "clf", "repeat", "method"])
               [["accept_rate", "selective_risk"]].first())
        a = g.xs("crc", level="method")["accept_rate"]
        r = g.xs("crc", level="method")["selective_risk"]
        tot = g.xs("accept_all", level="method")["selective_risk"]
        return float(((tot - a * r) / (1 - a).replace(0, np.nan)).mean())

    rows = []
    for fam in FAMILIES:
        for coh in ["ALL"] + sorted(sub[sub.family == fam]["cohort"].unique()):
            s = sub[(sub.family == fam) & (sub.cohort == coh if coh != "ALL" else True)]
            if s.empty:
                continue
            def acc_of(method):     # mean acceptance over evaluations
                return float(s[s.method == method]["accept_rate"].mean())
            orc = acc_of("oracle")
            rows.append({
                "family": fam, "cohort": coh,
                "accept_crc": acc_of("crc"),
                "accept_oracle": orc,
                "accept_naive": acc_of("naive"),
                "accept_conformal": acc_of("conformal_set"),
                "accept_mondrian": acc_of("crc_mondrian"),
                "frac_of_oracle": acc_of("crc") / orc if orc > 0 else np.nan,
                "risk_crc": float(s[s.method == "crc"]["selective_risk"].mean()),
                "risk_accept_all": float(s[s.method == "accept_all"]["selective_risk"].mean()),
                "risk_oracle": float(s[s.method == "oracle"]["selective_risk"].mean()),
                "abstain_crc": float(s[s.method == "crc"]["empty"].mean()),
                "risk_deferred": deferred_risk(s),
            })
    t2 = pd.DataFrame(rows).round(4)
    log(t2.to_string(index=False))
    log("")
    log("   risk_deferred = error rate among the cells the controller sends to review")
    log(t2[["family", "cohort", "risk_crc", "risk_deferred", "risk_accept_all"]]
        .round(3).to_string(index=False))
    save(t2, "r04_t2_efficiency.csv")

    log("\n" + "=" * 96)
    log("3. THE EXCHANGEABILITY CONTROL, PER COHORT AND CLASSIFIER (cellsplit)")
    log("=" * 96)
    c = M[(M.family == "cellsplit") & np.isclose(M["alpha"], 0.10)]
    t = (c.pivot_table(index=["cohort", "clf"], columns="method",
                       values=["covered", "accept_rate"])
           .round(3))
    log(t.to_string())
    log("")
    log("   the same table for the cross-FOV family, where exchangeability is what")
    log("   breaks: coverage should fall below 0.90")
    w = M[(M.family == "within") & np.isclose(M["alpha"], 0.10)]
    log(w.pivot_table(index=["cohort", "clf"], columns="method",
                      values=["covered", "accept_rate"]).round(3).to_string())

    log("\n" + "=" * 96)
    log("4. CALIBRATION: source vs deployment (mean ECE over folds and classifiers)")
    log("=" * 96)
    log(ECE.groupby(["family", "domain"]).agg(ece=("ece", "mean"), acc=("acc", "mean"),
                                              n=("n", "sum")).round(4).to_string())
    log("")
    log(ECE.pivot_table(index="cohort", columns=["family", "domain"],
                        values="ece").round(3).to_string())

    log("\n" + "=" * 96)
    log("5. MARKER-CONSISTENCY AUDIT (r06)")
    log("=" * 96)
    p = os.path.join(RES, "r06_marker_audit.csv")
    if os.path.exists(p):
        a = pd.read_csv(p)
        for reading in sorted(a["reading"].unique()):
            s = a[(a.coarse_class == "ALL_complete") & (a.reading == reading)]
            log("\n[evaluable rules only, ';' read as %s]" % reading.upper())
            log(s.pivot_table(index="dataset", columns="source",
                              values="violation_rate").round(3).to_string())
            log("   audit coverage (expert cells with a fully evaluable rule):")
            log(s[s.source == "expert"].set_index("dataset")["n"].to_string())
        save(a, "r04_t3_audit.csv")
        v = pd.read_csv(os.path.join(RES, "r06_violations.csv"))
        log("\nviolation rate by whether the prediction was right:")
        log(v.pivot_table(index=["dataset", "clf"], columns="group",
                          values="violation_rate").round(3).to_string())
        sl = pd.read_csv(os.path.join(RES, "r06_selective.csv"))
        log("\nviolation rate among accepted vs deferred cells (crc):")
        log(sl.pivot_table(index=["dataset", "clf"], columns="group",
                           values="violation_rate").round(3).to_string())
    else:
        log("(r06 output missing)")

    log("\n" + "=" * 96)
    log("6. ABLATIONS (r07)")
    log("=" * 96)
    p = os.path.join(RES, "r07_ablations.csv")
    if os.path.exists(p):
        b = pd.read_csv(p)
        for ab in ("n_cal", "grid", "delta", "min_accept", "score"):
            for fam in ("cellsplit", "within"):
                s = b[(b.ablation == ab) & (b.family == fam)]
                if s.empty:
                    continue
                t = (s.groupby(["value", "alpha"])
                       .agg(accept=("accept_rate", "mean"), coverage=("covered", "mean"))
                       .round(3))
                log("\n[%s / %s]" % (ab, fam))
                log(t.to_string())
        save(b, "r04_t4_ablations.csv")

    log("\n" + "=" * 96)
    log("7. MONTE-CARLO VALIDATION OF THE GUARANTEE (r09)")
    log("=" * 96)
    p = os.path.join(RES, "r09_simulation_summary.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        for setting in s["setting"].unique():
            log("\n[%s]" % setting)
            log(s[s.setting == setting]
                .pivot_table(index="method", columns="alpha", values="coverage")
                .round(3).to_string())
        save(s, "r04_t5_simulation.csv")

    log("\n" + "=" * 96)
    log("8. PANEL REDUCTION (r08)")
    log("=" * 96)
    p = os.path.join(RES, "r08_panel.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        t = (s.groupby(["clf", "frac", "alpha"])
               .agg(markers=("n_markers", "mean"), acc=("acc", "mean"),
                    accept=("accept_rate", "mean"), risk=("selective_risk", "mean"),
                    coverage=("covered", "mean")).round(3))
        log(t.to_string())
        save(s, "r04_t6_panel.csv")

    _fh.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()






