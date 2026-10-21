"""Build and cache the per-cohort cell tables; emit the data-audit numbers."""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from errctl import data as D, paths, markerlogic


def main():
    out = os.path.join(paths.WORK, "r01_dataset_audit.txt")
    fh = open(out, "w", encoding="utf-8")

    def log(*parts):
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    summary = []
    for ds in paths.MANUAL_DATASETS:
        log("=" * 90)
        log("##### %s" % ds)
        m = D.build_master(ds)
        m.to_parquet(os.path.join(paths.DATASETS, "master_%s.parquet" % ds), index=False)
        markers, _ = D.marker_table(
            ds, kind="HighQualityLabels" if ds in paths.HQ_DATASETS else "ManualLabels")
        markers = [x for x in markers if x in m.columns]

        expert = m[m["is_expert"]]
        n_expert = len(expert)
        n_usable = int((expert["coarse_expert"] != "Unidentified").sum())
        log("  segmented cells: %d   images: %d" % (len(m), m["image"].nunique()))
        log("  expert-labelled cells: %d  (usable, non-Unidentified: %d)" % (n_expert, n_usable))
        log("  markers in panel: %d  -> %s" % (len(markers), markers))
        log("  expert class counts (raw):")
        for k, v in expert["class_expert"].value_counts().items():
            log("      %-22s %5d" % (k, v))
        log("  expert class counts (coarse):")
        for k, v in expert["coarse_expert"].value_counts().items():
            log("      %-22s %5d" % (k, v))
        if m["class_hq"].notna().any():
            hq = m[m["class_hq"].notna()]
            agree = (hq.loc[hq["is_expert"], "class_hq"].astype(str).values
                     == hq.loc[hq["is_expert"], "class_expert"].astype(str).values).mean()
            cagree = (hq.loc[hq["is_expert"], "coarse_hq"].astype(str).values
                      == hq.loc[hq["is_expert"], "coarse_expert"].astype(str).values).mean()
            log("  HQ(model) vs expert label agreement: exact=%.3f coarse=%.3f (n=%d)"
                % (agree, cagree, int(hq["is_expert"].sum())))
        # missing value report
        miss = expert[markers].isna().mean().sort_values(ascending=False)
        log("  markers with any missing values: %s"
            % (dict(miss[miss > 0].round(3)) or "none"))

        rules, failed = markerlogic.compile_rules(D.celltype_table(
            ds, kind="HighQualityLabels" if ds in paths.HQ_DATASETS else "ManualLabels"))
        missing_chan = {}
        for name, rule in rules.items():
            miss_in_panel = sorted(rule.idents - set(markers))
            if miss_in_panel:
                missing_chan[name] = miss_in_panel
        log("  marker rules compiled: %d (failed: %d)" % (len(rules), len(failed)))
        for ct, txt, err in failed:
            log("      FAILED %s: %r -> %s" % (ct, txt, err))
        if missing_chan:
            log("  rules referencing channels absent from the panel:")
            for k, v in missing_chan.items():
                log("      %-22s %s" % (k, v))

        summary.append({
            "dataset": ds,
            "segmented": int(len(m)),
            "images": int(m["image"].nunique()),
            "expert": int(n_expert),
            "expert_usable": n_usable,
            "markers": len(markers),
            "rules": len(rules),
            "rules_failed": len(failed),
        })

    df = pd.DataFrame(summary)
    log("\n" + "=" * 90)
    log("SUMMARY")
    log(df.to_string(index=False))
    log("total expert cells: %d ; usable: %d"
        % (df["expert"].sum(), df["expert_usable"].sum()))
    df.to_csv(os.path.join(paths.WORK, "r01_summary.csv"), index=False)
    fh.close()


if __name__ == "__main__":
    main()
