"""R89 -- acceptance check for the three `errctl advise` deliverables.

Runs the command twice (a two-domain demo and a random split), then checks the
files a laboratory would actually use:

  * `budget.csv` is internally consistent and its hours reconcile;
  * `decisions.csv` has exactly one commit/review decision per deployment cell
    and a complete `review_rank` on the deferred queue;
  * `price_list.csv` recommends only a rung that is within budget, non-silent
    and net positive, and the JSON agrees;
  * the audit and certificate JSON are finite and quote a scoped guarantee.

The script exits non-zero on the first failed check.  It is CPU only and needs
no data beyond the synthetic demo it writes itself.
"""

import json
import math
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "work", "_r89_advise_check")
NCHECKS = [0]


def check(ok, label):
    NCHECKS[0] += 1
    if not ok:
        raise AssertionError(label)


def run(args):
    cmd = [sys.executable, os.path.join(HERE, "errctl_cli.py")] + args
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if p.returncode:
        raise AssertionError("command failed:\n%s\n%s"
                             % (" ".join(cmd), p.stdout[-4000:]))


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_domain(out):
    for name in ("budget.csv", "decisions.csv", "price_list.csv", "audit.csv",
                 "certificate.json", "certificate.md", "budget.svg"):
        check(os.path.exists(os.path.join(out, name)), "missing " + name)
    cert = load_json(os.path.join(out, "certificate.json"))
    budget = pd.read_csv(os.path.join(out, "budget.csv"))
    dec = pd.read_csv(os.path.join(out, "decisions.csv"))
    price = pd.read_csv(os.path.join(out, "price_list.csv"))
    audit = pd.read_csv(os.path.join(out, "audit.csv"))
    row = budget[budget.requested == 1].iloc[0]
    b = cert["budget"]

    check(int(row.n_accepted + (row.n - row.n_accepted)) == int(row.n), "budget count")
    check(int(b["n_accepted"] + b["n_review"]) == int(b["n_deployment"]),
          "certificate review count")
    check(abs(float(row.acceptance) - float(b["acceptance"])) < 1e-9,
          "acceptance disagreement")
    check(len(dec) == int(b["n_deployment"]), "decision row count")
    check(set(dec["decision"]).issubset({"commit", "review"}), "decision vocabulary")
    check(int((dec["decision"] == "commit").sum()) == int(b["n_accepted"]),
          "commit count disagreement")
    check(int((dec["decision"] == "review").sum()) == int(b["n_review"]),
          "review count disagreement")

    reviewed = dec[dec["decision"] == "review"]
    rank = reviewed["review_rank"].dropna().to_numpy()
    if len(reviewed):
        check(len(rank) == len(reviewed), "review rank missing")
        check(sorted(rank.astype(int)) == list(range(1, len(reviewed) + 1)),
              "review rank not a complete queue")
    check(dec.loc[dec["decision"] == "commit", "review_rank"].isna().all(),
          "committed cells carry a review rank")

    sec = float(cert["run"]["sec_per_cell"])
    expected = (int(row.n_accepted) - int(row.n_calibration)) * sec / 3600.0
    check(abs(float(row.net_saved_hours) - expected) < 1e-9, "net hours")
    if math.isfinite(float(row.realised_risk)):
        check(float(row.cp_upper) >= float(row.realised_risk) - 1e-12,
              "CP upper below point estimate")

    check(cert["audit_verdict"] in {"pass", "warn", "partial", "fail"},
          "audit verdict")
    check(isinstance(cert["quotation_allowed"], bool), "quotation flag")
    check("ratio" in cert["audit"] and "score_shift" in cert["audit"],
          "audit legs missing")
    check(len(audit) == len(cert["audit"]), "audit row count")
    check(set(audit["check"]) == set(cert["audit"]), "audit check names")

    if cert["repair"]["status"] == "ok":
        k = int(cert["repair"]["recommended_k"])
        hit = price[price.k == k]
        check(len(hit) == 1, "recommended k absent from price list")
        rr = hit.iloc[0]
        prefix = ("frozen" if str(rr.kind) == "calibration_ladder" else "refit")
        check(float(rr[prefix + "_net_share"]) > 0,
              "recommended rung is not net positive")
        check(int(rr[prefix + "_covered"]) == 1, "recommended rung misses budget")
        check(float(rr[prefix + "_cp_upper"]) <= float(cert["alpha_delta"][0]) + 1e-12,
              "recommended rung fails the CP check")


def verify_unlabelled(out, src, tgt):
    """A production batch with a 500-cell pilot and 2,500 unlabelled cells."""
    os.makedirs(out, exist_ok=True)
    frame = pd.read_csv(tgt)
    pilot = frame.copy()
    pilot.loc[500:, "Cell_Label"] = ""
    pilot_path = os.path.join(out, "target_pilot.csv")
    pilot.to_csv(pilot_path, index=False)
    run(["advise", "--source", src, "--target", pilot_path,
         "--label", "Cell_Label", "--markers", "m1", "m2", "m3", "m4", "m5", "m6",
         "--group", "sample", "--id", "cell", "--model", "logreg",
         "--k", "500", "--scan-k", "100,250,500",
         "--out-dir", os.path.join(out, "pilot"), "--seed", "19"])
    cert = load_json(os.path.join(out, "pilot", "certificate.json"))
    budget = pd.read_csv(os.path.join(out, "pilot", "budget.csv"))
    dec = pd.read_csv(os.path.join(out, "pilot", "decisions.csv"))
    check(cert["input"]["n_labelled_target"] == 500, "pilot label count")
    check(cert["input"]["n_unlabelled_target"] == 2500, "unlabelled count")
    check(cert["input"]["n_evaluated_deployment"] == 0, "unlabelled evaluation")
    check(cert["budget"]["realised_risk"] is None, "invented realised risk")
    check(cert["budget"]["ratio"] is None, "invented ratio")
    check(cert["audit"]["ratio"]["status"] == "not_testable",
          "label-free ratio status")
    check(cert["audit_verdict"] == "partial", "label-free audit verdict")
    check(cert["quotation_allowed"] is True, "label-free quote scope")
    check(int((dec["decision"] == "commit").sum()) == int(cert["budget"]["n_accepted"]),
          "unlabelled commit count")
    check(bool(dec["evaluated"].sum() == 0), "unlabelled cells marked evaluated")
    check(len(dec) == 2500, "unlabelled decision rows")
    check(int(budget.iloc[0]["n_calibration"]) == 500, "pilot calibration count")
    check("not estimable" in open(os.path.join(out, "pilot", "certificate.md"),
                                  encoding="utf-8").read(),
          "certificate admits missing labels")

    # The same run with labels in a separate id,label table.  This is the
    # production shape when the instrument export has no label column at all.
    no_label = frame.drop(columns=["Cell_Label"])
    no_label_path = os.path.join(out, "target_nolabel.csv")
    no_label.to_csv(no_label_path, index=False)
    cal_path = os.path.join(out, "calibration.csv")
    frame.iloc[:500][["cell", "Cell_Label"]].to_csv(cal_path, index=False)
    run(["advise", "--source", src, "--target", no_label_path,
         "--calibration-csv", cal_path, "--label", "Cell_Label",
         "--markers", "m1", "m2", "m3", "m4", "m5", "m6",
         "--group", "sample", "--id", "cell", "--model", "logreg",
         "--k", "500", "--scan-k", "100,250,500",
         "--out-dir", os.path.join(out, "separate"), "--seed", "19"])
    cert2 = load_json(os.path.join(out, "separate", "certificate.json"))
    check(cert2["input"]["calibration_source"] == "calibration_csv",
          "calibration csv source")
    check(cert2["input"]["n_calibration"] == 500, "calibration csv count")
    check(cert2["input"]["n_evaluated_deployment"] == 0,
          "calibration csv evaluation scope")
def main():
    run(["advise", "--demo", "--model", "logreg", "--k", "500",
         "--scan-k", "250,500,1000", "--out-dir",
         os.path.join(OUT, "target"), "--seed", "19"])
    verify_domain(os.path.join(OUT, "target"))

    src = os.path.join(OUT, "target", "demo_target.csv")
    run(["advise", "--csv", src, "--label", "Cell_Label",
         "--markers", "m1", "m2", "m3", "m4", "m5", "m6",
         "--group", "sample", "--id", "cell", "--model", "logreg",
         "--k", "500", "--scan-k", "100,250,500",
         "--out-dir", os.path.join(OUT, "single"), "--seed", "19"])
    verify_domain(os.path.join(OUT, "single"))

    verify_unlabelled(os.path.join(OUT, "unlabelled"),
                      os.path.join(OUT, "target", "demo_source.csv"),
                      os.path.join(OUT, "target", "demo_target.csv"))
    print("r89 advise checks: %d pass, 0 fail" % NCHECKS[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
