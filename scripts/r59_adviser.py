"""errctl-adviser -- the guided decision procedure of the paper, as a program.

Standalone and self-contained: it does not import the pipeline, does not read
the released tables, and needs nothing but a set of scores. Run it and it walks
you through the six decisions the paper's procedure consists of:

  1. what to do before spending a single label
  2. which error budget to commit to, and what it costs in review and in error
  3. whether the certificate can be quoted at all on this batch
  4. in what order to read the review queue it produces
  5. whether the calibration sample is representative (the audit, once labels
     arrive), and what to do when it is not
  6. where to buy labels, at what price, and the one sampling rule that must not
     be broken

Input, either:
  --npz FILE   a .npz with cal_scores, cal_errors, dep_scores and optionally
               dep_labels (0/1 error indicators), where scores are 1 - max prob
  --demo       the released random-split unit of the paper (melanoma, fold 0,
               histogram gradient boosting), which is the case all the quoted
               numbers come from

The certificate used here is the paper's own, imported from src/errctl when the
repository is on the path and otherwise reimplemented inline with the same
Clopper-Pearson rule, so the program runs from a bare checkout.

Example:  python scripts/r59_adviser.py --demo
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# measured trade-off of the source-only certificate, pooled over the benchmark
BUDGET = {0.05: (0.653, 0.021), 0.10: (0.403, 0.061), 0.20: (0.136, 0.139)}
# review-queue capture, pooled over cells, from experiments/r57_triage_summary.csv
QUEUE = {0.05: (0.477, 0.749, 0.498, 0.118), 0.10: (0.386, 0.655, 0.568, 0.265),
         0.20: (0.309, 0.568, 0.794, 0.594)}
PRICE = ("Of the twelve cross-tissue pairs, four reach a valid non-vacuous state at 500 target "
         "labels, three more at 750, 1,500 or 2,000, and five never do inside the grid; three "
         "of those five are unreachable because the transferred model's own ranking accepts "
         "5% or less (Table 6).")


def cp_upper(x, n, d):
    """One-sided Clopper-Pearson upper limit for x errors in n accepted cells."""
    x, n = int(x), int(n)
    if n <= 0:
        return 1.0
    if x <= 0:
        return 1.0 - d ** (1.0 / n)
    from scipy import stats
    return float(stats.beta.ppf(1.0 - d, x + 1, n - x))


class InlineController(object):
    """The paper's selective-risk controller, Clopper-Pearson form, in 20 lines.

    lambda is the largest threshold in a grid whose one-sided upper limit on the
    selective risk of the accepted set is at most alpha, subject to accepting at
    least min_accept calibration cells. Imported from src/errctl when available.
    """

    def __init__(self, alpha=0.10, delta=0.10, min_accept=25, grid_size=120):
        self.alpha, self.delta, self.min_accept, self.grid_size = alpha, delta, min_accept, grid_size

    def fit(self, scores, errors):
        s = np.asarray(scores, dtype=float)
        e = np.asarray(errors, dtype=float)
        grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, self.grid_size)))
        best = -np.inf
        for lam in grid:
            m = s <= lam
            n = int(m.sum())
            if n < self.min_accept:
                continue
            if cp_upper(int(e[m].sum()), n, self.delta) <= self.alpha:
                best = lam
        self.lam = best
        self.n_cal = int(len(s))
        return self

    def accept(self, scores):
        s = np.asarray(scores, dtype=float)
        if not np.isfinite(self.lam):
            return np.zeros(len(s), dtype=bool)
        return s <= self.lam


def get_controller(alpha, delta, min_accept=25):
    if ROOT not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from errctl.conformal import SelectiveRiskController
        return SelectiveRiskController(alpha, delta, min_accept=min_accept, bound="cp"), True
    except Exception:
        return InlineController(alpha, delta, min_accept=min_accept), False


def ask(question, default, choices=None, yes=False):
    if yes:
        return default
    tail = " [%s]" % default
    if choices:
        tail = " (%s) [%s]" % ("/".join(str(c) for c in choices), default)
    while True:
        try:
            raw = input("%s%s: " % (question, tail)).strip()
        except EOFError:
            print("  (no input available, taking the default)")
            return default
        if not raw:
            return default
        if choices:
            if any(_same(raw, c) for c in choices):
                return type(default)(float(raw))
            if all(isinstance(c, (int, float)) for c in choices):
                try:
                    float(raw)
                except ValueError:
                    pass
        try:
            val = type(default)(raw)
        except ValueError:
            print("  not a valid value")
            continue
        if choices and not any(_same(str(val), str(c)) for c in choices):
            print("  choose one of: %s" % ", ".join(str(c) for c in choices))
            continue
        return val


def _same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def load_demo():
    path = os.path.join(ROOT, "work", "predictions", "cellsplit_Melanoma_Oct2522_fold0__hgb.npz")
    if not os.path.exists(path):
        raise SystemExit("demo file not found: %s" % path)
    d = np.load(path, allow_pickle=True)
    cal_rep, test_rep = d["cal_rep"], d["test_rep"]
    ci, ti = cal_rep[:, 0], test_rep[:, 0]
    return {"cal_scores": 1.0 - d["cal_probs"][ci].max(1),
            "cal_errors": (d["cal_probs"][ci].argmax(1) != d["cal_y"][ci]).astype(float),
            "dep_scores": 1.0 - d["test_probs"][ti].max(1),
            "dep_labels": (d["test_probs"][ti].argmax(1) != d["test_y"][ti]).astype(float),
            "name": "melanoma fold 0, random split (released unit)"}


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if "cal_scores" in d.files:
        out = {k: np.asarray(d[k], dtype=float) for k in ("cal_scores", "cal_errors", "dep_scores")}
        if "dep_labels" in d.files:
            out["dep_labels"] = np.asarray(d["dep_labels"], dtype=float)
        out["name"] = os.path.basename(path)
        return out
    if "cal_probs" in d.files:                                  # repository cache format
        ci, ti = d["cal_rep"][:, 0], d["test_rep"][:, 0]
        return {"cal_scores": 1.0 - d["cal_probs"][ci].max(1),
                "cal_errors": (d["cal_probs"][ci].argmax(1) != d["cal_y"][ci]).astype(float),
                "dep_scores": 1.0 - d["test_probs"][ti].max(1),
                "dep_labels": (d["test_probs"][ti].argmax(1) != d["test_y"][ti]).astype(float),
                "name": os.path.basename(path)}
    raise SystemExit("cannot read %s: expected cal_scores/cal_errors/dep_scores or a "
                     "repository prediction cache" % path)


def ks_distance(a, b):
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    grid = np.unique(np.concatenate([x, y]))
    cx = np.searchsorted(x, grid, side="right") / max(len(x), 1)
    cy = np.searchsorted(y, grid, side="right") / max(len(y), 1)
    return float(np.max(np.abs(cx - cy)))


def main():
    ap = argparse.ArgumentParser(description="guided decision procedure for a cell-annotation batch")
    ap.add_argument("--npz", help="scores file (cal_scores, cal_errors, dep_scores, optional dep_labels)")
    ap.add_argument("--demo", action="store_true", help="run on the released random-split unit")
    ap.add_argument("--alpha", type=float, default=None, help="error budget; skips the question")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--yes", action="store_true", help="non-interactive, take every default")
    ap.add_argument("--out", help="write the plan as JSON to this path")
    args = ap.parse_args()

    if not args.demo and not args.npz:
        args.demo = True
    data = load_demo() if args.demo else load_npz(args.npz)
    has_labels = "dep_labels" in data

    hr("errctl-adviser -- the decision procedure, one question at a time")
    print("batch: %s" % data["name"])
    print("calibration cells: %d   deployment cells: %d   deployment labels: %s"
          % (len(data["cal_scores"]), len(data["dep_scores"]),
             "yes" if has_labels else "no"))
    print(textwrap.fill(
        "Nothing below is a step in an argument. Each answer is a statistic, a table or a rule, "
        "and the program only reports what the paper measured for the branch you take.", 72))
    plan = {"batch": data["name"], "n_cal": int(len(data["cal_scores"])),
            "n_dep": int(len(data["dep_scores"]))}

    # ---- 1. before any label ------------------------------------------------
    hr("STEP 1  before a single label: is a repair worth attempting?")
    ctrl0, imported = get_controller(0.10, args.delta)
    ctrl0.fit(data["cal_scores"], data["cal_errors"])
    src_accept = float(ctrl0.accept(data["dep_scores"]).mean())
    ks = ks_distance(data["cal_scores"], data["dep_scores"])
    print("source-only certificate acceptance : %.3f" % src_accept)
    print("KS distance, calibration vs deployment scores: %.3f" % ks)
    print(textwrap.fill(
        "These two statistics need no labels and they order units by whether a repair is worth "
        "attempting: over the 99 units of the cross-validated grid they separate the 47 that reach "
        "validity from the 52 that never do (Mann-Whitney p = 1e-6 and 2e-8). They order; they do "
        "not decide. The decision belongs to the audit in step 5.", 72))
    plan["step1"] = {"source_only_acceptance": src_accept, "ks_distance": ks}

    # ---- 2. the budget -----------------------------------------------------
    hr("STEP 2  choose the error budget. It is free in labels and is paid in error.")
    for a in (0.05, 0.10, 0.20):
        rv, er = BUDGET[a]
        print("   alpha = %.2f   review %.1f%% of the batch   error among retained labels %.3f"
              % (a, 100 * rv, er))
    alpha = float(args.alpha) if args.alpha else ask("commit to which error budget", 0.10, choices=[0.05, 0.10, 0.20], yes=args.yes)
    rv, er = BUDGET[alpha]
    print("\ncommitted: alpha = %.2f -> plan to review about %.1f%% of the batch and to keep about "
          "%.1f%% of the batch wrong-but-committed at a realised error of %.3f"
          % (alpha, 100 * rv, 100 * rv * er, er))
    plan["step2"] = {"alpha": alpha, "expected_review_fraction": rv, "expected_error_among_retained": er}

    # ---- 3. can the certificate be quoted at all ---------------------------
    hr("STEP 3  fit the certificate on this batch")
    ctrl, imported = get_controller(alpha, args.delta)
    ctrl.fit(data["cal_scores"], data["cal_errors"])
    accepted = ctrl.accept(data["dep_scores"])
    n_acc, n_dep = int(accepted.sum()), int(len(accepted))
    print("controller: %s" % ("src/errctl.SelectiveRiskController" if imported else "inline Clopper-Pearson fallback"))
    print("accepts %d of %d deployment cells (%.3f), defers %.3f"
          % (n_acc, n_dep, n_acc / max(n_dep, 1), 1.0 - n_acc / max(n_dep, 1)))
    if n_acc == 0:
        print(textwrap.fill(
            "The accepted set is empty: the certificate can be quoted but certifies nothing. This is "
            "the honest reading of a score whose ranking is too weak for this budget, and it is what "
            "the paper reports for the wrapped image classifier at the tight budgets.", 72))
    lam = None
    for attr in ("lam", "lambda_hat", "lam_hat", "lam_", "thr"):
        v = getattr(ctrl, attr, None)
        if v is not None and np.isfinite(np.asarray(v, dtype=float)).all():
            lam = float(np.asarray(v, dtype=float).ravel()[0])
            break
    plan["step3"] = {"lambda": lam, "acceptance": n_acc / max(n_dep, 1)}

    # ---- 4. the review queue ----------------------------------------------
    hr("STEP 4  read the review queue in descending uncertainty, and cut it from the end")
    c25, c50, head, tail = QUEUE[alpha]
    print("expected, pooled over the benchmark at alpha = %.2f:" % alpha)
    print("   first quarter of the queue  ->  %.3f of the residual errors" % c25)
    print("   first half of the queue     ->  %.3f of the residual errors" % c50)
    print("   mean error rate at the head ->  %.3f     at the tail -> %.3f" % (head, tail))
    print(textwrap.fill(
        "Read in that order, not in image or section order; if the queue has to be shortened, shorten "
        "it from the end, because the tail is where the errors are scarcest. Cross-learner disagreement "
        "and score entropy do not beat the certificate's own score as orderings.", 72))
    if has_labels:
        err = np.asarray(data["dep_labels"], dtype=float)
        order = np.argsort(-np.asarray(data["dep_scores"])[~accepted], kind="stable")
        de = err[~accepted]
        nd, ne = len(de), int(de.sum())
        if ne:
            cum = np.cumsum(de[order] > 0)
            got25 = cum[max(1, int(round(0.25 * nd))) - 1] / ne
            got50 = cum[max(1, int(round(0.50 * nd))) - 1] / ne
            q = max(1, int(round(0.25 * nd)))
            print("\nmeasured on THIS batch: deferred error rate %.3f, capture %.3f in the first "
                  "quarter and %.3f in the first half, head error rate %.3f against tail %.3f"
                  % (de.mean(), got25, got50, de[order[:q]].mean(), de[order[-q:]].mean()))
            plan["step4_measured"] = {"deferred_error_rate": float(de.mean()), "capture_q25": float(got25),
                                      "capture_q50": float(got50)}
    print(textwrap.fill(
        "Nothing here preserves the certificate: a cell that is neither committed nor reviewed "
        "carries no guarantee, and the tail still carries errors.", 72))

    # ---- 5. the audit ------------------------------------------------------
    hr("STEP 5  once labels arrive: is the calibration sample representative?")
    if not has_labels:
        print(textwrap.fill(
            "No deployment labels in this input, so the audit cannot run yet. When they arrive, form the "
            "ratio of the certificate's acceptance to the acceptance of the best threshold those labels "
            "support. Under validity the ratio cannot exceed one, because the oracle is the best threshold "
            "there is. We observe 0.744 for the random split, 0.766 across fields of view and 3.05 under "
            "cross-tissue transfer: the last is a strict proof that the calibration sample does not "
            "represent the deployment population.", 72))
        plan["step5"] = {"audit": "not runnable without deployment labels",
                         "benchmark_ratios": {"random_split": 0.744, "across_fields": 0.766, "cross_tissue": 3.05}}
    else:
        dep = np.asarray(data["dep_scores"], dtype=float)
        err = np.asarray(data["dep_labels"], dtype=float)
        # The oracle is the paper's own definition (scripts/r03_analyse.py,
        # oracle_acceptance): the largest threshold that meets the budget when the
        # deployment labels themselves choose it, scanned over every attainable
        # threshold rather than a fixed grid. A grid can only understate the
        # oracle, and therefore overstate the ratio.
        _ord = np.argsort(dep, kind="mergesort")
        _s, _e = dep[_ord], err[_ord]
        _k = np.arange(1, _s.size + 1)
        _ok = (np.cumsum(_e) / _k <= alpha) & (_k >= 25)
        best = float(_s[np.nonzero(_ok)[0][-1]]) if _ok.any() else -np.inf
        oracle_accept = float((dep <= best).mean()) if np.isfinite(best) else 0.0
        ratio = (n_acc / max(n_dep, 1)) / oracle_accept if oracle_accept > 0 else float("inf")
        print("certificate acceptance %.3f, oracle acceptance %.3f, ratio %.3f"
              % (n_acc / max(n_dep, 1), oracle_accept, ratio))
        if ratio > 1 + 1e-9:
            print(textwrap.fill(
                "The ratio exceeds one, so the calibration sample is not representative of this "
                "deployment population. This is a proof rather than a screen, and no amount of "
                "re-calibration on this sample fixes it; the repairs of step 6 are the available move.", 72))
        else:
            print(textwrap.fill(
                "The ratio is at most one, which is what validity requires: the certificate is inside "
                "its budget on this batch and no repair is indicated.", 72))
        plan["step5"] = {"audit_ratio": ratio, "oracle_acceptance": oracle_accept}

    # ---- 6. the price ------------------------------------------------------
    hr("STEP 6  if a repair is indicated: buy uniformly drawn target labels")
    print(textwrap.fill(PRICE, 72))
    print(textwrap.fill(
        "The one rule that must not be broken: draw the labels uniformly from the deployment cohort, "
        "never from the review queue. The deferred set is selected by the score, so a calibration "
        "sample taken from it is exactly the score-selected sample whose failure the audit just "
        "measured; under the bin-stratified permutation test that assumption fails in 178 of 180 units.", 72))
    print(textwrap.fill(
        "The labels are also cheaper than the review you have already accepted: on the cross-field-of-view "
        "family at a 10% budget the certificate routes 46.6% of the pool to a human -- 925 cells in a mean "
        "unit -- against the 500 labels the repair needs, and those 500 leave acceptance where it was.", 72))
    plan["step6"] = {"price": PRICE, "sampling_rule": "uniform over the deployment cohort, never from the review queue"}

    hr("PLAN")
    for k in ("step1", "step2", "step3", "step4_measured", "step5", "step6"):
        if k in plan:
            print("  %-16s %s" % (k, json.dumps(plan[k], sort_keys=True)))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(plan, fh, indent=2, sort_keys=True)
        print("\nwritten to %s" % args.out)


if __name__ == "__main__":
    main()
