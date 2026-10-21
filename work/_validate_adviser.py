# -*- coding: utf-8 -*-
"""Static + numeric audit of both guided interfaces and the sample CSV."""

import io
import json
import math
import os
import re
import sys

import numpy as np
from scipy.stats import beta as _beta

ROOT = r"D:\TT paper\0926Distribution-free error"
PAGES = {"zh": os.path.join(ROOT, "docs", "adviser.html"),
         "en": os.path.join(ROOT, "docs", "adviser_en.html")}
SAMPLE = os.path.join(ROOT, "docs", "adviser_sample.csv")
sys.path.insert(0, os.path.join(ROOT, "src"))

COUNT = [0]
fails = []


def chk(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- numeric core
def betacf(a, bb, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + bb, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for mm in range(1, MAXIT + 1):
        m2 = 2 * mm
        aa = mm * (bb - mm) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS: break
    return h


_G = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]


def lgamma(x):
    y, t = x, x + 5.5
    t -= (x + 0.5) * math.log(t)
    ser = 1.000000000190015
    for j in range(6):
        y += 1.0
        ser += _G[j] / y
    return -t + math.log(2.5066282746310005 * ser / x)


def betai(a, bb, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(lgamma(a + bb) - lgamma(a) - lgamma(bb) + a * math.log(x) + bb * math.log(1 - x))
    return bt * betacf(a, bb, x) / a if x < (a + 1) / (a + bb + 2) else 1 - bt * betacf(bb, a, 1 - x) / bb


def beta_quantile(p, a, bb):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if betai(a, bb, mid) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def cp_upper(x, n, d):
    return 1 - d ** (1.0 / n) if x <= 0 else beta_quantile(1 - d, x + 1, max(n - x, 1))


def cp_lower(x, n, d):
    return beta_quantile(d, x, n - x + 1)


def fit_lambda(sc, er, alpha, delta=0.10, min_accept=25, grid_size=120):
    idx = sorted(range(len(sc)), key=lambda i: (sc[i], i))
    ss = [sc[i] for i in idx]; ee = [er[i] for i in idx]
    n = len(ss); lo = max(min_accept, 2)
    if lo > n: return None
    steps = min(grid_size, n - lo + 1)
    raw = [round(lo + (n - lo) * i / (steps - 1)) for i in range(steps)] if steps > 1 else [lo]
    cand = sorted(set(int(v) for v in raw))
    dm = delta / (2.0 * len(cand))
    run = []; cum = 0.0
    for v in ee:
        cum += v; run.append(cum)
    lam, chosen = float("-inf"), None
    for k in cand:
        up = cp_upper(round(run[k - 1]), n, dm); low = cp_lower(k, n, dm)
        if (up / low if low > 0 else float("inf")) <= alpha:
            lam, chosen = ss[k - 1], k
    return lam, chosen, len(cand)


def oracle_acc(ls, le, alpha, min_accept=25):
    o = np.argsort(ls, kind="mergesort")
    ss, ee = ls[o], le[o]
    k = np.arange(1, ss.size + 1)
    ok = (np.cumsum(ee) / k <= alpha) & (k >= min_accept)
    return float((ls <= float(ss[np.nonzero(ok)[0][-1]])).mean()) if ok.any() else 0.0


# ---------------------------------------------------------------- per language
pages = {}
for lang, path in PAGES.items():
    b = open(path, "rb").read()
    s = b.decode("utf-8")
    pages[lang] = s
    tag = "[%s] " % lang
    chk(tag + "no BOM", b[:3] != b"\xef\xbb\xbf")
    chk(tag + "LF only", b.count(b"\r\n") == 0, b.count(b"\r\n"))
    for ph in ("__TITLE__", "__DEMO__", "__TEXT_JSON__", "__SAMPLE_JSON__", "__LANG__"):
        chk(tag + "placeholder %s replaced" % ph, ph not in s)

    tm = re.search(r"const T=(\{.*?\});\n", s, re.S)
    chk(tag + "T object present", tm is not None)
    T = json.loads(tm.group(1))
    used = set(re.findall(r'data-t="([A-Za-z0-9_]+)"', s)) | set(re.findall(r'data-th="([A-Za-z0-9_]+)"', s))
    chk(tag + "every data-t/data-th key is in T", used <= set(T), sorted(used - set(T)))
    tref = set(re.findall(r"\bT\.([A-Za-z0-9_]+)", s))
    chk(tag + "every T.<key> reference is in T", tref <= set(T), sorted(tref - set(T)))
    chk(tag + "T carries no unreferenced static keys",
        (used | tref) == set(T), sorted(set(T) - (used | tref)))

    ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', s))
    wanted = set(re.findall(r'getElementById\("([A-Za-z0-9_]+)"\)', s))
    chk(tag + "every getElementById target exists", wanted <= ids, sorted(wanted - ids))
    handlers = set(re.findall(r'on\w+="([A-Za-z0-9_]+)\(', s))
    defined = set(re.findall(r"function\s+([A-Za-z0-9_]+)\s*\(", s))
    chk(tag + "every onclick handler defined", handlers <= defined, sorted(handlers - defined))
    steps_src = re.search(r"const STEPS=\[(.*?)\];", s, re.S).group(1)
    step_ids = re.findall(r'\["([A-Za-z0-9_]+)",', steps_src)
    chk(tag + "STEPS ids are real cards", set(step_ids) <= ids, sorted(set(step_ids) - ids))
    chk(tag + "STEPS ids unique", len(step_ids) == len(set(step_ids)), step_ids)
    chk(tag + "paint() hides every step card",
        'document.getElementById(p[0]).classList.add("hide")' in s)
    chk(tag + "back() cannot re-enter itself", 'if(t===CUR)t="s0";' in s)

    script = s[s.rfind("<script>"):s.rfind("</script>")]
    for op, cl, nm in (("{", "}", "braces"), ("(", ")", "parens"), ("[", "]", "brackets")):
        chk(tag + "script %s balanced" % nm, script.count(op) == script.count(cl),
            "%d/%d" % (script.count(op), script.count(cl)))

    demo = json.loads(re.sub(r"([A-Za-z]+):", r'"\1":', re.search(r"const DEMO=(\{.*?\});", s, re.S).group(1)))
    chk(tag + "demo cal size", len(demo["calScores"]) == len(demo["calErrors"]) == 1843)
    chk(tag + "demo dep size", len(demo["depScores"]) == len(demo["depErrors"]) == 1843)
    chk(tag + "demo error codes 0/1",
        set(demo["calErrors"]) <= {0, 1} and set(demo["depErrors"]) <= {0, 1})

    embedded = json.loads(re.search(r"const SAMPLE=(\".*?\");\n", s, re.S).group(1))
    chk(tag + "embedded sample equals the shipped CSV",
        embedded == io.open(SAMPLE, encoding="utf-8").read())

    # certificate parity with the paper implementation
    from errctl.conformal import SelectiveRiskController
    calS = np.array(demo["calScores"]); calE = np.array(demo["calErrors"])
    depS = np.array(demo["depScores"])
    # The page never hard-codes a threshold: it recomputes lambda_hat from DEMO
    # at run time.  So the honest check is that the in-page re-implementation
    # agrees with src/errctl (exact) and that the 4-decimal figure the page
    # prints is the one errctl actually produces.
    printed4 = {0.05: "0.0371", 0.10: "0.1899", 0.20: "0.7311"}
    lam10 = None
    for a in (0.05, 0.10, 0.20):
        c = SelectiveRiskController(alpha=a, delta=0.10, min_accept=25, grid_size=120,
                                    num="cp", denom="cp", select="bonferroni")
        c.fit(calS, calE)
        out = fit_lambda(demo["calScores"], demo["calErrors"], a)
        chk(tag + "fitLambda == errctl at a=%.2f" % a,
            out is not None and abs(out[0] - c.lambda_hat) < 1e-12 and out[1] == c.chosen_,
            "js=%s errctl=(%r,%s)" % ((round(out[0], 8), out[1]) if out else None, c.lambda_hat, c.chosen_))
        chk(tag + "page prints the errctl lambda at a=%.2f (%s)" % (a, printed4[a]),
            "%.4f" % c.lambda_hat == printed4[a] and printed4[a] in s, c.lambda_hat)
        if a == 0.10:
            lam10 = c.lambda_hat

    keep = depS <= lam10
    acc = float(keep.mean())
    derr = np.array(demo["depErrors"], dtype=float)
    oa = oracle_acc(depS, derr, 0.10)
    chk(tag + "demo acceptance at 10% is 0.7400976668", abs(acc - 0.7400976668) < 1e-9, acc)
    chk(tag + "demo oracle at 10% is 0.8166033641", abs(oa - 0.8166033641) < 1e-9, oa)
    chk(tag + "demo audit ratio at 10% is 0.9063122924", abs(acc / oa - 0.9063122924) < 1e-9, acc / oa)
    for val in ("74.0%", "26.0%", "0.1899", "0.906"):
        chk(tag + "preview shows " + val, val in s)
    chk(tag + "the result panel says the two parts add to the whole batch",
        ("两者相加就是全批" if lang == "zh" else "The two add up to the whole batch") in s)
    chk(tag + "the result panel limits the guarantee to the accepted part",
        ("保证只覆盖前一部分" if lang == "zh" else "The guarantee covers only the first part") in s)

chk("zh and en tables have identical key sets",
    set(json.loads(re.search(r"const T=(\{.*?\});", pages["zh"], re.S).group(1))) ==
    set(json.loads(re.search(r"const T=(\{.*?\});", pages["en"], re.S).group(1))))
chk("zh checklist pins Table 6 (5 pairs)",
    "5 \u5bf9</b>\u4e0d\u503c\u5f97\u82b1\u6807\u7b7e" in pages["zh"])
chk("zh checklist carries the denser rungs", "750\u20132,000" in pages["zh"])
chk("zh checklist no longer says 4 pairs", "4 \u5bf9\u4e0d\u503c\u5f97\u4fee" not in pages["zh"])
chk("en checklist pins Table 6 (5 pairs)",
    "the remaining <b>5</b> are not worth the labels" in pages["en"])
chk("en checklist carries the denser rungs", "750\u20132,000" in pages["en"])
chk("en page is not Chinese prose",
    "\u7ec6\u80de\u6ce8\u91ca" not in pages["en"])

# ---------------------------------------------------------------- sample CSV
raw = io.open(SAMPLE, encoding="utf-8").read()
lines = raw.strip().split("\n")
chk("sample header is set,score,error", lines[0] == "set,score,error")
cal = [l.split(",") for l in lines[1:] if l.startswith("cal,")]
dep = [l.split(",") for l in lines[1:] if l.startswith("dep,")]
chk("sample has 800 calibration rows", len(cal) == 800, len(cal))
chk("sample has 1200 deployment rows", len(dep) == 1200, len(dep))
chk("sample every cal row labelled", all(len(c) == 3 and c[2] in ("0", "1") for c in cal))
nlab = sum(1 for d in dep if len(d) == 3 and d[2] in ("0", "1"))
chk("sample has 400 labelled deployment rows", nlab == 400, nlab)
# The four numbers per alpha are recorded properties of the shipped file, not
# free parameters: they were measured when the sample was generated, and the
# validator fails if the CSV is ever regenerated with different behaviour.
cs = np.array([float(c[1]) for c in cal]); ce = np.array([int(c[2]) for c in cal])
ds = np.array([float(d[1]) for d in dep])
dl = np.array([-1 if d[2] == "" else int(d[2]) for d in dep])
lam_hat = {}
for a, want_lam, want_acc in ((0.05, 0.444118, 0.4150), (0.10, 0.687611, 0.6867),
                              (0.20, 0.905703, 0.9158)):
    c = SelectiveRiskController(alpha=a, delta=0.10, min_accept=25, grid_size=120,
                                num="cp", denom="cp", select="bonferroni")
    c.fit(cs, ce)
    lam_hat[a] = c.lambda_hat
    k = ds <= c.lambda_hat
    chk("sample lambda a=%.2f" % a, abs(c.lambda_hat - want_lam) < 1e-4, c.lambda_hat)
    chk("sample acceptance a=%.2f" % a, abs(k.mean() - want_acc) < 5e-4, float(k.mean()))
chk("sample lambda is monotone in alpha",
    lam_hat[0.05] < lam_hat[0.10] < lam_hat[0.20],
    (lam_hat[0.05], lam_hat[0.10], lam_hat[0.20]))
m = dl >= 0
kk = ds <= lam_hat[0.10]
ls, le, la = ds[m], dl[m].astype(float), kk[m]
oa = oracle_acc(ls, le, 0.10)
chk("sample audit ratio at 10% is 0.9228", abs(la.mean() / oa - 0.9228) < 5e-4, la.mean() / oa)
chk("sample labelled acceptance at 10% is 0.7175", abs(la.mean() - 0.7175) < 5e-4, la.mean())
chk("sample realised risk at 10% is 0.0767", abs(le[la].mean() - 0.0767) < 5e-4, le[la].mean())

print()
print("CHECKS=%d FAILS=%d" % (COUNT[0], len(fails)))
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
