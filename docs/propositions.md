# Propositions

This section states and proves three results about the selective-risk certificate of
`src/errctl/conformal.py`. Proposition 1 fixes what the conditional-binomial reading
of the plug-in baseline assumes, exhibits a finite-support law on which that reading
is invalid as deployed, and states the exact guarantee that the certificate gives
instead. Proposition 2 is a comparison theorem between the two multiplicity
conventions of the same controller, the Bonferroni share over the grid and the
fixed-sequence (step-down) scan, proved under an explicit condition on the feasible
set which is then measured on the stored benchmark. Proposition 3 prices a bounded
likelihood ratio, the weakest exchangeability substitute the paper considers, and
matches a positive weighted bound with an order-tight lower bound. All three
propositions separate what is proved from what is only measured; a list of what is
not proved is given at the end. Every number quoted below appears in a verification table with the artefact and
the filter it came from, and every number was recomputed by `work/_r30_verify.py`,
whose transcript is `work/_r30_out.txt`.

Notation follows `docs/methods.md`. A cell contributes a nonconformity score
`s = 1 - max_c proba`, where `max_c proba` is the largest predicted class
probability, so a small `s` means a confident label; acceptance at a threshold is
the event `s <= lambda`, the accepted set is `{s <= lambda}`, and
`err = 1{predicted class != reference class}`. The selective risk of a threshold is
the error rate among its accepted cells,

    R(lambda) = E[err . 1{s <= lambda}] / E[1{s <= lambda}] = A(lambda) / B(lambda),

where `A(lambda) = E[err . 1{s <= lambda}]` and `B(lambda) = E[1{s <= lambda}]` are
each a mean of independent bounded variables although their ratio is not. `alpha` is
the error budget and `delta` the confidence level, so `1 - delta` is the nominal
coverage; `m` is the number of grid points, `min_accept` the floor on the accepted
count, and `n` the calibration size. Candidates are acceptance counts
`k_1 < k_2 < ... < k_m` built by `_grid_indices`,
`k = unique(round(linspace(lo, n, min(grid_size, n - lo + 1))))` with
`lo = max(min_accept, 2)`, so the candidate thresholds `lambda_j`, the `k_j`-th
smallest calibration scores, ascend in `j`; `lambda_hat` is the delivered threshold.
Unless stated otherwise the calibration and deployment cells are exchangeable, drawn
from one law `P`; under covariate shift the weighted controller replaces `P` by an
importance-weighted law.


## Proposition 1

**Assumption (C), constancy of the conditional error rate.** There is one number
`pi` with `P(E = 1 | S = s) = pi` for `P_S`-almost every `s` in the acceptance
region `{s <= lambda_hat}`; equivalently, conditionally on the accepted set, the
errors of the accepted cells are independent Bernoulli(`pi`) draws with a common
`pi`. Under (C) the number of wrong accepted cells is `Binomial(k, pi)` with `k` the
accepted count, and Sharpe's inversion

    pi_upper = CP(1 - delta_m; n_wrong + 1, k - n_wrong),   P(pi <= pi_upper) >= 1 - delta_m,

is exact; that is the plug-in reading. A weaker structure licenses the same binomial
tail without (C): errors that are independent Bernoulli draws whose probabilities
are all at most a `pi` known a priori, or a bounded-difference martingale whose
predictable variance is at most `k pi (1 - pi)` for a known `pi`. What does not
substitute for (C) is a variance or rate proxy estimated from the accepted cells
themselves, since that is the quantity heterogeneity moves, nor a pointwise bound on
the accepted mean alone.

**Proposition 1.** (i) The plug-in reading is not distribution-free: the
conditional-binomial limit is a valid upper confidence limit for the selective risk
only under (C), or under an equivalent martingale or sub-Gaussian structure with a
priori constants. (ii) There is a law `P` on a finite support for which
`P(E = 1 | s <= lambda)` is strictly increasing in `lambda` over the acceptance
region and the deployed coverage of the plug-in is 0.5693 against a nominal 0.90, the
mechanism being that the delivered acceptance set is strictly larger than the
specified block of cells whose count fixes the bound. (iii) The certificate of
`SelectiveRiskController` needs no such assumption: under exchangeability, or correct
importance weights, it satisfies `P(R(lambda_hat) <= alpha) >= 1 - delta`, with the
algebra below, and the scope of that statement is the accepted set whose counts the
bounds were taken at. (iv) On the stored benchmark the constancy assumption is
rejected in a large majority of evaluations while the plug-in covers; the
counterexample in (ii) is a possibility result and the stored numbers are its
empirical motivation, not a demonstration of it.

### The negative half

The exact obstruction is that the errors of accepted cells need not be identically
distributed. Write the accepted cells' error probabilities as `pi_1, ..., pi_k` with
`pi_bar = k^{-1} sum_i pi_i`; the error count is Poisson-binomial, and that law is not
the binomial with the same mean. For four accepted cells with `pi = (0.5, 0.5, 0, 0)`,
so `pi_bar = 0.25`,

    P(sum <= 1) = 0.7500  >  0.7383 = P(Binomial(4, 0.25) <= 1),

so no `Binomial(k, pi_bar)` stochastically dominates the count, and the
Clopper-Pearson inversion has no licence of that form. The direction of the failure
matters: heterogeneity across the `pi_i` moves mass towards low counts, and a low
count is exactly what the inversion reads as evidence of a low error rate, which the
cells of high `pi_i` then falsify. On the four-atom law's own accepted set at
`lambda = 2/3` (`k = 300`, `pi_bar = 0.1683`) the largest excess of the mixture over
the same-mean binomial is +0.03973 at `t = 56`.

That algebraic obstruction does not by itself produce under-coverage, and we do not
claim that it does. At the crossing point of the Clopper-Pearson inversion, where the
critical count `x(pi)` is the smallest count whose limit reaches `pi`, a violation
needs `P(sum <= x(pi_bar) - 1) > delta_m`; sweeping 60 `(k, pi, delta_m)` cells and
3,710 vectors that share the homogeneous mean exactly, the worst value of
`P(sum <= x - 1) / delta_m` is 0.970, and the vector attaining it is the homogeneous
one. A score-dependent error rate on an otherwise regular score therefore did not
break this plug-in in any cell we examined: on four continuous-score controls, the labels
`flat`, `step_cont`, `ramp3` and `ramp1` of `_cont_law` in
`work/_r30_verify.py`, whose profiles are defined in that function, the plug-in
covers 0.9980 to 1.0000 against a nominal 0.90.

What does break it is a score with atoms. The delivered threshold is the score of the
`k_j`-th sorted calibration cell, but the accepted set is `{s <= lambda_hat}`, which
contains the whole tied block at that score; the plug-in's trial count `k_j` and its
counts then refer to a strict sub-block of the set it actually accepts, and the
omitted part of the block is admitted without appearing in the bound. The gap is
measured directly: the test-set acceptance mass exceeds the chosen rank divided by
`n` by +0.1183 for the plug-in and by +0.0852 for the certificate. On the four-atom
law

    P(s = 0) = P(s = 1/3) = P(s = 2/3) = P(s = 1) = 0.25,
    P(E = 1 | s) = 0.005, 0.05, 0.45, 0.45 for those four atoms,

the conditional error rate over the acceptance region is strictly increasing,

    P(E = 1 | s <= 0)   = 0.0050,   P(E = 1 | s <= 1/3) = 0.0275,
    P(E = 1 | s <= 2/3) = 0.1683,   P(E = 1 | s <= 1)   = 0.2388,

in the orientation of `docs/methods.md`, where acceptance is `s <= lambda` and small
scores are the confident ones; the statement mirrors if the score is oriented the
other way. With `alpha = delta = 0.10`, `n = 400`, `min_accept = 25`, `grid_size = 20`
and 4,000 repetitions under the single seed 20260927, the plug-in's coverage of
`R(lambda_hat) <= alpha` is 0.5693 against the nominal 0.90, with mean delivered
acceptance mass 0.6055 and mean delivered population risk 0.0884; the certificate, on
the same repetitions, is at 0.8050. Both failures are the tie: recomputing the same
two rules with the trial count taken from the whole tied block at the delivered
threshold, so that the rank and the accepted set coincide, gives coverage 1.0000 for
both. Two further controls on the same atoms bound the claim. With the same atoms and
a constant error rate 0.05 the plug-in covers 1.0000, but the delivered set is 0.9933
of the mass and the tie gap is only +0.0062; with a constant rate 0.095 the plug-in
covers 1.0000 while 0.9808 of the repetitions certify nothing at all, so that figure
is vacuous. A constant rate cannot produce a mid-block threshold that has both
acceptance and a large tie gap, so the constant-rate controls confirm the mechanism
rather than isolate it. The two decisive controls are the tie-complete recomputation
(1.0000) and the continuous-score family (0.9980 to 1.0000): the failure needs the
delivered set to be strictly larger than the specified one, which an atom at the
delivered threshold produces and a continuous score cannot.

### The positive half

The certificate does not invert a conditional binomial. It bounds the two
unconditional probabilities `A(lambda)` and `B(lambda)`, each of which is a mean of
independent bounded variables, and it pays a share for the grid. For candidate `j`
let `N_+(lambda_j)` be the number of calibration cells with `err = 1` and
`s <= lambda_j`. The controller computes

    U_j = CP(1 - delta/(2m); N_+(lambda_j) + 1, n - N_+(lambda_j)),        (1)
    L_j = CP(delta/(2m); k_j, n - k_j + 1),                                (2)
    ratio_j = U_j / max(L_j, 1e-12),                                       (3)

and `select="bonferroni"` delivers the largest `j` with `ratio_j <= alpha`. Three
facts give the guarantee. (i) At a fixed threshold `lambda`, `N_+(lambda)` is
`Binomial(n, A(lambda))` and the accepted count is `Binomial(n, B(lambda))`, both
exactly, because each is a sum of independent bounded indicators; the
Clopper-Pearson limits are therefore exact one-sided limits at level
`1 - delta/(2m)`, and (2) is also the exact `delta/(2m)` quantile of
`Beta(k_j, n - k_j + 1)`, which is the law of `B(lambda_j)` for a continuous score.
(ii) The choice of `lambda_hat` is data-dependent, so the `m` statements in each tail
are held jointly by a union bound, `delta/(2m)` per tail per grid point, for a total
of `2 m . delta/(2m) = delta`; this identity was verified directly, with
`delta_m = delta/(2m) = 0.1/40 = 0.002500` and `2 m delta_m = 0.1000 = delta`.
(iii) The ratio in (3) is a deterministic monotone algebraic step from the two
bounded quantities and so cannot invalidate them. Every feasible candidate therefore
satisfies `R(lambda_j) <= alpha` simultaneously with probability at least
`1 - delta`, and the rule inherits that property,

    P(R(lambda_hat) <= alpha) >= 1 - delta,

with no assumption on the data distribution beyond exchangeability of the
calibration and deployment cells, or correct importance weights, and none at all on
the conditional error rate. The three closed forms in the code were checked against
the definitions on a fitted instance, with maximum absolute difference 0.000e+00 in
each leg and in the ratio, and the candidate ranks ascend in `lambda`.

### What the certificate does not promise

Two limitations belong in the statement of the guarantee.

First, exchangeability is not decoration. Every stored arm loses coverage under
covariate shift, the certificate included: `crc` covers 0.9213 on `shift_mild_1000`
and 0.7512 on `shift_strong_1000` at a 20% budget, against 1.0000 and 0.9988 on
`exchangeable_2000`, and the certificate's coverage on the `transfer` family of the
real benchmark is 0.3972 at a 10% budget. The conditional plug-in is not singled out
by those numbers; it is weaker than the certificate there as well, 0.6388 and 0.3387
against 0.9213 and 0.7512.

Second, the counts in (1) and (2) are those of the first `k_j` sorted calibration
cells. For a score with no atom at the delivered threshold that block is the
delivered set, and the guarantee is about the deployed rule. For an atomic score it
is not: the guarantee is then about the specified block rather than the deployed set.
On the four-atom law of the negative half the deployed coverage of the certificate
is 0.8050 against a nominal 0.90, and taking both counts from the delivered block
restores 1.0000 over the same 4,000 repetitions. That variant is verified
empirically only, and is not proved here.

### Empirical corroboration on the stored benchmark

The stored numbers are the empirical motivation for the negative half and, on the
same evaluations, a check on the positive half. On the exchangeable `cellsplit`
family at `alpha = delta = 0.10` the certificate accepts a mean 0.5594 of the
evaluation cells with coverage 1.0000 and no violations, while the Clopper-Pearson
plug-in accepts 0.5844 with the same coverage, a paired difference of +0.0249 with
unit-cluster 95% confidence interval [+0.0194, +0.0308] over 570 evaluations in 19
units; the plug-in is strictly ahead in 0.9474 of units and 0.7579 of evaluations.
Averaged over the nine `(alpha, delta)` cells of that family the plug-in accepts
between 0.3661 and 0.8316 against the certificate's 0.3077 to 0.8099, with coverage
within 0.0035 of the certificate's in every cell and equal to it in six of the nine.
The two other plug-in arms behave the same way: at the same configuration the
Wilson margin accepts 0.5779 and the Hoeffding margin 0.2097 against the
certificate's 0.5594, with coverage 1.0000 for all three (V1.14). So the
Clopper-Pearson plug-in is systematically the more permissive rule, by the paired
+0.0249 of accepted cells at a 10% budget (V1.15), and it does not fail coverage
here. What the r26 benchmark is an empirical demonstration of is the failure of
the constancy assumption and the systematic permissiveness of the plug-in, not a
coverage failure of it; the coverage failure is demonstrated only by the
counterexample above.

What the stored benchmark does show about the reading is that its assumption is
false. Testing the constancy of the error rate among the accepted calibration cells
at the certificate's own threshold at `alpha = delta = 0.10`, with a chi-square test
of homogeneity over predicted classes and over score deciles within the accepted
set, the test rejects in 0.7751 of evaluations by predicted class and 0.9287 by score
decile on `cellsplit`, 0.9143 and 0.9500 on `within`, and 0.9394 and 0.9409 on
`transfer`, over 1,389 evaluations. The mean spread of the error rate across
predicted classes is 0.2016 on `cellsplit`, 0.2430 on `within` and 0.2095 on
`transfer`, and across score deciles 0.2048, 0.2367 and 0.2360, against accepted
risks of 0.0559, 0.0642 and 0.0662. The conditional error rate is not constant, by a
wide margin relative to its own level, in three quarters to over nine tenths of
evaluations.

Where the stored simulations do show under-coverage, it is covariate shift that
causes it and it affects every unweighted arm. The conditional plug-in covers 0.6068
and 0.3740 at a 20% budget on `shift_mild` and `shift_strong` in
`scripts/r09_simulation.py`; in the corresponding two settings of
`scripts/r17_simulation.py` it covers 0.6388 and 0.3387 while the certificate covers
0.9213 and 0.7512. In every non-shift setting, exchangeable or heterogeneous, the
plug-in's coverage is 0.9785 to 1.0000 in `r09` and 0.9962 to 1.0000 in `r17`.
Heterogeneity of the error rate among cells that share a score does not break the
plug-in in these simulations; what breaks it is a change of measure between
calibration and deployment. This matters for how the negative half should be read:
the four-atom law above is a possibility result about the reading, and the stored
heterogeneity settings are not a demonstration of it.

### Verification table 1

Shorthand: `S26` = `experiments/r26_plugin_baseline_summary.csv`, `E26` =
`experiments/r26_plugin_baseline.csv`, `S27` = `experiments/r27_fixedseq_summary.csv`,
`R27` = `experiments/r27_fixedseq.csv`, `R09`/`R17` =
`experiments/r09_simulation_summary.csv` / `experiments/r17_simulation_summary.csv`,
`P` = `work/predictions/*.npz`, `V` = `work/_r30_verify.py`. The last column is the
section of the transcript `work/_r30_out.txt` in which the value was recomputed.

| # | quantity | value | source: file; filter / expression | at |
|---|---|---|---|---|
| V1.1 | atom law, weights | s in {0, 1/3, 2/3, 1}, each with P = 0.25 | V, `P1_S` and `P1_W` | [1] |
| V1.2 | atom law, error rate | P(E=1\|s) = 0.005, 0.050, 0.450, 0.450 on those atoms | V, `P1_P` | [1] |
| V1.3 | acceptance-region rate | 0.0050, 0.0275, 0.1683, 0.2388 at lambda = 0, 1/3, 2/3, 1 (strictly increasing) | V, `_p1_pop(lam)` for the four atoms | [1] |
| V1.4 | exact non-dominance | P(sum <= 1) = 0.7500 > 0.7383 = P(Binomial(4, 0.25) <= 1) for pi = (0.5, 0.5, 0, 0) | V, `_pb_le` and `_binom_le` | [2] |
| V1.5 | mixture excess on the law | k = 300, pi_bar = 0.1683, largest excess +0.03973 at t = 56 | V, `_pb_le(ps2, t) - _binom_le(300, pi2, t)` | [2] |
| V1.6 | crossing-point sweep | worst `P(sum <= x_pi - 1)/delta_m` = 0.970 over 60 (k, pi, delta_m) cells and 3,710 vectors, argmax homogeneous | V, `_p1_crossing_sweep`, seed 20260927, 120 Dirichlet vectors per cell | [3] |
| V1.7 | Monte Carlo, atom law | plug-in coverage of `R(lambda_hat) <= alpha` = 0.5693; of the held-out risk 0.5695 (nominal 0.90) | V, `_arms_mc`, seed 20260927, n = 400, alpha = delta = 0.10, min_accept = 25, grid = 20, 4000 reps | [4] |
| V1.8 | Monte Carlo, atom law | certificate coverage 0.8050; mean delivered acceptance mass 0.6055; mean delivered population risk 0.0884 | V, `_arms_mc` and `SelectiveRiskController` on the same stream | [4] |
| V1.9 | tie gap | test acceptance mass minus chosen rank/n: +0.1183 plug-in, +0.0852 certificate | V, `_arms_mc`, `gap_plugin` and `gap_cert` | [4] |
| V1.10 | tie-complete recomputation | plug-in coverage 1.0000, certificate coverage 1.0000 | V, `_tiecomplete_mc` and `_setvalued_cert_mc` | [4] |
| V1.11 | continuous-score controls | flat 1.0000/1.0000/1.0000; step_cont 0.9990/0.9940/1.0000; ramp3 1.0000/0.9930/1.0000; ramp1 0.9980/0.9920/1.0000 (plug-in population / plug-in test / certificate) | V, `_cont_law` (profiles defined in the code) and `_arms_mc`, 1000 reps, seed 20260927 | [5] |
| V1.12 | constant-rate controls | rate 0.095: 1.0000 with 0.9808 of repetitions vacuous; rate 0.05: 1.0000, 0.1478 vacuous, mean mass 0.9933, tie gap +0.0062; rate 0.25: 1.0000, 1.0000 vacuous | V, `_atom_draw_p`, 4000 reps, seed 20260927 | [K] |
| V1.13 | certificate closed forms | max abs deviation 0.000e+00 for `upper_n`, `lower_d` and `bound_ratio`; delta_m = 0.1/40 = 0.002500; 2 m delta_m = 0.1000 = delta; chosen rank 514 | V, `SelectiveRiskController` on a fitted n = 800 ramp3 draw, seed 20260927, denominator guarded by `max(L, 1e-12)` | [6] |
| V1.14 | r26 headline arms | certificate 0.5594/1.0000/0.0576/0 violations; CP plug-in 0.5844/1.0000/0.0633/0; Wilson 0.5779/1.0000/0.0618; Hoeffding 0.2097/1.0000/0.0292 (accept/coverage/realised risk/violations) | S26, family = cellsplit & alpha = 0.10 & delta = 0.10 & arm in {certificate, plugin_cp_cond, plugin_wilson_cond, plugin_hoeffding_cond} | [B] |
| V1.15 | r26 paired difference | delta_accept +0.0249, unit-cluster 95% CI [+0.0194, +0.0308], n_eval 570, n_units 19; unit-level strictly ahead 0.9474, evaluation-level 0.7579 | E26, paired per (unit, clf, repeat) against the certificate, 2000 resamples | [B] |
| V1.16 | r26 nine cells | cellsplit: certificate accept 0.3077 to 0.8099 with coverage 0.9965 to 1.0000; CP plug-in 0.3661 to 0.8316 with coverage 0.9930 to 1.0000; largest shortfall 0.0035, equal in 6 of 9 | S26, family = cellsplit, all alpha, delta in {0.05, 0.10, 0.20} | [B2] |
| V1.17 | r26 other families | within alpha = 0.10: certificate 0.5338/0.8167, CP 0.5554/0.7833, Wilson 0.5493/0.8000; transfer alpha = 0.10: certificate 0.3676/0.3972, CP 0.3974/0.3657 | S26, family in {within, transfer}, delta = 0.10 (accept/coverage) | [A] |
| V1.18 | constancy, cellsplit | 1,389 evaluations over the three families, of which n = 449 here with 16 units; accepted risk 0.0559; spread across classes 0.2016 (median 0.1581), rejected 0.7751; across deciles 0.2048 (median 0.2049), rejected 0.9287 | P, certificate threshold at alpha = delta = 0.10, chi-square p < 0.05, groups with >= 10 accepted cells | [C] |
| V1.19 | constancy, within | n = 280 evals, 17 units; accepted risk 0.0642; 0.2430/0.2000/0.9143 and 0.2367/0.2327/0.9500 | P, same test | [C] |
| V1.20 | constancy, transfer | n = 660 evals, 41 units; accepted risk 0.0662; 0.2095/0.1780/0.9394 and 0.2360/0.2411/0.9409 | P, same test | [C] |
| V1.21 | r17 exchangeable and heterogeneous | `exchangeable_2000` crc_cond 0.9988/0.1260 (a = 0.10) and 0.9988/0.7219 (a = 0.20); `hetero_strong` 1.0000/0.0000 and 1.0000/0.0424; `exchangeable_2000` crc 1.0000 and 0.9988 | R17, method = crc_cond, setting as named (coverage/mean_accept) | [7] |
| V1.22 | r17 covariate shift | `shift_mild_1000` a = 0.20: crc_cond 0.6388/0.4136, crc 0.9213/0.2644; `shift_strong_1000` a = 0.20: crc_cond 0.3387/0.2694, crc 0.7512/0.1449; `shift_labelshift` a = 0.20: crc_cond 0.0000, crc 0.0600 | R17, method in {crc, crc_cond} | [7] |
| V1.23 | plug-in on non-shift settings | crc_cond coverage 0.9785 to 1.0000 in r09 over exchangeable_200/500/2000 and hetero_coarse/mild/strong; 0.9962 to 1.0000 in r17 over exchangeable_200/2000 and hetero_strong | R09 and R17, method = crc_cond, setting not starting with `shift` | [9] |
| V1.24 | plug-in under shift, r09 | `shift_mild` a = 0.20 crc_cond 0.6068 (mean_accept 0.4114); `shift_strong` a = 0.20 crc_cond 0.3740 (0.2674) | R09, method = crc_cond | [9] |
| V1.25 | manuscript rounding of V1.24 | the figures 0.607 and 0.374 quoted in the manuscript are its rounding of 0.6068 and 0.3740, which are covariate-shift settings, not heterogeneity settings | `docs/manuscript.md`; the sentence attributing them to the controlled heterogeneity of the Monte-Carlo study; the underlying values are V1.24 | [9] |

## Proposition 2

**Definitions.** Every arm of `scripts/r27_fixedseq.py` shares the candidate grid
`k_1 < ... < k_m` of the preceding section, hence ascending candidate thresholds
`lambda_1 <= ... <= lambda_m`. For a confidence share `t` write

    ratio_j(t) = CP(1 - t; N_+(lambda_j) + 1, n - N_+(lambda_j))
                 / max(CP(t; k_j, n - k_j + 1), 1e-12)

and call candidate `j` `t`-feasible when `ratio_j(t) <= alpha`. The Bonferroni arm
(`select="bonferroni"`) sets `t = delta/(2m)` at every candidate and delivers the
largest feasible index, `j_B = max {j : j is delta/(2m)-feasible}`. The step-down arm
(`select="fixed_sequence"`) sets `t = delta/2` and scans `j = 1, 2, ...`, stopping at
the first infeasible candidate, so it delivers `j_F =` (that index) `- 1`, or `m` if
the scan never fails.

**Prefix-feasible.** For the feasible set `F` of one arm let `j^- = min F` and
`j^+ = max F`. The geometry stored by `geometry` in `scripts/r27_fixedseq.py` is
`infeas_below = j^- - 1` (the script's `idx[0]`), `infeas_above = m - j^+` and
`feas_count = |F|`. We call `F` a *prefix* when `infeas_below = 0`, and a *strict
prefix* when in addition `feas_count = m - infeas_above`, which is the statement that
no feasible candidate sits above an infeasible one, so `F = {1, ..., h}` for some `h`.
For the step-down arm the scan delivers the top of its own feasible set exactly when
that set is a prefix, so on that arm `prefix` is equivalent to `j_F = j^+`. In the
analysis of `r27_fixedseq.csv` the flag `prefix` is this condition evaluated on the
step-down arm's own feasibility (the script's `base_ok`, the same `ok` array from
which the scan picks its point), while `base_prefix` and `base_strict_prefix` are the
same two conditions evaluated on the Bonferroni arm.

**Proposition 2.** Fix `alpha`, `delta`, `n` and the candidate grid.
(a) [pointwise dominance] `ratio_j(delta/2) <= ratio_j(delta/(2m))` for every `j`, so
the Bonferroni feasible set is contained in the step-down feasible set.
(b) [non-negative gain] If the Bonferroni feasible set is a strict prefix then
`j_F >= j_B`, so the step-down delivers at least as many candidates as Bonferroni, and
hence at least as many accepted cells in every test set.
(c) [reversal] The conclusion fails as soon as the Bonferroni feasible set is not a
prefix: if the first infeasible candidate in the ascending scan lies below `j_B`, then
`j_F < j_B`, and in the extreme case where the first candidate is infeasible the
step-down delivers nothing while Bonferroni delivers a later feasible point.

**Proof.** (a) `delta/(2m) <= delta/2` because `m >= 1`. `CP(1 - t; x+1, n-x)` is the
`1 - t` quantile of a Beta distribution and so non-increasing in `t`, while
`CP(t; k, n-k+1)` is the `t` quantile and so non-decreasing in `t`; raising `t` from
`delta/(2m)` to `delta/2` therefore cannot increase the numerator or decrease the
denominator, and the guard `max(., 1e-12)` preserves the inequality. (b) By (a) every
`delta/(2m)`-feasible `j` is `delta/2`-feasible, so `F_B` is contained in `F_F`. If
`F_B = {1, ..., h}` then `j_B = h`, every `j <= h` is `delta/2`-feasible, and the
scan's first failure is at an index greater than `h`; hence `j_F >= h = j_B`. (c) If
`F_B` contains a feasible `j_B` above an infeasible `j < j_B`, the scan fails at or
before `j`, so `j_F <= j - 1 < j_B`.

**The price of the grid.** The two rules differ only in the share charged per
candidate, `delta/(2m)` against `delta/2`, which is the multiplicity penalty of a
family of `m` candidates against a family of one. In the empirical-Bernstein legs of
the controller the same two shares appear as `log(3m/delta)` and `log(3/delta)`, so
the grid enters the Bonferroni margin through `log m` and does not enter the
step-down margin at all; the exact Clopper-Pearson legs inherit the same comparison
through the tail `delta/(2m)` against `delta/2` at a fixed count pair. Refining the
grid therefore widens the Bonferroni margin and leaves the step-down margin
unchanged, which is where the step-down's advantage grows, and it is the only place
it is bought: the step-down's protection rests on the ordered-scan argument, which
charges only the first false claim in the prespecified order, and that argument is
what requires the feasible set to be a prefix.

**A falsifiable prediction.** (P2-i) On any evaluation whose Bonferroni feasible set
is a strict prefix, the paired difference
`accept(step-down) - accept(Bonferroni)` is non-negative, so the count of evaluations
with a strictly negative difference is zero. (P2-ii) On evaluations whose Bonferroni
feasible set is not a prefix the prediction has no force, and if the scan's first
failure lies inside a hole the difference is negative.

**Test.** The data support the strict-prefix form exactly and refute the form the
paper's prose states. Over the 16,056 pooled evaluations at `delta = 0.10` whose
Bonferroni feasible set is a strict prefix, the step-down count is below the
Bonferroni count in a fraction 0.0000 of them, with a worst deficit of 0 cells, as
P2-i requires; the mean gain there is +0.0359 with 0.497 improved and 0.000 worsened.
By floor, over 18,090 evaluations per floor, the strict-prefix subsets number 0 at
`min_accept = 25`, 1,195 at 50, 5,099 at 100 and 9,762 at 200, with a deficit
fraction of 0.0000 in each and mean gains of +0.0152, +0.0271 and +0.0430 for the
last three. P2-ii is also observed: the number of non-prefix Bonferroni sets is 0, 0,
100 and 871 by floor, and over the pooled data the non-prefix Bonferroni sets carry a
deficit fraction of 0.0006 with a worst deficit of 808 cells.

**Weakening.** The proposition cannot be stated on the step-down arm's own prefix
condition, which is the condition `docs/methods.md` names and the condition the `r27`
column reports. On the 38,754 pooled evaluations whose step-down feasible set is a
prefix the mean gain is +0.0644, with 0.644 improved and 0.006 worsened and a deficit
fraction of 0.0059 whose worst value is 19,506 cells; that sample is dominated by
cases in which both feasible sets are prefixes, but it contains cases in which the
Bonferroni set has a hole below a higher feasible point, and there the scan stops
early. The empirically supported statement is therefore the strict Bonferroni-prefix
form of (b): where the Bonferroni feasible set is a strict prefix the step-down never
delivers fewer cells (0 in 16,056), and on the larger but weaker condition the gain is
positive on average, +0.0964 with unit-cluster 95% interval [+0.0780, +0.1188] over
96 units, but not pointwise. Outside the condition the mirror statement is negative
and large: on the 33,606 evaluations whose step-down feasible set is not a prefix the
mean gain is -0.2142, no evaluation improves, 0.403 worsen, and the unit-cluster
interval is [-0.3715, -0.2633] over 99 units.

**The floor.** The condition's rate and the gain move together with the floor on the
accepted count, which is what the prefix condition depends on. Averaged over all
families, budgets, deltas and grids the ascending step-down beats Bonferroni
(paired CI strictly above zero) in 0 of 81 configurations at `min_accept = 25`
(mean gain -0.3549), 14 of 81 at 50 (-0.0908), 60 of 81 at 100 (+0.0389) and 81
of 81 at 200 (+0.0575). At the headline
configuration (`cellsplit`, `alpha = delta = 0.10`, `min_accept = 100`,
`grid_size = 120`) the gain is +0.0735 with interval [+0.0572, +0.0900] over 570
evaluations in 19 units, with 0.8088 improved and 0.0018 worsened, and the same sign
holds on `within` (+0.0511 [+0.0359, +0.0694]) and `transfer` (+0.0475
[+0.0299, +0.0654]). At `min_accept = 25` the Bonferroni feasible set is never a
strict prefix, the first candidate is infeasible, and the scan delivers nothing, so
the floor is not a tuning device but the price of the prefix condition.

**The negative control.** Reversing the prespecified order of the same candidates
tests the same relative risks and removes the gain everywhere: the descending arm
beats Bonferroni in 0 of 81 configurations at every floor, with mean gains of -0.4958,
-0.4940, -0.4876 and -0.4699, no evaluation improving and worsening between 0.7028
and 0.7137 (V2.4). Conditioning on its own geometry, its gain is -0.4157 over 30,376 evaluations whose
feasible set reaches the top of its own scan order and -0.4924 over 41,984 whose scan
stops below. Since the descending arm tests the same candidates with the same ratios
and differs only in the order, this control shows that the gain is carried by the
direction of the order and not by the ratio.

**Grid refinement.** With `alpha = delta = 0.10` and `min_accept = 100`, the gap
between step-down and Bonferroni acceptance is +0.0503, +0.0735 and +0.0896 on
`cellsplit` at `m = 20, 120, 400`, +0.0326, +0.0511 and +0.0627 on `within`, and
+0.0255, +0.0475 and +0.0612 on `transfer`, while the descending control stays at
0.0046 on `cellsplit`, 0.0011 on `within` and 0.0017 on `transfer`, the same
at every `m`. The step-down acceptance itself is almost flat in `m`
(0.6185, 0.6335, 0.6350), so the widening gap is Bonferroni's loss of acceptance
(0.5683, 0.5600, 0.5454), which is what the `delta/(2m)` share and its `log m`
counterpart predict. The same refinement also shows the cost: the step-down's
coverage on `cellsplit` is 0.9632 at a 10% budget and 0.9842 at a 20% budget against
the certificate's 1.0000, so the extra acceptance is not free.

**An explicit reversal.** The reversal in (c) is not hypothetical. On the transfer
unit `transfer_Melanoma_Oct2522_to_Breast_Jun0524_fold1` with classifier `hgb`,
`min_accept = 200`, `grid_size = 20` and `alpha = 0.10`, the Bonferroni feasible flags
are `[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, ...]`: candidate index 6 is infeasible
while indices 10 and 11 are feasible, so the feasible set is not a prefix and
`j_B = 11`. The step-down scan fails at index 6 and delivers `j_F = 5` at
`delta = 0.05` and `0.10`, and `j_F = 6` at `delta = 0.20`. The delivered thresholds
are 0.205670 against 0.133262, and on the 7,057 test cells Bonferroni accepts 2,608
against the step-down's 1,848, a deficit of 760 cells; the stored `r27` rows for that
unit reproduce both counts.

### Verification table 2

Shorthand as in Table 1.

| # | quantity | value | source: file; filter / expression | at |
|---|---|---|---|---|
| V2.1 | configuration counts | ascending: 324 configurations, 155 with the paired gain above 0 and the CI above 0, 109 with the CI entirely below 0; both scan orders: 648, 155, 433 | S27, select = fixed_sequence, one row per (family, alpha, delta, min_accept, grid_size, scan_order) | [A] |
| V2.2 | headline configuration | cellsplit, alpha = delta = 0.10, min_accept = 100, grid = 120: n_eval 570, 19 units; bonferroni 0.5600, step-down 0.6335, descending 0.0046; paired +0.0735, CI [+0.0572, +0.0900]; improved 0.8088, worsened 0.0018; coverage 1.0000 (bonferroni) against 0.9632 (step-down) | S27, family = cellsplit & alpha = 0.10 & delta = 0.10 & min_accept = 100 & grid_size = 120, mean_accept / delta_accept / delta_lo / delta_hi | [I] |
| V2.3 | headline, other families | within: 360 evals, 20 units, 0.5334/0.5845/0.0011, +0.0511 [+0.0359, +0.0694], coverage 0.8333 against 0.7333; transfer: 1080 evals, 60 units, 0.3707/0.4181/0.0017, +0.0475 [+0.0299, +0.0654], coverage 0.3917 against 0.3917 | S27, same filter with family changed | [I] |
| V2.4 | floor table | ascending better (paired CI strictly above zero) 0/81 (-0.3549), 14/81 (-0.0908), 60/81 (+0.0389), 81/81 (+0.0575) for min_accept = 25, 50, 100, 200; descending 0/81 everywhere (-0.4958, -0.4940, -0.4876, -0.4699), with mean frac_worsened 0.7137, 0.7137, 0.7137, 0.7028 | S27, select = fixed_sequence, scan_order and min_accept as named; 81 cells = 3 families x 3 alphas x 3 deltas x 3 grids | [B] |
| V2.5 | prefix split | delta = 0.10, ascending: step-down prefix n = 38,754, +0.0644, improved 0.644, worsened 0.006; not prefix n = 33,606, -0.2142, 0.000, 0.403; strict prefix n = 36,013, +0.0580, 0.623, 0.000; Bonferroni prefix n = 17,027, +0.0393, 0.523, 0.001; Bonferroni strict prefix n = 16,056, +0.0359, 0.497, 0.000 | R27, delta = 0.10, ascending, paired per (unit, clf, repeat) against the Bonferroni arm | [C] |
| V2.6 | pointwise deficits | step-down count below Bonferroni: Bonferroni strict prefix 0.0000 (worst 0); Bonferroni prefix 0.0006 (worst 808); step-down prefix 0.0059 (worst 19,506) | R27, same pairing, `n_accepted` difference | [C] |
| V2.7 | unit-cluster bootstrap | step-down prefix: 96 units, +0.0964, CI [+0.0780, +0.1188]; not prefix: 99 units, -0.3170, [-0.3715, -0.2633]; Bonferroni strict prefix: 91 units, +0.0519, [+0.0424, +0.0621]; Bonferroni not strict: 99 units, -0.1034, [-0.1261, -0.0814] | R27, delta = 0.10, ascending, unit-cluster percentile bootstrap, 2000 resamples, seed 20260927 | [C] |
| V2.8 | hypothesis counts by floor | strict-Bonferroni-prefix n = 0, 1,195, 5,099, 9,762 of 18,090 at floors 25, 50, 100, 200, with deficit fraction 0.0000 (worst 0) and mean gains +0.0152, +0.0271, +0.0430 for floors 50, 100, 200; non-prefix Bonferroni sets n = 0, 0, 100, 871 | R27, delta = 0.10, ascending, pooled over families, alphas and grids, one floor at a time | [J] |
| V2.9 | descending control | feasible set reaches the top: n = 30,376, -0.4157, improved 0.000, worsened 0.452; stops below the top: n = 41,984, -0.4924, 0.000, 0.850 | R27, scan_order = descending, geometry of its own scan | [D] |
| V2.10 | grid refinement | gap step-down minus Bonferroni at m = 20/120/400, alpha = delta = 0.10, min_accept = 100: cellsplit +0.0503/+0.0735/+0.0896 (0.5683/0.5600/0.5454 against 0.6185/0.6335/0.6350); within +0.0326/+0.0511/+0.0627; transfer +0.0255/+0.0475/+0.0612; descending 0.0046 on cellsplit, 0.0011 on within and 0.0017 on transfer, the same at every m | S27, family and grid_size as named, mean_accept difference | [E] |
| V2.11 | coverage at the headline configuration | delta = 0.10, min_accept = 100, grid = 120: cellsplit a = 0.10 bonferroni 1.0000, ascending 0.9632, descending 1.0000, and a = 0.20 ascending 0.9842; transfer a = 0.20 bonferroni 0.2861, ascending 0.2630, descending 0.8500 | S27, family and alpha as named | [F] |
| V2.12 | pointwise dominance | 106,920 comparisons over 297 prediction files and delta in {0.05, 0.10, 0.20} (35,640 each), 0 violations of `ratio_fs > ratio_bonf + 1e-15` | P, one `SelectiveRiskController(select="bonferroni")` and one `(select="fixed_sequence")` per file at alpha = 0.10, min_accept = 25, grid = 120 | [G] |
| V2.13 | explicit reversal | transfer_Melanoma_Oct2522_to_Breast_Jun0524_fold1, clf hgb, repeat 0, alpha = 0.10, min_accept = 200, grid = 20: j_B = 11, j_F = 5 at delta = 0.05/0.10 and 6 at 0.20; feasible flags have a hole at index 6 (ratio 0.11904) with index 11 feasible (0.0955) at delta = 0.10; lambda 0.205670 against 0.133262; test cells 2,608 against 1,848 of 7,057, a deficit of 760 cells | P, controller `fit` on the unit's calibration folds; R27 rows for the same unit/clf/repeat/alpha reproduce n_accepted 2608 (bonferroni) and 1848 (fixed_sequence) | [H] |



## Proposition 3

**Assumption (W), bounded likelihood ratio.** The deployment law `Q` is absolutely
continuous with respect to the calibration law `P`, and its density ratio
`w(x) = dQ/dP(x)` satisfies `0 <= w(x) <= M` for `P`-almost every `x`. The
calibration cells `(X_i, Y_i)` are iid `P`, `Y` is the error indicator, and the
weights are known. If the weights are estimated, `M` bounds the estimate and the
sup-norm error enters as an additional term of the same order as the radius below.
Exchangeability itself is not assumed; the bounded ratio is the substitute.

For a fixed threshold `lambda` write `A_lambda = {s <= lambda}` and define

    A_Q(lambda) = E_P[w Y 1{A_lambda}],       B_Q(lambda) = E_P[w 1{A_lambda}],
    R_Q(lambda) = A_Q(lambda) / B_Q(lambda),
    V_A(lambda) = E_P[w^2 Y 1{A_lambda}],     V_B(lambda) = E_P[w^2 1{A_lambda}].

`V_A` and `V_B` are uncentred second moments, which is the form the Bernstein
radius uses. The target acceptance mass is `B_Q`, and `beta` is a lower bound on
it.

**Proposition 3.** (i) The weights control the second moments by the first:

    V_A(lambda) <= M A_Q(lambda),       V_B(lambda) <= M B_Q(lambda).

If in addition `R_Q(lambda) <= alpha` and `B_Q(lambda) >= beta`, then

    V_A(lambda) <= M alpha B_Q(lambda),       V_B(lambda) <= M B_Q(lambda).

(ii) [achievability] A weighted Bernstein certificate that uses these null
variance envelopes has coverage at least `1 - delta` and is non-vacuous once

    n >= C M log(1/delta) / (alpha beta)

for a universal constant `C`, uniformly over `P, Q` with `0 <= w <= M`.
(iii) [lower bound] For every `M >= 1`, `alpha in (0, 1/2)` and
`delta in (0, 1/2)`, there are `P, Q` with `0 <= w <= M` and a target acceptance
set of mass one such that every procedure with the uniform `1 - delta` guarantee
and a non-vacuous certificate at risk `alpha` needs

    n >= c M log(1/delta) / alpha

calibration cells, for a universal constant `c`.
(iv) [order tightness] The upper and lower bounds agree in `M` and `alpha` up to
constants and logarithms. The label price is therefore `Theta(M/(alpha beta))`
for `beta` bounded away from zero; the lower bound is stated at `beta = 1`, and
the constants and the logarithm are not claimed tight.

**Proof of (i).** Since `Y in {0,1}` and `w <= M`,
`w^2 Y 1{A} <= M w Y 1{A}` pointwise, so taking expectations gives
`V_A <= M A_Q`. The same argument with `Y` removed gives `V_B <= M B_Q`.
Under the risk constraint `A_Q = R_Q B_Q <= alpha B_Q`, the first inequality
becomes `V_A <= M alpha B_Q`.

**Proof of (ii).** For iid `Z_i in [0, M]` with mean `mu` and variance `v`, the
one-sided Bernstein inequality gives, for `L = log(2/delta')`,

    P( |mean(Z) - mu| >= sqrt(2 v L / n) + 2 M L / (3 n) ) <= 2 exp(-L).

Apply it to `Z_i = w_i Y_i 1{A}` with `mu = A_Q` and `v <= V_A`, and to
`Z_i = w_i 1{A}` with `mu = B_Q` and `v <= V_B`. On the same event, the sum of
the two radii is at most

    sqrt(2 M alpha B_Q L / n) + 2 M L / (3 n)
      + sqrt(2 M B_Q L / n) + 2 M L / (3 n).

Take the union over the `m` thresholds and the two sides, so
`delta' = delta/(2m)`. Requiring the first radius to be at most
`alpha B_Q / 4` and the second at most `B_Q / 4` gives
`n >= C M L / (alpha B_Q)`, and `B_Q >= beta` gives the stated rate. The
denominator envelope is replaced by the upper confidence limit `B^+` in an
implementation; on the event `B^+ >= B_Q`, which the same union covers, the
argument is unchanged. The accepted set is the largest grid threshold whose
upper numerator is at most `alpha` times its lower denominator, so the bound is
simultaneous over the data-dependent choice.

**Proof of (iii).** Let `P` put mass `1/M` on a stratum `H` and `1 - 1/M` on a
stratum `L`, set `w = M` on `H` and `w = 0` on `L`, and let `Q` be `P`
conditioned on `H`. Then `Q << P`, `0 <= w <= M`, and `E_P[w] = 1`, so `Q` is a
probability law. The target is supported on `H`. Take the acceptance set to be
all of `H`, so `B_Q = 1` and the selective risk is
`theta = P(Y = 1 | H)`. Calibration cells outside `H` have weight zero; the only
informative cells are the `m` cells from `H`, and conditional on `m` their
labels are Bernoulli(`theta`).

Consider `theta_0 = alpha` and `theta_1 = alpha (1 + c)` with `c in (0, 1/2)`.
The Bernoulli Kullback-Leibler divergence is at least `c_0 alpha` for a
universal `c_0 > 0`. Any test that certifies `theta <= alpha` at `theta_0` and
refuses the same claim at `theta_1`, with error at most `delta` at both points,
therefore needs `m >= c_1 log(1/delta) / alpha` informative labels. Since
`m` is Binomial(`n, 1/M`). A standard Poissonized version makes `m` Poisson
with mean `n/M`; the same two-point bound then gives
`n >= c_2 M log(1/delta) / alpha` for a procedure with uniform error at most
`delta`, and de-Poissonization changes only the constants. This is the lower bound at `beta = 1`.

**Proof of (iv).** Put `beta = 1` in (ii) and compare with (iii): both sides have
the same `M / alpha` dependence and the same `log(1/delta)` factor. The upper
bound is positive and constructive; the lower bound is a two-point impossibility
for any procedure, not for the particular controller. The two sides do not claim
the same constant, and the logarithm is not optimized.

**What the proposition says, and what it does not.** It is a positive result:
bounded likelihood ratio is enough for a finite-sample certificate, and the
price is labels rather than validity. It is also an explanation of the observed
vacuity. On the 60 transfer pair-folds of `r19`, the median effective sample
fraction is 0.365, so `1 / (ESS/n)` is about 2.74, and at `alpha = 0.10` the
median ratio of the effective sample needed to the effective sample observed is
2.23. The theorem predicts a price of this order, not a free repair. The
range-only Hoeffding arm used in the paper replaces the error-weighted variance
`M alpha B` by the range bound `M^2`, so its radius is larger by a factor of
about `sqrt(M / (alpha B))`; closing that gap is necessary for a
non-vacuous weighted arm but is not sufficient, because `r19` also shows that
the weights are not estimable across fields of view (blocked domain AUC median
0.568, below chance in 27 of 60 pair-folds). The proposition therefore prices
the assumption; it does not claim that the estimated weights of the transfer
pairs are good enough to pay it.

**Why not the sequential route.** The alternative would promote the e-process of
`r32` to a cross-batch main theorem. The stored benchmark does not support that
promotion: at `cellsplit`, `alpha = delta = 0.10`, the e-process needs 0.9114 of
the pool to reach a realised acceptance of 0.75, while the honest union-bound
rival needs 0.8646, a paired loss of +0.0468 [0.0229, 0.0715] for the
e-process. A theorem can be proved for the sequential object, but its empirical
face would be a loss to a simpler valid competitor, so it is not the main
theorem of this paper.

### Verification table 3

| # | quantity | value | source: file; filter / expression | at |
|---|---|---|---|---|
| V3.1 | weight price on the real transfer pairs | 60 pair-folds; median ESS/n 0.3651 (q10 0.1040, q90 0.6740); median w_p99 4.374; median w_max 5.886; maximum 34.26 | R19W, `experiments/r19_weights.csv`, `ess_pool / n_src_pool`, `w_p99`, `w_max` | [K] |
| V3.2 | weighted certificate at three budgets | alpha = 0.05 / 0.10 / 0.20: non-vacuous 1 / 30 / 80 of 180 units; median finite ESS*/ESS_obs 2.125 / 2.229 / 2.619; median ESS_obs 966.14; median rho* 0.2200 | R19E, `experiments/r19_ess.csv`, filter on `alpha`, finite `ess_star_ratio` | [K] |
| V3.3 | order check for the price | median 1/(ESS/n) = 2.739; observed median ESS*/ESS_obs at alpha = 0.10 = 2.229; ratio 0.814 | R19W and R19E, medians over the 60 pair-folds and the 180 units | [K] |
| V3.4 | two-point lower bound, alpha = 0.05 / 0.10 / 0.20 | c = 0.5, delta = 0.10: Bernoulli KL 0.005062 / 0.010896 / 0.025732; informative labels m 317.97 / 147.71 / 62.55 | exact arithmetic from the formula in the proof, recomputed by `scripts/r73_label_price.py` | [L] |

## What is not proved here

- No proof is offered that the plug-in must under-cover whenever the conditional
  error rate varies with the score. The exact-arithmetic obstruction is proved
  (V1.4), and the crossing-point sweep (V1.6) and the continuous-score controls
  (V1.11) are evidence that a varying rate on an otherwise regular score does not
  produce under-coverage in the classes examined.
- The coverage figures on the four-atom law (V1.7 to V1.10) are Monte Carlo with one
  seed and 4,000 repetitions, not exact arithmetic; they are many Monte Carlo
  standard errors below nominal, but they are not a proof.
- The set-valued (tie-complete) variant of the certificate is empirically valid on
  the atom law but is not proved; the positive half applies to the counts at the
  selected candidate, and the set-valued variant's statistics are not the same
  functions of those counts.
- Proposition 2 is proved for the strict Bonferroni-prefix hypothesis. No proof is
  offered for the step-down arm's own prefix condition, and V2.5 shows that the
  statement fails pointwise there.
- Proposition 3 is order-tight, not constant-tight. Its lower bound is proved for
  the all-accept decision at `beta = 1` and uses a rare-stratum construction; the
  extension to a general accepted set and to estimated weights is not proved here.
  The empirical r19 numbers are a consistency check on the order, not a validation
  of the weighted controller on the transfer pairs.
- No cross-batch online error-control theorem is claimed. The stored sequential
  benchmark is reported in `r32` and is not promoted, because it loses to the
  honest union-bound rival at the pool sizes in this paper.
- The `0.607` and `0.374` figures that `docs/manuscript.md` attributes to controlled
  heterogeneity are its rounding of the conditional plug-in's coverage under
  covariate shift, 0.6068 and 0.3740 (V1.24, V1.25).
  In the stored heterogeneous and exchangeable settings the plug-in covers 0.9785 to
  1.0000 (V1.23). The attribution is noted here because Proposition 1(iv) depends on
  it; the manuscript itself is outside the scope of this section.


## Reproduction

The transcript is `work/_r30_out.txt` and is reproducible from the repository root
with

    python -u work/_r30_verify.py r26 r27 r27b p1

or, to write it as ASCII without a byte-order mark,
`python -u work/_r30_runner.py work/_r30_out.txt`, where `python` is any Python 3.10
with numpy, pandas and scipy. The section tags `[1]` to `[9]` for the derived laws, `[A]` to
`[C]` for r26, `[A]` to `[H]` for r27 and `[I]` to `[K]` for the two controls are
section labels of that transcript. The stored artefacts are
`experiments/r26_plugin_baseline.csv` and `_summary.csv`,
`experiments/r27_fixedseq.csv` and `_summary.csv`,
`experiments/r09_simulation_summary.csv`, `experiments/r17_simulation_summary.csv`
and the prediction files `work/predictions/*.npz`. The Monte-Carlo laws of
Proposition 1 are defined in `work/_r30_verify.py` (`P1_S`, `P1_P`, `P1_W`,
`P1_P_CONST`, `P1_P_CONST05`, `P1_P_CONST095`, `_atom_draw`, `_atom_draw_p`,
`_cont_law`) and are driven by the single seed 20260927; the constancy test of `[C]`
uses two worker processes. Table values are quoted as printed by the script.