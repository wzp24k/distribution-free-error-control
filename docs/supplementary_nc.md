# Supplementary Information

Distribution-free error control for cell type annotation in spatial proteomics

This file holds the extended results that are summarised in the main text. It carries eleven supplementary figures and twenty supplementary tables, and it uses the same certificate, split fractions, repeat counts and metric definitions as the main text throughout. Every table names the released result file it is derived from, so each number can be recomputed from the `experiments/` directory without rerunning the pipeline. The display items are numbered in their own series, Supplementary Figs. 1 to 11 and Supplementary Tables 1 to 20, and are cited from the main text by those numbers.

## Contents

- Supplementary Fig. 1. The controller
- Supplementary Fig. 2. Label free diagnosis of the weighted certificate
- Supplementary Fig. 3. The label cost of restoring the guarantee
- Supplementary Fig. 4. Acceptance against realised risk, and the certificate yield per target label
- Supplementary Fig. 5. The fixed-sequence step-down, its grid dependence and its validity
- Supplementary Fig. 6. Marker consistency and downstream effects
- Supplementary Fig. 7. Marker rule violation under definition and threshold conventions
- Supplementary Fig. 8. Marker audit across label sources and populations
- Supplementary Fig. 9. Accepted versus deferred marker consistency
- Supplementary Fig. 10. Does the accepted set distort the spatial readout
- Supplementary Fig. 11. Targeted repair and the label cost of a transfer repair
- Supplementary Table 1. The cross-field-of-view family at field-of-view resolution
- Supplementary Table 2. Cross-tissue transfer with a target-domain calibration sample
- Supplementary Table 3. Label-shift reweighting as an alternative to the weighted certificate, cross-tissue transfer
- Supplementary Table 4. Which cohort pairs can be repaired, and at what label cost
- Supplementary Table 5. Model adaptation against calibration of the frozen model, at equal label cost
- Supplementary Table 6. The label cost of the guarantee, by split family, at a 10% budget
- Supplementary Table 7. Acceptance at a matched realised risk, random split
- Supplementary Table 8. Acceptance at a matched guarantee, on the budget grid
- Supplementary Table 9. The fixed-sequence step-down across split families
- Supplementary Table 10. Ensemble confidence versus the certificate, random split, nominal 10% budget
- Supplementary Table 11. Conservative plug-in baselines and the confidence-level sweep
- Supplementary Table 12. The price of validity at a controlled calibration size
- Supplementary Table 13. The certificate around seven models, random split, at the shared budgets
- Supplementary Table 14. Identified sets and convention-free verdicts for the marker audit
- Supplementary Table 15. Targeted-repair arms within and across cohorts
- Supplementary Table 16. The targeted-repair arms at field-of-view resolution
- Supplementary Table 17. External validation: acceptance, risk and coverage
- Supplementary Table 18. The layer on an independent study, where the split is exchangeable
- Supplementary Table 19. Two further independent cohorts, a shared-panel control and a real change of antibody panel
- Supplementary Table 20. The layer around two published classifiers, on two cohort pairs

## Supplementary Figures

**Supplementary Fig. 1 | The controller. A** The distribution of the score s(x) = 1 - p_max on the calibration set, with misclassified cells highlighted, and the selected threshold. B The empirical selective risk as a function of the threshold, with the certified upper bound; the threshold is the largest value at which the bound meets the budget. C Schematic of the accept / abstain split and the selection rule.

%%IMG: figures/svg/fig3_method_full.png

**Supplementary Fig. 2 | Label free diagnosis of the weighted certificate. A** Effective sample size of the estimated density ratio against the domain classifier area under the curve, for every cross tissue pair, where the markers separate the random cell split from the held out whole images and most units fall below an effective sample size of 0.5, so the weights concentrate and the ratio is not estimable. B Fraction of the weight mass held by the top five per cent of cells for each ordered cohort pair, where a maximum weight of 2 to 17 shows that a few cells carry the extrapolation. C Effective sample size the weighted bound would need in order to be non vacuous at each budget, against the effective sample size observed in the source pool, where every unit sits above the diagonal, so the weights would have to vanish. D Error curve total variation that needs target labels against the score marginal total variation that is visible without target labels, with a Spearman correlation of 0.35, so the violation is invisible without labels.

%%IMG: figures/svg/figS5_weight_shift_full.png

**Supplementary Fig. 3 | The label cost of restoring the guarantee.** The controller is calibrated on k cells drawn from the deployment domain and evaluated on the remainder, for the random-split, cross-field-of-view and cross-tissue families. A Acceptance as a function of k, with the oracle as a dashed line and the k = 0 source-only certificate as a filled square. B The fraction of evaluations in which the certificate is willing to answer at all; it is zero for every family below a hundred labels, and non-vacuous coverage falls no lower than 0.92 (the within family at a hundred labels) and 0.996 (transfer at two thousand), so the cost of validity is paid in throughput rather than in correctness. C Per unit, the number of target labels needed to reach validity against the label-free score shift, measured as the Kolmogorov-Smirnov distance between the source calibration scores and the deployment scores; units that never reach validity inside the grid are drawn as open symbols on the top line, and the Spearman correlation is computed with their k censored at the largest balanced value.

%%IMG: figures/svg/fig8_label_ladder_full.png

**Supplementary Fig. 4 | Acceptance against realised risk, and the certificate yield per target label. A** Mean acceptance as a function of pooled realised risk at a 10% budget for the random-split, cross-field-of-view and cross-tissue families; the target-calibrated certificate, the oracle (dashed), the k = 0 source-calibrated certificate (filled diamonds) and the partial-unit points of the ladder (open circles) are distinguished, and the dotted diagonal is acceptance equal to risk. B Certificate yield, as auto-annotated cells per target label reviewed, against the number of target labels k at a 10% budget; solid segments are the balanced range in which every unit of the family contributes, dashed segments are partial-unit points and are shown but never claimed, and the right axis is the marginal return in extra auto-annotated cells per hundred additional labels. Panel A is the efficiency comparison and Panel B is the same accounting in the units a practitioner spends: the certificate's yield is negative for the exchangeable families and positive only under transfer, where what it buys first is validity.

%%IMG: figures/svg/fig10_yield_full.png

**Supplementary Fig. 5 | The fixed-sequence step-down, its grid dependence and its validity. A** Mean acceptance against candidate-grid size for Bonferroni, the ascending step-down and the reversed step-down, one line per split family at a 10% budget, a 0.10 confidence level and a 100-cell floor; the gap to Bonferroni widens with the grid because only Bonferroni carries log(m). B Coverage of the budget against the nominal 1 - delta for every configuration and arm, with Wilson intervals, so that the constructions that remain valid are separable from the ones that only accept more.

%%IMG: figures/svg/figS7_fixedseq_full.png

**Supplementary Fig. 6 | Marker consistency and downstream effects. A** Marker-rule violation rate by cohort and label source. B Violation rate among accepted cells against deferred cells. C Neighbourhood enrichment computed from predicted labels against the same statistic computed from expert labels; points off the diagonal indicate that annotation errors distort the reported biology.

%%IMG: figures/svg/fig6_markers_full.png

**Supplementary Fig. 7 | Marker rule violation under definition and threshold conventions. A** Violation rate of the expert label against its own published rule, per cell class, for each cohort, with the cohort pool drawn as a horizontal bar and the 5% gate as a dotted line, where classes with no fully evaluable rule are drawn as open circles. B Violation rate over all evaluable classes against the positivity percentile, 60 to 95, for each cohort, where solid lines use all cells, dashed lines use the mixture positive cells and the dash dot line marks the mixture crossing point that is the default.

%%IMG: figures/svg/figS3_marker_sensitivity_full.png

**Supplementary Fig. 8 | Marker audit across label sources and populations. A** Cell pooled violation rate by cohort and label source, where bars read the depot semicolon as a disjunction and circles read it as a conjunction, and the 5% gate is the dashed line. B Per population violation rate for the same conventions with the repaired rule text, against the 5% gate, restricted to populations of at least 50 cells. C Pooled rate over all sixty eight positivity, reading and rule text conventions, per cohort and label source, with the median of each group as a horizontal bar. D Agreement of the depot's two label sources on the same cells, where the coarse kappa is 0.66 for colorectal and 0.82 for GVHD, and the bars give the fraction of shared cells that agree, 0.73 and 0.71 for colorectal and 0.85 and 0.78 for GVHD.

%%IMG: figures/svg/figS4_marker_sources_full.png

**Supplementary Fig. 9 | Accepted versus deferred marker consistency. A** Paired violation rate among deferred cells minus the rate among accepted cells, per cohort and classifier, for each of the sixty eight positivity, reading and rule text conventions at the three budgets, where a positive value is the ordering the manuscript reports, the units are the distinct lambda configuration pools and estimability requires at least 75 evaluable cells on both sides. B The median of the same paired difference per convention, where the dashed line is zero and the marginal histograms show that the direction is universal and the effect size is not.

%%IMG: figures/svg/figS6_accept_deferred_full.png

**Supplementary Fig. 10 | Does the accepted set distort the spatial readout. A** Absolute distortion of the neighbourhood enrichment statistic, computed from predicted labels in place of expert labels, for the cells the controller accepts against all cells, one point per unit and budget, the dashed diagonal marking equality, so a point below the line is a unit whose accepted cells distort the spatial readout less than the batch does. B The pooled and per-cohort difference in that distortion, all cells minus accepted cells, as a cell-weighted mean with a 95% interval bootstrapped over images and the number of cells behind each estimate; a negative value means the accepted cells distort the readout more than the batch they were drawn from. Sources: `experiments/r21_downstream.csv` and the matched-weight re-run `scripts/r21b_matched_downstream.py`.

%%IMG: figures/svg/fig9_downstream_full.png

**Supplementary Fig. 11 | Targeted repair and the label cost of a transfer repair. A** Coverage of the error budget on the cross-field-of-view family by arm (the y axis is named 'coverage of the error budget'), under the certificate's own vacuity-inclusive readout (filled circles), under the joint readout that scores an empty accepted set as a failure (open squares), and under the conditional readout P(covered | accepted set non-empty) (open triangles), at 5% and 10% budgets with the nominal line drawn. B The paired difference in the joint quantity against the plain certificate for the three score-stratified and Mondrian arms, with unit-bootstrap intervals, all crossing zero in the published twenty-unit design; the same comparison at field-of-view resolution is Table 18. C The transfer label-cost ladder: the smallest target sample at which each arm returns to covering the budget under the vacuity-inclusive readout, in which an empty accepted set counts as covering, against the error budget, with the acceptance attained at that cost tabulated in Table 17; under the non-vacuous readout no arm reaches the budget at any k in the grid.

%%IMG: figures/svg/figS8_repair_full.png

## Supplementary Tables

**Supplementary Table 1 |** The cross-field-of-view family at field-of-view resolution

| method | error budget | coverage | coverage lo | coverage hi | joint coverage | joint lo | joint hi | non-vacuous fraction | mean accept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crc | 0.05 | 0.84 | 0.79 | 0.89 | 0.72 | 0.64 | 0.78 | 0.87 | 0.53 |
| crc | 0.10 | 0.79 | 0.72 | 0.85 | 0.72 | 0.64 | 0.79 | 0.93 | 0.73 |
| crc | 0.20 | 0.86 | 0.80 | 0.91 | 0.82 | 0.76 | 0.89 | 0.97 | 0.92 |
| crc_cond | 0.05 | 0.81 | 0.75 | 0.87 | 0.71 | 0.63 | 0.78 | 0.90 | 0.55 |
| crc_cond | 0.10 | 0.77 | 0.70 | 0.83 | 0.71 | 0.64 | 0.78 | 0.94 | 0.74 |
| crc_cond | 0.20 | 0.84 | 0.78 | 0.90 | 0.82 | 0.76 | 0.88 | 0.98 | 0.93 |
| crc_mondrian | 0.05 | 0.85 | 0.79 | 0.91 | 0.65 | 0.56 | 0.72 | 0.79 | 0.22 |
| crc_mondrian | 0.10 | 0.82 | 0.76 | 0.89 | 0.75 | 0.66 | 0.82 | 0.93 | 0.40 |
| crc_mondrian | 0.20 | 0.91 | 0.86 | 0.95 | 0.88 | 0.82 | 0.93 | 0.97 | 0.65 |
| crc_temp | 0.05 | 0.86 | 0.81 | 0.91 | 0.73 | 0.66 | 0.80 | 0.87 | 0.53 |
| crc_temp | 0.10 | 0.79 | 0.72 | 0.85 | 0.72 | 0.64 | 0.79 | 0.93 | 0.73 |
| crc_temp | 0.20 | 0.86 | 0.80 | 0.91 | 0.83 | 0.76 | 0.89 | 0.97 | 0.92 |
| conformal_set | 0.05 | 0.76 | 0.70 | 0.81 | 0.71 | 0.65 | 0.77 | 0.95 | 0.56 |
| conformal_set | 0.10 | 0.76 | 0.69 | 0.82 | 0.73 | 0.67 | 0.79 | 0.98 | 0.70 |
| conformal_set | 0.20 | 0.90 | 0.86 | 0.94 | 0.88 | 0.82 | 0.92 | 0.98 | 0.86 |
| naive | 0.05 | 0.55 | 0.47 | 0.63 | 0.52 | 0.44 | 0.61 | 0.98 | 0.67 |
| naive | 0.10 | 0.56 | 0.47 | 0.64 | 0.54 | 0.46 | 0.63 | 0.99 | 0.83 |
| naive | 0.20 | 0.83 | 0.76 | 0.89 | 0.82 | 0.75 | 0.88 | 0.99 | 0.95 |
| oracle | 0.05 | 0.997 | 0.99 | 1.00 | 0.94 | 0.90 | 0.98 | 0.94 | 0.67 |
| oracle | 0.10 | 1.00 | 1.00 | 1.00 | 0.95 | 0.92 | 0.99 | 0.95 | 0.81 |
| oracle | 0.20 | 1.00 | 1.00 | 1.00 | 0.98 | 0.95 | 0.997 | 0.98 | 0.93 |
| crc_eb | 0.05 | 1.00 | 1.00 | 1.00 | 0.04 | 0.01 | 0.07 | 0.04 | 0.03 |
| crc_eb | 0.10 | 0.95 | 0.91 | 0.98 | 0.51 | 0.44 | 0.59 | 0.57 | 0.44 |
| crc_eb | 0.20 | 0.91 | 0.86 | 0.95 | 0.79 | 0.71 | 0.86 | 0.88 | 0.86 |
| accept_all | 0.05 | 0.05 | 0.02 | 0.08 | 0.05 | 0.02 | 0.08 | 1.00 | 1.00 |
| accept_all | 0.10 | 0.26 | 0.19 | 0.34 | 0.26 | 0.19 | 0.34 | 1.00 | 1.00 |
| accept_all | 0.20 | 0.78 | 0.70 | 0.85 | 0.78 | 0.70 | 0.85 | 1.00 | 1.00 |

Notes: For four cohorts and 96 field-of-view units (`scripts/r48_fov_units.py`), one annotated image held out per unit: coverage of the error budget and the joint readout P(covered and non-empty), each with its own cluster-bootstrap interval over units (2,000 resamples), the conditional readout P(covered | non-empty), the vacuity fraction and mean acceptance, for the certificate, its conditional and Mondrian variants, the temperature-scaled and conformal-set baselines, the naive threshold and the oracle, at 5%, 10% and 20% budgets; 1,728 evaluations per cell, being the 288 stored prediction files crossed with the six calibration draws. Sources: `experiments/r48_metrics.csv` and `experiments/r48_coverage_ci.csv`; `scripts/r48_fov_units.py`.

**Supplementary Table 2 |** Cross-tissue transfer with a target-domain calibration sample

| budget α | calibration | acceptance | non-vacuous fraction | coverage (all) | coverage (non-vacuous) |
| --- | --- | --- | --- | --- | --- |
| 0.05 | source only (k = 0) | 0.18 | 0.49 | 0.51 | 0.00 |
| 0.05 | source only, doubled pool | 0.21 | 0.53 | 0.47 | 0.01 |
| 0.05 | weighted source | 0.00 | 0.01 | 0.994 | 0.00 |
| 0.05 | score-conditional | 0.08 | 0.37 | 0.68 | 0.13 |
| 0.05 | uncalibrated | 0.36 | 0.84 | 0.22 | 0.07 |
| 0.05 | oracle | 0.06 | 0.53 | 0.95 | 0.90 |
| 0.05 | target, k = 200 | 0.00 | 0.00 | 1.00 | n/a |
| 0.05 | target, k = 500 | 0.00 | 0.00 | 1.00 | n/a |
| 0.05 | target, k = 1,000 | 0.00 | 0.01 | 1.00 | 1.00 |
| 0.05 | target, k = 2,000 | 0.01 | 0.07 | 1.00 | 1.00 |
| 0.10 | source only (k = 0) | 0.37 | 0.64 | 0.39 | 0.04 |
| 0.10 | source only, doubled pool | 0.41 | 0.69 | 0.34 | 0.05 |
| 0.10 | weighted source | 0.08 | 0.17 | 0.84 | 0.07 |
| 0.10 | score-conditional | 0.22 | 0.57 | 0.48 | 0.10 |
| 0.10 | uncalibrated | 0.57 | 0.91 | 0.16 | 0.07 |
| 0.10 | oracle | 0.12 | 0.72 | 0.97 | 0.96 |
| 0.10 | target, k = 200 | 0.00 | 0.00 | 1.00 | n/a |
| 0.10 | target, k = 500 | 0.01 | 0.02 | 1.00 | 1.00 |
| 0.10 | target, k = 1,000 | 0.02 | 0.09 | 1.00 | 1.00 |
| 0.10 | target, k = 2,000 | 0.04 | 0.16 | 0.999 | 0.997 |
| 0.20 | source only (k = 0) | 0.68 | 0.79 | 0.27 | 0.08 |
| 0.20 | source only, doubled pool | 0.72 | 0.86 | 0.23 | 0.10 |
| 0.20 | weighted source | 0.28 | 0.44 | 0.65 | 0.21 |
| 0.20 | score-conditional | 0.45 | 0.79 | 0.38 | 0.22 |
| 0.20 | uncalibrated | 0.84 | 0.98 | 0.11 | 0.09 |
| 0.20 | oracle | 0.33 | 0.93 | 0.98 | 0.97 |
| 0.20 | target, k = 200 | 0.01 | 0.02 | 1.00 | 1.00 |
| 0.20 | target, k = 500 | 0.05 | 0.15 | 1.00 | 1.00 |
| 0.20 | target, k = 1,000 | 0.10 | 0.28 | 1.00 | 1.00 |
| 0.20 | target, k = 2,000 | 0.15 | 0.39 | 0.999 | 0.997 |

Notes: 180 evaluations per cell (12 cohort pairs × 5 folds × 3 classifiers × 10 random draws). Non-vacuous coverage restricts to evaluations whose accepted set is non-empty; an empty accepted set counts as a vacuous success in coverage (all). Nominal coverage is 1 −δ = 0.90. Sources: `experiments/r14_targetcal_summary.csv`, with the per-evaluation rows in `experiments/r14_targetcal.csv`; `scripts/r14_targetcal.py`.

**Supplementary Table 3 |** Label-shift reweighting as an alternative to the weighted certificate, cross-tissue transfer

| arm | non-vacuous (5% / 10% / 20%) | holds at 10% | ESS/n |
| --- | --- | --- | --- |
| covariate density ratio (the weighted certificate) | 1 / 30 / 80 | 2 of 30 | 0.36 |
| label shift, EM priors | 6 / 23 / 42 | 0 of 23 | 0.28 |
| label shift, true priors (oracle control) | 2 / 16 / 75 | 0 of 16 | 0.62 |

Notes: Non-vacuous units are the units in which the bound can be quoted at all, out of 180 certificate units per budget; holds counts the non-vacuous units whose realised target risk is at or below the budget; ESS/n is the pooled effective sample size of the weights inside the source pool. The oracle-prior arm uses the true deployment class frequencies, so it removes the prior-estimation error and isolates the effect of the correction itself. Sources: `experiments/r24b_labelshift.csv`; `scripts/r24b_labelshift.py`.

**Supplementary Table 4 |** Which cohort pairs can be repaired, and at what label cost

| cohort pair | oracle acceptance (10%) | k to reach validity (10%) | oracle acceptance (20%) | k to reach validity (20%) |
| --- | --- | --- | --- | --- |
| breast to GVHD | 0.32 | 500 | 0.68 | 200 |
| colorectal to breast | 0.21 | 500 | 0.55 | 200 |
| colorectal to GVHD | 0.14 | 500 | 0.33 | 250 |
| melanoma to GVHD | 0.19 | 500 | 0.39 | 250 |
| GVHD to breast | 0.13 | 750 | 0.53 | 500 |
| melanoma to breast | 0.15 | 2,000 | 0.44 | 500 |
| melanoma to colorectal | 0.10 | 1,500 | 0.25 | 500 |
| breast to colorectal | 0.05 | never | 0.23 | 1,000 |
| breast to melanoma | 0.01 | never | 0.05 | never |
| colorectal to melanoma | 0.00 | never | 0.16 | never |
| GVHD to colorectal | 0.12 | never | 0.29 | 500 |
| GVHD to melanoma | 0.02 | never | 0.05 | never |

Notes: The `tgt_k` arm of the same run as Table 3. A pair *reaches* a budget when, at some k <= 2,000, it returns at least 5 non-vacuous decisions out of 150 and covers at least 90% of them; `never` means no k in the grid reaches it. Oracle acceptance is what a threshold chosen with knowledge of the evaluation labels achieves on the same cells, averaged over all 150 evaluations. The grid is k = 0, 200, 250, 500, 750, 1,000, 1,500 and 2,000 target labels (`scripts/r14c_ladder_dense.py`); 25, 50 and 100 are also run and never reach. Sources: `experiments/r14c_ladder_dense.csv` (the dense grid), with the `tgt_k` arm of `experiments/r14_targetcal_summary.csv`; `scripts/r14c_ladder_dense.py`.

**Supplementary Table 5 |** Model adaptation against calibration of the frozen model, at equal label cost

| budget | k | adapted (k/2 adapted, k/2 calibrated) | frozen, all k labels for calibration | ratio | oracle, adapted | oracle, frozen |
| --- | --- | --- | --- | --- | --- | --- |
| 5% | 1,000 | 0.01 | 0.00 | 4.8 | 0.35 | 0.09 |
| 5% | 2,000 | 0.12 | 0.01 | 8.8 | 0.43 | 0.10 |
| 10% | 1,000 | 0.18 | 0.02 | 7.4 | 0.52 | 0.15 |
| 10% | 2,000 | 0.36 | 0.04 | 9.2 | 0.60 | 0.17 |
| 20% | 1,000 | 0.47 | 0.10 | 4.8 | 0.79 | 0.36 |
| 20% | 2,000 | 0.66 | 0.15 | 4.5 | 0.85 | 0.39 |

Notes: The adapted arm refits the classifier on the source training set plus half of the k target cells and calibrates the controller on the other half; the frozen arm spends all k labels on calibration. The frozen column is the `tgt_k` arm of the Table 3 run, so the two columns are separate executions of the same folds and seeds rather than two draws from one code path, and the ratio column is a comparison across those two executions. The matched half-budget arm of the adaptation run, which calibrates the frozen model on the same k/2 = 1,000 cells, gives 0.03 at k = 2,000, against 0.02 for the Table 3 run at k = 1,000, so the two runs agree, and inside the adaptation run it fixes the adapted arm at 0.36 against 0.03, a 13.7-fold lift. The stored adapted level is reproducible: with the BLAS thread pool pinned to a single thread, two independent executions return 0.36 exactly and agree to 0.000e+00 on mean acceptance and on both coverage readouts (Methods, Reproducibility), so the cross-execution ratio in this table is a reproducible comparison. The 0.15 reported for the same recipe in an earlier run is the logistic-regression-only subtotal of the classifier mix, not an instance of thread drift; measured with the pool pinned, that subtotal is 0.16 over the two repeats run here, and the two figures are within the sampling error of the same quantity (Methods, Reproducibility). Oracle columns are thresholds chosen with knowledge of the evaluation labels, on the adapted and on the frozen score respectively. Sources: `experiments/r14b_targetadapt_summary.csv`, with the frozen arm of `experiments/r14_targetcal_summary.csv`; `scripts/r14_targetcal.py` (`--adapt`).

**Supplementary Table 6 |** The label cost of the guarantee, by split family, at a 10% budget

| family | k | acceptance | coverage (non-vacuous) | oracle acceptance |
| --- | --- | --- | --- | --- |
| random split | 0 | 0.56 | 1.00 | 0.75 |
| random split | 200 | 0.19 | 1.00 | 0.84 |
| random split | 500 | 0.34 | 1.00 | 0.71 |
| cross-FOV | 0 | 0.53 | 0.76 | 0.70 |
| cross-FOV | 100 | 0.01 | 0.92 | 0.71 |
| cross-FOV | 200 | 0.14 | 1.00 | 0.71 |
| cross-FOV | 500 | 0.53 | 1.00 | 0.81 |
| cross-FOV | 1,000 | 0.43 | 1.00 | 0.66 |
| cross-tissue | 0 | 0.37 | 0.04 | 0.12 |
| cross-tissue | 500 | 0.01 | 1.00 | 0.12 |
| cross-tissue | 1,000 | 0.02 | 1.00 | 0.12 |
| cross-tissue | 2,000 | 0.04 | 0.997 | 0.12 |

Notes: The controller is calibrated on k cells drawn from the deployment domain and evaluated on the rest. Coverage is restricted to non-vacuous decisions; the k = 0 row is the source-only certificate. The oracle peeks at the evaluation labels and is an upper bound, not a method. The cross-tissue rows are the run of Table 3; the random-split and cross-field-of-view rows are a second run whose k = 0 rows reproduce Table 1, so the two families are internally consistent but are not drawn from the same code path as the cross-tissue rows. n = 19, 20 and 60 units for the random-split, cross-field-of-view and cross-tissue families, and that is the k = 0 row: a unit whose pool cannot supply k labels is dropped from the ladder, so a large-k row rests on fewer units, at a 10% budget the cross-field-of-view k = 500 row is 15 of the 20 units and the random-split k = 500 row is 7 of its 19. Table 6 carries the unit count next to every row. Sources: `experiments/r20_ladder.csv`; `scripts/r20_label_ladder.py`.

**Supplementary Table 7 |** Acceptance at a matched realised risk, random split

| realised risk | certificate | oracle | uncalibrated | certificate / oracle | certificate / uncalibrated |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.33 | 0.59 | 0.58 | 0.57 | 0.57 |
| 0.10 | 0.56 | 0.75 | 0.75 | 0.74 | 0.75 |
| 0.15 | 0.70 | 0.85 | 0.84 | 0.82 | 0.83 |
| 0.20 | 0.80 | 0.91 | 0.91 | 0.88 | 0.88 |
| 0.30 | 0.88 | 0.97 | 0.97 | 0.91 | 0.91 |

Notes: The certificate, the oracle and the uncalibrated baseline are each held to the same realised selective risk; acceptance is the mean fraction of cells retained, and the ratio is certificate over oracle. Sources: `experiments/r16_matched_risk_ratio.csv`, with the per-evaluation rows in `experiments/r16_matched_risk.csv`; `scripts/r16_review_budget.py`.

**Supplementary Table 8 |** Acceptance at a matched guarantee, on the budget grid

| risk target | split family | certificate | certificate conditional | conformal prediction set | uncalibrated threshold | temperature scaled | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | random split | 0.01 | 0.03 | infeasible | infeasible | infeasible | infeasible |
| 0.10 | random split | 0.56 | 0.58 | infeasible | infeasible | infeasible | 0.59 |
| 0.20 | random split | 0.80 | 0.83 | infeasible | 0.41 | 0.41 | 0.91 |
| 0.10 | cross field of view | 0.09 | 0.11 | infeasible | infeasible | infeasible | 0.53 |
| 0.20 | cross field of view | 0.53 | 0.38 | 0.27 | infeasible | 0.36 | 0.87 |
| 0.10 | cross tissue transfer | infeasible | infeasible | infeasible | infeasible | infeasible | infeasible |

Notes: Every arm is held to the same guarantee, that the realised selective risk sits at or below the risk target on 1.00 of the evaluation units of its family, and the entry is the largest mean acceptance over the budget grid 0.02 to 0.50 that reaches it, so a larger number is the more useful rule at the same promise. An empty accepted set counts as covered, which is why a feasible arm can carry a small acceptance. `infeasible` means that no budget on the grid holds the target on every unit. The random split is 570 evaluations over 19 units, 3 classifiers and 10 calibration repeats, the cross-field-of-view family 360 over 20 units, 3 classifiers and 6 repeats, and cross-tissue transfer 1,080 over 60 cohort pairs, 3 classifiers and 6 repeats. The uncalibrated and temperature-scaled arms place their threshold on the calibration selective risk at the same nominal budget. Sources: `experiments/r68_equal_guarantee.csv`; `scripts/r68_equal_guarantee.py`.

**Supplementary Table 9 |** The fixed-sequence step-down across split families

| family | error budget | confidence | Bonferroni | ascending step-down | reversed step-down | uncalibrated | paired delta accept | delta lo | delta hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random-split | 0.05 | 0.05 | 0.31 | 0.42 | 0.00 | 0.54 | 0.11 | 0.07 | 0.16 |
| random-split | 0.05 | 0.10 | 0.54 | 0.62 | 0.01 | 0.70 | 0.07 | 0.05 | 0.10 |
| random-split | 0.05 | 0.20 | 0.77 | 0.80 | 0.13 | 0.87 | 0.03 | 0.01 | 0.05 |
| random-split | 0.10 | 0.05 | 0.33 | 0.44 | 0.00 | 0.54 | 0.11 | 0.07 | 0.15 |
| random-split | 0.10 | 0.10 | 0.55 | 0.63 | 0.01 | 0.70 | 0.08 | 0.05 | 0.12 |
| random-split | 0.10 | 0.20 | 0.78 | 0.81 | 0.13 | 0.87 | 0.03 | 0.02 | 0.06 |
| random-split | 0.20 | 0.05 | 0.36 | 0.46 | 0.00 | 0.54 | 0.11 | 0.07 | 0.15 |
| random-split | 0.20 | 0.10 | 0.56 | 0.65 | 0.01 | 0.70 | 0.09 | 0.05 | 0.14 |
| random-split | 0.20 | 0.20 | 0.78 | 0.83 | 0.14 | 0.87 | 0.05 | 0.02 | 0.09 |
| cross-field-of-view | 0.05 | 0.05 | 0.35 | 0.40 | 0.00 | 0.49 | 0.05 | 0.02 | 0.08 |
| cross-field-of-view | 0.05 | 0.10 | 0.53 | 0.58 | 0.00 | 0.64 | 0.06 | 0.04 | 0.08 |
| cross-field-of-view | 0.05 | 0.20 | 0.73 | 0.76 | 0.07 | 0.79 | 0.03 | 0.01 | 0.05 |
| cross-field-of-view | 0.10 | 0.05 | 0.36 | 0.42 | 0.00 | 0.49 | 0.06 | 0.03 | 0.09 |
| cross-field-of-view | 0.10 | 0.10 | 0.53 | 0.59 | 0.00 | 0.64 | 0.06 | 0.04 | 0.09 |
| cross-field-of-view | 0.10 | 0.20 | 0.73 | 0.76 | 0.07 | 0.79 | 0.03 | 0.01 | 0.06 |
| cross-field-of-view | 0.20 | 0.05 | 0.37 | 0.44 | 0.00 | 0.49 | 0.07 | 0.04 | 0.09 |
| cross-field-of-view | 0.20 | 0.10 | 0.54 | 0.60 | 0.00 | 0.64 | 0.06 | 0.04 | 0.09 |
| cross-field-of-view | 0.20 | 0.20 | 0.73 | 0.77 | 0.07 | 0.79 | 0.04 | 0.01 | 0.07 |
| cross-tissue | 0.05 | 0.05 | 0.17 | 0.21 | 0.00 | 0.32 | 0.04 | 0.02 | 0.06 |
| cross-tissue | 0.05 | 0.10 | 0.36 | 0.41 | 0.00 | 0.52 | 0.05 | 0.04 | 0.07 |
| cross-tissue | 0.05 | 0.20 | 0.66 | 0.70 | 0.04 | 0.80 | 0.04 | 0.02 | 0.07 |
| cross-tissue | 0.10 | 0.05 | 0.18 | 0.22 | 0.00 | 0.32 | 0.05 | 0.02 | 0.07 |
| cross-tissue | 0.10 | 0.10 | 0.37 | 0.43 | 0.00 | 0.52 | 0.06 | 0.04 | 0.07 |
| cross-tissue | 0.10 | 0.20 | 0.66 | 0.71 | 0.04 | 0.80 | 0.05 | 0.02 | 0.08 |
| cross-tissue | 0.20 | 0.05 | 0.19 | 0.24 | 0.00 | 0.32 | 0.05 | 0.03 | 0.08 |
| cross-tissue | 0.20 | 0.10 | 0.38 | 0.45 | 0.00 | 0.52 | 0.07 | 0.05 | 0.10 |
| cross-tissue | 0.20 | 0.20 | 0.67 | 0.72 | 0.04 | 0.80 | 0.06 | 0.03 | 0.09 |

Notes: For the random-split, cross-field-of-view and cross-tissue families, at 5%, 10% and 20% error budgets, confidence levels 0.05, 0.10 and 0.20, 25-, 50-, 100- and 200-cell floors and 20-, 120- and 400-candidate grids: mean acceptance and coverage of the budget for Bonferroni, the ascending and the reversed step-down and the uncalibrated baseline, the paired acceptance difference against Bonferroni with its unit-bootstrap interval, and the geometry of the feasible set on the grid, including whether it is a prefix of the scan order. Sources: `experiments/r27_fixedseq_summary.csv`, with the per-evaluation rows in `experiments/r27_fixedseq.csv`; `scripts/r27_fixedseq.py`.

**Supplementary Table 10 |** Ensemble confidence versus the certificate, random split, nominal 10% budget

| base learner | ensemble score | ECE | acceptance | coverage (controller, same score) | acceptance (uncalibrated) | coverage (uncalibrated) |
| --- | --- | --- | --- | --- | --- | --- |
| hgb | predictive entropy | 0.04 | 0.67 | **1.00** | 0.82 | 0.63 |
| hgb | max mean probability | 0.04 | 0.67 | **1.00** | 0.83 | 0.66 |
| hgb | mutual information | 0.04 | 0.64 | **1.00** | 0.81 | 0.63 |
| logreg | predictive entropy | 0.05 | 0.52 | **1.00** | 0.74 | 0.53 |
| logreg | max mean probability | 0.05 | 0.53 | **1.00** | 0.75 | 0.53 |
| logreg | mutual information | 0.05 | 0.49 | **1.00** | 0.71 | 0.54 |

Notes: Coverage is the fraction of evaluations whose realised selective risk met the budget; the controller is applied to the same ensemble score as the uncalibrated baseline. Sources: `experiments/r11_ensemble.csv`; `scripts/r11_ensemble.py`.

**Supplementary Table 11 |** Conservative plug-in baselines and the confidence-level sweep

| family | error budget | confidence | certificate | uncalibrated | oracle | plug-in CP | plug-in Hoeffding | plug-in Wilson | certificate vacuity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random-split | 0.05 | 0.05 | 0.998 | 0.56 | 0.99 | 0.996 | 1.00 | 0.998 | 0.40 |
| random-split | 0.05 | 0.10 | 1.00 | 0.58 | 0.99 | 1.00 | 1.00 | 1.00 | 0.22 |
| random-split | 0.05 | 0.20 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 0.14 |
| random-split | 0.10 | 0.05 | 0.996 | 0.56 | 0.99 | 0.996 | 1.00 | 0.996 | 0.36 |
| random-split | 0.10 | 0.10 | 1.00 | 0.58 | 0.99 | 1.00 | 1.00 | 1.00 | 0.21 |
| random-split | 0.10 | 0.20 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 0.13 |
| random-split | 0.20 | 0.05 | 0.996 | 0.56 | 0.99 | 0.993 | 1.00 | 0.996 | 0.33 |
| random-split | 0.20 | 0.10 | 1.00 | 0.58 | 0.99 | 0.998 | 1.00 | 1.00 | 0.21 |
| random-split | 0.20 | 0.20 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 0.12 |
| cross-field-of-view | 0.05 | 0.05 | 0.85 | 0.64 | 1.00 | 0.83 | 1.00 | 0.83 | 0.34 |
| cross-field-of-view | 0.05 | 0.10 | 0.85 | 0.57 | 0.98 | 0.80 | 0.95 | 0.80 | 0.24 |
| cross-field-of-view | 0.05 | 0.20 | 0.92 | 0.80 | 1.00 | 0.90 | 0.92 | 0.90 | 0.18 |
| cross-field-of-view | 0.10 | 0.05 | 0.83 | 0.64 | 1.00 | 0.83 | 1.00 | 0.83 | 0.33 |
| cross-field-of-view | 0.10 | 0.10 | 0.82 | 0.57 | 0.98 | 0.78 | 0.95 | 0.80 | 0.22 |
| cross-field-of-view | 0.10 | 0.20 | 0.92 | 0.80 | 1.00 | 0.90 | 0.92 | 0.90 | 0.18 |
| cross-field-of-view | 0.20 | 0.05 | 0.83 | 0.64 | 1.00 | 0.82 | 1.00 | 0.83 | 0.31 |
| cross-field-of-view | 0.20 | 0.10 | 0.80 | 0.57 | 0.98 | 0.78 | 0.93 | 0.80 | 0.21 |
| cross-field-of-view | 0.20 | 0.20 | 0.92 | 0.80 | 1.00 | 0.90 | 0.92 | 0.90 | 0.18 |
| cross-tissue | 0.05 | 0.05 | 0.55 | 0.22 | 0.92 | 0.50 | 1.00 | 0.54 | 0.54 |
| cross-tissue | 0.05 | 0.10 | 0.40 | 0.16 | 0.89 | 0.37 | 0.60 | 0.38 | 0.39 |
| cross-tissue | 0.05 | 0.20 | 0.29 | 0.11 | 0.92 | 0.25 | 0.30 | 0.25 | 0.25 |
| cross-tissue | 0.10 | 0.05 | 0.54 | 0.22 | 0.92 | 0.49 | 1.00 | 0.52 | 0.54 |
| cross-tissue | 0.10 | 0.10 | 0.40 | 0.16 | 0.89 | 0.37 | 0.57 | 0.37 | 0.39 |
| cross-tissue | 0.10 | 0.20 | 0.29 | 0.11 | 0.92 | 0.23 | 0.30 | 0.25 | 0.24 |
| cross-tissue | 0.20 | 0.05 | 0.52 | 0.22 | 0.92 | 0.48 | 1.00 | 0.51 | 0.52 |
| cross-tissue | 0.20 | 0.10 | 0.38 | 0.16 | 0.89 | 0.36 | 0.54 | 0.37 | 0.37 |
| cross-tissue | 0.20 | 0.20 | 0.27 | 0.11 | 0.92 | 0.23 | 0.29 | 0.24 | 0.23 |

Notes: For the random-split, cross-field-of-view and cross-tissue families at 5%, 10% and 20% error budgets and confidence levels 0.05, 0.10 and 0.20: mean acceptance, coverage of the budget, mean realised risk and the vacuity rate for the certificate, the uncalibrated threshold, an empirical oracle threshold, and plug-in thresholds with exact Clopper-Pearson, Hoeffding and Wilson margins on the conditional error rate among accepted cells, together with the paired differences and their unit-bootstrap intervals and the per-predicted-class homogeneity tests of the plug-in's constancy assumption. Sources: `experiments/r26_plugin_baseline_summary.csv`, with the per-evaluation rows in `experiments/r26_plugin_baseline.csv`; `scripts/r26_plugin_baseline.py`.

**Supplementary Table 12 |** The price of validity at a controlled calibration size

| cohort | oracle | 25 | 50 | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **10% error budget** | | | | | | | | |
| breast | 0.98 | 0.00 | 0.00 | 0.04 | 0.42 | 0.83 | 0.90 | – |
| GVHD | 0.94 | 0.00 | 0.00 | 0.03 | 0.35 | 0.71 | 0.80 | 0.80* |
| melanoma | 0.75 | 0.00 | 0.00 | 0.00 | 0.06 | 0.28 | 0.46 | 0.57 |
| colorectal | 0.41 | 0.00 | 0.00 | 0.00 | 0.01 | – | – | – |
| **20% error budget** | | | | | | | | |
| breast | 1.00 | 0.32 | 0.39 | 0.79 | 0.97 | 0.995 | 0.998 | – |
| GVHD | 1.00 | 0.21 | 0.12 | 0.55 | 0.90 | 0.98 | 0.998 | 0.999* |
| melanoma | 0.97 | 0.05 | 0.02 | 0.24 | 0.50 | 0.73 | 0.87 | 0.92 |
| colorectal | 0.64 | 0.00 | 0.00 | 0.09 | 0.20 | – | – | – |

Notes: Balanced panel per cohort: the test set is a fixed quarter of the pool and the calibration set is drawn from the remainder at the stated size, ten draws per evaluation unit, with the oracle recomputed on the same test cells. Acceptance is a mean over the cohort's evaluation units, the convention of Table 1, and the tuned gradient-boosting learner of Table 15 is the score throughout. Coverage at a 10% budget lies in 0.95-1.00 for every cell of the panel. A dash marks a size the cohort's pool cannot supply for at least three evaluation units and an asterisk a two-unit cell. Sources: `experiments/r53_calibration_size.csv`; `scripts/r53_calibration_size.py`.

**Supplementary Table 13 |** The certificate around seven models, random split, at the shared budgets

| model | acceptance 5% | acceptance 10% | acceptance 15% | acceptance 20% | oracle 10% | oracle 20% | coverage 10% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic regression | 0.25 | 0.50 | 0.65 | 0.74 | 0.68 | 0.88 | 1.00 |
| random forest | 0.35 | 0.57 | 0.70 | 0.81 | 0.79 | 0.93 | 1.00 |
| gradient boosting (tuned) | 0.41 | 0.61 | 0.75 | 0.85 | 0.79 | 0.92 | 1.00 |
| perceptron, 256-128 | 0.29 | 0.52 | 0.66 | 0.77 | 0.72 | 0.91 | 1.00 |
| soft vote of the three | 0.41 | 0.62 | 0.77 | 0.84 | 0.80 | 0.95 | 1.00 |
| bagged gradient boosting (5) | 0.42 | 0.62 | 0.77 | 0.85 | 0.80 | 0.95 | 1.00 |
| bagged vote (5) | 0.37 | 0.60 | 0.74 | 0.82 | 0.79 | 0.96 | 1.00 |
| **three-learner mean (headline)** | **0.33** | **0.56** | **0.70** | **0.80** | **0.75** | **0.91** | **1.00** |

Notes: Acceptance and oracle acceptance are means over the nineteen evaluation units of the random-split family, the convention of Table 1, and coverage is the vacuity-inclusive readout. Every model is refit from scratch on the same folds as the headline, and the three base learners reproduce it to the fourth decimal. The last row is the equal-weight mean of the first three, which is the headline configuration. Sources: `experiments/r52_stronger.csv`; `scripts/r52_stronger_score.py`.

**Supplementary Table 14 |** Identified sets and convention-free verdicts for the marker audit

| cohort | label source | cell type | min rate | max rate | n conventions above 5% | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Colorectal | expert | APC | 0.00 | 0.00 | 0 | below |
| Colorectal | expert | CD3T | 0.07 | 1.00 | 68 | above |
| Colorectal | expert | Endothelial | 0.05 | 0.60 | 68 | above |
| Colorectal | HQ non-expert | APC | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | Bcell | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | CD3T | 0.06 | 0.06 | 4 | above |
| Colorectal | HQ non-expert | CD4T | 0.00 | 0.01 | 0 | below |
| Colorectal | HQ non-expert | CD8T | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | Endothelial | 0.01 | 0.01 | 0 | below |
| Colorectal | HQ non-expert | Fibroblast | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | Neutrophil | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | Plasma | 0.00 | 0.00 | 0 | below |
| Colorectal | HQ non-expert | Treg | 0.00 | 0.00 | 0 | below |
| GVHD | expert | Neutrophil | 0.26 | 0.93 | 68 | above |
| GVHD | expert | Plasma | 0.20 | 0.95 | 68 | above |
| GVHD | HQ all | APC | 0.13 | 0.998 | 68 | above |
| GVHD | HQ non-expert | APC | 0.14 | 0.14 | 4 | above |
| GVHD | HQ non-expert | Bcell | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | CD3T | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | CD4T | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | CD8T | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Endothelial | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Fibroblast | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Macrophage | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Mast | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Monocyte | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Neutrophil | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Plasma | 0.00 | 0.00 | 0 | below |
| GVHD | HQ non-expert | Treg | 0.00 | 0.00 | 0 | below |
| Melanoma | expert | CD8T | 0.20 | 0.90 | 68 | above |
| Breast | expert | all evaluable | 0.00 | 0.88 | 52 | mixed |
| Colorectal | expert | all evaluable | 0.01 | 0.90 | 64 | mixed |
| Colorectal | HQ reference | all evaluable | 0.00 | 0.88 | 56 | mixed |
| Colorectal | HQ all | all evaluable | 0.00 | 0.80 | 60 | mixed |
| Colorectal | HQ non-expert | all evaluable | 0.00 | 0.07 | 2 | mixed |
| GVHD | expert | all evaluable | 0.02 | 0.90 | 60 | mixed |
| GVHD | HQ reference | all evaluable | 0.00 | 0.90 | 58 | mixed |
| GVHD | HQ all | all evaluable | 0.00 | 0.91 | 60 | mixed |
| GVHD | HQ non-expert | all evaluable | 0.00 | 0.25 | 2 | mixed |
| Melanoma | expert | all evaluable | 0.04 | 0.87 | 64 | mixed |

Notes: For every cohort, label source and cell type: the minimum and maximum violation rate over the sixty-eight positivity, reading and rule-text conventions, the conventions attaining each extreme, the exact Clopper-Pearson interval at each, the number of conventions in which the rate exceeds a 5% gate, and whether the verdict is the same under every convention. Complements Supplementary Figs. 7 and 9. Sources: `experiments/r28_bounds_by_marker.csv` and `experiments/r28_bounds_pooled.csv`; `scripts/r28_marker_bounds.py`.

**Supplementary Table 15 |** Targeted-repair arms within and across cohorts

| family | arm | error budget | coverage all | coverage nonvac | coverage cond | mean accept | delta nonvac vs plain | delta lo | delta hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross-field-of-view | plain | 0.05 | 0.83 | 0.50 | 0.75 | 0.36 | - | - | - |
| cross-field-of-view | plain | 0.10 | 0.82 | 0.59 | 0.76 | 0.53 | - | - | - |
| cross-field-of-view | plain | 0.20 | 0.92 | 0.74 | 0.90 | 0.73 | - | - | - |
| cross-field-of-view | sc3 | 0.05 | 0.95 | 0.54 | 0.92 | 0.26 | 0.03 | -0.05 | 0.13 |
| cross-field-of-view | sc3 | 0.10 | 0.92 | 0.63 | 0.88 | 0.42 | 0.03 | -0.05 | 0.13 |
| cross-field-of-view | sc3 | 0.20 | 0.98 | 0.76 | 0.97 | 0.61 | 0.03 | -0.05 | 0.13 |
| cross-field-of-view | sc10 | 0.05 | 0.98 | 0.32 | 0.93 | 0.08 | -0.05 | -0.15 | 0.07 |
| cross-field-of-view | sc10 | 0.10 | 0.96 | 0.64 | 0.95 | 0.32 | -0.05 | -0.15 | 0.07 |
| cross-field-of-view | sc10 | 0.20 | 1.00 | 0.73 | 1.00 | 0.57 | -0.05 | -0.15 | 0.07 |
| cross-field-of-view | mond_class | 0.05 | 0.95 | 0.50 | 0.91 | 0.11 | 0.04 | -0.07 | 0.19 |
| cross-field-of-view | mond_class | 0.10 | 0.94 | 0.71 | 0.93 | 0.22 | 0.04 | -0.07 | 0.19 |
| cross-field-of-view | mond_class | 0.20 | 0.93 | 0.77 | 0.92 | 0.41 | 0.04 | -0.07 | 0.19 |
| cross-tissue | plain | 0.05 | 1.00 | 0.01 | 1.00 | 0.00 | - | - | - |
| cross-tissue | plain | 0.10 | 1.00 | 0.09 | 1.00 | 0.02 | - | - | - |
| cross-tissue | plain | 0.20 | 1.00 | 0.27 | 1.00 | 0.10 | - | - | - |
| cross-tissue | mond_class | 0.05 | 1.00 | 0.02 | 1.00 | 0.01 | 0.04 | 0.02 | 0.06 |
| cross-tissue | mond_class | 0.10 | 1.00 | 0.29 | 1.00 | 0.08 | 0.04 | 0.02 | 0.06 |
| cross-tissue | mond_class | 0.20 | 1.00 | 0.50 | 1.00 | 0.15 | 0.04 | 0.02 | 0.06 |
| cross-tissue | prop_mond | 0.05 | 1.00 | 0.02 | 1.00 | 0.00 | 0.04 | 0.03 | 0.06 |
| cross-tissue | prop_mond | 0.10 | 1.00 | 0.28 | 1.00 | 0.08 | 0.04 | 0.03 | 0.06 |
| cross-tissue | prop_mond | 0.20 | 1.00 | 0.53 | 1.00 | 0.15 | 0.04 | 0.03 | 0.06 |
| cross-tissue | twostage_nw | 0.05 | 1.00 | 0.00 | 1.00 | 0.00 | -0.02 | -0.03 | -0.01 |
| cross-tissue | twostage_nw | 0.10 | 1.00 | 0.02 | 1.00 | 0.01 | -0.02 | -0.03 | -0.01 |
| cross-tissue | twostage_nw | 0.20 | 1.00 | 0.11 | 1.00 | 0.04 | -0.02 | -0.03 | -0.01 |

Notes: The cross-field-of-view family here is the published twenty-unit design; its field-of-view-resolution re-run is Table 18. For the cross-field-of-view and cross-tissue families and 5%, 10% and 20% budgets: coverage of the budget under the certificate's own readout, under the joint readout and under the conditional reporting convention, with mean acceptance beside each; the paired differences against the plain certificate with unit-bootstrap intervals; the realised risk and the fraction of non-empty evaluations over budget for the transfer arms; and the target label cost at which each arm returns to covering the budget with a non-empty accepted set. Sources: `experiments/r31_within_summary.csv`, `experiments/r31_transfer_summary.csv`, `experiments/r31_paired.csv`, `experiments/r31_paired_transfer.csv` and `experiments/r31_ladder_summary.csv`; `scripts/r31_targeted_repair.py`.

**Supplementary Table 16 |** The targeted-repair arms at field-of-view resolution

| arm | error budget | coverage all | coverage nonvac | coverage cond | mean accept | delta nonvac vs plain | delta lo | delta hi | delta accept vs plain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 0.05 | 0.84 | 0.72 | 0.82 | 0.53 | - | - | - | - |
| plain | 0.10 | 0.79 | 0.72 | 0.78 | 0.73 | - | - | - | - |
| plain | 0.20 | 0.86 | 0.82 | 0.85 | 0.92 | - | - | - | - |
| sc3 | 0.05 | 0.98 | 0.71 | 0.97 | 0.32 | -0.01 | -0.07 | 0.06 | -0.21 |
| sc3 | 0.10 | 0.97 | 0.84 | 0.96 | 0.50 | 0.12 | 0.07 | 0.19 | -0.22 |
| sc3 | 0.20 | 0.97 | 0.92 | 0.97 | 0.71 | 0.10 | 0.05 | 0.15 | -0.20 |
| sc10 | 0.05 | 0.993 | 0.05 | 0.88 | 0.02 | -0.66 | -0.74 | -0.58 | -0.51 |
| sc10 | 0.10 | 0.99 | 0.54 | 0.97 | 0.31 | -0.18 | -0.26 | -0.11 | -0.42 |
| sc10 | 0.20 | 0.998 | 0.84 | 0.998 | 0.60 | 0.02 | -0.04 | 0.08 | -0.31 |
| mond_class | 0.05 | 0.95 | 0.51 | 0.91 | 0.17 | -0.20 | -0.30 | -0.11 | -0.36 |
| mond_class | 0.10 | 0.89 | 0.73 | 0.86 | 0.31 | 0.01 | -0.07 | 0.09 | -0.42 |
| mond_class | 0.20 | 0.91 | 0.86 | 0.91 | 0.55 | 0.03 | -0.03 | 0.10 | -0.37 |

Notes: For the same 96 units and the same three budgets, the `plain`, `sc3`, `sc10` and `mond_class` arms of `scripts/r31_targeted_repair.py` re-run on the field-of-view units by `scripts/r48b_fov_repair.py`: the three coverage readouts and mean acceptance for each arm, and the paired unit-bootstrap differences against `plain` in the joint non-vacuous coverage and in acceptance (4,000 resamples over units), both pooled over the three budgets and computed separately for each of them. Sources: `experiments/r48b_fov_repair_summary.csv` with `experiments/r48b_fov_repair_paired.csv` and `experiments/r48b_fov_repair_paired_by_alpha.csv`.

**Supplementary Table 17 |** External validation: acceptance, risk and coverage

| arm | error budget | mean accept | non-vacuous fraction | realised risk | coverage | cells accepted | accept ratio | evaluations over budget | file checks passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transfer (tonsil/Barrett) | 0.05 | 0.49 | 0.97 | 0.40 | 0.20 | 709920 | 2.01 | 144 | 10/10 |
| transfer (tonsil/Barrett) | 0.10 | 0.84 | 1.00 | 0.49 | 0.03 | 1166808 | 2.34 | 174 | 10/10 |
| transfer (tonsil/Barrett) | 0.20 | 0.99 | 1.00 | 0.52 | 0.00 | 1354242 | 2.08 | 180 | 10/10 |
| within intestine | 0.05 | 0.51 | 1.00 | 0.03 | 0.87 | 182993 | 0.81 | 12 | 10/10 |
| within intestine | 0.10 | 0.73 | 1.00 | 0.08 | 0.66 | 262756 | 0.91 | 31 | 10/10 |
| within intestine | 0.20 | 0.96 | 1.00 | 0.16 | 0.68 | 345584 | 0.994 | 29 | 10/10 |

Notes: For the within-intestine, transfer-to-Barrett and transfer-to-tonsil arms at three budgets: mean acceptance, the fraction of non-empty evaluations, realised risk, coverage of the budget and the number of cells evaluated and accepted; the evaluation-level ratio diagnostic with its over-budget, flagged, missed and undefined counts; and the archived-file checks. Sources: `experiments/r34_phase2_certificate_summary.csv`, `experiments/r34_phase2_ratio.csv` and `experiments/r34_phase2_checks.csv`; `scripts/r34_external.py`.

**Supplementary Table 18 |** The layer on an independent study, where the split is exchangeable

| split family | method | acceptance | vacuous | coverage | coverage (non-vacuous) | risk (non-vacuous) |
| --- | --- | --- | --- | --- | --- | --- |
| cell-level random split, intestine (exchangeable) | `crc` | **0.85** | 0.00 | **1.00** | 1.00 | 0.09 |
| cell-level random split, intestine (exchangeable) | `crc_cond` | 0.86 | 0.00 | 1.00 | 1.00 | 0.09 |
| cell-level random split, intestine (exchangeable) | `crc_mondrian` | 0.67 | 0.00 | 1.00 | 1.00 | 0.06 |
| cell-level random split, intestine (exchangeable) | `conformal_set` | 0.69 | 0.00 | 1.00 | 1.00 | 0.05 |
| cell-level random split, intestine (exchangeable) | `naive` | 0.89 | 0.00 | **0.49** | 0.49 | 0.10 |
| cell-level random split, intestine (exchangeable) | `oracle` | 0.89 | 0.00 | 1.00 | 1.00 | 0.10 |
| target-calibrated, tonsil + Barrett oesophagus (exchangeable) | `crc` | **0.31** | 0.28 | **1.00** | 1.00 | 0.07 |
| target-calibrated, tonsil + Barrett oesophagus (exchangeable) | `naive` | 0.35 | 0.00 | **0.55** | 0.55 | 0.10 |
| target-calibrated, tonsil + Barrett oesophagus (exchangeable) | `oracle` | 0.35 | 0.00 | 1.00 | 1.00 | 0.10 |
| image-grouped within intestine (cross-field-of-view analogue) | `crc` | 0.73 | 0.00 | **0.66** | 0.66 | 0.08 |

Notes: The STELLAR CODEX cell tables (Zenodo record 6863872, CC0-1.0), a different laboratory, platform and tissue from the benchmark, scored with the identical pipeline. The exchangeable rows are the direct analogue of the benchmark's random-split family: the model is trained on the intestine cohort's other images and the held-out pool is then divided at random into calibration and evaluation halves, 10,000 cells spent on calibration, over 150 evaluations at each budget. The target-calibrated rows train on the intestine cohort and divide the tonsil and Barrett oesophagus cohorts at random, over 60 evaluations at each budget. The last row is the image-grouped arm on the same intestine cells, reported in the external-validation analysis above: the same cohorts, the same pipeline and the same certificate, with the split the only thing that differs. The columns follow the conventions of Table 1 and the rows shown are the 10% budget. Robustness to the feature transform is in `experiments/r55_stellar_norm_robustness.csv`: coverage at a 10% budget is 1.00 under standardisation without the log and under the raw released values as well as under the r34 transform. Sources: `experiments/r55_stellar.csv` and `experiments/r55_stellar_summary.csv`; `scripts/r55_stellar_cellsplit.py`.

**Supplementary Table 19 |** Two further independent cohorts, a shared-panel control and a real change of antibody panel

| arm | cohort | error budget | acceptance | vacuous | realised risk | coverage | evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exchangeable, own panel | ion beam microarray | 0.05 | 0.84 | 0.00 | 0.04 | **1.00** | 150 |
| exchangeable, own panel | ion beam microarray | 0.10 | 0.97 | 0.00 | 0.08 | **0.993** | 150 |
| exchangeable, own panel | ion beam microarray | 0.20 | 1.00 | 0.00 | 0.10 | **1.00** | 150 |
| exchangeable, own panel | liver biopsies | 0.05 | 0.93 | 0.00 | 0.04 | **1.00** | 150 |
| exchangeable, own panel | liver biopsies | 0.10 | 0.995 | 0.00 | 0.06 | **1.00** | 150 |
| exchangeable, own panel | liver biopsies | 0.20 | 1.00 | 0.00 | 0.07 | **1.00** | 150 |
| exchangeable, shared channels | ion beam microarray | 0.05 | 0.75 | 0.00 | 0.04 | **1.00** | 150 |
| exchangeable, shared channels | ion beam microarray | 0.10 | 0.91 | 0.00 | 0.09 | **0.993** | 150 |
| exchangeable, shared channels | liver biopsies | 0.05 | 0.87 | 0.00 | 0.04 | **1.00** | 150 |
| exchangeable, shared channels | liver biopsies | 0.10 | 0.99 | 0.00 | 0.07 | **1.00** | 150 |
| panel swap, 13 shared channels | ion beam microarray | 0.05 | 0.00 | 1.00 | undefined | **1.00** | 30 |
| panel swap, 13 shared channels | ion beam microarray | 0.10 | 0.00 | 1.00 | undefined | **1.00** | 30 |
| panel swap, 13 shared channels | ion beam microarray | 0.20 | 0.02 | 0.67 | 0.11 | **1.00** | 30 |
| panel swap, 8 shared channels | liver biopsies | 0.05 | 0.01 | 0.87 | 0.02 | **1.00** | 30 |
| panel swap, 8 shared channels | liver biopsies | 0.10 | 0.02 | 0.67 | 0.02 | **1.00** | 30 |
| panel swap, 8 shared channels | liver biopsies | 0.20 | 0.07 | 0.33 | 0.26 | **0.67** | 30 |

Notes: Both cohorts are scored with the identical pipeline, certificate, split fraction, repeat count and metric definitions as the depot and the STELLAR cohort. The exchangeable arm groups each cohort's fields into five image folds and divides the held-out pool at random into calibration and evaluation halves, over ten calibration repeats and three classifiers; the shared-channel arm is the same protocol with the feature set restricted to the channels the cohort shares with the source; the panel-swap arm trains on the depot cohort with the largest shared panel and deploys unchanged on the new cohort, over one unit and three classifiers, with the classes restricted to those the two vocabularies share. `vacuous` is the fraction of evaluations whose accepted set is empty, and an empty accepted set counts as covered, so a coverage of 1.00 with an acceptance of zero is a certificate that declines to answer rather than one that succeeds, and the realised risk is undefined where every evaluation is vacuous. Coverage intervals are Wilson. Sources: `experiments/r69_external2_log1p_z.csv` and `experiments/r69_external2_log1p_z_summary.csv`, with the panel overlaps in `experiments/r69_external2_panel_overlap.csv` and the asserted checks in `experiments/r69_external2_checks.csv`; `scripts/r69_external2.py`.

**Supplementary Table 20 |** The layer around two published classifiers

| classifier | cohort pair | arm | budget | evaluations | coverage | realised risk | acceptance (certificate) | acceptance (oracle) | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CellTune | breast to colorectal | exchangeable | 0.05 | 40 | 1.00 | 0.03 | 0.81 | 0.94 | 0.87 |
| CellTune | breast to colorectal | exchangeable | 0.10 | 40 | 1.00 | 0.06 | 0.96 | 1.00 | 0.96 |
| CellTune | breast to colorectal | exchangeable | 0.20 | 40 | 1.00 | 0.08 | 1.00 | 1.00 | 1.00 |
| CellTune | breast to colorectal | cross-field-of-view | 0.05 | 600 | 0.86 | 0.03 | 0.86 | 0.91 | 0.96 |
| CellTune | breast to colorectal | cross-field-of-view | 0.10 | 600 | 0.87 | 0.07 | 1.00 | 0.98 | 1.02 |
| CellTune | breast to colorectal | cross-field-of-view | 0.20 | 600 | 1.00 | 0.07 | 1.00 | 1.00 | 1.00 |
| CellTune | breast to colorectal | cross-tissue (target) | 0.05 | 40 | 0.00 | 0.51 | 0.40 | 0.02 | 20.00 |
| CellTune | breast to colorectal | cross-tissue (target) | 0.10 | 40 | 0.00 | 0.56 | 0.96 | 0.02 | 39.87 |
| CellTune | breast to colorectal | cross-tissue (target) | 0.20 | 40 | 0.00 | 0.57 | 0.99 | 0.05 | 21.51 |
| CellTune | GVHD to colorectal | exchangeable | 0.05 | 40 | 1.00 | 0.02 | 0.68 | 0.82 | 0.82 |
| CellTune | GVHD to colorectal | exchangeable | 0.10 | 40 | 1.00 | 0.07 | 0.88 | 0.98 | 0.90 |
| CellTune | GVHD to colorectal | exchangeable | 0.20 | 40 | 1.00 | 0.11 | 1.00 | 1.00 | 1.00 |
| CellTune | GVHD to colorectal | cross-field-of-view | 0.05 | 960 | 0.81 | 0.03 | 0.73 | 0.77 | 0.93 |
| CellTune | GVHD to colorectal | cross-field-of-view | 0.10 | 960 | 0.75 | 0.07 | 0.91 | 0.93 | 1.00 |
| CellTune | GVHD to colorectal | cross-field-of-view | 0.20 | 960 | 1.00 | 0.11 | 1.00 | 1.00 | 1.00 |
| CellTune | GVHD to colorectal | cross-tissue (target) | 0.05 | 40 | 0.00 | 0.26 | 0.33 | 0.00 | undefined |
| CellTune | GVHD to colorectal | cross-tissue (target) | 0.10 | 40 | 0.00 | 0.32 | 0.64 | 0.12 | 5.48 |
| CellTune | GVHD to colorectal | cross-tissue (target) | 0.20 | 40 | 0.00 | 0.42 | 1.00 | 0.24 | 4.15 |
| CellSighter | breast to colorectal | exchangeable | 0.05 | 40 | 1.00 | undefined | 0.00 | 0.68 | 0.00 |
| CellSighter | breast to colorectal | exchangeable | 0.10 | 40 | 1.00 | 0.07 | 0.81 | 0.90 | 0.89 |
| CellSighter | breast to colorectal | exchangeable | 0.20 | 40 | 1.00 | 0.14 | 1.00 | 1.00 | 1.00 |
| CellSighter | breast to colorectal | cross-field-of-view | 0.05 | 600 | 0.94 | 0.15 | 0.03 | 0.72 | 0.00 |
| CellSighter | breast to colorectal | cross-field-of-view | 0.10 | 600 | 0.80 | 0.07 | 0.83 | 0.86 | 0.91 |
| CellSighter | breast to colorectal | cross-field-of-view | 0.20 | 600 | 0.87 | 0.13 | 1.00 | 0.98 | 1.03 |
| CellSighter | breast to colorectal | cross-tissue (target) | 0.05 | 40 | 1.00 | undefined | 0.00 | 0.02 | 0.00 |
| CellSighter | breast to colorectal | cross-tissue (target) | 0.10 | 40 | 0.00 | 0.55 | 0.65 | 0.06 | 10.58 |
| CellSighter | breast to colorectal | cross-tissue (target) | 0.20 | 40 | 0.00 | 0.62 | 1.00 | 0.10 | 10.06 |
| CellSighter | melanoma to GVHD | exchangeable | 0.05 | 6 | 1.00 | undefined | 0.00 | 0.00 | undefined |
| CellSighter | melanoma to GVHD | exchangeable | 0.10 | 6 | 1.00 | undefined | 0.00 | 0.27 | 0.00 |
| CellSighter | melanoma to GVHD | exchangeable | 0.20 | 6 | 1.00 | 0.11 | 0.30 | 0.44 | 0.68 |
| CellSighter | melanoma to GVHD | cross-field-of-view | 0.05 | 24 | 1.00 | undefined | 0.00 | 0.18 | 0.00 |
| CellSighter | melanoma to GVHD | cross-field-of-view | 0.10 | 24 | 1.00 | undefined | 0.00 | 0.29 | 0.00 |
| CellSighter | melanoma to GVHD | cross-field-of-view | 0.20 | 24 | 0.50 | 0.28 | 0.20 | 0.39 | 0.17 |
| CellSighter | melanoma to GVHD | cross-tissue (target) | 0.05 | 6 | 1.00 | undefined | 0.00 | 0.00 | undefined |
| CellSighter | melanoma to GVHD | cross-tissue (target) | 0.10 | 6 | 1.00 | undefined | 0.00 | 0.00 | undefined |
| CellSighter | melanoma to GVHD | cross-tissue (target) | 0.20 | 6 | 0.00 | 0.42 | 0.39 | 0.00 | undefined |

Notes: The colorectal to GVHD direction is the mirror of the third pair and is valid but silent: the certificate covers in every evaluation and declines to answer at a 5% and a 10% budget, while an oracle threshold could accept 0.07 and 0.32 of cells, so there the source model's own ranking is the limit rather than the calibration. The GVHD to colorectal rows use CellTune's own exported cell table, the per-cell marker means and the area, under the benchmark's published log1p plus per-domain z-score, because the colorectal high-quality release ships no images and the compartment features cannot be recomputed for it. Every row uses the identical certificate, split fraction and metric definitions as the benchmark, and the arm level detail of the benchmark itself is Supplementary Tables 17 to 19. CellSighter is its published repository vendored unmodified and driven through its own modules on image crops, and CellTune is a reimplementation of the classifier its documentation specifies, its documented per-cell compartment statistics and its default XGBoost and CatBoost pair averaged into one prediction, because the released tool is a desktop application rather than an importable package. Both were fitted on the source cohort alone, over 40 calibration repeats, 15 held-out fields of view and the 12 markers and 9 coarse classes the breast and colorectal panels share, while the melanoma to GVHD pair is an earlier CellSighter run over 24 shared markers and 10 coarse classes with 6 calibration repeats. The realised risk is the mean over evaluations with a non-empty accepted set and is undefined where every evaluation abstains, and an empty accepted set counts as covered, so a coverage of 1.00 with an acceptance of 0.00 is a certificate that declines to answer rather than one that succeeds; the ratio is undefined where the oracle accepts nothing. Sources: `experiments/r51_cellsighter_summary_breast2crc.csv`, `experiments/r51b_celltune_summary.csv` and `experiments/r51b_celltune_summary_gvhd2crc_z.csv` and `experiments/r51_cellsighter_summary.csv`, with the per-evaluation rows beside them; `scripts/r51_cellsighter.py` and `scripts/r51b_celltune.py`.
