# Distribution-free error control for cell type annotation in spatial proteomics

A post-hoc calibration layer that wraps any cell-type classifier for multiplexed
imaging and returns an accepted set carrying a finite-sample, distribution-free
bound on its selective error rate, plus an abstention set routed to human review.

## Manuscript

- [Main manuscript](1020manuscript.docx)
- [Supplementary Information](1022supplementary_NC.docx)

The submitted versions of both Word files are stored in this repository. The
analysis code below reproduces every released result table and figure.
Superseded drafts and the separate static figure exports are deliberately not
tracked: all final figures are embedded in the two Word files, and the plotting
scripts regenerate them from the released tables.

The guarantee rests on exchangeability between the calibration and deployment
populations. This repository contains the experiments that establish where that
assumption holds (exchangeable random splits), where it degrades
(cross-field-of-view) and where it fails (cross-tissue transfer); what it costs in
target-domain labels to restore it in each family; a label-free audit that says
whether the calibration sample represents the batch; a labelled pilot that prices
a repair; and the review budget the resulting certificate implies.

## Layout

```
src/errctl/        the package
  conformal.py       selective risk controllers: Clopper-Pearson on the two
                     unconditional probabilities (bonferroni or fixed-sequence
                     grid), a range-aware weighted variant with a stratum-exact
                     leg and an unlabelled-target denominator, a score-conditional
                     variant; ablations: cond, eb, mondrian, temperature
  markerlogic.py     parser and evaluator for Boolean marker rules
  preprocess.py      log1p + standardisation, DomainNormaliser
  classifiers.py     logistic regression, random forest, histogram gradient boosting
  calibration.py     ECE, reliability curves, temperature scaling
  markercheck.py     marker-rule consistency audit
  data.py labels.py paths.py weights.py eval.py

scripts/           one script per experiment, numbered in dependency order
  s00_inspect_data.py, s01_inventory.py, s02_probe_demo_and_join.py,
  s03_probe_images.py      data inventory and integrity audit
  r01_build_datasets.py    assemble the per-cohort master tables
  r02_run_experiments.py   train/deploy every cohort pair; write the prediction cache
  r03_analyse.py           run every controller on the cached predictions
  r04_report.py            consolidate into the report and tables
  r05_cellsplit.py         the exchangeable random-split family
  r06_marker_audit.py      marker-rule violation rates
  r07_ablations.py         grid / delta / min_accept / score / calibration size
  r08_panel.py             controlled marker-panel reduction: throughput cost
  r09_simulation.py        Monte-Carlo validity study
  r10_figures.py           main and extended-data figures
  r11_ensemble.py          ensemble confidence versus the certificate
  r12_labelnoise.py        reference-label-noise sensitivity
  r13_coverage_ci.py       cluster-bootstrap intervals for coverage
  r14_targetcal.py         transductive target-domain calibration; model adaptation
  r14c_ladder_dense.py     Table 6 recomputed on the denser label grid
  r15_marker_sensitivity.py marker violation under 16 definition/threshold conventions
  r16_review_budget.py     review budget, matched-risk price of validity, fixed-sequence
  r17_simulation.py        second Monte-Carlo study of the rebuilt controllers
  r18_marker_sources.py    marker audit across label sources and populations
  r19_weight_shift_diag.py label-free diagnosis of the weighted certificate
  r20_label_ladder.py      the label cost of the guarantee, by split family
  r21_downstream_repair.py downstream neighbourhood enrichment under the repair
  r21b_matched_downstream.py the same statistic under one weight per type
  r22_fig8.py              rebuild of the label-ladder figure
  r24_efficiency.py        matched-realised-risk accounting and certificate yield
  r24b_labelshift.py       label-shift reweighting as an alternative to calibration
  r25_accept_deferred.py   accepted-versus-deferred marker consistency, 68 conventions
  r26_plugin_baseline.py   conservative plug-in baselines and the confidence-level sweep
  r27_fixedseq.py          fixed-sequence step-down: acceptance, validity, grid dependence
  r28_marker_bounds.py     identified sets and convention-free verdicts for the marker audit
  r31_targeted_repair.py   targeted repair of the two conformal failures
  r32_sequential.py        anytime-valid sequential certificate
  r33_two_source.py        second annotation source; partial identification of the audit
  r34_external.py          external validation on STELLAR CODEX
  r35_figures.py           supplementary figures S8-S10 from the round-2 tables
  r43_threadpin.py         BLAS thread pinning; is the adaptation arm reproducible?
  r44_gvhd_delivery.py     do the shipped segmentation masks cover the shipped labels?
                           (the GVHD delivery defect)
  r48_fov_units.py         cross-field-of-view coverage at field-of-view resolution
  r48b_fov_repair.py       the targeted-repair arms re-run on those field-of-view units
  r49_markerlogic.py       the marker-logic (panel-invariant) representation, all four spaces
  r49b_panelinvariance.py  is the marker-rule audit layer panel invariant?
  r50_review_loop.py       review budget versus repair price, in the same units
  r51_cellsighter.py       the layer around a published image-based classifier (CellSighter)
  r52_stronger_score.py    seven models through the same controller: does a stronger score
                           compress the review?
  r53_calibration_size.py  the price of validity at a controlled calibration size
  r54_ranking.py           the strongest tabular rankers on the same folds: can a better
                           ranking lower the oracle?
  r55_stellar_cellsplit.py the independent cohorts on the exchangeable split: does the
                           guarantee reproduce outside the depot?
  r56_neighbour_context.py the representation axis: run, then withdrawn from the
                           manuscript (a null, and a cross-cell feature that would
                           void the binomial law); tables kept on disk as a record
  r57_multiplicity.py      multiplicity and selection correction over the released
                           tables: per-unit, family-level and gate families
  r57b_triage.py           what to read first in the review queue, in the
                           certificate's own score order
  r59_adviser.py           the guided decision procedure, one question at a time
  r60_adviser_html.py      builds the guided interface (Chinese + English)
                            and the downloadable sample batch it offers
  r61_shared_panel.py      a real re-panelling: both cohorts trained,
                            calibrated and deployed on the channels they share
  r62_source_family.py     the fourth partition family: one cell set under
                            two label sources, the repair that covers both,
                            and the per-fold transfer diagnostic
  r63_reproduce_check.py   one command that re-checks the four negative results,
                            the cross-field-of-view shortfall, the positive
                            replications, the priced cross-panel deployment, the
                            repair pilot on both pairs and the tool demonstration
                            against fixed expected values, from the released
                            tables alone
  r70_e2e_demo.py          the layer run end to end on a public cohort, with the
                            review fraction converted into expert hours
  r71_crosspanel_price.py  a real change of antibody panel, priced in target
                            labels across a 100 to 8,000 label ladder
  r72_panel_repair.py      the same change of panel with the labels spent on
                            the model instead of the calibrator, four further
                            panel changes that add no tissue at all: colon and
                            breast cores of one published MIBI-TOF array, each
                            against its own depot cohort, a tonsil pair that
                            also changes platform, MIBI-TOF against CODEX, and
                            a second source for the colon core from STELLAR
                            intestine with one whole source region held out
  r73_label_price.py       the label price of a bounded likelihood ratio, with
                            the r19 effective-sample summary and the exact
                            two-point lower bound of Proposition 3
  r74_repair_triage.py     a 250-label target-labelled pilot ranks repairs that
                            repay their labels above those that do not, whereas
                            the certificate-to-oracle ratio does not; reports
                            cluster-bootstrap intervals, an unlabelled-denominator
                            robustness check, a retrospective cut and a
                            leave-one-fold-out threshold rule
  r76_pilot_multiplicity.py  the same pilot under Bonferroni and fixed-sequence,
                            with deployment and pilot-only ratios separated;
                            the ratio gains support but stays at chance
  r77_pilot_decision.py    the pilot as a decision rule: four-fold threshold
                            selection applied to the fifth, a calibration-
                            silence ladder and a failed net-gain lower bound
  r78_external_repair_pair.py  the same pilot on a second repair pair, breast
                            and tonsil, with the colon cut transported
                            unchanged and a within-pair threshold cross-fitted;
                            it fixes when the pilot is worth running
  r80_immucan_breas_export.R  export the 21,731-cell IMMUcan BREAS repair set
  r80_immucan_breast_repair.py  a large-target repair test on that set: 1,000
                            labels are 4.6% of the target, 20 draws, three
                            classifiers, protocol frozen in
                            docs/submission/preregistration_immucan_breast_repair.md
  r81_immucan_breast_sourcefold.py  the post-primary five-source-fold
                            robustness check for the same repair
  r89_verify_advise.py     one-command acceptance check for the three `advise`
                            products: budget, decisions and price list
  r75_immucan_export.R     export the public IMMUcan panel 1 V3 labelled cells
  r75_immucan_lock.py      the protocol-locked first evaluation on IMMUcan, five
                            sample-grouped folds x ten draws x three classifiers;
                            plan and code hashes in
                            docs/submission/preregistration_immucan.md
  errctl_cli.py            the layer as a command line tool (certify / deploy /
                            repair / demo), writing decisions, a report and a
                            figure
  run_pipeline.py          the data-stage driver (one place for the run order)

docs/              manuscript, methods, and the submission package
  adviser.html       the guided decision interface (single file, offline)
  adviser_en.html    the same interface in English, for referees
  adviser_sample.csv a synthetic three-column batch to try the interface on
  submission/reproducibility.md
                     what reproduces from the released tables alone, and the
                     expected value of each headline claim
experiments/       all result tables (CSV) and the consolidated report
tools/CellSighter/ the published CellSighter repository, vendored unmodified for r51
figures/           figures as PDF (vector) and PNG
work/              cached datasets, predictions and logs (not redistributable data)
```

## Reproducing

Python 3.10, CPU only, no GPU. Dependencies: numpy, pandas, scikit-learn, scipy,
tifffile, matplotlib.

The flags below are the invocations that reproduce the released tables, which
are not always the script defaults (r03's eight-alpha grid, r14's ten
replicates, r14b's three-size adaptation grid, r33's bootstrap budget).  Where
a script has no --jobs, none is passed.  `python scripts/run_pipeline.py
--list` prints the same table; Round 28 of docs/manuscript_number_audit.md
records how it was established.

```
python scripts/r01_build_datasets.py
python scripts/r02_run_experiments.py --units within,transfer --clfs logreg,rf,hgb --norm z_dataset --seed 0
python scripts/r05_cellsplit.py --clfs logreg,rf,hgb --norm z_dataset --seed 0
python scripts/r03_analyse.py --alphas 0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50 --jobs 8
python scripts/r04_report.py
python scripts/r06_marker_audit.py --alphas 0.05,0.10,0.20 --clfs logreg,rf,hgb
python scripts/r07_ablations.py --jobs 8
python scripts/r08_panel.py --jobs 8
python scripts/r09_simulation.py --reps 4000 --jobs 8
python scripts/r10_figures.py
python scripts/r11_ensemble.py --jobs 8
python scripts/r12_labelnoise.py --jobs 8
python scripts/r13_coverage_ci.py
python scripts/r14_targetcal.py --reps 10 --jobs 8
python scripts/r14_targetcal.py --adapt --adapt-ks 200,1000,2000 --reps 2 --out r14b_targetadapt --jobs 8
python scripts/r14_targetcal.py --ks 0,25,50,100,200,250,500,750,1000,1500,2000 --reps 10 --out r14_targetcal_ext --jobs 8
python scripts/r14c_ladder_dense.py
python scripts/r15_marker_sensitivity.py --jobs 8 --boot 2000
python scripts/r16_review_budget.py --jobs 8
python scripts/r17_simulation.py --reps 800 --jobs 8
python scripts/r18_marker_sources.py --jobs 8 --boot 2000
python scripts/r19_weight_shift_diag.py --jobs 6
python scripts/r20_label_ladder.py --jobs 6
python scripts/r21_downstream_repair.py
python scripts/r21b_matched_downstream.py
python scripts/r22_fig8.py
python scripts/r24_efficiency.py --jobs 6
python scripts/r24b_labelshift.py --jobs 4
python scripts/r25_accept_deferred.py --jobs 4
python scripts/r26_plugin_baseline.py
python scripts/r27_fixedseq.py
python scripts/r28_marker_bounds.py
python scripts/r31_targeted_repair.py --jobs 4
python scripts/r32_sequential.py --procs 4
python scripts/r33_two_source.py --boot 200 --boot-ci 120 --rand 8000 --lca-boot 200 --lca-rand 8 --lca-rand-ci 6 --pool 24
python scripts/r34_external.py --stage all
python scripts/r35_figures.py
python scripts/r43_threadpin.py --threads 1 --reps 2 --tag p1a
python scripts/r43_threadpin.py --aggregate
python scripts/r44_gvhd_delivery.py --check
python scripts/r48_fov_units.py --jobs 6 --write --analyse
python scripts/r48b_fov_repair.py --jobs 10
python scripts/r49_markerlogic.py --space published --jobs 8 --write --analyse
python scripts/r49_markerlogic.py --space rule      --jobs 8 --write --analyse
python scripts/r49_markerlogic.py --space augz      --jobs 8 --write --analyse
python scripts/r49_markerlogic.py --space augraw    --jobs 8 --write --analyse
python scripts/r49_markerlogic.py --report
python scripts/r49b_panelinvariance.py
python scripts/r50_review_loop.py
python scripts/r51_cellsighter.py --stage all --epochs 20
python scripts/r52_stronger_score.py --jobs 14
python scripts/r53_calibration_size.py --jobs 8 --reps 10
python scripts/r54_ranking.py --jobs 14
python scripts/r55_stellar_cellsplit.py --only ext_targetcal
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs logreg --tag logreg
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs rf     --tag rf
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs hgb    --tag hgb
python scripts/r55_stellar_cellsplit.py --norm z_train --tag z_train
python scripts/r55_stellar_cellsplit.py --norm raw     --tag raw
# r56 (the representation axis) is withdrawn from the manuscript; the commands are
# kept only so the record can be reproduced
python scripts/r56_neighbour_context.py --cohort Breast_Jun0524
python scripts/r56_neighbour_context.py --cohort Melanoma_Oct2522
python scripts/r56_neighbour_context.py --cohort CRC_Feb2623
python scripts/r56_neighbour_context.py --cohort GVHD_Feb1323
python scripts/r56_neighbour_context.py --analyse
python scripts/r57_multiplicity.py
python scripts/r57b_triage.py
python scripts/r59_adviser.py --demo     # the guided interface, on the command line
python scripts/r60_adviser_html.py       # regenerate the interface and sample
python scripts/r61_shared_panel.py --jobs 8
python scripts/r61_shared_panel.py --pair Melanoma_Oct2522,GVHD_Feb1323 --tag _mg --jobs 8
python scripts/r62_source_family.py --jobs 8
python scripts/r68_equal_guarantee.py    # the matched-guarantee reading
python scripts/r69_external2.py --jobs 8 # two further external cohorts
python scripts/r63_reproduce_check.py    # self-check from the released tables
```

## Round-2 analyses

Fifteen experiments (eighteen scripts) added after the first submission round, one
of them (r56) withdrawn and kept only as a record. Two further scripts carry the
decision procedure to a reader: `r59_adviser.py` on the command line and
`r60_adviser_html.py`, which builds it as a bilingual, offline page
(`docs/adviser.html`, `docs/adviser_en.html`) with a downloadable sample batch
(`docs/adviser_sample.csv`). Of the fifteen, three
sharpen headline claims -- by measuring what a stronger wrapped model buys, how much
calibration the guarantee needs, and whether a stronger ranker lowers the ceiling;
the rest sharpen a validity statement, bound what a claim can be made to cover,
or state a result in the units a laboratory plans in. Each writes tables to `experiments/` and a report to
`work/`.

- `scripts/r31_targeted_repair.py` - targeted repair of the two conformal
  failures (cross-field-of-view `within`, cross-tissue `transfer`). Report:
  `work/r31_report.txt`. Outcome: cross-tissue transfer is unrepaired and still
  needs 500 target labels at a 10% budget for the plain, Mondrian,
  proportional-Mondrian and two-stage arms (200 at a 20% budget; the pooled
  importance-weighted arm reaches the criterion at 25 labels but only under the
  vacuity-inclusive readout, at an acceptance of 0.006). Within a cohort the repair
  does hold under the conventionally-reported readout (0.7643 to 0.9467 at a 10%
  budget, at an acceptance of 0.320 against 0.534) but not under the joint readout,
  where in that twenty-unit design no arm reaches 0.90 and none of those three
  paired differences against the plain certificate excludes zero (the
  field-of-view-resolution re-run is the exception for the three-stratum arm; see
  the r48 entry below).
- `scripts/r32_sequential.py` - an anytime-valid e-process certificate under
  optional stopping. Report: `work/r32_report.txt`. Outcome: it needs fewer
  labels than the fixed-budget certificate used at t = n (cellsplit, alpha = 0.10;
  the label cost of a 0.75 realised acceptance is 0.9114 of the pool, against the
  whole pool for the fixed-sample object) but loses to the honest sequential
  fixed-budget rival `cert_fixed_union` (+0.0468 of the pool needed
  [+0.0229, +0.0715]), so there is no log-log advantage at these pool sizes.
- `scripts/r33_two_source.py` - partial identification from the second annotation
  source (colorectal and GVHD only). Report: `work/r33_report.txt`. Outcome: the
  convention tie is not broken. Exact sharp sets only lower the worst case
  (colorectal endothelial 0.604 to 0.227) and never raise the floor: every sharp
  set reaches 0. The convention-free marker verdicts are statements about the
  shipped reference labels, not latent cell membership.
- `scripts/r34_external.py` - external validation on STELLAR CODEX (Zenodo record
  6863872, DOI 10.5061/dryad.g4f4qrfrc, licence CC0-1.0) plus a provenance audit
  of the CellTune demo. Report: `work/r34_report.txt`. Outcome: the guarantee is
  not reproduced externally (coverage 0.6556 within, 0.0333 transfer at a 10%
  budget); the ratio diagnostic misses 60 of 174 over-budget transfer
  evaluations; and the CellTune demo is not an independent study (38 of 75 images
  shared with the depot, identical 40-marker panel).
- `scripts/r43_threadpin.py` - is the adaptation arm reproducible once the BLAS
  thread pool is pinned? Report: `work/r43_threadpin_report.txt`. Outcome: yes.
  Two independent pinned processes agree to 0.000e+00 on mean acceptance and on
  both coverage readouts across all five arms, and each reproduces the stored
  `adapt_k` headline of 0.3551 exactly; the 0.155 in `work/r31_report.txt` is the
  logistic-regression-only subtotal of the classifier mix (0.163 measured with the
  pool pinned), not thread drift. The
  non-reproducibility caveat the manuscript used to carry has been withdrawn.
- `scripts/r44_gvhd_delivery.py` - do the shipped segmentation masks cover the
  shipped labels? Report: `work/r44_delivery_report.txt`, tables
  `experiments/r44_gvhd_delivery.csv` and `experiments/r44_gvhd_delivery_summary.csv`,
  write-up `docs/submission/gvhd_delivery_defect.md`. Outcome: they do for breast
  (7,610 of 7,610), colorectal (2,505 of 2,505) and melanoma (21,204 of 21,204), and
  they do not for GVHD, where 1,794 of 10,283 labelled cells (17.5%) name a segment
  that exists in the shipped mask and every one of the 48 fields is affected, so the
  join that a pipeline would take for granted is broken for that cohort. The analysis
  never reads the GVHD mask, which is why this is a defect in the delivered data and
  not in the pipeline.
- `scripts/r49_markerlogic.py` (with `scripts/r49b_panelinvariance.py`) - is
  cross-panel transfer a feature-scale artefact that a panel-invariant
  representation repairs, and is the marker-rule audit layer panel invariant?
  Reports: `work/r49_report.txt`, `work/r49b_report.txt`. Outcome: no, and
  partly. Every unit is rebuilt with each cell described by the depot's 59
  Boolean rules. Concatenating the rule vector onto the shared panel leaves both
  families where they stand (within coverage 0.850/0.872/0.900 against the
  published 0.833/0.817/0.917; transfer 0.513/0.405/0.292 at acceptance
  0.187/0.357/0.655 against the published 0.541/0.397/0.286 at
  0.179/0.367/0.669, the second scaling within 0.004 of the first). The rule-only space is natively panel invariant and it does
  strip the transferred score's false confidence - transfer acceptance at a 10%
  budget falls from 0.367 to 0.019 - but by abstaining: its non-vacuous coverage
  on transfer is 0.000, and within a cohort even the oracle certifies only 3.9%
  of cells at a 5% budget in it. What does transport is the audit layer: of the 29
  (rule, cohort) combinations carrying at least 30 cells of the rule's own type,
  26 satisfy their own rule at 0.95 or more (minimum 0.795, the alternate plasma
  rule in GVHD; largest cross-cohort disagreement 0.111, the endothelial rule),
  for 40 of the 59 rules in at least one cohort and 6 in all four.
- `scripts/r48_fov_units.py` (with `scripts/r48b_fov_repair.py`) - the
  cross-field-of-view family holds out a *fold* of images at a time, twenty units
  in all, and at that resolution the cluster bootstrap cannot separate the
  shortfall from the nominal level. Holding out one image per unit puts the same
  family on 96 units. Reports: `work/r48_report.txt`, `work/r48b_report.txt`.
  Outcome: the shortfall is resolved - coverage 0.844 [0.788, 0.894] at a 5%
  budget and 0.792 [0.724, 0.855] at 10%, both excluding the nominal 0.90 - and on
  those units the three-stratum repair arm is the first in the study whose
  improvement in the joint non-vacuous coverage excludes zero at the 10% and 20%
  budgets (+0.123 [+0.067, +0.186] and +0.095 [+0.047, +0.149]; +0.070
  [+0.027, +0.116] pooled over the three). That readout is 0.845 at the 10% budget
  and 0.920 at the 20% one, the latter above nominal although its interval still
  includes it.
- `scripts/r50_review_loop.py` - the review budget and the label price of the
  repair in the same units, read from the released tables rather than recomputed.
  Report: `work/r50_report.txt`. Table: `experiments/r50_review_loop.csv`. Outcome:
  on the cross-field-of-view family at a 10% budget the certificate routes 46.6% of
  the evaluation pool to review (925 cells in a mean unit of 1,984) while 500
  uniformly drawn target labels per unit restore coverage from 0.8167 to 1.0000 at
  an unchanged acceptance of 0.5338 to 0.5346, so the repair is free in throughput
  and costs 1.85 times less than the review the certificate already generates. At a
  5% budget no count in the ladder restores throughput, and under cross-tissue
  transfer none does at any budget, which is the ranking limit Table 7 addresses.
- `scripts/r51_cellsighter.py` - the layer around a classifier of a different
  family: CellSighter, a published convolutional network that reads image crops
  rather than marker tables (doi:10.1038/s41467-023-40066-7, vendored unmodified
  at `tools/CellSighter`). The network is trained on melanoma fields with four of
  the ten held out, and evaluated on the melanoma-to-GVHD pair through the same
  controller. Report: `work/r51_report.txt`. Table:
  `experiments/r51_cellsighter_summary.csv`. Outcome: a negative result about
  the network rather than about the layer. The wrapper takes a foreign classifier
  without modification and abstains on it - the certificate accepts nothing at a 5%
  or a 10% budget on any of its evaluations, where an oracle threshold would accept
  0.271 of the same cells at the 10% budget,
  because the most confident decile of the held-out melanoma fields already carries
  a risk of 0.101 against a 10% budget - and the reason is visible in the model:
  it over-predicts two mid-frequency classes five- to seven-fold on its own
  held-out fields (neutrophils 1,493 predicted against 203 present), which is the
  signature of CPU training for twenty epochs without the published ensemble. Its
  GVHD accuracy is 0.491 against 0.58-0.67 for the tabular regressor on the same
  pair, so the section demonstrates the interface and not the operating point.
- `scripts/r52_stronger_score.py` - what sets the headline throughput, and what a
  stronger model buys. Seven models are refit from scratch on the random-split
  family's own folds -- the three base learners, their soft vote, a bagged
  gradient boosting, a bagged vote and a perceptron -- and scored the paper's
  way through the same controller and oracle. Tables: `experiments/r52_stronger.csv`
  and `experiments/r52_stronger_summary.csv`. Outcome:
  the three base learners reproduce the headline to the fourth decimal, and the
  headline turns out to be an equal-weight mean that includes a weak linear model.
  The tuned learner alone certifies 0.609 at a 10% budget against the mean's 0.559
  (pooled review 40.3% to 32.6%, and 13.6% to 6.6% at a 20% budget, coverage
  1.000 throughout), so the labour claim becomes 67.4% and 93.4% of the review
  removed. Ensembling adds about one point at the tight budget and nothing at the
  loose one: the best-calibrated score in the set (ECE 0.040) is not the one that
  certifies the most.

- `scripts/r53_calibration_size.py` - the price of validity at a controlled
  calibration size. A balanced panel per cohort holds the test quarter fixed and
  re-draws the calibration set at 25 to 1600 cells, with the oracle recomputed on
  the same cells. Tables: `experiments/r53_calibration_size.csv` and
  `experiments/r53_calibration_size_summary.csv`.
  Outcome: the gap between the certificate and the oracle is not a constant of the
  construction but a decreasing function of the number of labelled cells the
  certificate is given. At a 10% budget with the tuned learner it is the whole of
  the oracle at 25 cells, falls to 14.1 points on breast and 22.6 on GVHD by 400
  cells and to 7.6 and 13.9 by 800, and is slower on melanoma (29.2 at 800, 18.2 at
  1600); colorectal does not respond at any size its 756-cell pool supports,
  because its oracle is already 0.413. Coverage stays at or above 0.95 in every
  cell of the panel at that budget. The pooled 15.9-point price is therefore the
  value of this curve at one calibration size, and on three of the four cohorts a
  few hundred target labels remove most of it.

- `scripts/r54_ranking.py` - can a stronger ranker lower the oracle? Six arms --
  the paper's tuned learner, a widened histogram gradient boosting, a random forest
  grown to the leaf, extra trees, xgboost and their soft vote -- are refit on the
  random-split family's own folds with the same controller and oracle as r52.
  Tables: `experiments/r54_ranking.csv` and `experiments/r54_ranking_summary.csv`;
  Its released table. Outcome: a null result, and a useful one. At a 10%
  budget the six arms accept 0.536 to 0.612 and span 4.4 points of oracle (extra
  trees 0.748 to the tuned learner 0.792), and the highest oracle belongs to the
  learner the paper already had. The vote comes closest on acceptance (0.612 against
  0.609) yet certifies less at a 20% budget (0.847 against 0.851), while being both
  the most accurate and the best-calibrated arm (0.839, ECE 0.041) - so swapping the
  learner is not a route to the residual review the paper leaves open, and the floor
  is a property of the features and labels rather than of the model. The tuned
  learner's row reproduces r52's to the fourth decimal. LightGBM is excluded: its C
  API raises an access violation inside worker processes on this platform, a failure
  reproducible outside the script.

- `scripts/r57_multiplicity.py` - the three families of comparisons the manuscript
  quotes are each a maximum or a minimum over many units, and none of them carried a
  multiplicity correction. This script re-reads the released tables (no refits) and
  charges every family the correction it needs: a per-comparison one-sided
  Clopper-Pearson limit, a family-level Bonferroni share, and a max-statistic cluster
  bootstrap over units. Tables: `experiments/r57_multiplicity.csv` and
  `experiments/r57_multiplicity_summary.csv`. Report: `work/r57_report.txt`. Outcome:
  no benchmark unit breaks its per-unit limit and the corrected limits stay clear of
  nominal (per-unit 0.926 to 0.839 pooled over 19 units, pooled 0.996 to 0.985, the five
  STELLAR units 0.926 to 0.878); in the repair family 5 of the 12 pairs certify within a
  10% budget before and after the correction, the pooled risk upper moving 0.0355 to
  0.0358; and the marker gate -- a minimum over 68 conventions, then selected over 48
  population-by-cohort cells -- keeps hard bounds on three of five populations (GVHD
  neutrophils 26.35% to 14.26%, GVHD plasma 19.84% to 13.89%, melanoma CD8 T 19.78% to
  16.86%) while the two smallest (6.90% and 5.21%) lose theirs.
- `scripts/r57b_triage.py` - the review queue is ordered by the certificate's own score,
  and this script measures what that ordering is worth: a pooled read of the deferred
  set in descending uncertainty. Tables: `experiments/r57_triage.csv` and
  `experiments/r57_triage_summary.csv`. Outcome: at a 10% budget the first quarter of
  the queue carries 0.386 of the residual errors and the first half 0.655, against
  0.250 and 0.500 for an uninformed order, with a head error rate of 0.568 against a
  tail of 0.265; cross-learner disagreement and score entropy do not beat it. The
  ordering saves reading time and does not preserve the certificate -- a cell that is
  neither committed nor read still carries no guarantee.

- `scripts/r55_stellar_cellsplit.py` - the external evidence was counterexample-only,
  because both external arms used splits that are hard by construction (image-grouped
  folds and cross-cohort transfer) and the exchangeable family the guarantee is stated
  for had never been run outside the depot. This script runs it: the STELLAR CODEX
  cohorts (Zenodo 6863872, CC0-1.0), a different laboratory, platform and tissue,
  trained on the cohort's other images and then split at random into calibration and
  evaluation halves, with the identical certificate, split fraction, repeat count and
  metric definitions. Tables: `experiments/r55_stellar.csv`,
  `experiments/r55_stellar_summary.csv`, `experiments/r55_stellar_norm_robustness.csv`
  and `experiments/r55_stellar_checks.csv`. Report:
  `work/r55_report.txt`. Outcome: the guarantee transfers. Coverage is 1.000 at a 10%
  budget over 150 evaluations on the intestine cohort, with a realised risk of 0.086
  and 85.1% of cells retained, and 1.000 on the target-calibrated tonsil and Barrett
  oesophagus cohorts; the uncalibrated threshold matches the certificate on throughput
  (0.890 against 0.851) but holds the budget in 0.487 of evaluations against 1.000; the
  ratio diagnostic makes no false positive and no miss; and the image-grouped arm on the
  same cells still covers 0.656, so the positive replication does not soften the
  failure. Robust to the feature transform: coverage at a 10% budget is 1.000 under
  standardisation without the log and under the raw released values as well.

- `scripts/r56_neighbour_context.py` (**withdrawn from the manuscript**) - the model
  sweeps bound the learner axis and
  the image wrap bounds the architecture axis, so this bounds the representation
  axis, which is the objection a referee makes first: every model in the paper reads
  one cell's marker vector, so the floor could be missing tissue context. Each cell
  is augmented with the mean normalised marker vector of its ten nearest neighbours
  in the same image and the log mean neighbour distance -- coordinates and
  intensities only, no label information -- and run through the identical
  certificate, oracle, folds, pool and calibration draws. Tables:
  `experiments/r56_neighbour.csv`, `experiments/r56_neighbour_summary.csv` and
  `experiments/r56_neighbour_checks.csv`. Report:
  `work/r56_report.txt`. Outcome: the context does not lower the floor. The oracle
  moves by -0.0094, -0.0137 and +0.0018 at the 5, 10 and 20 percent budgets over 190
  paired evaluations, accuracy by -0.0056, and the review fraction at a 10% budget
  by 0.15 points; the baseline arm reproduces r52's tuned-learner row exactly, which
  is what makes the comparison like for like.
  Withdrawn from the manuscript on two grounds: the effect is a null, and the
  neighbour features are computed across cells, which makes the accepted-and-wrong
  count non-binomial and voids the law the family is quoted for (the condition is
  stated in `scripts/r54_ranking.py`). The scripts and tables stay on disk as a
  record and are not cited anywhere in the manuscript.

## Round-3 additions

Two scripts were added after the second round, and both are cited in the manuscript.

- `scripts/r68_equal_guarantee.py` - the outline asked whether the certificate costs
  throughput, and the answer depends on what the two arms are matched on. Matching
  pooled realised risk lets an arm that does not hold its budget spend risk freely;
  matching the guarantee instead fixes a risk target and asks on what fraction of
  evaluations it is met. Holding both arms to a target of 0.10 on every one of the 570
  evaluations of the random-split family, the certificate keeps 0.559 of the batch
  against the oracle's 0.589, and the uncalibrated threshold cannot reach full coverage
  at any budget on the grid, so its entry is infeasible rather than large; loosening the
  match to 0.975 admits it at 0.585 against 0.559, and only by letting 11 of the 570
  evaluations run over the target, the worst of them at a realised risk of 0.235 against
  a budget of 0.10. Tables: `experiments/r68_equal_guarantee.csv`. Report: `work/r68_report.txt`.

- `scripts/r69_external2.py` - extends the external evidence from one independent
  source to three. Two further cohorts, neither sharing a laboratory or a platform with
  the depot or with STELLAR, are read as flat per-cell tables with the identical
  certificate, split fraction, repeat count and metric definitions: a multi-tissue
  tissue microarray acquired by MIBI-TOF (Zenodo 5945388, CC-BY-4.0) and human liver
  allograft biopsies acquired by imaging mass cytometry (Zenodo 7570242, CC-BY-4.0).
  Tables: `experiments/r69_external2_log1p_z.csv`, `..._summary.csv`,
  `experiments/r69_external2_panel_overlap.csv` and
  `experiments/r69_external2_checks.csv`. Report:
  `work/r69_report.txt`. Outcome: on the exchangeable split the guarantee reproduces on
  both cohorts, coverage 1.000 and 0.993 over 150 evaluations each at acceptance 0.928
  and 0.844, and restricting the panel to the shared channels costs acceptance rather
  than coverage; a real swap of antibodies onto the depot leaves the certificate valid
  and almost silent, accepting 0.008 to 0.021 of cells, and the liver swap breaches the
  budget at a 20% budget on 10 of 30 evaluations.

## Reproducing the headline results

`python scripts/r63_reproduce_check.py` re-derives, from the released tables in
`experiments/` alone, the four negative results (cross-tissue coverage, the
efficiency comparison, the marker violation level and the external cohort), the
cross-field-of-view shortfall, the positive replications on four independent
sources, the priced cross-panel deployment, the repair pilot and the tool
demonstration, against fixed expected values. It is stdlib-only and exits non-zero on any mismatch; the
exhaustive companion is `work/_preflight.py`. The prose, with the expected value
and the column behind each claim, is `docs/submission/reproducibility.md`.

## Using the layer as a tool

The layer is released as a command line tool, not only as a set of experiments.
It runs from a bare checkout, needs numpy, pandas and scikit-learn, and uses no
GPU.

```
python scripts/errctl_cli.py demo --out-dir out
python scripts/errctl_cli.py advise --demo --out-dir out
python scripts/errctl_cli.py certify --csv cells.csv --label Cell_Label \
    --markers CD3 CD20 CD8 --alpha 0.10 --out-dir out
python scripts/errctl_cli.py deploy --source train.csv --target batch.csv \
    --label Cell_Label --k 500 --alpha 0.10 --out-dir out
python scripts/errctl_cli.py advise --source train.csv --target batch.csv \
    --label Cell_Label --k 500 --scan-k 250,500,1000,2000 \
    --out-dir out
python scripts/errctl_cli.py advise --source train.csv \
    --target unlabelled_batch.csv --calibration-csv pilot_labels.csv \
    --label Cell_Label --calibration-id cell --k 500 --out-dir out
```

`certify` splits one labelled table at random into a training, a calibration and
a deployment part and reports the certificate on the deployment part; `deploy`
trains on a source table, spends `k` labels of a new batch on calibration and
reports the certificate on the rest of that batch; `demo` runs the whole
procedure on a synthetic two-batch problem so the tool can be tried with no data
of your own. Each of those runs writes `decisions.csv` (one row per deployment
cell, with the committed or deferred decision), `report.txt` (acceptance,
realised risk, coverage of the budget, the review the certificate routes, the
diagnostic ratio against an oracle, and the same review in hours under a stated
per-cell assumption) and `budget.svg` (the review budget as a figure, with real
text so the labels stay editable).

`advise` is the three-product entry point. It writes:

- `decisions.csv`, with `decision` (`commit` or `review`), the selected error
  budget and `review_rank` (read rank 1 first);
- `budget.csv`, with acceptance, realised risk, the Clopper-Pearson upper limit,
  review hours and net hours saved;
- `price_list.csv`, with target labels spent, frozen and refitted acceptance,
  net review removed, realised risk and expert hours for every ladder rung;
- `audit.csv`, one row per audit check with its interval where available;
- `certificate.md` and `certificate.json`, with the finite-sample guarantee, its
  scope, the audit verdict, the recommended next action and a machine-readable
  record;
- `budget.svg`.

The audit has four label-free legs (score shift, distance to the
predicted-class centroid, refit stability and an optional marker-logic table)
and one labelled-pilot leg, the certified-to-oracle ratio. A ratio above one
sets `quotation_allowed` to false. `--rules` accepts a `CellTypes` table with
`CellType` and `PrimaryMarker` columns. `--norm rank` uses a per-marker
within-domain rank transform, which is the safer choice for a real antibody
panel change. Calibration labels must be drawn uniformly from the batch, never
from the review queue; the certificate prints that rule and the scope of its
guarantee.

For a production batch, the target table may leave the label column blank for
the cells that will actually be deployed. `advise` treats the non-empty labels
as the calibration pilot and still writes a decision for every unlabelled cell.
If the instrument export has no label column at all, pass a separate
`--calibration-csv` with a cell id and a label column; `--calibration-id` and
`--calibration-label` name those columns. In this mode the certificate is
label-free: `realised_risk`, `cp_upper` and the certified-to-oracle ratio are
reported as `not estimable` unless some deployment cells happen to carry
labels. The price list is then a pilot cross-fit estimate rather than a second
certificate, and the certificate says so.

The demonstration cohort of the manuscript is the same procedure pointed at a
public dataset: `scripts/r70_e2e_demo.py` runs it on 28,646 immune-labelled cells
of a liver transplant imaging mass cytometry cohort over four held-out donor
sets, and `scripts/r71_crosspanel_price.py` runs it across a real change of
antibody panel while `scripts/r72_panel_repair.py` asks where those labels are
best spent and adds a same-tissue control. The numbers are in
`work/r70_report.txt`, `work/r71_report.txt` and `work/r72_report.txt`.

## Independent audits of the manuscript

Three audits were run after the round-2 integration. Each recomputes what it can
from the released tables and states its own verdict; none was written by the agent
that produced the text being checked.

- `work/r37_verify.txt` - number audit of every claim in the new passages: 91 claim
  rows, 90 MATCH, 1 MISMATCH, 0 UNVERIFIABLE, as recorded in the report's 2026-09-29
  addendum. The mismatch (the self-split coverage pair) was corrected in
  `docs/manuscript.md`. The last two rows to fall, the class/decile dispersion and the
  chi-square rejection rates, were unverifiable only because no released table carried
  per-class or per-decile accepted-calibration error counts; `experiments/r26_constancy_bins.csv`
  now does, and both recompute from it alone (`work/_v123.py` reproduces the five
  statistics from that table and exits non-zero on any mismatch).
- `work/r38_framing_audit.txt` - framing audit: 18 findings (3 HIGH, 8 MEDIUM,
  7 LOW), no numerical error. Every finding is a description or scope defect and
  each is now fixed in the manuscript.
- `work/r39_figure_audit.txt` - figure audit of `figS8`-`figS10`: 286 value checks
  with 0 MISMATCH, plus 10 P2 and 5 P3 layout and caption defects, corrected in
  `scripts/r35_figures.py` and the figure legends.
- `docs/manuscript_number_audit.md` is the cumulative record of the corrections
  these audits triggered.

## Manuscript trimmed to the outline

`docs/manuscript.md` and `manuscript.docx` now follow the approved outline's
backbone. Every Results section that answers one of the outline's R1-R7 items,
the external-validation evidence row or the wrapping of a published classifier
is kept in full. The six sections whose subject the outline never planned
(the stronger-model sweep, the reference-label-noise and ensemble objection,
the proposition development, the anytime-valid sequential controller, the two-source
partial identification and the closing decision-procedure section) are
compressed to one or two sentences each, their tables and figures are dropped,
and the surviving tables and figures are renumbered continuously.

- Results: nineteen subsections remain, six of them now a single short paragraph,
  and the two external-validation readings share one heading.
- Tables 26 to 22, renumbered 1-22. Figures 19 to 17, renumbered 1-17 by
  `docs/figure_assets.json` and rebuilt in `figures/` by
  `scripts/r67_figure_names.py`.
- Dropped: the conditional-binomial correction note that used to close the
  manuscript, and the tables and figures behind the compressed sections. Their
  Methods subsections are compressed to one sentence each and every released
  result file is unchanged.
- Condensed pass: prose-only compression of the target-domain calibration section, the
  calibration-size section, the targeted-repair section, the four longest Methods
  subsections, six Discussion paragraphs, three duplicated passages in the efficiency
  section and the Abstract. No table, figure, result or citation was dropped.
- `docs/manuscript_pre_outline_trim.md` keeps the pre-trim text.

## Submission formatting

`manuscript.docx` is rebuilt from `docs/manuscript.md` by
`scripts/r64_manuscript_docx.py`, which enforces the author's formatting brief.

- Body text is Times New Roman 11 pt, left aligned, with a two-character first-line
  indent on every body paragraph; the manuscript title is 16 pt.
- Bold is reserved for the title, the section and subsection headings, the Abstract
  heading, the `Keywords:` label, the four Abstract part labels and the `Figure N |`
  and `Table N |` caption labels. Nothing else is bold, so inline emphasis in the
  Markdown source does not leak into the document.
- Every table is a three-line table, its number and title above it and its notes
  below it; table and figure numbers run 1..22 and 1..17 continuously with no `S`
  series, in order of first appearance, and every one of them is cited in the prose.
- Figure captions sit under their image, and each caption's panels are the letters
  of the standalone `figures/Figure N<L>.svg` files.
- Numbers in the text carry two decimals, except values between 0.990 and 1.000,
  which keep three so that a near-perfect coverage is never printed as 1.00, and
  non-zero magnitudes below 0.005, which keep three so that a small value is never
  printed as 0.00. `P >= 0.001` keeps three decimals and `P < 0.001` is written
  `P < 0.001`. `work/_preflight.py` normalises decimal precision on both sides of
  every text pin, so the pins stay valid at either precision.
- All 50 references carry a DOI; each was checked twice, once by resolving
  `https://doi.org/...` and once against the Crossref or DataCite record for its
  title and year. Ten are arXiv preprints registered with DataCite and the rest are
  publisher DOIs registered with Crossref.
- The manuscript closes with `Author contributions` and `Ethics statement`; the
  author names in the contributions paragraph are placeholders, because no author
  name appears anywhere in this repository.
- `work/_preflight.py` follows the new numbering: `CHECKS=610 FAILS=0`.

## Data

The cohorts come from CellTuneDepot and are licensed for non-commercial academic
use only. They are **not** redistributed here; the retrieval and preprocessing
scripts reproduce every dataset in this repository from the public archive. See
`docs/submission/data_availability.md`.

## Licence

Code: MIT. See `docs/submission/code_availability.md`.
