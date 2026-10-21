# Code availability

## Licence

The analysis code is released under the **MIT licence**; the licence text is in
the repository root.

## Repository

GitHub: https://github.com/wzp24k/distribution-free-error-control

The repository contains the analysis code, released result tables, figure
generation scripts, `1020manuscript.docx` and `1022supplementary_NC.docx`.

## Archived release

A permanent, citable archive of the exact code version used for this manuscript will be deposited on **Zenodo** upon acceptance.

<PLACEHOLDER: Zenodo DOI — to be minted at acceptance>

The repository is public. No clinical, patient-level or identifiable material is
included.

## Computational requirements

The entire pipeline is **CPU-only** and requires no GPU. The CellSighter wrap is also run on the CPU, in place of the GPU the published code assumes; see `docs/methods.md`, `## An external, image-based classifier`. <PLACEHOLDER: the CellSighter repository ships no licence file, so `tools/CellSighter` should be excluded from the released archive unless the authors confirm terms> It is implemented in **pure Python 3.10** and depends only on general scientific libraries:

- `numpy`, `scipy`
- `pandas`, `pyarrow` (parquet caching)
- `scikit-learn`
- `matplotlib`, `seaborn` (figures)
- `tifffile` (TIFF decoding)
- three off-the-shelf graders from scikit-learn, defined in `src/errctl/classifiers.py`: logistic regression, random forest, and histogram gradient boosting
- `torch`, `torchvision`, `opencv-python`, `scikit-image` and `tensorboard` (CPU builds), for the wrap of a published image-based classifier in `scripts/r51_cellsighter.py`; the published CellSighter repository is vendored unmodified at `tools/CellSighter` and cited in `docs/methods.md`
- `xgboost` (CPU build), for the strongest-ranker sweep in `scripts/r54_ranking.py` and for the reimplemented CellTune classifier in `scripts/r51b_celltune.py`. LightGBM is deliberately **not** a dependency: its C API raises an access violation in `LGBM_DatasetSetField` inside spawned worker processes on Windows, a failure reproducible outside the sweep, so `r54` keeps the strongest learners that fit reliably here and Table 13 records the omission
- `catboost` (CPU build), for the same reimplementation: CellTune's documentation names XGBoost and CatBoost as its default model pair, and `scripts/r51b_celltune.py` reproduces that pair and averages its two probability matrices.
- `R` 4.6.1 with Bioconductor `SingleCellExperiment`, used only by `scripts/r75_immucan_export.R` to export the public IMMUcan labelled cells. The released `data/external3/immucan_v3_cells.csv` is sufficient for `scripts/r75_immucan_lock.py`, which is pure Python.

The IMMUcan `.rds` object and the exported cell table are not included in the
code archive; both are recreated from Zenodo record 12912567 by the export
script. This keeps the repository free of the large public source file and of any
sample-level material that is not needed to reproduce the released summaries.

Reference hardware: a 16-logical-core x86-64 desktop with 16 GB RAM; the full experiment grid (`scripts/r02`, `scripts/r05`, `scripts/r07`–`r09`, `scripts/r11`) runs on this machine without accelerator hardware. <PLACEHOLDER: confirm the exact machine specification quoted in Methods>

## Repository layout

```
src/errctl/          the errctl package (the contribution under test)
  paths.py           filesystem layout and configuration
  labels.py          label harmonisation and the COARSE_MAP to common cell types
  markerlogic.py     expert marker-rule parsing/compilation to Boolean predicates
  data.py            CellTuneDepot table loading and expert/HQ label joining
  preprocess.py      intensity extraction and decile normalisation
  classifiers.py     regularised classifier wrappers and hyperparameter grids
  calibration.py     temperature scaling and calibration diagnostics (ECE, reliability)
  conformal.py       conformal risk control: selective-risk thresholds and bounds
  weights.py         covariate-shift density-ratio weighting
  markercheck.py     marker-consistency audit (rule violation accounting)
  eval.py            coverage / efficiency / risk metrics and AUROC, AURC

scripts/
  s00_inspect_data.py, s01_inventory.py, s02_probe_demo_and_join.py, s03_probe_images.py
                     data inventory and integrity audit
  r01_build_datasets.py     build and cache per-cohort cell tables (data audit)
  r02_run_experiments.py    train graders and store per-fold scores/predictions
  r03_analyse.py            compute risk-coverage metrics, ECE, conformal controllers
  r04_report.py             consolidate all result tables into the report
  r05_cellsplit.py          pooled random cell-level splits: pooled and per-cohort
  r06_marker_audit.py       marker-rule violation rates: by source, class, accepted/deferred
  r07_ablations.py          ablations: calibration size, threshold grid, delta, score, min-accept
  r08_panel.py              marker-panel reduction: the throughput cost of a
                            smaller panel under exchangeability
  r09_simulation.py         Monte-Carlo validation of the guarantee (exchangeability, shift)
  r10_figures.py            main and extended-data figures
  r11_ensemble.py           bagged-ensemble graders
  r12_labelnoise.py         reference-label-noise sensitivity
  r13_coverage_ci.py        cluster-bootstrap intervals for coverage
  r14_targetcal.py          transductive target-domain calibration and model adaptation
  r14c_ladder_dense.py      Table 6 recomputed on the denser label grid (reads
                            experiments/r14_targetcal_ext.csv)
  r15_marker_sensitivity.py marker-rule violation under 16 definition/threshold conventions
  r16_review_budget.py      review budget and the matched-risk price of validity
  r17_simulation.py         second Monte-Carlo study of the rebuilt controllers
  r18_marker_sources.py     marker audit across label sources and cell populations
  r19_weight_shift_diag.py  label-free diagnosis of the weighted certificate
  r20_label_ladder.py       the label cost of the guarantee, by split family
  r21_downstream_repair.py  downstream neighbourhood enrichment under the repair
  r21b_matched_downstream.py the same statistic under one weight per type
  r22_fig8.py               rebuild of the label-ladder figure
  r24_efficiency.py         matched-realised-risk accounting and certificate yield
  r24b_labelshift.py        label-shift reweighting as an alternative to calibration
  r25_accept_deferred.py    accepted-versus-deferred marker consistency, 68 conventions
  r26_plugin_baseline.py    conservative plug-in baselines and the confidence-level sweep
                            and the constancy companion, r26_constancy_bins.csv
  r27_fixedseq.py           fixed-sequence step-down: acceptance, validity, grid dependence
  r28_marker_bounds.py      identified sets and convention-free verdicts for the marker audit
  r31_targeted_repair.py    targeted repair of the within and transfer conformal failures
  r32_sequential.py         anytime-valid (sequential) selective-risk certificate
  r33_two_source.py         two annotation sources and partial identification of the marker audit
  r34_external.py           external validation on STELLAR CODEX and the CellTune demo provenance audit
  r35_figures.py            supplementary figures S8-S10 from the round-2 tables
  r43_threadpin.py          BLAS thread pinning and the reproducibility of the adaptation arm
  r44_gvhd_delivery.py      do the shipped segmentation masks cover the shipped
                            labels?  The GVHD delivery defect, written up in
                            `docs/submission/gvhd_delivery_defect.md`
  r48_fov_units.py          cross-field-of-view coverage at field-of-view resolution
  r48b_fov_repair.py        the targeted-repair arms re-run on those field-of-view units
  r49_markerlogic.py        the marker-logic (panel-invariant) representation: published control,
                            rule-only, and two augmented spaces
  r49b_panelinvariance.py   the marker-rule audit layer across the four panels
  r50_review_loop.py        the review budget against the label price of the repair,
                            in the same units (derived from r16 and r24 tables)
  r51_cellsighter.py        the layer around a published image-based classifier (CellSighter)
  r51b_celltune.py          the layer around a second published classifier, the CellTune
                            classifier rebuilt from its documentation (XGBoost and CatBoost
                            over documented per-cell compartment statistics)
  r52_stronger_score.py     seven models through the same controller on the random-split
                            family's own folds: what a stronger score buys
  r53_calibration_size.py   the price of validity at a controlled calibration size
  r54_ranking.py            the strongest tabular rankers on the same folds: can a better
                            ranking lower the oracle?
  r55_stellar_cellsplit.py  the independent cohorts on the exchangeable split: the
                            guarantee reproduced outside the depot
  r56_neighbour_context.py  spatial neighbourhood features through the same
                            certificate (withdrawn from the manuscript; kept as a record)
  r57_multiplicity.py       multiplicity and selection correction over the released tables
  r57b_triage.py            the worth of the certificate's own ordering of the review queue
  r59_adviser.py            the guided decision procedure on the command line
  r60_adviser_html.py       builds the bilingual interface and its sample batch,
                            from the released demo unit
  r70_e2e_demo.py           the layer run end to end on a public cohort, with
                     the review fraction converted into expert hours under a
                     stated per-cell assumption
  r71_crosspanel_price.py   a real change of antibody panel, priced in target
                     labels: coverage at every rung of a 100 to 8,000 label
                     ladder, against the arm that calibrates on the source
  r72_panel_repair.py       the same change of panel with the labels spent on
                     the model rather than the calibrator, and four further
                     panel changes that hold the tissue fixed or move the
                     platform: a colon adenocarcinoma core and two breast
                     ductal carcinoma cores of one MIBI-TOF tissue microarray
                     against their own depot cohorts on 13 and 8 shared
                     markers, the array's own tonsil cores against the
                     STELLAR tonsil cohort on 11 shared markers, which is
                     MIBI-TOF against CODEX, and a second independent source
                     for the colon core from STELLAR intestine on 10 shared
                     markers with one whole source region held out
  r73_label_price.py       the label price of a bounded likelihood ratio:
                    the effective-sample summary of the r19 transfer weights
                    and the exact two-point lower bound of Proposition 3
  r74_repair_triage.py     a 250-label pilot before a panel repair: source-level
                    separation, cluster-bootstrap intervals, threshold
                    sensitivity and a leave-one-fold-out threshold rule
  r76_pilot_multiplicity.py  the same pilot under the Bonferroni and
                    fixed-sequence corrections, with deployment and
                    pilot-only ratios reported separately
  r77_pilot_decision.py   the pilot as a decision rule: four-fold threshold
                    selection applied to the fifth, a calibration-silence
                    ladder and a failed net-gain lower bound
  r78_external_repair_pair.py  the same 250-label pilot on a second repair
                    pair, breast and tonsil, with the colon cut transported
                    unchanged and a within-pair threshold cross-fitted
  r80_immucan_breas_export.R  export the BREAS six-class repair set from the
                    public IMMUcan SingleCellExperiment
  r80_immucan_breast_repair.py  the large-target repair test on 21,731 IMMUcan
                    BREAS labels: 1,000 labels (4.6%), 500 refit + 500
                    calibrate, twenty draws and three classifiers
  r81_immucan_breast_sourcefold.py  the post-primary five source-image-fold
                    robustness check for the same repair
  r75_immucan_export.R     export the public IMMUcan panel 1 V3 labelled cells
                    from the deposited SingleCellExperiment
  r75_immucan_lock.py      the pre-registered first evaluation on IMMUcan:
                    five sample-grouped folds, ten draws, three classifiers
  errctl_cli.py            the layer as a command line tool: certify a labelled
                     table, deploy on a new batch with k target labels, repair
                     a shifted batch, or run `advise` once to write the
                     commit/review budget, the target-label price list and the
                     label-free audit (with a labelled-pilot ratio), including
                     `certificate.md`, `certificate.json`, `budget.csv`,
                     `decisions.csv`, `price_list.csv`, `audit.csv` and
                     `budget.svg`.  The target table may leave labels blank
                     for deployment cells; a separate `--calibration-csv`
                     supplies an id,label pilot when the export has no label
                     column, and unlabelled cells are never imputed
  r61_shared_panel.py       a real re-panelling: training, calibration and
                            deployment on the channels two cohorts share
  r62_source_family.py      the fourth partition family: one cell set under
                            two label sources, with the two-source repair
                            and the per-fold transfer diagnostic
  r63_reproduce_check.py    one command that re-checks the four negative results,
                            the cross-field-of-view shortfall, the positive
                            replications, the repair pilot and the pre-registered
                            IMMUcan evaluation against fixed expected values
                            (reads only experiments/*.csv)
  r85_duodenum_crosspanel.py  the real antibody-panel change: trains on STELLAR
                            small intestine and deploys to the adult duodenum
                            CODEX atlas on 14 shared markers
  r86_verify_duodenum.py    re-derives the duodenum summary from the released
                            per-evaluation CSV (220 assertions)
  r87_duodenum_ranknorm.py  the rank-normalised rerun: per-marker within-domain
                            rank transform plus the three-model ensemble
  r88_verify_duodenum_ranknorm.py  re-derives the rank-normalised summary from
                            its per-evaluation CSV (734 assertions)
  run_pipeline.py           the data-stage driver (stage order, logs, table diffs)

experiments/        result tables produced by r03/r06/r07/r08/r09/r11, plus the
                    round-2 analyses: r31_*.csv (13 files), r32_*.csv (4 files),
                    r33_*.csv (11 files), r34_*.csv (17 files) and
                    r34_phase2_*.csv (10 files), and r49_*.csv (8 files: metrics and
                    coverage intervals for the four marker-logic spaces) together with
                    r49b_panelinvariance.csv, r48_*.csv (3 files: metrics, ECE and coverage
                    intervals for the field-of-view-resolution family) and
                    r48b_fov_repair_*.csv (4 files: detail, summary with
                    unit-bootstrap intervals, pooled paired differences, and the
                    same differences per budget), and r50_review_loop.csv
                    (review fraction and repair price per family and budget),
                    and r51_cellsighter.csv with r51_cellsighter_summary.csv
                    (the wrapped CellSighter network, by split family and budget) and
                    r51b_celltune.csv with r51b_celltune_summary.csv (the reimplemented
                    CellTune classifier on the same units), and
                    r52_stronger.csv with r52_stronger_summary.csv (seven models
                    through the certificate and the oracle, by budget), and
                    r53_calibration_size.csv with r53_calibration_size_summary.csv
                    (acceptance against calibration size on a balanced panel), and
                    r54_ranking.csv with r54_ranking_summary.csv (the strongest
                    rankers through the certificate and the oracle, by budget), and
                    r55_stellar.csv with r55_stellar_summary.csv (the independent
                    cohorts on the exchangeable and target-calibrated splits), together
                    with r55_stellar_norm_robustness.csv (the same result under three
                    feature transforms) and r55_stellar_checks.csv, and
                    r56_neighbour.csv with r56_neighbour_summary.csv (the
                    own-markers baseline against the same score with ten-nearest-
                    neighbour context, by budget) and r56_neighbour_checks.csv
                    (r56 is withdrawn from the manuscript and kept as a record), and
                    r57_multiplicity.csv with r57_multiplicity_summary.csv and
                    r57_triage.csv with r57_triage_summary.csv (multiplicity and
                    selection correction, and the order to read the queue in), and
                    r74_repair_triage.csv with r74_repair_triage_summary.csv
                    (the labelled pilot that prices a panel repair), and
                    r75_immucan_evals.csv with r75_immucan_summary.csv (the
                    pre-registered IMMUcan evaluation), and
                    r61_shared_panel.csv with r61_shared_panel_summary.csv and
                    r61_shared_panel_mg.csv with r61_shared_panel_mg_summary.csv
                    (the real re-panelling on the shared channels, for the
                    colorectal/GVHD and melanoma/GVHD pairs), and
                    r62_source_family.csv with r62_source_family_summary.csv
                    (the annotation-source family: the same cells under two
                    label sources, and the repair that covers both) and
                    r62_fold_transfer.csv (the leave-one-fold-out transfer
                    accuracy of every fold and classifier), and
                    r21_downstream.csv with r21b_matched_downstream.csv (the
                    neighbourhood-enrichment distortion under the accepted set, and
                    the same statistic under one weight per type on both scopes), and
                    r44_gvhd_delivery.csv with r44_gvhd_delivery_summary.csv
                    (whether the shipped segmentation masks cover the shipped labels,
                    per cohort and field of view), and
                    r26_constancy_bins.csv (the accepted and wrong cell counts
                    behind the two constancy statistics, by predicted class and
                    by score decile), and
                    r14_targetcal_ext.csv with r14c_ladder_dense.csv (the
                    label-ladder grid behind Table 6, extended to eleven rungs)
figures/            figure output (PDF + PNG)
tools/CellSighter/  the published CellSighter repository, vendored unmodified for r51
work/               intermediate caches, logs and data-audit reports
```

## Guided interface

`docs/adviser.html` (Chinese) and `docs/adviser_en.html` (the same interface in
English, for referees) are single-file, offline, dependency-free decision aids for
a laboratory that has to triage a batch of cells. It asks for the error budget, fits the
paper's own certificate, and returns an acceptance fraction, a threshold, a review queue
ordered by the certificate's own score, and -- when deployment labels are available --
the audit ratio. The certificate is a port of `src/errctl/conformal.py`
(`SelectiveRiskController` with `num="cp"`, `denom="cp"`, `select="bonferroni"`,
`delta=0.10`, `min_accept=25`, and a 120-point acceptance grid); on the released unit
shipped in the page it reproduces the Python implementation to every printed digit
(lambda = 0.18987 at a 10% budget, acceptance 0.74010, oracle acceptance 0.81660, audit
ratio 0.90631). No data leaves the browser. The same procedure is on the command line as
`scripts/r59_adviser.py` (`--demo`, `--npz`, `--alpha`, `--yes`, `--out`), which
additionally prices a repair in target labels. The page is generated by
`scripts/r60_adviser_html.py`, which embeds the released melanoma random-split unit at
full double precision through the same loader the command-line adviser uses, so the two
report the same numbers on the same cells; `work/_preflight.py` reads the unit back out
of the finished HTML and reruns `src/errctl` on it. Both files carry the same text
table, so the two languages cannot drift apart.

`docs/adviser_sample.csv` is a 2,001-row synthetic batch in the page's own
three-column format (`set,score,error`): 800 calibration cells, all labelled, and
1,200 deployment cells of which 400 carry labels. It exists so that a first-time user
can press the download button, load the file straight back, and watch the whole
procedure run on data that owes nothing to the manuscript. The label column is
deliberately half empty in the deployment block, which is what makes the audit ratio
demonstrable without any labels of the user's own; when a file is only partly
labelled, the page reports its realised risk against the labelled subset and says so.
Every number the page returns on that file -- acceptance 0.4150 / 0.6867 / 0.9158 and
threshold 0.4441 / 0.6876 / 0.9057 at a 5 / 10 / 20 % budget, and on the 400 labelled
rows acceptance 0.7175, realised risk 0.0767 and audit ratio 0.9228 at 10 % -- is
pinned by `work/_preflight.py` and by the generator `scripts/r60_adviser_html.py`,
which rewrites the CSV and both pages from a fixed seed (20260928).

## Reproducing the results

```powershell
# 1. obtain the data locally (see data_availability.md) and place it under data/
# 2. build the cached cell tables and the data-audit report
python scripts/r01_build_datasets.py
# 3. train graders across folds (writes work/predictions/*.npz)
python scripts/r02_run_experiments.py
python scripts/r05_cellsplit.py
# 4. analyse, audit and report
python scripts/r03_analyse.py --alphas 0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50 --jobs 8
python scripts/r06_marker_audit.py
python scripts/r07_ablations.py
python scripts/r08_panel.py
python scripts/r09_simulation.py
python scripts/r04_report.py
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
python scripts/r11_ensemble.py --jobs 8
# round-2 analyses
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
python scripts/r51_cellsighter.py --stage all --epochs 20
python scripts/r51b_celltune.py --stage all --hold-every 2 --cal-repeats 40 --threads 14
python scripts/r51b_celltune.py --space table --norm log1p_z --src GVHD_Feb1323 --tgt CRC_Feb2623 --hold-every 2 --cal-repeats 40 --threads 14 --suffix _gvhd2crc_z
python scripts/r51b_celltune.py --space table --norm log1p_z --src CRC_Feb2623 --tgt GVHD_Feb1323 --hold-every 2 --cal-repeats 40 --threads 14 --suffix _crc2gvhd_z
python scripts/r52_stronger_score.py --jobs 14
python scripts/r53_calibration_size.py --jobs 8 --reps 10
python scripts/r54_ranking.py --jobs 14
python scripts/r55_stellar_cellsplit.py --only ext_targetcal
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs logreg --tag logreg
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs rf     --tag rf
python scripts/r55_stellar_cellsplit.py --only ext_cellsplit --clfs hgb    --tag hgb
python scripts/r55_stellar_cellsplit.py --norm z_train --tag z_train
python scripts/r55_stellar_cellsplit.py --norm raw     --tag raw
python scripts/r56_neighbour_context.py --cohort Breast_Jun0524
python scripts/r56_neighbour_context.py --cohort Melanoma_Oct2522
python scripts/r56_neighbour_context.py --cohort CRC_Feb2623
python scripts/r56_neighbour_context.py --cohort GVHD_Feb1323
python scripts/r56_neighbour_context.py --analyse
python scripts/r70_e2e_demo.py --jobs 4 --models xgb,cat,hgb
python scripts/r71_crosspanel_price.py --jobs 6
python scripts/r72_panel_repair.py --part abcde --jobs 6
python scripts/r73_label_price.py
python scripts/r74_repair_triage.py
python scripts/r76_pilot_multiplicity.py
python scripts/r77_pilot_decision.py
python scripts/r78_external_repair_pair.py --jobs 12
Rscript scripts/r75_immucan_export.R
python scripts/r75_immucan_lock.py
python scripts/r75_immucan_lock.py --from-evals
python scripts/errctl_cli.py demo --out-dir out
python scripts/errctl_cli.py advise --demo --out-dir out
python scripts/errctl_cli.py advise --source train.csv --target batch.csv \
    --label Cell_Label --k 500 --scan-k 250,500,1000 --out-dir out
python scripts/r89_verify_advise.py
# the same labels on a refitted model, on a source and a shifted batch
python scripts/errctl_cli.py repair --source train.csv --target batch.csv \
    --label Cell_Label --k 500 --scan-k 250,500,1000 --out-dir out
python scripts/r61_shared_panel.py --jobs 8
python scripts/r61_shared_panel.py --pair Melanoma_Oct2522,GVHD_Feb1323 --tag _mg --jobs 8
python scripts/r62_source_family.py --jobs 8
# 5. verify a released checkout from the result tables alone (no data, no GPU)
python scripts/r63_reproduce_check.py
# 6. figures
python scripts/r10_figures.py
```

Result tables are written to `experiments/` and `work/`; the consolidated
human-readable report is `experiments/r04_report.txt`.

## Verifying a released checkout

Two checks read the released tables in `experiments/` and need no dataset, no
prediction cache and no accelerator:

- `python scripts/r63_reproduce_check.py` re-derives the four negative results
  (cross-tissue coverage, the efficiency comparison, the marker violation level
  and the external cohort), the cross-field-of-view shortfall, the positive
  replications including the real duodenum panel change, the repair pilot and
  the pre-registered IMMUcan evaluation, against fixed expected values. It is
  stdlib-only, prints one line per assertion, and exits non-zero on any
  mismatch (254 assertions).
- `python scripts/r86_verify_duodenum.py` re-derives coverage, acceptance,
  realised risk and Wilson intervals for every duodenum arm from
  `experiments/r85_duodenum_crosspanel.csv` alone (220 assertions).
- `python scripts/r88_verify_duodenum_ranknorm.py` does the same for the
  rank-normalised rerun in `experiments/r87_duodenum_ranknorm.csv` (734
  assertions).
- `work/_preflight.py` re-derives every number quoted in `docs/manuscript_nc.md`
  from the same tables and resolves the registrations of the later scripts
  (621 checks). It is a working-tree tool, not part of the released
  archive.

The prose behind both, with the expected value and tolerance for each headline
claim, is `docs/submission/reproducibility.md`.

## Figure file mapping

Main figures are `figures/figN_*.pdf` and `.png` for N = 1-8. **Figure 9** of the
manuscript is `figures/fig10_yield.pdf` / `.png`; the file keeps its original
sequence number, which is why the two differ. A downstream-effect figure
(`figures/fig9_downstream.*`) was dropped from the paper, and its result is
reported in text as not supporting the repair claim, so there is no file
`fig9_*` in use.

Supplementary figures map to files with a one-step offset after S1:
`figures/figS1_labelnoise.*` is Supplementary Figure 1,
`figures/figS3_marker_sensitivity.*` is Supplementary Figure 2,
`figures/figS4_marker_sources.*` is Supplementary Figure 3,
`figures/figS5_weight_shift.*` is Supplementary Figure 4,
`figures/figS6_accept_deferred.*` is Supplementary Figure 5,
`figures/figS7_fixedseq.*` is Supplementary Figure 6,
`figures/figS8_repair.*` is Supplementary Figure 7 (drawn by
`scripts/r35_figures.py` from `experiments/r31_*.csv`),
`figures/figS9_sequential.*` is Supplementary Figure 8 (from `r32_*.csv`), and
`figures/figS10_external.*` is Supplementary Figure 9 (from `r34_*.csv`). There
is no `figS2_*` file; the label was retired with an earlier draft.

## Table provenance

Every main and supplementary table is backed by released result files, and each
table's legend in `docs/manuscript_nc.md` names its own file. The same mapping, in
one place, with the script that writes each file (`experiments/` throughout):

| table | backing result file(s) | script |
|---|---|---|
| Table 1 | `r03_summary.csv` (acceptance, coverage, risk) and `r13_coverage_ci.csv` (the 95% unit-bootstrap interval) | `r03_analyse.py`, `r13_coverage_ci.py` |
| Table 2 | `r16_review_budget.csv` and `r24_efficiency.csv` | `r50_review_loop.py` |
| Table 3 | `r28_bounds_by_marker.csv` and `r28_bounds_pooled.csv`; per-convention rows in `r28_bounds.csv` | `r28_marker_bounds.py` |
| Table 4 | `r34_phase2_certificate_summary.csv`, `r55_stellar_summary.csv`, `r69_external2_log1p_z_summary.csv`, `r72_panel_repair.csv`, `r75_immucan_summary.csv`, `r82_immucan_v1_summary.csv`, `r85_duodenum_crosspanel_summary.csv` and `r87_duodenum_ranknorm_summary.csv` | `r34_external.py`, `r55_stellar_cellsplit.py`, `r69_external2.py`, `r72_panel_repair.py`, `r75_immucan_lock.py`, `r82_immucan_v1.py`, `r85_duodenum_crosspanel.py`, `r87_duodenum_ranknorm.py` |
| the tool demonstration | `r70_e2e_demo.csv` with `r70_e2e_units.csv` | `r70_e2e_demo.py` |
| the cross-panel price ladder | `r71_crosspanel_price.csv` | `r71_crosspanel_price.py` |
| the panel repair and its four further panel changes | `r72_panel_repair.csv` | `r72_panel_repair.py` |
| the labelled repair pilot | `r74_repair_triage.csv` with `r74_repair_triage_summary.csv` | `r74_repair_triage.py` |
| the pilot under both multiplicity corrections | `r76_pilot_multiplicity.csv` with `r76_pilot_multiplicity_summary.csv` | `r76_pilot_multiplicity.py` |
| the pilot decision rule and the silence ladder | `r77_pilot_decision.csv` with `r77_calibration_silence.csv` | `r77_pilot_decision.py` |
| the pilot on a second repair pair | `r78_external_pair_summary.csv` with `r78_external_repair_pair.csv` | `r78_external_repair_pair.py` |
| the pre-registered IMMUcan evaluation | `r75_immucan_evals.csv` with `r75_immucan_summary.csv` | `r75_immucan_lock.py`; export by `r75_immucan_export.R` |
| Supplementary Table 1 | `r28_bounds_by_marker.csv` and `r28_bounds_pooled.csv` | `r28_marker_bounds.py` |
| Supplementary Table 2 | `r26_plugin_baseline_summary.csv`; per-evaluation rows in `r26_plugin_baseline.csv` | `r26_plugin_baseline.py` |
| Supplementary Table 3 | `r27_fixedseq_summary.csv`; per-evaluation rows in `r27_fixedseq.csv` | `r27_fixedseq.py` |
| Supplementary Table 4 | `r31_within_summary.csv`, `r31_transfer_summary.csv`, `r31_paired.csv`, `r31_paired_transfer.csv`, `r31_ladder_summary.csv` | `r31_targeted_repair.py` |
| Supplementary Table 5 | `r32_sequential_summary.csv`; per-run rows in `r32_sequential.csv`, simulation in `r32_sim.csv` | `r32_sequential.py` |
| Supplementary Table 6 | `r33_sharp.csv`, `r33_headline_summary.csv`, `r33_agreement.csv` | `r33_two_source.py` |
| Supplementary Table 7 | `r34_phase2_certificate_summary.csv`, `r34_phase2_ratio.csv`, `r34_phase2_checks.csv` | `r34_external.py` |
| Supplementary Table 8 | `r48_metrics.csv` and `r48_coverage_ci.csv` | `r48_fov_units.py` |
| Supplementary Table 9 | `r48b_fov_repair_summary.csv` with `r48b_fov_repair_paired.csv` and `r48b_fov_repair_paired_by_alpha.csv` | `r48b_fov_repair.py` |

The withdrawn r56 tables (`r56_neighbour*.csv`) and the r52, r53, r54 and r57
result files back no table in the current four-table manuscript: they are kept
on disk as a record only.

## Randomness and determinism

All experiments fix the numpy/scikit-learn random seeds per fold and repeat, so re-running the pipeline on the same depot produces byte-identical result tables. The external STELLAR CODEX cohorts in `scripts/r34_external.py` are subsampled once to 20,000 cells each with seed 20240618. <PLACEHOLDER: state the seed policy explicitly in Methods>
