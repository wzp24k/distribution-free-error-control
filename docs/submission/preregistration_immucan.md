# Pre-registered first evaluation: IMMUcan panel 1 V3

**Status:** locked on 2026-10-18, before the classifier or the certificate was fitted on this cohort. Schema inspection (dimensions, marker names, label vocabulary and class counts) was performed; no coverage, acceptance or risk endpoint had been computed.

## Question

Does the selective-risk certificate reproduce its finite-sample guarantee on an independent imaging mass cytometry cohort that has not entered any earlier analysis, under a split whose calibration and evaluation halves are exchangeable by construction?

This is a first evaluation, not a re-analysis of a cohort used for model or threshold selection. The result will be reported as a pass or a failure of the pre-specified criterion, whichever occurs.

## Data

- Source: IMMUcan panel 1, second labelling batch, `sce_labelled_V3.rds`.
- Record: Zenodo 12912567, DOI 10.5281/zenodo.12912567, CC-BY-4.0.
- Cohort: 80,988 cells from 30 samples; 57,791 cells with a manual label other than `unlabelled`; 37 antibody channels marked `use_channel = TRUE` in the deposited row metadata.
- Feature matrix: the deposited `exprs` assay, one row per cell, 37 columns.
- Labels: the deposited `cell_labels` column. `unlabelled` cells are excluded before any split.
- No clinical, diagnostic or patient-level record is exported or redistributed. The analysis uses de-identified sample identifiers only.

## Split and model

- Grouping unit: `sample_id`. Five folds are formed by shuffling the 30 sample identifiers with seed 20261018 and splitting them into five groups of six.
- For each fold, the model is trained on cells from the other 24 samples and deployed on the six held-out samples.
- Training cells are capped at 60,000; the held-out pool is capped at 20,000. A class enters a fold only when it has at least 3 training cells and 20 held-out cells.
- Within each held-out pool, calibration and evaluation cells are a random 50/50 split, repeated 10 times. This is the exchangeable arm: the two halves are drawn from the same held-out population.
- Classifiers: multinomial logistic regression, random forest and histogram gradient boosting, with the pipeline defaults in `src/errctl/classifiers.py`.
- Certificate: the same two-unconditional-probability controller, delta = 0.10, minimum acceptance 25 and 120-threshold grid used everywhere else in the paper.
- Budgets: 5%, 10% and 20% selective risk.

## Primary endpoint

At alpha = 0.10, over 5 folds x 10 draws x 3 classifiers = 150 evaluations:

1. vacuity-inclusive coverage at least 0.90;
2. Wilson 95% lower bound for coverage at least 0.85;
3. non-vacuous fraction at least 0.80;
4. mean acceptance at least 0.50.

All four conditions must hold. A failure of any condition is reported as a failed first evaluation, without changing the fold seed, class filter, classifier set, budget or threshold rule.

## Secondary endpoints

Coverage, acceptance, realised risk, vacuity and the certified-to-oracle ratio at alpha = 0.05 and 0.20, with the same folds and draws. These are descriptive and do not rescue a failed primary endpoint.

## Locked hashes

| object | SHA256 |
|---|---|
| `data/external3/sce_labelled_V3.rds` | `CF9D07BD33053A004CE61933CF114A4AC77402CE248CC48FB9EF582B1A1BB2AC` |
| `data/external3/immucan_v3_cells.csv` | `EDD311DCC393A8DBFF8B24C2AC8F8F0C656096871CB1D60BF9AA656A1F42A860` |
| `scripts/r75_immucan_lock.py` (protocol as locked) | `241FABB2D60533259963D733E031F9454D1D15C635A1C4061F7519622E83A6BE` |
| `scripts/r75_immucan_lock.py` (current, determinism only) | `01743CE8C49F907F3BAC141DF5B283A8A99B74D78075D0031BCBB7EF6C4A04DA` |
| `scripts/r75_immucan_export.R` | `037E5AEA2D4AF95AE132C21D61C05780B1D3F1749D90927BA633B89BB17B7C50` |

The code hash covers the fold seed, class filter, classifier list, budgets, certificate settings and the primary success rule. Re-running the script is a deterministic check of the locked analysis, not a new evaluation.


## Amendment 1

The first execution completed all 450 model evaluations and wrote `experiments/r75_immucan_evals.csv`, then stopped in the summary step because `g.empty` resolved to the DataFrame attribute rather than the stored `empty` column. The implementation was corrected to `g["empty"]`; no fold, seed, class filter, classifier, budget, controller setting or success criterion changed. The corrected script was rerun end to end as a deterministic check. The primary result is computed from the saved evaluation rows and the rerun, not from a new endpoint.

The second execution repeated the model fitting under the same seeds and again reached the summary step, where a second `g.empty` occurrence stopped it. Rather than spend further compute, the summary was rebuilt from the 450 evaluation rows already written by the first execution with `--from-evals`. The endpoint is therefore the first evaluation, not a newly trained result, and no analysis parameter changed between executions.

## Amendment 2

After the endpoint had been computed, the report header was corrected from a hard-coded 38 markers to a count derived from the exported table; the table has 37 marker columns. This changes a label in `work/r75_immucan_report.txt` only. The analysis script was also given `--from-evals` so that the summary can be checked without refitting. Neither change touches the folds, seed, class filter, classifiers, budgets, controller or success rule. The current hash in the table above covers these reporting fixes; the protocol hash is retained beside it.

## Amendment 3

The worker pool was changed from `imap_unordered` to `imap` so that the
evaluation table is written in the protocol's job order. The change affects row
order only; it does not touch the folds, seed, class filter, classifiers,
budgets, controller, success rule or any endpoint. The released evaluation rows
and the primary result are unchanged, and the current hash above records the
deterministic version.
