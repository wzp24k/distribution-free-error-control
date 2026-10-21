# Pre-registered replication: IMMUcan panel 1 V1

**Status:** protocol locked on 2026-10-19, before any coverage, acceptance or
risk endpoint was computed on this batch. Only the deposited object schema
(dimensions, marker names, label vocabulary, sample identifiers) will be
inspected before the endpoint.

## Question

Does the selective-risk certificate reproduce its finite-sample guarantee on
the first IMMUcan annotation batch (V1), a batch that has not entered any
earlier analysis in this manuscript?

The V3 batch supplied the locked first evaluation (R75). V1 is a separate
annotation round from the same consortium and the same antibody panel. It is a
replication of precision and breadth, not a new cross-panel test.

## Data

- Source: IMMUcan panel 1, first labelling batch, `sce_labelled_V1.rds`.
- Record: Zenodo 12912567, DOI 10.5281/zenodo.12912567, CC BY 4.0.
- Published size: 179 patients and more than 340,000 manually annotated cells
  across 14 cell types.
- Feature matrix: the deposited `exprs` assay restricted to channels with
  `use_channel = TRUE`.
- Labels: the deposited `cell_labels` column; `unlabelled` cells are removed.
- Independence rule: any `sample_id` that also occurs in the V3 batch is
  removed before the primary endpoint, so that the replication sample is
  disjoint from the locked first evaluation. The number of removed samples is
  reported.

## Split and model

- Grouping unit: `sample_id`.
- Five folds formed by shuffling sample identifiers with seed 20261018.
- For each fold, train on the other samples and deploy on the held-out samples.
- Training cells are capped at 60,000 and the held-out pool at 20,000.
- A class enters a fold only with at least 3 training cells and 20 held-out
  cells.
- Within each held-out pool, calibration and evaluation cells are a random
  50/50 split, repeated 10 times.
- Classifiers: multinomial logistic regression, random forest and histogram
  gradient boosting, with the pipeline defaults.
- Certificate: two-unconditional-probability controller, delta 0.10, minimum
  acceptance 25, 120-threshold grid.
- Budgets: 5%, 10% and 20% selective risk.

## Primary endpoint

At alpha = 0.10, over 5 folds x 10 draws x 3 classifiers = 150 evaluations:

1. vacuity-inclusive coverage at least 0.90;
2. Wilson 95% lower bound for coverage at least 0.85;
3. non-vacuous fraction at least 0.80;
4. mean acceptance at least 0.50.

All four conditions must hold. A failure is reported as a failed replication
without changing the seed, class filter, classifier set, budget or threshold
rule.

## Secondary endpoints

Coverage, acceptance, realised risk, vacuity and the certified-to-oracle ratio
at alpha = 0.05 and 0.20. These are descriptive and do not rescue a failed
primary endpoint.

## Locked hashes

| object | SHA256 |
|---|---|
| `data/external3/sce_labelled_V1.rds` | `956B0F4249934DA33272A324CFBA52F73AE74F54C2EDEDC1623D7E370D0380FC` |
| `data/external3/immucan_v1_cells.csv` | `192CED17A61F254744F5D5BCDF785F4E0D28214926C6EFBFA95B440D34F544D0` |

The deposited file has MD5 `bdd982c3b307245032a050445b673b22`, matching the
Zenodo record, and 365,957,908 bytes.

## Result

The V1 batch contains 290,226 labelled cells from 149 samples and 37 markers.
None of the 149 sample identifiers occurs in V3, so the replication sample is
disjoint from the locked first evaluation.

At alpha = 0.05, 0.10 and 0.20, vacuity-inclusive coverage is 1.000
[0.975, 1.000] over 150 evaluations, the non-vacuous fraction is 1.000, and
the realised selective risk is 0.040, 0.065 and 0.065. Mean acceptance is
0.935, 0.999 and 1.000.

The pre-registered primary endpoint passes all four conditions.
