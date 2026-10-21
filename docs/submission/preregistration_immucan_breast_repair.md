# Pre-specified secondary repair evaluation: IMMUcan BREAS

**Status:** protocol fixed on 2026-10-19 after class counts and marker names
were inspected, before any classifier was fitted or any net, acceptance or risk
endpoint was computed on this target.

## Question

Does the 1,000-label repair remain net positive on a second breast target whose
batch is large enough that the same label spend is below five per cent of the
target? This is a secondary repair evaluation, not a new prospective cohort and
not an independent validation of the certificate.

## Data and label mapping

- Target: IMMUcan panel 1 V3, `Indication == BREAS`, 28,920 manual labels over
  14 samples (Zenodo 12912567, DOI 10.5281/zenodo.12912567, CC-BY-4.0).
- Analysis set: the six target labels with a one-to-one source class,
  `Tumor`, `CD4`, `CD8`, `B`, `plasma` and `MacCD163`, total 21,731 cells.
  Labels without a clean source counterpart are excluded before fitting:
  `BnT` because it mixes B and T, `Treg` because the source has no Treg class,
  `DC`, `pDC`, `HLADR`, `NK`, `Mural` and `Neutrophil` for the same reason or
  because the source class has fewer than 30 examples.
- Source: CellTuneDepot `Breast_Jun0524`, mapped as
  `TumorEpithelial -> Tumor`, `CD4T -> CD4`, `CD8T -> CD8`,
  `Bcell -> B`, `Plasma -> plasma`, `Macrophage -> MacCD163`.
- Shared channels: `CD3`, `CD4`, `CD8a` (source `CD8`), `CD20`, `CD38`,
  `CD14`, `CD15`, `CD68`, `Ecad` (source `ECAD`), `HLADR` (source
  `HLADRDPDQ`), `Ki67` and `SMA`.
- No clinical, patient-level or diagnostic field is exported. The target file
  contains only a de-identified sample identifier, the six-way label and the
  twelve marker intensities.

## Split, estimator and accounting

- The source model is fitted once per classifier on all mapped source cells.
- The target analysis set is split by a uniformly random permutation. The first
  1,000 cells are the label budget: 500 refit the model on the source plus
  target cells and 500 calibrate the certificate. The remaining 20,731 cells
  are the deployment pool. This is the same exchangeable within-target split
  used by `scripts/r72_panel_repair.py`.
- Twenty independent draws, three classifiers (`logreg`, `rf`, `hgb`), and
  budgets 0.05, 0.10 and 0.20. `delta = 0.10`, `min_accept = 25` and the
  standard 120-point threshold grid.
- Net is `(accepted - 1,000) / 20,731`, the paper's net-of-calibration
  accounting. The label fraction is `1,000 / 21,731 = 0.0460`.

## Endpoints

Primary, at alpha = 0.10:

1. for each tree ensemble (`rf`, `hgb`), mean net is positive and the 95%
   bootstrap interval over the twenty draws excludes zero;
2. the realised risk among accepted cells is at or below 0.10 in at least
   95% of non-empty draws;
3. at least 18 of 20 draws are net positive for each tree ensemble.

Logistic regression, the 0.05 and 0.20 budgets, the frozen-model arm and the
source-model accuracy are secondary and are reported without changing the
primary rule. A failure is reported as a failed replication.

## Locked hashes

The protocol is frozen against the following files. Hashes are filled after the
two analysis files are written and before the endpoint is executed.

| object | SHA256 |
|---|---|
| `scripts/r80_immucan_breas_export.R` | `3CFF219AB9CDA374FFB30E736D127C50B332C64B91EAEDB20845414DDB8C7A17` |
| `scripts/r80_immucan_breast_repair.py` (model run) | `444B6193FEA091C257A9209B2E046340916C7BEF17510988D6F6872DB09B5DDA` |
| `scripts/r80_immucan_breast_repair.py` (current, summary denominator only) | `71AF307E7F4D9C68392CD24F58DD0486B5FB7680DCDB80A1E7A7E310F5B881B2` |
| `data/external3/immucan_breas_repair.csv` | `4482A83A4DB23723AB3B39FF457B8F23E6551C5726957C292CE7C1A1918CEB3C` |

## Amendment 1

The model run completed before the summary was rebuilt. The first summary used
the raw `n_pool` field, which is the full 21,731-cell target, as the denominator
of net. The pre-specified denominator is the 20,731-cell deployment pool after
the 1,000 labels are set aside. The summary was corrected and rebuilt with
`--from-csv`; no model was refitted, and no draw, classifier, threshold or
endpoint changed. The corrected denominator is the one reported.

## Post-primary robustness (not confirmatory)

After the primary endpoint was computed, the same target protocol was repeated
with five disjoint source-image folds and five target draws per fold
(`scripts/r81_immucan_breast_sourcefold.py`). This is an exploratory robustness
check, not a second confirmatory test. Histogram gradient boosting and random
forest were net positive in 25 of 25 draws and in all five source-fold means;
logistic regression was positive in 18 of 25 draws and in all five fold means
but had a wider spread. The stable claim remains the tree-ensemble one.

A complete rerun of both scripts on 2026-10-19 reproduced all six released
output files byte-for-byte, with a maximum absolute numeric difference of zero
in the raw and summary tables (`work/r80_r81_determinism_check.txt`).
