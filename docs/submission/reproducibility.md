# Reproducing the manuscript from the released tables

**What this note is for.** Four of this manuscript's headline results are
negative results -- the guarantee fails under cross-tissue transfer, it buys no
throughput over the baselines, and the shipped marker rules disagree with the
shipped labels -- while the guarantee itself replicates positively on the
independent public cohorts, and two published
classifiers wrap into the same guarantee and the same failure, and the layer
run end to end on a public cohort converts the review fraction into hours. A
reader who suspects any of them is entitled to re-derive it without downloading
a dataset, training a model or owning a GPU. Every such number is
reproducible from the released result tables in `experiments/` alone, and this
note states, per claim, the file, the column and the expected value.

## The one command

```
python scripts/r63_reproduce_check.py
```

It reads only `experiments/*.csv`, is stdlib-only, and prints one line per
assertion. On an intact checkout the tail is:

```
ASSERTIONS=254 FAILS=0
REPRODUCED: 4 of 4 negative results, the cross-field-of-view
shortfall, the positive replications and a real cross-panel
deployment, the real panel swap, the priced
cross-panel deployment, the repair of that deployment by a refit and
the tool demonstration, and the protocol-locked IMMUcan evaluation
-- all 254
assertions hold.
```

Any mismatch prints `BAD`, lists the offending row, and exits with status 1.
`python scripts/r63_reproduce_check.py --list` names the pinned claims without
running them. The exhaustive companion is `work/_preflight.py` (622 checks),
which re-derives every number quoted in `docs/manuscript_nc.md`, including the ones
r63 does not cover; it is a working-tree tool rather than part of the archive.
The layer itself ships as a command line tool, `scripts/errctl_cli.py`, with a
synthetic demonstration that runs with no data of the reader's own; the two
tables above are what it prints when pointed at a real cohort.

## What is and is not reproduced here

- **Reproducible from the released tables, with no data and no compute**: every
  number in every main and supplementary table and in every figure, because each
  table is a deterministic summary of the CSVs in `experiments/`. The mapping is
  one-to-one and is listed in `docs/submission/code_availability.md` under
  "Table provenance", and named in each table's own legend in `docs/manuscript.md`.
- **Reproducible with the data**: the tables themselves, by running the pipeline
  in `docs/submission/code_availability.md`. The full grid is CPU-only and fits on
  a 16-core, 16 GB desktop.
- **Not reproduced here**: the underlying cohorts. The CellTuneDepot data are
  licensed for non-commercial academic use and are not redistributed; the
  retrieval and preprocessing path that rebuilds every dataset from the public
  archive is `scripts/r01_build_datasets.py` (see
  `docs/submission/data_availability.md`). The independent STELLAR CODEX cohorts
  are public (Zenodo record 6863872, CC0-1.0) and are retrieved by
  `scripts/r34_external.py`.

## The four negative results and the positive replications

Each row below is a claim the manuscript makes, the released file and column it
comes from, and the value r63 pins. Tolerances are the precision the manuscript
quotes.

### N1 -- the guarantee fails under cross-tissue transfer

Source: `experiments/r03_summary.csv`, filtering `kind == "transfer"` and
`method == "crc"`; the non-vacuous figure is `experiments/r24_efficiency.csv`
with `family == "transfer"`, `arm == "src"`, `k == 0`.

| quantity | column | alpha | expected |
|---|---|---|---|
| coverage of the budget | `coverage` | 0.05 | 0.5407 |
| coverage of the budget | `coverage` | 0.10 | 0.3972 |
| coverage of the budget | `coverage` | 0.20 | 0.2861 |
| acceptance | `mean_accept` | 0.10 | 0.3676 |
| realised selective risk | `mean_risk` | 0.10 | 0.3164 |
| coverage among committing decisions | `cov_nonvac` | 0.10 | 0.01364 |

The direction matters as much as the level: coverage *falls* as the budget
loosens (0.541 to 0.397 to 0.286), which is the signature of a bound that has
stopped tracking the quantity it claims to bound.

### N2 -- the certificate buys no throughput

Source: `experiments/r16_matched_risk_ratio.csv`, `family == "cellsplit"`,
`risk == 0.10`, where the certificate, an oracle and the uncalibrated baseline
are all held to the same realised selective risk.

| quantity | column | expected |
|---|---|---|
| certificate / oracle acceptance | `ratio_crc_oracle` | 0.7444 |
| certificate acceptance | `crc_accept` | 0.5594 |
| oracle acceptance | `oracle_accept` | 0.7516 |
| matched-risk rows where the certificate beats the oracle | (count over the file) | 0 |

The guarantee is a bound, not an efficiency claim: at a matched risk the
certificate retains 0.744 of what an oracle threshold retains, and it never
retains more. That is why the paper's claim is validity plus a review budget,
not "fewer labels than the baselines".

### N3 -- the marker violation level is convention-dependent

Sources: `experiments/r28_bounds_pooled.csv` (`scope == "summary"`,
`source == "expert"`, `cls == "ALL_evaluable"`) for the identified sets, and
`experiments/r28_bounds_by_marker.csv` (`scope == "type"`,
`source == "expert"`) for the per-type verdicts. The sweep is 68 positivity,
reading and rule-text conventions per cohort.

| cohort | `rate_min` | `rate_max` |
|---|---|---|
| Breast_Jun0524 | 0.0044295 | 0.8783893 |
| CRC_Feb2623 | 0.0058406 | 0.9023780 |
| GVHD_Feb1323 | 0.0209386 | 0.9002525 |
| Melanoma_Oct2522 | 0.0398367 | 0.8722766 |

Over the 48 expert type rows, exactly five clear the 5% gate under all 68
conventions, and colorectal APC clears it under none; the median of
`ratio_span_to_mean_ci` over those 48 rows is 12.7029, so the binding uncertainty
in this audit is the convention rather than the sample.

### N4 -- the guarantee is not reproduced on the external cohort

Sources: `experiments/r34_phase2_certificate_summary.csv` and
`experiments/r34_phase2_ratio.csv`, both at `alpha == 0.10`. The cohort is
STELLAR CODEX, a different laboratory, platform and tissue.

| quantity | file / filter | expected |
|---|---|---|
| within-intestine coverage | certificate summary, `ext_within` | 0.6556 |
| within-intestine acceptance | certificate summary, `ext_within` | 0.7264 |
| cross-tissue coverage | certificate summary, `ext_transfer` | 0.0333 |
| cross-tissue acceptance | certificate summary, `ext_transfer` | 0.8394 |
| cross-tissue over-budget evaluations | ratio, `level="eval"`, `ext_transfer` | 174 |
| of those, the ratio diagnostic misses | ratio, `level="eval"`, `ext_transfer` | 60 |
| within-cohort over-budget evaluations the ratio misses | ratio, `level="eval"`, `ext_within` | 2 |

### N5 -- the cross-field-of-view shortfall is resolved at field-of-view resolution

Source: `experiments/r48_coverage_ci.csv`, `method == "crc"`. The published
cross-field-of-view family is twenty image-fold units and cannot separate its
shortfall from nominal; holding out one image per unit puts it on 96 units.

| quantity | alpha | expected |
|---|---|---|
| coverage | 0.05 | 0.84375 |
| coverage CI lower | 0.05 | 0.7882 |
| coverage CI upper (excludes 0.90) | 0.05 | 0.8941 |
| coverage | 0.10 | 0.79225 |
| coverage CI upper (excludes 0.90) | 0.10 | 0.8547 |

### P1 -- the positive replication

Source: `experiments/r55_stellar_summary.csv` (the headline row is
`norm == "log1p_z"`, `family == "ext_cellsplit"`, `alpha == 0.10`,
`method == "crc"`) and `experiments/r55_stellar_norm_robustness.csv`.

| quantity | file / filter | expected |
|---|---|---|
| exchangeable-split coverage | summary, `ext_cellsplit` | 1.000 |
| acceptance | summary, `ext_cellsplit` | 0.8514 |
| realised selective risk | summary, `ext_cellsplit` | 0.08646 |
| target-calibrated coverage | summary, `ext_targetcal` | 1.000 |
| coverage under standardisation without the log | robustness, `z_train` | 1.000 |
| coverage under the raw released values | robustness, `raw` | 1.000 |

The image-grouped arm on the same cells still covers only 0.656, so the positive
replication on the exchangeable split does not soften the failure under the
splits that are hard by construction.

### P2 to P5 -- the further independent material and the wrapped classifiers

Sources: `experiments/r69_external2_log1p_z_summary.csv` (the two further
independent cohorts), `experiments/r51_cellsighter_summary_breast2crc.csv` (the
wrapped CellSighter network on the breast to colorectal pair),
`experiments/r51b_celltune_summary.csv` (the CellTune classifier rebuilt from its
documentation on the same pair) and `experiments/r51_cellsighter_summary.csv`
(the earlier melanoma to GVHD wrap).

| quantity | file / filter | expected |
|---|---|---|
| MIBI-TOF acceptance, exchangeable | r69, `ext2_cellsplit`, `EXT2_MIBITOF_Angelo`, a=0.05 | 0.8444 |
| MIBI-TOF coverage, exchangeable | r69, `ext2_cellsplit`, `EXT2_MIBITOF_Angelo`, a=0.10 | 0.99333 |
| liver IMC acceptance, exchangeable | r69, `ext2_cellsplit`, `EXT2_LiverTx_IMC`, a=0.05 | 0.928 |
| liver IMC coverage, exchangeable | r69, `ext2_cellsplit`, `EXT2_LiverTx_IMC`, a=0.10 | 1.000 |
| MIBI-TOF acceptance and vacuity, panel swap | r69, `ext2_panelswap`, `EXT2_MIBITOF_Angelo`, a=0.05 | 0.000, vacuous 1.000 |
| liver IMC acceptance and vacuity, panel swap | r69, `ext2_panelswap`, `EXT2_LiverTx_IMC`, a=0.10 | 0.0199, vacuous 0.667 |
| CellSighter acceptance and coverage, exchangeable | r51 breast2crc, a=0.10 | 0.80589, 1.000 |
| CellSighter ratio, target transfer | r51 breast2crc, a=0.10 | 10.575 |
| CellTune acceptance and coverage, exchangeable | r51b, a=0.05 | 0.81438, 1.000 |
| CellTune acceptance, exchangeable | r51b, a=0.10 | 0.96152 |
| CellTune ratio, target transfer | r51b, a=0.10 | 39.872 |
| CellTune table entry acceptance and coverage, exchangeable | r51b GVHD to colorectal, a=0.10 | 0.87809, 1.000 |
| CellTune table entry ratio, target transfer | r51b GVHD to colorectal, a=0.10 | 5.4831 |
| CellTune table entry, mirror direction | r51b colorectal to GVHD, a=0.10 | accepts 0.000, coverage 1.000 by the empty-set convention |

### P7, P8 and P9 -- the tool demonstration, the cross-panel price ladder and the panel repair

Sources: `experiments/r70_e2e_demo.csv` (the end to end demonstration on the
public liver cohort, one row per deployment unit, arm, label count and budget)
and `experiments/r71_crosspanel_price.csv` (the real panel change on the 27
channels two depot panels share, one row per fold, classifier, repeat, rung and
budget). The repair is priced against the same ladder by
`experiments/r72_panel_repair.csv` (one row per part, cohort pair, fold,
classifier, repeat, rung, budget and arm, where the arms are the frozen model,
the frozen model with half the calibration labels, and the model refitted on
the other half; the five parts are the depot colorectal to GVHD change, a colon
core and two breast cores of one MIBI-TOF array against their own depot
cohorts, that array's tonsil cores against a CODEX tonsil cohort, and a
second source for the colon core from the STELLAR intestine table). All
three are released tables, so all three are re-derived by the same command.

| quantity | file / filter | expected |
|---|---|---|
| deployment cells in the demonstration | r70, `arm=layer`, `alpha=0.05`, `k=500` | 28646 |
| layer acceptance, 5% budget | r70, same rows | 0.91305 |
| coverage, 5% budget | r70, same rows | 1.000 |
| expert hours removed, 5% budget | r70, same rows, accepted minus the 2,000 calibration labels at 20 s | 134.0167 |
| expert hours removed, 20% budget | r70, `alpha=0.20`, `k=500` | 147.6722 |
| tuned-threshold coverage, 5% budget | r70, `arm=strong`, `alpha=0.05`, `k=500` | 0.500 |
| cross-panel coverage, every rung of the ladder | r71, `tgt_k*` at k = 100 to 4,000, alpha = 0.05 to 0.20 | 1.000 |
| cross-panel acceptance, 8,000 target labels | r71, `tgt_k8000`, `alpha=0.20` | 0.19946 |
| cross-panel acceptance, 4,000 target labels | r71, `tgt_k4000`, `alpha=0.10` | 0.02755 |
| source-calibrated acceptance | r71, `src_cal`, `alpha=0.10` | 0.40259 |
| source-calibrated coverage | r71, `src_cal`, `alpha=0.10` | 0.300 |
| source-calibrated coverage | r71, `src_cal`, `alpha=0.20` | 0.23333 |
| panel-repair rows pinned | r72, all five parts | 11340 |
| refit acceptance, same tissue, 1,000 labels, 20% budget | r72, `part=B_same_tissue_panel`, `arm=adapt_k1000`, `alpha=0.20` | 0.70473 |
| refit realised risk, same tissue | r72, same rows | 0.11323 |
| refit coverage, same tissue | r72, same rows | 1.000 |
| refit Clopper-Pearson validity, same tissue | r72, same rows | 1.000 |
| refit net of the labels read, same tissue | r72, same rows, accepted minus 1,000 labels over a pool of 5,458 | 0.39239 |
| frozen acceptance, same tissue | r72, `arm=tgt_k1000`, `alpha=0.20` | 0.23264 |
| frozen net of the labels read, same tissue | r72, same rows | 0.00680 |
| refit acceptance, cross tissue, 1,000 labels, 20% budget | r72, `part=A_cross_tissue_panel`, `pair=CRC_Feb2623->GVHD_Feb1323`, `arm=adapt_k1000` | 0.84083 |
| refit net of the labels read, cross tissue | r72, same rows | 0.63033 |
| frozen acceptance, cross tissue | r72, `arm=tgt_k1000`, same pair | 0.04379 |
| shared marker counts | r72, 13 same-tissue, 27 cross-tissue | 13, 27 |
| second organ, refit acceptance at a 5% budget | r72, `part=C_same_tissue_panel_breast`, `arm=adapt_k1000`, `alpha=0.05` | 0.51177 |
| second organ, refit net of the labels read at 5% | r72, same rows | 0.44627 |
| second organ, refit realised risk at a 5% budget | r72, same rows | 0.01811 |
| second organ, refit coverage at a 5% budget | r72, same rows | 1.000 |
| second organ, frozen arm at a 5% budget | r72, `arm=tgt_k1000`, `alpha=0.05` | 0.000 |
| second organ, refit acceptance at a 10% budget | r72, `arm=adapt_k1000`, `alpha=0.10` | 0.61738 |
| shared marker count, breast pair | r72, `part=C_same_tissue_panel_breast` | 8 |
| cross-platform frozen acceptance, every rung | r72, `part=D_cross_platform_tonsil`, `arm=tgt_k*`, every budget | 0.000 |
| cross-platform refit acceptance, 1,000 labels, 10% budget | r72, `arm=adapt_k1000`, `alpha=0.10` | 0.05220 |
| cross-platform refit realised risk, 10% budget | r72, same rows | 0.03338 |
| cross-platform refit net of the labels read, 10% budget | r72, same rows | 0.03613 |
| cross-platform refit acceptance, 1,000 labels, 20% budget | r72, `arm=adapt_k1000`, `alpha=0.20` | 0.30750 |
| cross-platform refit acceptance, 4,000 labels, 10% budget | r72, `arm=adapt_k4000`, `alpha=0.10` | 0.53407 |
| cross-platform refit net of the labels read, 4,000 labels | r72, same rows | 0.44038 |
| cross-platform coverage and Clopper-Pearson validity | r72, `part=D_cross_platform_tonsil` | 1.000, 1.000 |
| shared marker and class counts, cross-platform pair | r72, `part=D_cross_platform_tonsil` | 11, 6 |
| second source for the colon core, frozen acceptance, 20% budget | r72, `part=E_cross_platform_colon`, `arm=tgt_k1000`, `alpha=0.20` | 0.03653 |
| second source for the colon core, refit acceptance, 4,000 labels, 20% budget | r72, `arm=adapt_k4000`, `alpha=0.20` | 0.42180 |
| second source for the colon core, refit realised risk and coverage | r72, same rows | 0.13259, 1.000 |
| second source for the colon core, refit net of the labels read | r72, same rows | -0.27576 |
| second source for the colon core, shared markers and classes | r72, `part=E_cross_platform_colon` | 10, 6 |

The per-cell review time is the one assumption in the hours column, 20 seconds,
one minute per three cells; the released report also carries the 5 and 60 second
sensitivities. The demonstration cohort is not redistributed, and neither
`r70`, `r71` nor `r72` needs it: all three are re-derived from the released CSVs alone.

CellTune is a reimplementation of the classifier its documentation specifies, not
the shipped desktop application: the per-evaluation rows are in
`experiments/r51b_celltune.csv`, and `work/r51b_report.txt` records the feature
definition, the model pair and every deviation. Both wraps therefore read as one
claim, that the layer is agnostic to the model above it, and one diagnostic, that
a stronger base classifier over-commits further once exchangeability is broken.

### P10 to P13 -- the pre-registered cohort and the repair pilot

Sources: `experiments/r75_immucan_summary.csv` (the locked IMMUcan panel 1 V3
evaluation), `experiments/r74_repair_triage_summary.csv` (the 250-label pilot,
with cluster-bootstrap intervals), `experiments/r76_pilot_multiplicity_summary.csv`
(the Bonferroni and fixed-sequence pilot at 250 labels),
`experiments/r77_pilot_decision.csv` (the cross-fitted decision rule and the
failed net-gain lower bound) and `experiments/r77_calibration_silence.csv`
(the calibration-silence ladder), and `experiments/r78_external_pair_summary.csv`
with `experiments/r78_external_repair_pair.csv` (the same pilot on a second
repair pair, breast and tonsil). The IMMUcan protocol, including its amendment
record, is `docs/submission/preregistration_immucan.md`.

| quantity | file / filter | expected |
|---|---|---|
| evaluations at a 10% budget | r75 summary, `alpha=0.10` | 150 |
| coverage at a 10% budget | r75 summary | 1.000 |
| Wilson lower bound on coverage | r75 summary | 0.97503 |
| acceptance at a 10% budget | r75 summary | 0.98592 |
| realised selective risk | r75 summary | 0.06827 |
| non-vacuous evaluations | r75 summary | 1.000 |
| oracle-proxy AUC at 20%, 250 pilot labels | r74 summary, `outcome_k1000`, `oracle_auc` | 0.94902 |
| oracle-proxy AUC lower bound | r74 summary, same row, `lo` | 0.89196 |
| units where the 250-label certificate accepts nothing at 20% | r74 summary, `cert_accept_zero` | 80 |
| units where the ratio is zero or undefined at 20% | r74 summary, `ratio_zero_or_nan` | 80 |
| units where the pilot-only ratio is zero or undefined at 20% | r74 summary, `pilot_ratio_zero_or_nan` | 80 |
| nonzero deployment ratios at 20% | r74 summary, `ratio_support` | 1 |
| nonzero pilot-only ratios at 20% | r74 summary, `pilot_ratio_support` | 1 |
| units where the 250-label certificate accepts nothing at 10% | r72, parts B and E, `arm=tgt_k250` | 81 |
| units where the ratio is zero or undefined at 10% | r72, same filter | 81 |
| units where the unlabelled-denominator certificate is silent at 20% | r74 summary, `cert_accept_ud_zero` | 75 |
| unlabelled-denominator ratio AUC at 20% | r74 summary, `ratio_ud_auc` | 0.52696 |
| fixed-sequence support at 20% | r76 summary, `select=fixed_sequence`, `denom=cal` | 17 |
| fixed-sequence deployment ratio AUC at 20% | r76 summary, same row, `auc_deploy` | 0.50523 |
| fixed-sequence pilot ratio AUC at 20% | r76 summary, same row, `auc_pilot` | 0.50523 |
| fixed-sequence unlabelled support at 20% | r76 summary, `denom=ud` | 19 |
| fixed-sequence unlabelled ratio AUC at 20% | r76 summary, same row, `auc_deploy` | 0.49967 |
| accuracy of the descriptive 0.35 cut | r74 summary, `accuracy` | 0.88889 |
| accuracy with a threshold chosen on four folds | r74 summary, `lofo_accuracy` | 0.90123 |
| units still silent at 10% with 4,000 labels | r77 silence, `alpha=0.10`, `k=4000` | 62 |
| units still silent at 20% with 4,000 labels | r77 silence, `alpha=0.20`, `k=4000` | 19 |
| units silent at 20% with 1,000 labels | r77 silence, `alpha=0.20`, `k=1000` | 48 |
| LOFO go units | r77 decision, `policy=lofo`, `part=ALL` | 45 |
| LOFO TPR / TNR / accuracy | r77 decision, same row | 0.86275 / 0.96667 / 0.90123 |
| LOFO net, same-tissue | r77 decision, `part=B_same_tissue_panel` | 0.37081 |
| LOFO delta vs always, same-tissue | r77 decision, same row, `delta_mean` | -0.021579 |
| LOFO net, cross-platform | r77 decision, `part=E_cross_platform_colon` | -0.031161 |
| LOFO delta vs always, cross-platform | r77 decision, same row, `delta_mean` | 0.063906 |
| LOFO cell-weighted delta vs always, all units | r77 decision, `part=ALL`, `delta_cell` | 0.024961 |
| LOFO cell-weighted delta interval | r77 decision, same row, `delta_cell_lo/hi` | 0.003296 / 0.043056 |
| naive net-gain LCB go units | r77 decision, `policy=net_lcb`, `part=ALL` | 67 |
| naive LCB TNR | r77 decision, same row, `tnr` | 0.433333 |
| naive LCB delta vs always | r77 decision, same row, `delta_mean` | 0.008639 |
| net gain of triage over always repairing the E source, fixed cut | r74 summary, policy row | 0.06135 |
| second-pair units, breast and tonsil | r78 summary, `policy=lofo_CD`, parts C and D | 45 / 45 |
| positive-net units, breast and tonsil | r78 repair table, `alpha=0.20`, `arm=adapt_k1000` | 30 / 24 |
| certificate-valid units, breast and tonsil | r78 repair table, same filter, `cp_valid` | 45 / 45 |
| LOFO net, breast | r78 summary, `policy=lofo_CD`, part C | 0.497882 |
| LOFO delta vs always, breast | r78 summary, same row, `delta` | -0.095351 |
| LOFO delta upper bound, breast | r78 summary, same row, `delta_hi` | -0.011845 |
| LOFO net, tonsil | r78 summary, `policy=lofo_CD`, part D | 0.232805 |
| LOFO delta vs always, tonsil | r78 summary, same row, `delta` | -0.054728 |
| LOFO delta upper bound, tonsil | r78 summary, same row, `delta_hi` | 0.001018 |
| pooled second-pair go units | r78 summary, `policy=lofo_CD`, `part=ALL` | 67 |
| pooled second-pair delta vs always | r78 summary, same row, `delta` | -0.075040 |
| units sent to repair by the transported colon cut | r78 summary, `policy=transport_BE_0.3725`, `part=ALL`, `go` | 0 |
| oracle headroom, breast | r78 summary, `policy=oracle`, part C, `delta` | 0.010831 |
| oracle headroom, tonsil | r78 summary, `policy=oracle`, part D, `delta` | 0.005344 |

The pilot is not a label-free test. Its usable statistic is the target-labelled
oracle ceiling. At 250 labels the certificate is silent in 80 of 81 units at a
20% budget and in all 81 at 10%. The silence is not a bug but the finite-sample
price: at 10% even 4,000 labels leave 62 of 81 units silent, and at 20% 1,000
labels leave 48 of 81. The calibration-only ratio therefore becomes estimable
only when the pilot costs as much as the repair (AUC 0.73 at 1,000 labels), which
is why the pilot uses the oracle ceiling instead. A leave-one-fold-out threshold
(0.3625-0.3725) reaches TPR 0.863, TNR 0.967 and accuracy 0.901; it retains
0.371 of the same-tissue batch against 0.392 for always repairing, and reduces
the cross-platform net loss from 0.095 to 0.031, a saving of 0.064 [0.032,
0.092]. The cross-platform arm remains net negative: the pilot reduces the loss
rather than turning it into a gain. A naive Clopper-Pearson net-gain lower bound
is too permissive (67 of 81 go, TNR 0.433, delta 0.009) because the 250-label
oracle is a ceiling, not a lower bound on the 1,000-label refit; it is retained
as a failed comparator. The leave-one-fold-out rule is a cross-fitted transport
check, not an externally prospective test; a true prospective evaluation on a
new cohort pair remains future work. The fixed 0.35 cut was inspected after the
result and is reported only as a descriptive comparator.

The second pair fixes the operating regime. On the breast and tonsil panel
changes the 1,000-label refit is profitable in 30 of 45 and 24 of 45 units and
every certificate stays within its 20% budget, so the repair itself replicates.
The decision rule does not transport: the colon threshold of 0.3725 sends none of the
90 units to repair because both cohorts sit below its pilot ceiling, and a
threshold cross-fitted inside each pair sends 30 of 45 breast and 37 of 45
tonsil units and loses 0.095 and 0.055 of the batch against repairing whole.
That loss is not a ranking failure. An oracle that knows which units repay can
add only 0.011 and 0.005 over repairing whole, because a failed repair costs
about 0.04 and 0.02 of the batch while a successful one returns 0.91 and 0.55.
When most units repay, the pilot's 250 labels are a cost with almost nothing to
buy, so the rule is a gate for deployments whose repairs often fail, as in the
cross-platform colon pair, rather than a universal first step.

### P14 -- the real antibody-panel change, STELLAR to duodenum

Sources: `experiments/r85_duodenum_crosspanel.csv` (per-evaluation rows) and
`experiments/r85_duodenum_crosspanel_summary.csv` (the summary), written by
`scripts/r85_duodenum_crosspanel.py`; the released summary is re-derived
independently by `scripts/r86_verify_duodenum.py` (220 assertions) and, for the
rank-normalised rerun, `scripts/r88_verify_duodenum_ranknorm.py` (734
assertions). The source
cohort is the STELLAR CODEX small intestine table (donor B004, 235,874 labelled
cells, 48 markers); the target is the adult human proximal-intestine CODEX atlas
(Zenodo record 10724499, 211,424 labelled cells, 42 channels, 3 donors, 107
ROIs). Fourteen markers are shared. No target label trains the classifier;
target labels enter only the calibration half.

| quantity | file / filter | expected |
|---|---|---|
| shared markers | `work/duodenum_shared_markers.csv`, `confidence=high` | 14 |
| target-calibrated coverage at a 10% budget | r85 summary, `arm=cross_random`, `alpha=0.10` | 1.000 |
| target-calibrated acceptance at a 10% budget | r85 summary, same row | 0.207 |
| target-calibrated realised risk at a 10% budget | r85 summary, same row | 0.077 |
| target-calibrated coverage at a 20% budget | r85 summary, `alpha=0.20` | 1.000 |
| target-calibrated acceptance at a 20% budget | r85 summary, same row | 0.628 |
| source-only coverage at a 10% budget | r85 summary, `arm=cross_source` | 0.000 |
| source-only realised risk at a 10% budget | r85 summary, same row | 0.312 |
| ROI-disjoint coverage at a 10% budget | r85 summary, `arm=cross_roi` | 0.953 |
| ROI-disjoint coverage at a 20% budget | r85 summary, same row | 0.807 |
| acceptance with 500 target labels at 10% | r85 summary, `arm=cross_k500` | 0.000 |
| acceptance with 10,000 target labels at 10% | r85 summary, `arm=cross_k10000` | 0.173 |
| in-domain acceptance on the same 14 markers | r85 summary, `arm=within_target` | 0.809 |

The operational `FAIL` printed at the end of `work/r85_duodenum_crosspanel_report.txt`
refers to the script's pre-set throughput screen (`acceptance >= 0.50`), not to
coverage: the certificate is valid at 1.000, and the shortfall is in the
acceptance that a real panel change costs. The manuscript reports the two
separately and does not claim an efficiency gain for this deployment.

### P15 -- why the duodenum acceptance was 0.21, and the rank-normalised rerun

The z-score pipeline aligns the target to the source by subtracting the source
mean and dividing by the source standard deviation after log1p. That assumes the
two panels have the same shape. They do not. With 14 markers and 60,000 training
cells the linear model is overconfident: the certificate is valid, but its
confidence no longer ranks errors, and its oracle acceptance at a 10% budget is
only 0.025. The unweighted mean over the three classifiers is therefore pulled
down by a base model that contributes almost no acceptance.

R87 keeps every other choice and replaces the transform with a per-marker
within-domain rank: log1p, empirical quantile, standard normal. The transform
uses no labels and is fitted separately inside each domain. Sources:
`experiments/r87_duodenum_ranknorm.csv` (per-evaluation rows) and
`experiments/r87_duodenum_ranknorm_summary.csv` (per-classifier summary),
written by `scripts/r87_duodenum_ranknorm.py` and re-derived by
`scripts/r88_verify_duodenum_ranknorm.py` (734 assertions).

| quantity | file / filter | expected |
|---|---|---|
| ensemble coverage at a 10% budget | r87 summary, `arm=cross_random`, `clf=ens` | 1.000 |
| ensemble acceptance at a 10% budget | r87 summary, same row | 0.495 |
| ensemble realised risk at a 10% budget | r87 summary, same row | 0.087 |
| random-forest acceptance at a 10% budget | r87 summary, `clf=rf` | 0.479 |
| linear-model acceptance at a 10% budget | r87 summary, `clf=logreg` | 0.339 |
| gradient-boosting acceptance at a 10% budget | r87 summary, `clf=hgb` | 0.328 |
| mean over the three base models | r87 summary, same arm and budget | 0.382 |
| ensemble acceptance at a 20% budget | r87 summary, `alpha=0.20` | 0.827 |
| ROI-disjoint coverage at a 10% budget | r87 summary, `arm=cross_roi`, `clf=ens` | 0.880 |
| source-only coverage at a 10% budget | r87 summary, `arm=srcnull`, `clf=ens` | 0.000 |
| source-only realised risk at a 10% budget | r87 summary, same row | 0.254 |
| acceptance with 10,000 labels at 10% | r87 summary, `arm=crossk`, `clf=ens` | 0.466 |
| acceptance with 500 labels at 10% | r87 summary, same filter | 0.000 |
| in-domain acceptance at a 10% budget | r87 summary, `arm=within_target`, `clf=ens` | 0.830 |

The two runs answer different questions. R85 shows that the certificate is valid
under a real panel change at the cost of throughput. R87 shows that the size of
that cost is set by the normalisation and the confidence ranking, not by the
certificate: the same guarantee, on the same cells, admits twice as many labels
once the marker distributions are aligned by rank. The manuscript reports both
rows, and the z-score arm remains the pre-specified pipeline.

## Why this can be re-derived without the data

The released tables carry the per-evaluation rows as well as the summaries, so a
reader can recompute every summary statistic rather than trust it. Two examples:

- the N2 ratio at a 10% realised risk is
  `mean(crc_accept) / mean(oracle_accept)` over
  `experiments/r16_matched_risk_ratio.csv` at `family == "cellsplit"`,
  `risk == 0.10`, and equals 0.7444;
- the N5 coverage is the evaluation-level mean of the `crc` rows of
  `experiments/r48_metrics.csv` grouped by `unit` at `alpha == 0.10`, whose
  unit-level mean is 0.79225 with the cluster-bootstrap interval quoted above.

Row counts are quoted in the table legends (150, 180, 190, 1,728 and so on), so
a reader can also check that the summarised row set is the one the manuscript
says it is.

## Determinism

Every experiment fixes its numpy/scikit-learn seeds per fold and repeat. The
measured property is narrower than "the values reproduce exactly", and it is now
characterised rather than asserted, because the whole pipeline was re-run end to
end from the shipped code and the shipped data (`scripts/run_pipeline.py --mode
full --jobs 16`; 37 stages, every one exit 0, `work/logs/round28_full.out`).

**What reproduces, and at what granularity.** `work/datasets/*.parquet` is 5/5
md5-identical to its manifest, but all 297 files under `work/predictions/` differ
by floating-point last bits when the models are retrained on identical inputs with
identical seeds (mean 605 bytes smaller per file). A byte-identical re-derivation
of every released table is therefore not available through the prediction route;
the claim is value-level. `experiments/r09_simulation.csv` shows what that means
at its cleanest -- 5,760,000 numeric cells compared, 0 different, only its row
order moved. Across the 27 tables that moved, the largest genuine residual is
`experiments/r13_coverage_ci.csv` cov_lo 0.0194 (a Wilson endpoint); `r03_summary`
coverage 0.0167, `r26_plugin_baseline_summary` n_violations 6, `r12_labelnoise`
9.0e-6 and `r16_*` 2.6e-4 are next. None of them moves a number quoted in
`docs/manuscript.md`.

**Byte stability.** A stage that writes a released table collects its worker
results in submission order. Twenty-four stages used
`multiprocessing.Pool.imap_unordered` and accumulated rows in worker-completion
order, so a large per-unit table came out ordered differently on every run. Every
stage now uses the ordered pool; the three that can be re-run cheaply (r20, r27,
r62) were re-run and two consecutive executions return identical bytes, and
`experiments/r26_constancy_bins.csv` is byte-identical when r26 is re-run in
isolation. The decile spreads are quoted to three decimals (0.2047 / 0.2374 /
0.2360) for the same reason: they do not reproduce in the fourth.

**The r20 crash.** `scripts/r20_label_ladder.py` raised
`ValueError: cannot convert float NaN to integer` on a clean run: on the
`within / frac / min_accept=5` grid no coordinate is present in every unit, so the
balanced range is empty, `kmax` is -1, `_at` returns NaN and `int(round(...))`
fails. The count fields now fall back to the same -1 their neighbours already
carried, and the stage exits 0. Re-running it also showed that the released
`experiments/r20_ladder.csv` was stale on its nine `min_accept=5, grid=frac`
rows, and that `experiments/r20_diag.csv` carried a column set
(`k_max_balanced_frac`) the shipped script does not emit; both are regenerated,
and the baseline snapshot that `--reconcile` restores was re-cut to match
(`work/baseline_recut/RECUT.txt`). No manuscript number moved: every
`min_accept=25` ladder row is unchanged, the per-unit table that Table 5's k-rows
are read from is set-identical, and Table 5's cross-tissue rows come from the
Table 3 run.

**One stage exits 0 on a released artifact that records a known failing check.**
`scripts/r34_external.py` re-derives `experiments/r34_checks.csv`, whose C15 row
has always been `pass=0, expect_pass=1` -- the accept/oracle ratio misses two
marginally over-budget `demo_transfer` units and raises no false alarm. The row
stays in the CSV and in the report; the stage's exit code counts only unexpected
failures.

The one place a released number depended on BLAS thread scheduling is the
model-adaptation arm of Table 7; `scripts/r43_threadpin.py` pins the thread pool
to a single thread and shows that two independent executions return the stored
0.3551 exactly, and the non-reproducibility caveat the draft used to carry has
been withdrawn. The STELLAR cohorts in `scripts/r34_external.py` are subsampled
once to 20,000 cells each with a fixed seed. `docs/submission/code_availability.md`
states the seed policy in full.
