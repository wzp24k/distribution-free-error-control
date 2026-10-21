# Methods

## Data

All experiments used CellTuneDepot (Keren Lab, Weizmann Institute of Science; Bussi et al., bioRxiv 2025; celltune.org/celltunedepot). It provides `ManualLabels/<cohort>`, fully manual annotations (`Tables/Labels.csv`: `image`, `cellID`, `class`) with per-FOV folders holding a `segmentation_labels` mask and one TIFF per channel, and `HighQualityLabels/<cohort>`, semi-supervised CellTune labels for whole datasets (`Tables/CellTable.csv`: per-cell mean intensities, `Area`, centroids, `class`), which exist only for CRC and GVHD.

Recomputed from the shipped tables (`scripts/r01_build_datasets.py`; `work/r01_summary.csv`, `work/r01_dataset_audit.txt`), the four cohorts - Breast_Jun0524, CRC_Feb2623, GVHD_Feb1323, Melanoma_Oct2522 - contain 7,610, 94,389, 175,548 and 21,204 segmented cells over 30, 35, 84 and 10 FOVs; 7,610, 2,505, 10,283 and 21,204 expert labels, Breast and Melanoma being exhaustive while CRC and GVHD cover 8 and 48 FOVs; 7,450, 2,397, 9,504 and 20,333 usable labels after excluding coarse `Unidentified`; panels of 22, 58, 40 and 40 markers; and 13, 12, 30 and 19 Boolean rules compiled with 0, 4, 1 and 0 failures. Totals are 41,602 expert-labelled and 39,684 usable cells, superseding the "31,319 manually annotated cells" of the study outline; we follow the depot. Semi-supervised labels agreed with expert labels at 0.709 exact and 0.725 coarse for CRC (n = 2,505) and 0.781 and 0.847 for GVHD (n = 10,283). `errctl.labels.COARSE_MAP` maps the four vocabularies onto a 16-entry coarse space.

Delivery integrity was audited by intersecting each labelled FOV's shipped `segmentation_labels` ID set with the `cellID` values in `Labels.csv`. Breast, CRC and Melanoma matched fully (7,610/7,610, 2,505/2,505, 21,204/21,204); GVHD matched at only 1,794/10,283 (0.175), its manual labels using a cell numbering other than the mask's. The version in `r10_figures.py` samples six FOVs per cohort and reports 0.067 for GVHD. No result depends on this, because CRC and GVHD features come from `CellTable.csv`, where labels join by `(image, cellID)` at 100% coverage (`work/s01_inventory.txt`); a user recomputing GVHD features from the shipped masks would not find most labelled cells.

The licence allows non-commercial academic and educational use at a degree-granting institution only, forbids redistribution and commercial use, and extends those terms to models and derivative datasets. No depot data are redistributed here; only a retrieval script and the download URL are provided.

## Preprocessing

`errctl.data.build_master` builds one table per cohort along two paths. CRC and GVHD take per-cell mean intensities, areas and centroids from `CellTable.csv`. Breast and Melanoma ship no such table, so `extract_crop_features` decodes each FOV's channel TIFFs and `segmentation_labels` mask and computes, per mask label, each channel's mean over that cell's pixels, the pixel count (`Area`) and the mask centroid, cached as parquet. The feature matrix is always the marker-intensity block alone; `Area` and centroids are carried but never used, and `data.feature_matrix` is called by no driver.

`errctl.preprocess.DomainNormaliser` applies `log1p` and then z-scores each marker, with the standard deviation floored at 1e-8. Statistics are stored under a `domain` key; the naming is looser than it looks, since `mode="z_dataset"` is the only mode used and behaves like any value other than `"none"`. In `within` and `cellsplit` the normaliser is fitted once on the training cells and applied under the same `"src"` key to training, calibration and deployment cells, so one source-fitted normaliser is used throughout. Only `transfer` fits two domains, `"src"` on source training cells and `"tgt"` on the target cohort: label-free, but it uses unlabelled deployment features and is transductive. The docstring claiming statistics come per domain from unlabelled cells is accurate for `transfer` and overstates the other families. Cross-cohort units intersect their panels on canonical channel names (`shared_panel`), and training sets are capped by class-proportional subsampling (`subsample_training`: 60,000 cells in `r02`/`r05`, 20,000 in `r08`/`r11`/`r12`).

## Marker logic

Each cohort's `Tables/CellTypes.csv` defines its populations by a Boolean expression over channels in `PrimaryMarker`. `errctl.markerlogic.parse` is a recursive-descent parser for `expr := term ("|" term)*`, `term := factor ("&" factor)*`, `factor := "!" factor | "(" expr ")" | IDENT`, with precedence `!`, then `&`, then `|`; unconsumed tokens or untokenisable characters raise `LogicError`. `CompiledRule` canonicalises every identifier inside the syntax tree through `canonical()`, folding panel-specific spellings such as `pan_keratin` and `hladr` onto a shared vocabulary; canonicalising the tree rather than only the recorded identifier set is what lets a rule written for one panel be evaluated on another. `compile_rules` reads `PrimaryMarker` only, skips `Unidentified`, and returns failures separately.

Four CRC definitions use `;` where the others use `|` or `&` (`CD3;CD4`, `Cytokeratin;NAKATPASE;MUC1`, `CD68;CD163;CD11b`, `SYP;CD56`), and they are exactly the four that fail to compile unmodified. The depot documents no meaning for that separator, but uses the same character elsewhere to separate alternatives: `MarkerTable.csv` carries `Expected_Expression` values such as `CD4T;Myeloid`, `NK;NKT` and `Proliferation;Activation`. OR is therefore the default reading (`build_groups` rewrites `;` to `|`), and both readings are reported, the sensitivity being confined to CRC, where reading `;` as AND raises the expert violation rate from 0.001 to 0.202. A code comment attributes the list-separator usage to `Channels.csv`, which the README describes but the shipped depot does not contain.

Positivity calls come from `markercheck.positivity`: a two-component Gaussian mixture on each channel's `log1p` intensities, thresholded at the crossing point of the two weighted components, with a fallback to the upper tertile for weak splits and an all-negative verdict for constant or all-zero channels. Non-protein channels and duplicate canonical names are dropped first, leaving 22/22, 56/58, 38/40 and 39/40 channels with a positivity call.

When a panel lacks a channel a rule needs, `CompiledRule.evaluate` raises `LogicError` from `Ident.evaluate`. `markercheck.evaluate_labels` then marks affected cells `missing_channel` and leaves them non-evaluable, with no fallback; the audit instead keeps, per coarse class, whichever published rules have all identifiers available on that panel and evaluates the OR over the survivors, so a class remains evaluable if one rule survives. That partial-rule fallback is the cross-panel mechanism, and it is why the audit reports `n_rules_used` (surviving rules for the class), `n_rules_total` (rules grouped for the class) and `rule_complete` (1 only when all survive). Headline rates therefore use only the `ALL_complete` subset, a partly checked class being able to report only the missing channel.

The sensitivity study (`scripts/r15_marker_sensitivity.py`,
`scripts/r18_marker_sources.py`) reports the violation rate under sixty-eight
definition-by-threshold conventions per cohort and, where the depot ships them,
for the semi-supervised labels as a second source. The convention axes are the
rule scope, the positivity definition, the reading of the depot's semicolon and
the shipped or repaired channel names, giving seventeen positivity settings
crossed with two readings and two rule texts. Label sources are the expert
panel, the depot's HighQualityLabels restricted to expert-labelled cells
(`hq_ref`), the same labels on every segmented cell (`hq_all`) and the same
labels on cells the expert never labelled (`hq_nonexpert`); the agreement
between the expert panel and `hq_ref` is reported as raw agreement and as
Cohen's kappa on coarse classes. The stability of the population ranking across
conventions is reported separately as the fraction of settings that preserve
the identity of the worst population and as the mean Spearman correlation of
the population-level rates, both on the full grid and on one-axis-at-a-time
sub-grids, because the ordering is not preserved and the instability needs to
be attributed to an axis rather than assumed. `ALL_complete` keeps classes all of whose rules
are evaluable; `ALL_strict` drops classes with a partly evaluable rule;
`ALL_evaluable` keeps every cell whose class retains at least one evaluable rule.
The positivity definitions are the mixture crossing point and the 60th to 95th
percentiles of all cells or of the positive population. Channel names are
additionally canonicalised a second time against the panel, which matters for
GVHD: two of its three epithelial rules reference `Morphology` and
`NAKATPASE_hi`, neither of which that panel carries, so under the shipped text a
single surviving sub-rule (`Mucin`) is used to judge 3,009 epithelial cells and
returns a 69.4% violation rate, whereas repairing the channel name to
`NAKATPASE` reduces that population to 0.33% and the cohort's fully evaluable
pooled rate from 23.97% to 2.09%. Every per-population rate carries a Wilson
interval and the pooled cohort figures a cluster bootstrap over cohorts.

The accepted-versus-deferred comparison is a paired statistic rather than a rate
(`scripts/r25_accept_deferred.py`). The unit is one accepted/deferred split per
distinct threshold configuration in the `repeat` column of `r03_metrics.csv`,
pooled exactly as in the deployed stage of `r15_marker_sensitivity.py`, which
reproduces all 48 rows of `r15_deployed.csv` exactly. A unit is estimable when
both sides carry at least 75 evaluable cells, mirroring the resolution floor used
in the text, with 30 and 200 reported as sensitivity. The statistic is the paired
sign test of accepted rate below deferred rate over the estimable units, with ties
dropped and counted separately, and the effect size is the median paired
difference. Because the same cells are re-audited under 68 conventions and three
budgets, the units are not independent and the p-value ranks the ordering rather
than certifying it; the Wilson intervals are computed on the pooled
configuration counts, which repeat each cell once per distinct threshold
configuration and are therefore narrower than a distinct-cell interval, with
`n_unique_cells` in the result table giving the distinct counts. A
class-standardised version, the cell-weighted mean of the
within-class differences over classes with at least 20 evaluable cells on both
sides, removes the cell-composition term.

`scripts/r28_marker_bounds.py` then treats the sixty-eight conventions as an
identification set rather than as a sensitivity analysis. For each cohort and
cell type the reported interval is the minimum and maximum violation rate over
the conventions, which is the range attainable under no assumption beyond "the
convention is one of the enumerated ones"; the exact Clopper-Pearson interval is
reported at each extreme and the conventions attaining them are named. A cell
type is robustly discordant when its rate exceeds the 5% gate under every
convention on which its rate is finite, and robustly concordant when it is at or
below the gate under every one; a group with fewer finite-rate conventions than
the grid is excluded from both claims and flagged. The threshold families -- the
mixture crossing point, the percentile-of-all-cells thresholds and the
percentile-of-positive-population thresholds -- are summarised separately,
because the verdict of 33 of the 48 expert cell types differs between the first
two. The between-convention spread is compared with the fixed-convention sampling
interval by the ratio of the two widths, and the channel-name repair is tested for
monotonicity across matched conventions.


## Marker-logic representation

`scripts/r49_markerlogic.py` rebuilds every experimental unit of the split-family
benchmark with each cell represented by the depot's Boolean definitions instead of
by channel intensities. The rule table is `CellTuneDepot/CellTypes_ALL.csv`, read
on `CellType`, `PrimaryMarker` and `AlternatePrimaryMarker`; rows whose marker text
is empty, `None` or `Various` are dropped, as are cell types whose name starts with
`Unidentified`, leaving 59 rules: 57 primaries plus two alternate forms
(`Plasma~alt` and `APC~alt`). For one cohort, `markercheck.positivity` is called
once over that cohort's own panel with the mixture threshold used everywhere else
in the paper, and every rule is evaluated on the resulting Boolean channels; the
call covers every cell of the cohort, held-out test images included, which is
label-free -- positivity reads marker intensities only -- but does make the rule
features transductive, unlike the per-fold channel transform of `published`. A
rule identifier is matched against the canonical form of the panel's channel
names, so a panel that spells a channel `HLADR` or `Fe56` counts as carrying
`HLADRDPDQ` or `Fe`, the convention the transfer panel intersection of
`scripts/r02_run_experiments.py` already uses. The
per-cell descriptor is two entries per rule: the first is 1 when every
canonicalised identifier of the rule has a positivity call on this panel, the
second is 1 when the rule additionally holds. A rule with an unavailable channel
contributes (0, 0), which is deliberately kept distinct from (1, 0), "evaluable and
false", because writing the two cases as one value would silently convert missing
channels into violations. Four feature spaces are run: `published` (the released
representation, used as a control), `rule` (the rule vector alone), `augraw` (the
shared canonicalised panel concatenated with the rule vector) and `augz` (the same
concatenation with the rule block standardised per domain). In `rule` and `augraw`
the published `log1p` plus per-domain z-score is applied to the channel block only,
`DomainNormaliser` being wrapped so that the trailing `2 x 59` columns carry no
per-domain standardisation -- they receive the same `log1p` as every other
column, which is monotone and identical in every panel. The point of the space is
that a source threshold on a rule flag means the same thing on the target. Folds,
`subsample_training`, the six calibration draws per unit, the certificate, the
baselines and every readout are imported from `scripts/r02_run_experiments.py`
and `scripts/r03_analyse.py` rather than re-implemented; the assembly of the
calibration pool is re-derived inline and was verified line by line against the
runner. The `published` space exercises exactly that imported path, so
that the four spaces are compared through one code path. `scripts/r49b_panelinvariance.py`
reports, per cohort and per rule, whether the rule is evaluable on that panel and
the fraction of the rule's own expert-labelled cells that satisfy it, each cohort
using its own positivity call. Outputs: `work/predictions_r49_<space>/`,
`experiments/r49_<space>_metrics.csv`, `experiments/r49_<space>_coverage_ci.csv`,
`work/r49_<space>_runs.csv`, `work/r49_report.txt`,
`experiments/r49b_panelinvariance.csv`, `experiments/r49b_pair_overlap.csv` and
`work/r49b_report.txt`.

## Selective risk control

The object of study is the selective risk of a threshold on a confidence score,

    R(lambda) = E[err . 1{s(X) <= lambda}] / E[1{s(X) <= lambda}],

where `s` is a nonconformity score (small values mean a confident label), `err = 1{predicted class != reference class}`, and the accepted set is `{s <= lambda}`. `A(lambda) = E[err . 1{s <= lambda}]` and `B(lambda) = E[1{s <= lambda}]` are each monotone in lambda, but their ratio is not, and it is not a mean of independent variables, so conformal risk control as usually stated, which needs a monotone loss bounded by 1, does not apply directly. The terms are bounded separately and uniformly over a grid, and the largest feasible threshold is taken:

    lambda_hat = max { lambda in grid : N_plus(lambda) <= alpha . D_minus(lambda) }.

`_grid_indices` builds candidates as acceptance counts over the sorted calibration scores, `k = unique(round(linspace(lo, n, min(grid_size, n - lo + 1))))` with `lo = max(min_accept, 2)`; `lambda_hat` is the score of the k-th smallest calibration cell, and acceptance is `s <= lambda_hat`.

Both bounds act on the unconditional probabilities, which are means of independent bounded variables. (i) The accepted-and-wrong count is `Binomial(n, A(lambda))`, so inverting the exact binomial test gives a Clopper-Pearson upper limit on `A(lambda)`, `_beta.ppf(1 - delta/(2m), n_wrong + 1, max(n - n_wrong, 1))`, with `m` the number of grid points. (ii) The accepted count is `Binomial(n, B(lambda))`, so the exact Clopper-Pearson lower limit `_beta.ppf(delta/(2m), k, n - k + 1)` bounds `B(lambda)`. (iii) Uniformity over the threshold grid is carried by the Bonferroni share `delta/(2m)`, applied to both families. (iv) The ratio is then recovered by a deterministic algebraic step, `upper_A / lower_B` (guarded at 1e-12, infinite when the denominator bound is zero), which involves no estimation and so cannot invalidate the bounds. If no candidate is feasible, `lambda_hat = -inf` and nothing is accepted.

Both bounds hold simultaneously at every grid point with probability at least `1 - delta`, so any data-dependent choice among them inherits them, and therefore `P(R(lambda_hat) <= alpha) >= 1 - delta`, with no assumption on the data distribution beyond exchangeability of the calibration and deployment cells, or correct importance weights. An empty accepted set has undefined risk and is a vacuous success, not a failure: `covered = (not isfinite(risk)) or risk <= alpha`, with an `empty` flag stored separately.

The variant `num="cond"` instead places the binomial bound on the error rate among accepted cells, treating `n_wrong` as `Binomial(k, p)` with the accepted count as the number of trials (Sharpe's construction). That is exact only if the error probability is constant across accepted cells; a confidence score is meant to make it vary with `s`, so the accepted set is a mixture, and `Binomial(k, p_bar)` does not stochastically dominate a mixture's error count, whose variance is larger. It therefore carries no finite-sample guarantee and is kept only as an ablation.

Two further threshold rules are implemented for `SelectiveRiskController`.
`select="bonferroni"` (the default) applies the share `delta/(2m)` at every grid
point as described. `select="fixed_sequence"` is step-down: candidates are
scanned in ascending `lambda`, the scan stops at the first failure, and the
certified set is the prefix up to the last success, with a `delta/2` share per
tail instead of `delta/(2m)`. Only the first true null met in a prespecified
order can be rejected incorrectly, so the family-wise error rate over the scan is
at most `delta` and the certified prefix inherits the bound. The rule assumes the
feasible set is a prefix of the candidate list. On this benchmark it is an
interval rather than a prefix, because a very small accepted count cannot support
the Clopper-Pearson lower bound on the denominator while a large one pushes the
numerator bound past `alpha . D_minus`, so the prefix rule certifies nothing
unless the scan may start above a prespecified floor on the accepted count. With
`min_accept = 100` and `grid_size = 120` it recovers about seven percentage
points of acceptance at unchanged nominal coverage (`scripts/r16_review_budget.py`,
`r16_fixedseq*.csv`).

`WeightedSelectiveRiskController` replaces the earlier weighted path. It bounds
the weighted numerator with a range-aware Hoeffding radius,
`sqrt(sum_i w_i^2 log(1/lev) / (2 n^2))`, which unlike a plain weighted Hoeffding
bound does not require the weights to be bounded by one; it takes the minimum of
that leg and a stratum-wise exact Clopper-Pearson leg, both valid, so the
combination stays valid; and it optionally accepts `target_scores=`, in which
case the denominator is an exact one-sided Clopper-Pearson limit computed from
the unlabelled deployment scores rather than from the weighted source. Its slack
must fall below `alpha . B(lambda) - A(lambda)`, which is small exactly where the
weights are hardest to estimate, so it has no advantage unless the density ratio
is estimable (`scripts/r19_weight_shift_diag.py`).

A second and more identifiable form of the same correction is run for comparison
(`scripts/r24b_labelshift.py`). Under a pure label shift the covariate ratio
collapses to one weight per class, `w(y) = q_tgt(y) / q_src(y)`, with `q_tgt`
estimated by the Saerens-Latinne-Decaestecker EM on the unlabelled deployment
score matrix and clipped at 20; the certificate then has the same form as the
weighted one, and an oracle-prior arm given the true deployment class frequencies
bounds what any correction built on the class labels alone could achieve. The
pool construction matches the `w_big` arm of `r14_targetcal.py` and the cell-level
density ratio is recomputed from the same code and seed as the diagnostic, which
agrees with it to floating-point and makes the two routes comparable on identical
units.

`ScoreConditionalController` assumes something weaker than exchangeability,
namely that `P(err | s)` is the same in the calibration and the deployment domain
while the marginal distribution of `s` may differ. It bins by pooled-score
quantiles, places an exact Clopper-Pearson upper limit on the per-bin error
probability at a Bonferroni share `delta/(2B)`, and integrates by parts over the
deployment score distribution using exact per-edge limits on its CDF, so the
assumption costs a single total-variation term rather than one margin per bin; a
minimum cumulative source count (`min_accept = 25`) is required before an edge is
certified. The assumption is testable, and the controller fails visibly when it
is false (see `shift_labelshift` below).


The remaining variants of `SelectiveRiskController`:

- `num="cp"` with `denom="cp"` is the default and produced every headline number; it is unavailable when `weighted=True`, since a Clopper-Pearson limit needs integer counts.
- `num="eb"` substitutes an empirical-Bernstein numerator (Bernstein radius under `bound="bernstein"`, the default, or a Hoeffding radius under `bound="hoeffding"`), reported as `crc_eb` and `crc_hoeffding`; `denom="eb"` does the same for the denominator.
- `weighted=True` reweights the calibration cells by an estimated target-to-source density ratio, restoring validity under covariate shift at the cost of effective sample size; `errctl.weights.density_ratio` fits a logistic domain classifier (`StandardScaler` plus `LogisticRegression`, three-fold `cross_val_predict`) on the concatenated normalised features, converts the class probability to `p/(1 - p)`, clips at 20 and normalises the mean to 1. Used only for `transfer` (`crc_weighted`), and valid only insofar as the weights are.
- `MondrianSelectiveRiskController` fits one controller per predicted class with floor `max(5, min_accept // 3)`, falling back to the global controller for sparse classes (`crc_mondrian`).
- `TemperatureScaler` fits one temperature by bounded scalar minimisation of the calibration negative log-likelihood using `log(p)` as logits, giving `temp_naive` and `crc_temp`.
- `conformal_set` is split-conformal prediction sets (score `1 - p(Y)`, finite-sample quantile `conformal_quantile`, cumulative-probability sets, singletons accepted only), which controls marginal coverage of the true label rather than selective risk.
- `naive` puts the threshold where the empirical calibration selective risk equals alpha, with no correction: the practitioner's default and the efficiency yardstick.
- `accept_all` abstains from nothing; `oracle` applies the same empirical rule to the deployment cells themselves, peeks at the labels being annotated, and bounds achievable acceptance from above.

Defaults: alpha was swept over 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40 and 0.50 in the full analysis, with 0.05/0.10/0.20 in the headline tables, 0.10/0.20 in the ablation, panel and ensemble studies and 0.05/0.10/0.20 in the label-noise study; `delta = 0.10`; `min_accept = 25`; `grid_size = 120`; `bound = "bernstein"`; and the score `s = 1 - max_k p_k`, with entropy and the top-two margin swept only in the score ablation. One comment is stale: `scripts/r03_analyse.py` still calls `crc` an empirical-Bernstein numerator with a Clopper-Pearson denominator, whereas the default is `num="cp"`.

## Split families

- `cellsplit` (`scripts/r05_cellsplit.py`) is the exchangeable control: FOVs are shuffled into five folds, one is held out while the model trains on the other four, and that fold's cells are split at random half to calibration and half to deployment in ten independent draws. Training never sees the held-out FOVs, but calibration and deployment cells come from the same FOVs, so exchangeability holds and the guarantee must be met. This family isolates the theory from the data's structure and measures the certificate's cost against the naive threshold.
- `within` (`scripts/r02_run_experiments.py`) holds the cohort fixed and breaks exchangeability: for fold `f` the deployment set is fold `f`, the calibration pool is fold `f+1`, and the remaining three folds train the model, so no FOV contributes cells to two roles. Panel, staining and case mix are constant, while FOV-to-FOV variation - batch effects, tissue composition, section-to-section drift - is not, so a failure is attributable to the loss of exchangeability rather than to the estimator.
- `crossfov_loo` (`scripts/r48_fov_units.py`) re-estimates the `within` family at field-of-view resolution, changing exactly one axis and leaving the rest alone: one annotated image is held out per unit, so coverage is estimated over all 96 annotated images of the four cohorts rather than over the twenty five-fold units. The model never trains on the held-out image nor on the images that supply its calibration cells, and the calibration subsample, the six calibration draws per unit, the certificate, the baselines and the readouts are the published ones, imported from `scripts/r02_run_experiments.py` and `scripts/r03_analyse.py`, so only the unit changes. `scripts/r48b_fov_repair.py` then re-runs the part A arms of `scripts/r31_targeted_repair.py` on those units by pointing its prediction directory at `work/predictions_fov/` and patching only the unit-name parser, so the arm implementations, the confidence level and the minimum-acceptance floor cannot drift from `experiments/r31_within_summary.csv`; the `ladder` arms are not run, because a label-count ladder has no meaning for a one-image-held-out unit. Its `plain` arm reproduces the field-of-view coverage bit for bit (0.843750, 0.792245 and 0.856481 at `alpha` 0.05, 0.10 and 0.20), the joint readout likewise (0.716435, 0.721644 and 0.824653) and acceptance to all printed digits (0.528162, 0.725899 and 0.918875); the conditional readout agrees to within 0.01, the two renderings weighting units and evaluations respectively. Outputs: `work/predictions_fov/` (288 prediction files), `experiments/r48_metrics.csv`, `r48_ece.csv`, `r48_coverage_ci.csv`, `r48b_fov_repair_detail.csv`, `r48b_fov_repair_summary.csv` (coverage and joint coverage each with a unit-bootstrap interval), `r48b_fov_repair_paired.csv` (pooled over the three budgets) and `r48b_fov_repair_paired_by_alpha.csv` (the same differences per budget), `work/r48_runs.csv`, `work/r48_report.txt` and `work/r48b_report.txt`.
- `transfer` (`scripts/r02_run_experiments.py`) trains and calibrates on a source cohort and deploys on a different one, restricted to the canonical panel intersection (11 to 27 channels over the twelve ordered pairs, the per-pair counts being shipped in `experiments/r49b_pair_overlap.csv`) and to coarse classes with at least 30 source and 20 target cells; all twelve ordered pairs of the four cohorts were run, each over five folds. Tissue, panel, staining and label vocabulary all change, shifting both the covariate and the reference-label distribution; this is the setting the study is about.

In `within` and `transfer` the class inventory comes from the training labels (`run_unit` keeps classes with at least three training cells), and a deployment cell whose label is absent is encoded as -1 and counted as an error; that bound in 675 of 119,052 `within` deployments (0.57%) and none of the 1,710,990 `transfer` deployments.


## Transductive calibration and the label-cost ladder

The `within` and `transfer` families are rerun in a transductive mode
(`scripts/r14_targetcal.py`, `scripts/r20_label_ladder.py`). For each unit, `k`
cells are drawn at random from the deployment pool, the controller is calibrated
on those `k` cells alone, and it is evaluated on the remaining deployment cells,
so calibration and deployment are exchangeable by construction at any `k`. The
`k = 0` arm reproduces the source-only certificate exactly, which is the
self-consistency check reported alongside the ladder. Ten independent draws are
used for the cross-tissue family and twenty for the three-family ladder; a unit
enters the ladder only while its deployment pool holds at least twice the
largest `k` evaluated. The ladder statistic quoted in the text is the smallest
`k` at which the controller is both valid (non-vacuous coverage at least
`1 - delta`) and useful (mean acceptance at least 0.05); a family that never
reaches both is reported as not restored at the grid's resolution. Because a
tight but valid certificate on a small evaluation remainder sits exactly at the
budget and so trips a per-draw 0/1 coverage flag about half the time, the
ladder is scored on the one-sided 95% Clopper-Pearson upper limit of the
realised risk among accepted cells, computed per unit and classifier, taken
worst-classifier-first and then averaged over units. Two further rules are
fixed. A family average is only quoted over the balanced range on which every
unit of the family has rows, so that a large cohort is never compared with a
small one at the same absolute `k`. And per pair, a budget is called *reached*
when at least 5 non-vacuous decisions out of 150 cover at least 90% of
themselves, `never` otherwise. The cross-tissue family is run a second time on
a denser grid, `k` = 0, 25, 50, 100, 200, 250, 500, 750, 1,000, 1,500 and
2,000, with the same folds, seeds and ten draws (`scripts/r14c_ladder_dense.py`,
`experiments/r14_targetcal_ext.csv`); the coarse sub-grid reproduces Table 6
cell for cell and the finer rungs move four of its twenty-four cells.

The efficiency accounting is expressed at a matched realised risk
(`scripts/r24_efficiency.py`). An arm's operating point at budget alpha is its
pooled realised risk, `sum(n_wrong) / sum(n_accepted)` over all units and
evaluations, against its mean acceptance; its curve is the upper monotone frontier
of those points, from which acceptance at a common risk is read by linear
interpolation and never extrapolated, and only points inside the balanced range
enter a claim. The same script reports the certificate yield per target label
spent, `mean auto-annotated cells per evaluation / k`, its marginal return
`100 * d(annotated cells per evaluation) / dk` over consecutive balanced points,
the fraction of evaluations whose realised risk is at or below the nominal budget,
and a unit-bootstrap interval on the certificate-to-oracle acceptance ratio that
resamples folds and cohort pairs together.

The two label-free statistics used to triage a pair before any labels are spent
are the acceptance of the source-only certificate at `k = 0` and the two-sample
Kolmogorov-Smirnov distance between the source calibration scores and the
deployment scores. Both are computed without target labels. The strongest
single-marker source-to-target separation is also recorded but is not used as a
predictor, because its pooled association with the label cost is carried by the
split family rather than by the shift and its sign is not stable within
families.

The same code path also runs a model-adaptation variant (`--adapt`), in which the
`k` drawn cells are split into two halves: one half is appended to the source
training set and the classifier is refitted, and the other half calibrates the
refitted model on deployment scores. Both halves are disjoint from the evaluation
cells, so the certificate stays valid, and the arm separates the two possible
causes of failure under transfer, namely a certificate that cannot be estimated
from source data and a model whose ranking has degraded.

The panel question is also measured on the channels two panels actually share
rather than by deleting channels from one cohort (`scripts/r61_shared_panel.py`,
`experiments/r61_shared_panel.csv` for colorectal/GVHD and
`experiments/r61_shared_panel_mg.csv` for melanoma/GVHD). For a pair of cohorts
the model is fitted, calibrated and deployed on the canonical markers both
panels carry and on nothing else, so which channels survive is fixed by the two
antibody panels rather than by any data-driven ranking; colorectal and GVHD
share 27 of them and melanoma and GVHD share 24, and the coarse classes kept are
those with at least 20 cells in both. Four arms separate the terms.
`within_<cohort>_full` and `within_<cohort>_shared` hold the cohort, the folds
and the protocol fixed, so only the panel changes. `cross_<src>_<tgt>` trains on
the source's shared channels and then splits the target's held-out images at
random into a calibration half and a deployment half, which is exchangeable
inside the target, so the only violation left is the change of cohort at a fixed
panel; `crossk_` repeats it with a fixed number of target labels. `srcnull_` is
the source-calibrated baseline of the transfer family, the arm the guarantee
does not cover. Every arm runs on the same five image folds, with five random
calibration splits per fold and the three classifiers, and validity is read both
ways, as elsewhere in the paper: the per-evaluation covered flag and the
one-sided 95% Clopper-Pearson upper limit of the realised risk among the
accepted cells.

The annotation-source family is run by `scripts/r62_source_family.py` (`experiments/r62_source_family.csv`, its summary and
`experiments/r62_fold_transfer.csv`, which carries the leave-one-fold-out
transfer accuracy of every fold and classifier and is what the manuscript
reads the colorectal cohort's thinness from). Only the two cohorts that ship a second source are used -- colorectal and GVHD, whose `ManualLabels` and `HighQualityLabels` tables label the same `(image, cellID)` pairs -- and the model is trained once per fold on the manual labels and then held fixed, so the accepted set never depends on which source the deployment will be scored against. The image folds are balanced by pool cells rather than by image count, because the certificate's feasibility is close to a fixed number of calibration cells and an equal split of image names had left one colorectal fold too small to use; twenty random calibration/deployment splits are drawn per fold. Four arms share the cells, the five image folds and the twenty splits: `same_expert` and `same_hq` calibrate on one source and are read against both, `consmax` fits one controller to the pointwise maximum of the two per-cell errors, and `consint` fits two controllers at half the failure budget each and accepts the intersection of their thresholds. Both repairs are valid without knowing the deployment source -- the first because the pointwise maximum bounds either reading and the second by a union bound -- and both are read against each source separately. Every row also carries the disagreement rate among its accepted cells, and the report checks the identity that the two sources' realised risks differ by at most that rate. A fold whose held-out pool is under 200 cells is dropped; with the balanced folds neither cohort loses one, and each keeps five.

## Classifiers

Three off-the-shelf classifiers were used, deliberately untuned for accuracy because the contribution is the calibration layer above them (`errctl.classifiers`): `LogisticRegression(max_iter=5000, C=1.0)`; `RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1)`; and `HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)`. The regularisation is necessary because the library defaults under-fit badly at this sample size, and the configuration was selected on training data only, never against calibration or deployment labels. `fit_classifier` disables early stopping and caps the model at 200 iterations when the least populated class holds a single member, which happens inside bootstrap resamples of rare populations. A `logreg_l2` variant (`C=0.1`) is defined but unused, and classes with fewer than three training cells are dropped before fitting.

The ensemble study (`scripts/r11_ensemble.py`) bags the two cheaper learners (`logreg`, `hgb`) five times per unit: bag 0 is the full training set, bags 1 to 4 are bootstrap resamples with replacement, each refit after restricting to the classes it contains, with absent classes filled as zero and the bag matrix renormalised. Three scores are formed - `ens_maxprob` (one minus the maximum of the mean distribution), `ens_entropy` (normalised entropy of the mean distribution) and `ens_mi` (entropy of the mean distribution minus the mean of the per-bag entropies, normalised by `log K`, the BALD score) - and each goes through the same controller on the exchangeable half/half split.

A fourth model is used in one place only and is not one of these tabular
scorers: an image-based network, described under `## An external, image-based
classifier`.

## Evaluation metrics

- Selective risk: the fraction of accepted cells whose predicted class differs from the reference label, undefined when nothing is accepted.
- Acceptance rate: the accepted fraction of deployment cells.
- Coverage: the fraction of evaluation units (unit x classifier x calibration repeat) whose realised selective risk is at most alpha. An empty accepted set has undefined risk and counts as covered; `empty` is stored so that vacuous success can be separated from a genuinely certified set, and the label-noise study also reports `cov_nonvac`, the probability of a non-empty accepted set that is within budget.
- Risk deferred to review: the error rate among non-accepted cells, recovered from the all-cells risk decomposition as `(risk_accept_all - accept_rate . risk_crc) / (1 - accept_rate)`.
- Fraction-of-oracle acceptance: the controller's acceptance divided by the oracle's, whose threshold is chosen with deployment labels in hand and therefore bounds rather than competes.
- ECE: `expected_calibration_error(conf, correct, n_bins=15, scheme="quantile")` bins the confidence (maximum predicted probability) by its own empirical quantiles into at most 15 bins with outer edges at infinity, and returns the bin-count-weighted mean absolute difference between mean confidence and empirical accuracy within bins, computed separately on the calibration (source) and deployment predictions of every unit.
- NLL and Brier: the reference-class negative log-likelihood (`calibration.nll`) is the temperature-scaling objective; the Brier score (`calibration.brier`) is the mean over cells of the squared difference between the predicted probability vector and the one-hot reference.
- AURC: `conformal.aurc` is the trapezoidal area under the risk-versus-coverage curve divided by the mean error rate, lower being better.
- Marker-rule violation rate: the fraction of cells, among those whose coarse label has a fully evaluable rule on that cohort's panel, for which the published Boolean definition of their own label is false under the mixture-based positivity calls. It is reported for expert labels, for semi-supervised labels where they exist, and for each classifier's predictions, separately for correct and incorrect predictions and for accepted versus deferred cells.

Neither Brier nor AURC appears in the reported tables, which carry only ECE and
accuracy. Intervals are attached in three places and only three: the
field-of-view cluster bootstrap of `eval.cluster_bootstrap_ci` for the per-cohort
pooled coverage, the unit bootstrap of `scripts/r13_coverage_ci.py` for the
coverage column of the headline table, and the fold-and-cohort-pair bootstrap of
`scripts/r24_efficiency.py` for the certificate-to-oracle acceptance ratio. The
cell-level `eval.bootstrap_ci` is deliberately left unused, because cells within a
field of view are not independent; everywhere else dispersion appears only as
spread across resampling units.


## Review budget

The review-budget analysis (`scripts/r16_review_budget.py`) converts the
certificate into the quantity a lab plans. At a budget `alpha` the fraction of
cells routed to review is one minus acceptance, and the realised error among
retained labels is the selective risk of the accepted set. The pooled figure sums
accepted and reviewed cells over the random-split family's held-out cells (19,669
cells across 19 units). Two comparisons are reported. Against reviewing
everything, the saving is the fraction of review work removed. Against the
oracle, acceptance is matched on realised risk rather than on the nominal budget,
and the ratio of the two is the price of validity; a ratio above one is
impossible for a valid certificate and is therefore a failure signature rather
than an efficiency claim. The threshold arm is cross-checked against
`scripts/r03_analyse.py` evaluation by evaluation (maximum absolute difference in
acceptance 1.1e-16).

`scripts/r50_review_loop.py` puts the review budget and the label price of the
repair in the same units, reading only released tables: the certificate's
acceptance, review fraction and evaluation-pool size by family and budget from
`experiments/r16_review_budget.csv`, and the target-label ladder from
`experiments/r24_efficiency.csv`. The repair price it reports is the smallest
label count at which the ladder restores coverage over all evaluations to at
least 0.90 *without* losing the source-only acceptance -- the count at which the
repair is free in throughput rather than merely valid -- and where no count in the
ladder meets that it reports the family as unrepaired by calibration together with
the best acceptance the ladder reaches. The deployment pool per unit is the
certificate run's evaluation cells divided by its evaluation count, which is
19,669 / 19 for the random-split family exactly. Unit counts are carried through,
because the ladder drops units whose pool cannot supply `k` labels. Outputs:
`experiments/r50_review_loop.csv`, `work/r50_report.txt`.

## Baseline selection rules and the fixed-sequence step-down

`scripts/r26_plugin_baseline.py` evaluates six acceptance rules on the identical
score, candidate grid and split as the certificate: the certificate itself, an
uncalibrated threshold placed so that the calibration selective risk equals the
budget, an empirical oracle threshold placed on the held-out selective risk, and
three conservative plug-ins that bound the conditional error rate among accepted
cells rather than the two unconditional probabilities. The plug-ins use an exact
Clopper-Pearson limit, a Hoeffding margin and a Wilson margin respectively, each
at confidence 1 - delta/(2m) over the m candidate thresholds, so the comparison
isolates the confidence margin and not the grid. Paired differences against the
certificate are computed on the same evaluations with a unit-cluster percentile
bootstrap over 2,000 resamples, and the constancy assumption the conditional
bounds rest on is tested by a Pearson chi-square on the error rate across
predicted classes and across score deciles within the accepted calibration set.
The `certificate` arm reproduces all 2,790 evaluations of the `bonferroni_ma25_gs120`
configuration of `experiments/r16_fixedseq.csv` exactly, and the Clopper-Pearson
plug-in reproduces `SelectiveRiskController(num="cond")` on all 18,090
evaluations.
The counts behind the constancy test -- accepted cells and wrong cells by
predicted class and by score decile, for every evaluation that entered it -- are
released in `experiments/r26_constancy_bins.csv`, so both the spread and the
chi-square recompute from a released file rather than from the report; groups
with fewer than ten accepted cells are carried there and excluded from the two
statistics.

`scripts/r27_fixedseq.py` evaluates the step-down in both scan orders on the same
grid. The ascending arm uses the controller's `select="fixed_sequence"` path,
which charges no multiplicity share for the grid; the descending arm is
reimplemented locally so that the only difference between the two arms is the
order in which the hypotheses are tested, and it is reported as a negative
control. The geometry of the feasible set on the grid is reported per evaluation
as the number of feasible candidates below and above the chosen one, so that
whether the feasible set is a prefix of the scan order is visible, and the
coverage of the budget is reported against the nominal 1 - delta for every arm
with a Wilson interval.

## Monte-Carlo validation

`scripts/r09_simulation.py` validates the guarantee on a known data-generating process at `delta = 0.10`, over alpha in 0.02, 0.05, 0.10 and 0.20, with 4,000 repetitions per setting, method and alpha and 1,000 deployment cells per repetition. Each cell draws a score `s` in [0, 1] and an independent covariate `u` in [0, 1] that the score never observes, and its top-1 label is wrong with probability `p = 0.02 + 0.40 . s + het . (u - 0.5)`, clipped to [0, 0.99]. Calibration scores are always Uniform(0, 1); deployment scores are Uniform(0, 1) in the exchangeable settings and Beta(2, 1) or Beta(3, 1), densities `2s` and `3s^2`, under shift. `het = 0` makes the error probability a function of the score alone, while `het > 0` leaves it varying among cells that share a score, so the accepted set is a mixture - the case the conditional binomial route cannot represent. Seeds are per repetition, `default_rng(7919 . rep + 13 . len(setting) + n_cal)`.

The eight settings:

- `exchangeable_200`, `exchangeable_500`, `exchangeable_2000`: same law on both sides, calibration size 200, 500 and 2,000 cells - the guarantee's cost in labels.
- `hetero_mild` (500 cells, `het = 0.4`) and `hetero_strong` (500, `het = 0.8`): exchangeability preserved, only error-rate heterogeneity grows.
- `hetero_coarse` (500, `het = 0.8`): scores additionally rounded to five levels so many cells share a score exactly - ties.
- `shift_mild` (1,000 cells, Uniform to Beta(2, 1)) and `shift_strong` (1,000, Uniform to Beta(3, 1)): covariate shift, where the unweighted controller has no guarantee and the weighted one does.

For the shift settings both the analytic density ratio (`2s`, `3s^2`) and one estimated by a logistic domain classifier on the score are supplied, giving `crc_weighted_known` and `crc_weighted_est`. Every setting is also evaluated with `accept_all`, `naive`, `crc`, `crc_cond`, `crc_eb`, `crc_hoeffding` and `oracle`, under the same coverage convention as the main analysis.


## Second Monte-Carlo validation

`scripts/r17_simulation.py` re-validates the guarantee against the rebuilt
controllers on the same data-generating process, at `delta = 0.10`, alpha in
0.05, 0.10 and 0.20, 800 repetitions per setting and 4,000 deployment cells per
repetition. Seven settings are used: `exchangeable_200` and `exchangeable_2000`;
`hetero_strong` (500 cells, `het = 0.8`); `shift_mild_1000` and `shift_mild_4000`
(Uniform to Beta(2,1)); `shift_strong_1000` (Uniform to Beta(3,1)); and
`shift_labelshift`, which breaks the score-conditional assumption by construction
by giving the deployment error probability `0.02 + 0.4 s + 0.12` while the
calibration domain keeps `0.02 + 0.4 s`. Eleven methods are compared, including
the unconditional controller under the Bonferroni and the fixed-sequence rules,
the conditional-binomial rival, the old and rebuilt weighted controllers (known
and estimated density ratio, with and without the unlabelled-target denominator),
the score-conditional controller, the uncalibrated threshold, accept-all and the
oracle. `shift_labelshift` is the designed-failure control: a method that assumes
score-conditional invariance must break there, and the unconditional controller
must not.

The real-data counterpart is `scripts/r19_weight_shift_diag.py`, which reports,
for each of the twelve cross-tissue pairs and without using any target label, the
effective sample size of the estimated density ratio, the separability of the two
domains under a two-sample classifier on the normalised features, and the
precision that the weighted bound would need in order to be non-vacuous; it then
reports the same pairs' score-conditional coverage. This is what turns the
weighted route's failure from an observation into a criterion that can be
evaluated before any labels are spent.

## Proposition 1: when the conditional-binomial plug-in is valid

The conditional-binomial plug-in is a valid upper confidence limit for the
selective risk only if the conditional error rate is constant over the accepted
cells. `docs\propositions.md` states the licence as Assumption (C): one number
`pi` with `P(E = 1 | S = s) = pi` for almost every accepted `s`, equivalently
independent Bernoulli(`pi`) errors among the accepted cells with a common `pi`.
Under (C) Sharpe's inversion of the accepted count is exact, and the same
binomial tail survives under independent errors whose probabilities are bounded
by a known `pi`, or a bounded-difference martingale whose predictable variance is
at most `k pi (1 - pi)` for a known `pi`. What does not substitute for (C) is a
variance or rate proxy estimated from the accepted cells themselves, because
that is exactly the quantity heterogeneity moves. The corrected mechanism is
that validity is a property of the constancy of the conditional error rate over
the accepted set, and the condition the certificate of `SelectiveRiskController`
needs, and the plug-in needs as well, is the exchangeability of the deployment
and calibration populations.

The proved failure channel is an atom in the score. The delivered threshold is
the score of the `k_j`-th sorted calibration cell, but the accepted set is
`{s <= lambda_hat}`, which contains the whole tied block at that score; the trial
count `k_j` and the counts that fix the bound then refer to a strict sub-block of
the set that is actually accepted, and the omitted part of the block is admitted
without appearing in the bound. On the explicit four-atom law with
`P(s = 0) = P(s = 1/3) = P(s = 2/3) = P(s = 1) = 0.25` and
`P(E = 1 | s) = 0.005, 0.05, 0.45, 0.45` the plug-in covers 0.5693 against a
nominal 0.90 at `alpha = delta = 0.10`, `n = 400`, `min_accept = 25`,
`grid_size = 20` over 4,000 repetitions under one seed, with mean delivered
acceptance mass 0.6055 and mean delivered population risk 0.0884; the certificate
covers 0.8050 on the same repetitions. The test-set acceptance mass exceeds the
chosen rank divided by `n` by +0.1183 for the plug-in and +0.0852 for the
certificate, and recomputing the two rules with the trial count taken from the
whole tied block at the delivered threshold, so that rank and accepted set
coincide, restores coverage 1.0000 for both.

A score-varying conditional error rate alone did not break the plug-in in any
of the 60 `(k, pi, delta_m)` cells or four continuous-score controls examined
here. A violation needs `P(sum <= x(pi_bar) - 1) > delta_m` at the
crossing point of the Clopper-Pearson inversion, and sweeping 60
`(k, pi, delta_m)` cells and 3,710 vectors that share the homogeneous mean
exactly, the worst value of `P(sum <= x - 1)/delta_m` is 0.970, attained by the
homogeneous vector itself. On the four continuous-score controls `flat`,
`step_cont`, `ramp3` and `ramp1`, whose error profiles vary strongly, the plug-in
covers 1.0000, 0.9990, 1.0000 and 0.9980 respectively against the same nominal
0.90. The exact arithmetic that
heterogeneity moves mass towards low counts is real (for four accepted cells with
`pi = (0.5, 0.5, 0, 0)`, `P(sum <= 1) = 0.7500 > 0.7383 = P(Binomial(4, 0.25) <= 1)`),
but it does not by itself produce under-coverage, and the earlier phrasing that
the plug-in is invalid once errors vary among accepted cells is an overstatement.
The stored benchmark rejects constancy in 0.7751 of evaluations by predicted
class and 0.9287 by score decile on `cellsplit`, with mean spreads of 0.2016 and
0.2048 against an accepted risk of 0.0559, while the plug-in still covers there.

Where the stored simulations do show under-coverage of the plug-in, covariate
shift causes it rather than heterogeneity. The plug-in covers 0.6068 and 0.3740
at a 20% budget on `shift_mild` and `shift_strong` in
`scripts/r09_simulation.py`, and 0.6388 and 0.3387 in the corresponding settings
of `scripts/r17_simulation.py`, against 0.9785 to 1.0000 in every non-shift
setting of `r09` and 0.9962 to 1.0000 in `r17`. The 0.607 and 0.374 figures are
the rounding of those two covariate-shift coverages. The operative condition is
the exchangeability of the deployment and calibration populations, or correct
importance weights: the certificate holds under it with no assumption at all on
the conditional error rate, the plug-in needs it in addition to constancy, and
the four-atom law is a possibility result about the reading rather than a
demonstration that the stored heterogeneity settings should have failed.

## Proposition 2: prefix feasibility and gain

The two multiplicity conventions of the same controller charge different shares.
The Bonferroni arm sets `t = delta/(2m)` at every one of the `m` candidate grid
points; the step-down arm sets `t = delta/2` and scans the candidates upward,
stopping at the first infeasible one. Because `delta/(2m) <= delta/2` and both
Clopper-Pearson legs move in the safe direction when the tail is raised,
`ratio_j(delta/2) <= ratio_j(delta/(2m))` for every `j`, so the bound at each
candidate is pointwise no larger under the step-down share and the Bonferroni
feasible set is contained in the step-down feasible set. That inequality is
verified with 0 violations over the 297 prediction files of
`experiments/r27_fixedseq.csv` on the 120-point candidate grid at each of
`delta` in {0.05, 0.10, 0.20}, 106,920 comparisons in all
(`work/r30_report.txt`). In the empirical-Bernstein legs the same
two shares appear as `log(3m/delta)` and `log(3/delta)`, so the grid enters the
Bonferroni margin through `log m` and does not enter the step-down margin at all.

The consequence is a gain in delivered acceptance whenever the Bonferroni
feasible set is a strict prefix of the ascending grid, that is `F_B = {1, ..., h}`
for some `h`: every `j <= h` is then step-down feasible, the scan's first failure
lies above `h`, and the step-down delivers at least as many accepted cells in
every test set. Over the 16,056 pooled evaluations at `delta = 0.10` whose
Bonferroni feasible set is a strict prefix, the step-down count is below the
Bonferroni count in a fraction 0.0000 of them with a worst deficit of 0 cells,
and the mean gain is +0.0359. The gain grows with the grid because only
Bonferroni carries the `log m` penalty: at `alpha = delta = 0.10` and
`min_accept = 100` the gap between step-down and Bonferroni acceptance on
`cellsplit` is +0.0503, +0.0735 and +0.0896 at `m = 20, 120, 400`, while
step-down acceptance is almost flat at 0.6185, 0.6335 and 0.6350 and Bonferroni
falls from 0.5683 to 0.5454. The same refinement costs coverage: step-down
coverage on `cellsplit` is 0.9632 at a 10% budget and 0.9842 at a 20% budget
against the certificate's 1.0000.

The scope limit belongs in the statement. Proposition 2 is proved for the strict
Bonferroni-prefix hypothesis, not for the step-down arm's own prefix condition,
which is the condition the `r27` column reports and the condition the
corresponding prose names; there the clean statement fails pointwise, because
over the 38,754 pooled evaluations at `delta = 0.10` whose step-down feasible set
is a prefix the mean gain is +0.0644 while 0.006 of evaluations worsen, with a
count-deficit fraction of 0.0059 and a worst deficit of 19,506 cells. The
empirically supported form is the strict Bonferroni-prefix one, where there is no
deficit anywhere; on the larger but weaker condition the gain is positive on
average, +0.0964 with unit-cluster interval [+0.0780, +0.1188] over 96 units, but
not pointwise. Outside the condition the comparison reverses and is large: on the
33,606 evaluations whose step-down feasible set is not a prefix the mean gain is
-0.2142, no evaluation improves and 0.403 worsen.

The characterisation is about prefix feasibility rather than about the ratio. For
the step-down arm the ascending scan delivers the top of its own feasible set
exactly when that set is a prefix, so on that arm `j_F = j^+` is equivalent to
the prefix condition, and a non-negative paired gain is what that condition buys;
the pointwise dominance of the ratios holds either way, and the reversal outside
it is a hole in the feasible set below a higher feasible point, which stops the
scan early. The negative control isolates the same conclusion: reversing the
prespecified order of the identical candidates with the identical ratios removes
the gain in all 81 configurations at every floor, with mean gains of -0.4958,
-0.4940, -0.4876 and -0.4699 at floors 25, 50, 100 and 200, so the gain is
carried by the direction of the scan rather than by the ratio.

## Anytime-valid sequential control

The certificate of `SelectiveRiskController` is a fixed-budget object. It bounds
the two unconditional Bernoulli means `A(lambda) = P(err and s <= lambda)` and
`B(lambda) = P(s <= lambda)`, whose ratio is the selective risk `R(lambda)`, at
one prespecified calibration size `n`, and it cannot be stopped early.
`scripts/r32_sequential.py` adds the missing sequential arm on the same score
`s = 1 - max_c proba` and the same grid: cells of the target domain arrive one at
a time in a uniform random order and the decision is taken at the first look that
certifies. The instrument is an e-process on each of the two binary streams. With
a `Beta(1,1)` mixing prior and `X_i(lambda)` the indicator that cell `i` is
accepted and wrong, with the analogous accepted-only stream for the denominator,
`M_t(lambda, mu_0)` is a nonnegative mean-one martingale under every `mu_0`,
checked by summing against the exact binomial law to a worst `|E M_t - 1|` of
3.1e-12. Inverting `M_t` at level `1/delta'` gives one-sided limits `U_t(lambda)`
and `L_t(lambda)` by Ville's inequality, so a published threshold that meets
`U_tau <= alpha L_tau` has `R(tau) <= alpha`; charging `delta' = delta/(2m)` to
each of the `2m` one-sided bounds, exactly the share the fixed-budget certificate
charges, the union over the grid and over the two sides is an event of
probability at least `1 - delta` that is simultaneous in `t`. With probability at
least `1 - delta`, therefore, at every stopping time and for every grid
threshold the realised selective risk of the published threshold is at most
`alpha`.

Four sequential arms are reported. `seq_eproc_bonf` inverts the e-process with the
`delta/(2m)` share at every candidate; `seq_eproc_fs` drops the grid share and
scans upward, stopping at the first point that cannot be certified, which charges
`delta` in total and needs the anytime-validity of the tests rather than a
per-grid share; `seq_spend_cp` spends the level along the look schedule with
Clopper-Pearson limits; and `seq_cs_hoeffding` uses a Hoeffding confidence
sequence. They are compared with three fixed-budget rivals, `cert_fixed_full` and
`stepdown_full`, and the honest sequential competitor `cert_fixed_union`, which
is a union bound over the same prespecified look schedule, remains valid under
optional stopping and pays `log K` on top of the grid's `log m`. Three controls
are carried: `cert_fixed_peek` and `naive_peek_tight` reuse a fixed-sample bound
at every look without a time correction and so buy labels with validity, and
`naive_plug` is the no-bound control.

The engineering claim is label cost, and it holds against the fixed-budget
certificate used at `t = n`. On `cellsplit` at `alpha = delta = 0.10` the
e-process arm needs 0.9114 of the labelled pool to reach a realised deployment
acceptance of 0.75, against 1.0000 by construction for the fixed-budget object, a
paired difference of -0.0886 with unit-cluster 95% interval
[-0.1589, -0.0276]; at `alpha = 0.20` the same comparison is 0.5174 against
1.0000, -0.4826 [-0.6146, -0.3292]. The fixed-budget references cannot publish
anything before `t = n`, so their label cost of 1.000 is the honest reading of a
fixed-budget object and not a missing number.

The loss belongs beside the win. Against the honest rival `cert_fixed_union`,
which is itself valid under optional stopping, the e-process does not win: at
`cellsplit` and `alpha = 0.10` its `s3_t75_frac` is +0.0468 [0.0229, 0.0715],
positive meaning it needs a larger fraction of the pool than the union bound
does, and the mean over the nine (family, alpha) cells at `delta = 0.10` is
+0.0259. The predicted log-log advantage of a mixture confidence sequence is not
observed at these pool sizes, and every sequential arm loses acceptance at the
full budget: `seq_eproc_bonf` has a mean `d_accept_final` of -0.0994 over the 27
(family, alpha, delta) cells, losing with an interval excluding zero in 27 of 27
and winning in none, because a sequential test must hedge against every look it
did not use. The negative control behaves as it should: the peeking control that
reuses the same exact fixed-sample bound at every look, the arm the label-cost
tables call `naive_peek_tight` (named `control_peek_cp` in the Monte-Carlo
validation), crosses at an empirical rate of 0.2215 against a nominal 0.05 in the
Monte-Carlo stream. The construction is not a repair of exchangeability either,
since every bound is a bound on the labelled pool and the transfer family fails
in the same way the fixed-budget certificate does.

## Partial identification from two annotation sources

For CRC and GVHD the depot ships a second label table beside the manual one, and
the two are joined cell by cell on the key `(image, cellID)`. CRC_Feb2623 has
2,505 expert cells matching 2,505 of the HighQuality CellTable with 0 orphans,
and GVHD_Feb1323 has 10,283 matching 10,283 with 0 orphans. On coarse classes the
two passes agree on 0.7253 of CRC cells (Cohen's kappa 0.6615) and 0.8473 of
GVHD cells (kappa 0.8187), against native agreements of 0.7094 and 0.7811. The
expert subset is not a random sample of its images, so every number here is
conditional on the cells the expert actually labelled.

The natural point-identification model is a three-test latent-class (Hui-Walter)
model with `E` = expert says type `T`, `H` = HQ says type `T`, `F` = the cell
fails `T`'s marker rule and `Z` = the cell really is `T`. It carries seven
parameters for the seven free probabilities of the 2x2x2 table, so it is
saturated: an exact fit generically exists and its residual carries no evidence,
and the likelihood is exactly symmetric under the swap
`(p, aE, a0E, aH, a0H, pi, phi) -> (1 - p, a0E, aE, a0H, aH, phi, pi)`, so
conditional independence alone identifies `pi` only up to the swap of `pi` with
`phi`. Point identification therefore rests entirely on an assumed
informativeness restriction, that each labeller beats a coin flip about
membership, together with PIUS, that the rule fails no more often among true
members than among non-members. That route is rejected for `CRC_Feb2623` APC and
`GVHD_Feb1323` neutrophil, where no informativeness-consistent fit exists and the
best in-box fit still misses the table by L1 misfit 2.67e-3 and 2.57e-3; where it
is not rejected, the selected branch sits close to its own informativeness
boundary in two cases (margins 0.0122 and 0.0156) and well inside it in the
third (0.3039), and the field-of-view bootstrap flips between the two mirror
branches,
so the point correction is not quotable.

The defensible route makes no conditional-independence assumption. Under
informativeness and PIUS alone, and because `pi` is a ratio of linear functionals
over a polytope, the sharp set of the latent violation rate is an exact linear
program. Comparing the identified set over the 68 conventions, which is a
statement about the shipped reference labels, with the sharp set of the
latent-membership violation rate at the primary convention, every sharp set
reaches 0, and reaching 0 is not vacuous because an explicit
conditional-independence fit attains it for three of the five populations. The
upper end drops where the linear program binds: CRC `CD3T` [0.0690, 1.0000]
becomes [0, 0.5000]; CRC endothelial [0.0521, 0.6042] becomes [0, 0.2271]; GVHD
neutrophil [0.2635, 0.9341] becomes [0, 0.5000]; GVHD plasma [0.1984, 0.9463]
becomes [0, 0.5000]; and the one convention-free concordant population, CRC APC
[0, 0], becomes [0, 0.1940].

Bounded conditional dependence sharpens nothing further. The model is saturated,
`kappa = 0` is admitted wherever an exact latent-class fit exists, and a chosen
dependence bound of 0.002 is the smallest value in the searched grid at which the
search returns a feasible split, returning one whose attained upper end is
0.0000, so the dependence grid is a sensitivity display for a chosen bound rather
than a data-determined quantity. The consequence for the paper is that
assumption-light sharp sets only lower the worst case and never raise the floor,
so the convention-free verdicts are statements about the shipped reference labels
and not about latent cell membership, and no positive true-violation rate is
established by these data.

## External validation and targeted-repair arms

The CellTune demo project is not an independent study. Its 40-marker panel is
identical, name for name, to `GVHD_Feb1323`; 38 of its 75 fields of view are the
depot's own images, with 80.6% of the shared marker TIFFs byte-identical to the
`HighQualityLabels` release and 0.0% to the `ManualLabels` release the benchmark
uses; it is a third annotation pass over the same material; and its segmentation
is not the depot's, so its 1,752 usable expert cells are recomputed objects of
which only 142 share `(FOV, cellID)` with the benchmark's labelled cells. On that
cohort the certificate is vacuous: `demo_within` accepts nothing at
`alpha = 0.05` and 0.10 and `demo_heldout` accepts nothing at any budget, so the
1.0000 coverage entries there are the empty-set convention and no claim can be
made from them.

The genuinely external data are the STELLAR CODEX cell tables, Zenodo record
6863872, DOI `10.5061/dryad.g4f4qrfrc`, licence CC0-1.0:
`B004_training_dryad.csv`, 248,285 usable cells over 48 markers from human
intestine, and `BE_Tonsil_l3_dryad.csv`, 219,926 usable cells over 44 markers
from human tonsil and Barrett's esophagus, each capped at a deterministic
20,000-cell subsample with seed 20240618. On the two hard splits that data does
not reproduce the certificate; on the family the guarantee is stated for it
does, which is the family below. `ext_within`, five image-grouped folds over the eight image-disjoint
intestine regions, accepts 0.7264 at a realised risk of 0.0778 but covers only
0.6556 at `alpha = 0.10`, against 0.8167 on the benchmark's corresponding
cross-field-of-view family and 1.0000 on its random-split family. `ext_transfer`,
source-calibrated across tissue, panel and platform on a shared panel of 26
markers, accepts 0.8394 at a realised risk of 0.4893 with coverage 0.0333, so the
budget is violated several-fold. The ratio diagnostic the paper's prose carries
does not survive the transfer: it misses 60 of 174 over-budget transfer
evaluations and 2 of 31 over-budget within evaluations at `alpha = 0.10`, with no
false alarms, so the perfect-detector reading is a benchmark result and not a
theorem.

The same two cohorts also carry the family the guarantee is stated for, which
neither the image-grouped nor the transfer arm provides, and there the certificate
holds. `scripts/r55_stellar_cellsplit.py` trains on the intestine cohort's other
images and then divides the held-out pool of 20,000 cells, 10,000 of them spent
on calibration -- an order of magnitude more than the benchmark's 2,500 -- at
random into calibration and evaluation halves, exactly as
`scripts/r05_cellsplit.py` does on the depot, with the same certificate, split
fraction, repeat count and metric definitions (`scripts/r03_analyse.evaluate`,
imported rather than re-implemented). Over 150 evaluations, being five
held-out-field folds crossed with ten repeats and three classifiers, coverage is
1.0000 at `alpha = 0.10`, 0.9933 at 0.05 and 1.0000 at 0.20, at acceptances of
0.8514, 0.6760 and 0.9962 and realised risks among retained cells of 0.0865,
0.0392 and 0.1423. The uncalibrated threshold matches the certificate on
throughput (0.8902 against 0.8514) and holds the budget in 0.4867 of evaluations
against 1.0000, which is the benchmark's own contrast reproduced on data from
another laboratory, platform and tissue. A second family (`ext_targetcal`)
trains on the intestine cohort and divides the tonsil and Barrett's esophagus
cohorts each at random, over their shared 26 markers and six shared coarse
classes; over 60 evaluations per budget coverage is 1.0000 at all three budgets
with a non-vacuous acceptance of 0.436, 0.436 and 0.519 and realised risks of
0.0359, 0.0684 and 0.1534. The result is not an artefact of the feature
transform: the released STELLAR values are already on an arcsinh-like scale, and
coverage at a 10% budget is 1.0000 under standardisation without the log and
under the raw values as well as under the transform the `r34` branch already
uses. The ratio diagnostic is sharp on this family, one flagged over-budget
evaluation out of 630 and no false positive, so its 60 transfer misses belong to
the transfer arm, where the oracle threshold accepts nothing and the denominator
of the ratio does not exist. Two limits are recorded with the result rather than
after it: the certificate's higher throughput here than on the depot (0.8514
against 0.559) tracks the tenfold larger calibration sample, so it is a reading
of the calibration-size curve rather than an independent-cohort effect, and the
target-calibrated arm's transferred model is weak on tonsil (0.26 to 0.33
accuracy over the six shared classes), so that arm shows the guarantee holding
rather than a tool lifting annotation.

The targeted-repair arms repair neither failure. Within the cross-field-of-view
family the vacuous-inclusive coverage rises from 0.8333 and 0.8167 at
`alpha = 0.05` and 0.10 to 0.9500 and 0.9167 for `sc3`, 0.9750 and 0.9639 for
`sc10` and 0.9500 and 0.9444 for `mond_class`, at no extra label cost, since
`sc3` and `sc10` read the denominator off the unlabelled deployment scores with
the error curve from calibration and `mond_class` splits the stored calibration
cells by predicted class. But in this twenty-unit design non-vacuous coverage
never reaches 0.90 at any budget for any arm, and no paired `covered_nonvac`
difference against `plain` excludes zero: `sc3` +0.0306 [-0.0519, +0.1287], `sc10` -0.0472
[-0.1472, +0.0667], `mond_class` +0.0444 [-0.0750, +0.1935]. The apparent repair
is a redistribution of vacuity rather than a gain in capability.

At field-of-view resolution the same four arms tell a different story, and the
difference is one of design rather than of statistics. `scripts/r48b_fov_repair.py`
re-runs `plain`, `sc3`, `sc10` and `mond_class` on the 96 field-of-view units. The
plain certificate again falls short (0.8438 and 0.7922 at `alpha = 0.05` and 0.10),
and `sc3` repairs the vacuity-inclusive readout at both budgets (0.9780 and 0.9688)
with the conditional readout at 0.9698 and 0.9643 against `plain`'s 0.8210 and
0.7765, at acceptances of 0.3173 and 0.5044. On the joint readout the paired
difference against `plain` is +0.0698 [+0.0270, +0.1161] over a unit bootstrap of
4,000 resamples, the first arm in this study whose improvement of that quantity
excludes zero, while `mond_class` gives -0.0530 [-0.1156, +0.0089] and `sc10`
-0.2774 [-0.3279, -0.2249]; `sc10` repairs coverage but at a 5% budget accepts
0.0212 of the batch, so it is valid and nearly silent. The partial repair reported
at twenty units therefore becomes a resolved but still partial one at 96: the
readout that counts abstention as a failure improves by about seven points, reaching
0.920 at a 20% budget, above the nominal 0.90 though its interval still includes it,
and 0.845 at the 10% budget, where it stays below nominal.

On transfer the label cost stays where the stored ladder put it: at
`alpha = 0.10` the arms that reach the nominal level with a non-empty accepted
set need 500 target labels (`mond_class`, `plain`, `prop_mond` and
`twostage_nw`; `pool_iw` reports 25 but accepts 0.0063 of the batch, and
`unc_iw` needs 1000), and the cost falls to 200 only at `alpha = 0.20`. The
realised-risk table identifies the cause, and it is not bound slack: the accepted
sets are genuinely wrong, with a pooled realised risk among accepted cells of
0.3347 for `src` beside a fraction of non-empty evaluations over budget of 0.986,
and 0.3790 for `pool_iw` and 0.3102 for `unc_iw`, so the labels are wrong rather
than the bound loose. Active label selection does not turn that around: the
selection arms' vacuous-inclusive coverage difference against `plain` is at most
+0.00002 and excludes zero in no case, and at `alpha = 0.10` their label cost is
still 500 for `prop_mond` and `twostage_nw`. One reproducibility note governs
the adaptation numbers, and it clears them: the logreg iteration count does
follow the BLAS thread pool, but pinning the pool to a single thread makes the
arm bit-reproducible -- see Reproducibility below -- and the 0.155 reported for
the `r31` re-run is the logistic-regression-only subtotal of the classifier mix
(measured with the pool pinned, that subtotal is 0.163 over two repeats), not
thread drift.

## An external, image-based classifier

The layer is claimed to wrap any cell-type classifier, but every model in the
main grid is one of three tabular scorers fitted on per-cell marker means.
`scripts/r51_cellsighter.py` tests the claim against a model of a different
kind: CellSighter [Amitay et al., *Nature Communications* 14 (2023),
doi:10.1038/s41467-023-40066-7], an ensemble of ResNet-50 networks over
multiplexed image crops.

The published repository is vendored at `tools/CellSighter` and its own modules
are imported unmodified - `model.Model`, `data.utils.load_crops`,
`data.data.CellCropsDataset`, `data.transform.train_transform` /
`val_transform`, and the loops `train.train_epoch` / `train.define_sampler` and
`eval.val_epoch`. The layer around it is the same
`errctl.conformal.SelectiveRiskController`, and the readouts come from
`scripts/r03_analyse.evaluate`, so neither the model nor the analysis can drift
from the main grid.

Because the depot ships per-cell marker means rather than image features for its
two field-of-view cohorts, the crops are rebuilt from the released TIFFs in the
format the CellSighter README specifies (`CellTypes/data/images`,
`CellTypes/cells`, `CellTypes/cells2labels`, plus `channels.txt`), restricted to
the shared marker panel and the shared coarse classes of the pair, exactly as
`scripts/r02_run_experiments.py` builds a transfer unit. Only the two cohorts
whose image folders cover their annotated cells can carry this: the Melanoma
folders return 19,775 of their 21,204 crops with a label (93%), while the GVHD
folders are partial crops of a larger acquisition and return 1,549 of 10,283
(15%), so Melanoma is the source and GVHD the target of the wrap.

Four deviations from the published recipe are forced by the host and are
recorded rather than hidden: the device is the CPU where their code hardcodes
`cuda`; `num_workers` is 0 because their dataset would pickle whole field arrays
out to each worker; one cohort pair is used instead of their five, with a
per-class cap of `size_data = 400` and a reduced epoch count; and a single
network is trained in place of their ensemble, so the score is that network's
softmax.

## Compute

All code is pure Python 3.10.11 with numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, tifffile 2025.5.10 and matplotlib 3.10.9. The main grid runs on the CPU and imports neither PyTorch nor LightGBM; the one exception is the external-classifier wrap described immediately above, which imports PyTorch 2.4.1+cpu, torchvision 0.19.1, OpenCV, scikit-image and TensorBoard and is likewise run on the CPU, in place of the GPU the published code assumes. The image I/O is TIFF decoding in `extract_crop_features` (Breast and Melanoma), in `scripts/r51_cellsighter.py`, and in the delivery audit; all three are cached or run once.

Parallelism is process-level: the drivers take `--jobs` and fan out over independent units with `multiprocessing.Pool`, each worker loading one stored prediction file or cohort cache. The 297 stored prediction files were analysed with 14 workers, the 32,000 Monte-Carlo repetitions over eight settings with eight, and the 40-job label-noise study with eight. `HistGradientBoostingClassifier` is pinned to `OMP_NUM_THREADS=2` per worker in `r12_labelnoise.py` to avoid oversubscribing the machine, while the random forest uses `n_jobs=-1` internally. The `work/cache/r08/<cohort>.npz` caches are built once by `r08_panel.py` and reused. The transductive and label-ladder experiments (`r14`, `r20`) reuse the stored prediction files and need no refitting except in the adaptation arm, which refits one classifier per drawn subsample; the review-budget, marker-sensitivity and weight-diagnosis experiments (`r16`, `r15`, `r18`, `r19`) are cache-only and finish in seconds to minutes on eight workers. The field-of-view-resolution family adds 288 stored prediction files under `work/predictions_fov/`, analysed with 10 workers; `scripts/r48b_fov_repair.py` fans out over the same 288 files with a process pool of up to 10, importing the arm code into each child rather than calling r31's driver, so its regression check against `experiments/r03_summary.csv` is not triggered. The review-loop table (`r50`) is cache-only and finishes in under a second.

## Reproducibility

Two things about the adaptation arm of `r14_targetcal.py` are worth stating,
because an earlier version of this section reported the first as a defect. First,
the iteration count of `LogisticRegression` follows the BLAS thread pool, so the
adapted model is not reproducible unless that pool is fixed; `r31_targeted_repair.py`
reported 0.155 acceptance where the stored run reports 0.355. Second, the two
numbers are not two executions of the same recipe: 0.155 is the
logistic-regression-only subtotal of a three-classifier mix, and the pinned
logreg-only subtotal measured here is 0.163 over two of its repeats, a
difference inside the sampling error of a three-repeat mean of that quantity. Pinning `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to 1 before
numpy is imported, and giving `LogisticRegression` an explicit `tol`, makes the
arm bit-reproducible: two independent processes agree to `0.000e+00` on mean
acceptance and on both coverage readouts across all five arms, and each
reproduces the stored `adapt_k` headline of 0.3551 exactly
(`scripts/r43_threadpin.py`, `work/r43_threadpin_report.txt`). A pin sweep from 1
to 4 threads moves acceptance by at most `1.4e-05`. Every number quoted from the
adaptation arm in the manuscript is therefore a reproducible quantity.

Beyond that arm, the released result tables are self-contained. Every main and
supplementary table is a deterministic summary of the CSVs in `experiments/`,
and `scripts/r63_reproduce_check.py` re-derives the manuscript's four negative
results, the cross-field-of-view shortfall and its one positive replication
from those files alone, in a single standard-library pass that needs no dataset
and exits non-zero on any mismatch; `docs/submission/reproducibility.md` gives
the file, column and expected value behind each claim, and `work/_preflight.py`
extends the same check to every number quoted in the manuscript.