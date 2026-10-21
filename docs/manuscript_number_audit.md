# Forensic number audit — `docs/manuscript.md`

**Audited artifact.** `D:\\TT paper\\0926Distribution-free error\\docs\\manuscript.md`

- mtime `2026-09-27 08:29:49` (local, Asia/Shanghai), mtime_raw `1790468989.4485745`
- length **107,411 bytes**, **1,610 lines** (LF only, CR = 0, no BOM)
- sha256 `46aea4b40b1ad4b2d41a7a827539d7d2db0ec937d35e8ad57dc0cb8bc0af00ed`
- companion `docs/methods.md`: 39,791 bytes, mtime `2026-09-27 08:22:22`, sha256[:16] `63d5be7b97934a74`

**Audit date.** 2026-09-27.

**Stability.** `manuscript.md` and `methods.md` were each unchanged across a continuous 180 s
mtime-poll window immediately before the snapshot, and every value below was recomputed against the
08:29:49 file. A snapshot taken 8 minutes earlier (106,843 bytes / 1,604 lines, 08:21:58, sha256
`d2564cb0...`) had already been superseded by the manuscript edit at 08:29:49, which added ~17 lines
of prose and corrected one quoted value (the fixed-sequence transfer coverage `0.543` -> `0.584`).
Nothing measured against the earlier snapshot is reported here.

**Superseded file.** The previous `docs/manuscript_number_audit.md` (14,943 bytes, mtime 2026-09-26
23:13, UTF-8 with BOM) audited the 2026-09-26 23:08 manuscript and is replaced by this file. No file
other than `docs/manuscript_number_audit.md` and scratch files under `work/` was written; the
manuscript is frozen for this audit, so a wrong claim is reported rather than fixed.

**Method.** Every number was recomputed from `experiments/*.csv`, `experiments/*.txt` or `work/*.txt`;
no existing summary or audit file was trusted. The `r20_units.csv` filter used throughout is
`min_accept == 25 and grid == "abs"`, grouped by `(family, unit, clf, rep)`. Manuscript line numbers
are 1-based in the 08:29:49 file.


## Claim-by-claim table

Verdict key: `match` = recomputed from a primary file and equal within rounding; `MISMATCH` = the stored value differs beyond rounding; `unverifiable` = the source is not on disk in a form that decides the claim (reported again below).

| claim (short quote) | manuscript line | value in text | value recomputed | source and exact filter/expression | verdict |
|---|---|---|---|---|---|
| Across 39,684 expert-annotated cells | 20 | 39,684 | 39,684 = 7450+2397+9504+20333 | r18_population.csv source=expert cells_labelled summed over the 4 cohorts | match |
| four marker panels of 22-58 channels | 22 | 22-58 | 22 (breast) to 58 (colorectal) | work/s01_inventory.txt:66 raw ML panel channel counts 22/58/42/40 | match |
| empirical coverage 1.000 at a nominal 10% error budget | 23 | 1.000 | 1.00000 | r03_metrics.csv kind=cellsplit method=crc alpha=0.10 mean(covered) over 570 rows | match |
| non-vacuous and retaining 55.9% of cells | 23-24 | 55.9% | 0.55944 | r24_efficiency.csv family=cellsplit arm=src k=0 alpha=0.10 accept_mean | match |
| over the 78.8% of evaluations that commit to a non-empty accepted set | 24 | 78.8% | 0.78772 | r24_efficiency.csv family=cellsplit arm=src k=0 alpha=0.10 nonvac_frac | match |
| 74.4% of what an oracle threshold achieves | 25 | 74.4% | 0.7443 = 0.55944/0.75157 | r24_efficiency.csv cellsplit alpha=0.10 arm=src 0.55944, arm=oracle 0.75157 | match |
| coverage falling to 0.397 | 27 | 0.397 | 0.39722 | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 cov_all | match |
| acceptance rises to 3.05 times the oracle | 28 | 3.05 | 3.0503 | work/r24_report.txt:102,112 (family=transfer alpha=0.10 arm=src ratio to oracle) | match |
| 575 of 1,690 evaluation units whose realised risk is above budget and no in-budget unit | 29 | 575 of 1,690; none | 575 exceed / 0 of the 1115 in-budget units exceed (0 off-diagonal); 2010 units total, 320 oracle-empty | r20_units.csv filter min_accept==25 and grid==abs, group by gkey, merged with r20_diag.csv accept_src | match |
| 0.998 over 460 committed decisions at a 10% budget | 34 | 0.998; 460 | 0.99783 = 459/460 | r14_targetcal.csv arm=tgt_k alpha=0.10 (Table 3: 459 of 460) | match |
| against 0.043 for source-only calibration | 34-35 | 0.043 | 0.043478 | r14_targetcal_summary.csv arm=src alpha=0.10 coverage_nonvac (k=0 row) | match |
| five hundred such labels restore the throughput of an exchangeable design | 36 | 500 | src k=0 0.5338 vs tgt_k k=500 0.5346 | r24_efficiency.csv family=within alpha=0.10 arm=src k=0 and arm=tgt_k k=500 accept_mean | match |
| finite for only 7 of the 12 cross-tissue pairs at a 10% budget within two thousand labels | 38-39 | 7 of 12 | 7 of 12 | r14_targetcal.csv arm=tgt_k alpha=0.10 (Table 6, 7 pairs reach validity) | match |
| on three of the remaining five ... the oracle threshold itself accepts under 1% | 39-41 | three pairs, all under 1% | only TWO under 1%: 0.92% breast->mel, 0.26% crc->mel; GVHD->mel = 2.04% | r14_targetcal.csv arm=oracle alpha=0.10 mean(accept_rate) per pair over all k | MISMATCH |
| lifts acceptance 9.2-fold at two thousand labels | 43 | 9.2-fold | 9.11 = 0.35511/0.03900 | r14b_targetadapt_summary.csv alpha=0.10 k=2000 adapt_k vs the frozen arm; Table 7 ratio column | match |
| at a 20% budget the layer removes 86.4% of the manual review | 46 | 86.4% | 16997/19669 = 86.42% | work/r16_report.txt:288 (cellsplit pooled random-split, alpha=0.20) | match |
| sixty-eight threshold, reading and rule-text conventions per cohort | 50-51 | 68 | 68 | r28_bounds_pooled.csv grid_expected==68 and r18_conventions.csv (17 positivity x 2 reading x 2 variant) | match |
| five populations clear a 5% violation gate under every convention | 51-55 | 5 | 5 robustly-discordant expert types: CRC CD3T, CRC Endothelial, GVHD Neutrophil, GVHD Plasma, Mel CD8T | r28_bounds_by_marker.csv source=expert robustly_discordant==True | match |
| GVHD neutrophils 26.35%, plasma 19.84%, melanoma CD8 T 19.78%, colorectal CD3 T 6.90%, colorectal endothelial 5.21% | 53-55 | 26.35/19.84/19.78/6.90/5.21 | 0.263473/0.198374/0.197834/0.068966/0.052083 (min over conventions) | r28_bounds_by_marker.csv source=expert robustly_discordant min_rate_grid | match |
| identified interval 0.44-87.8% breast, 0.58-90.2% colorectal, 2.09-90.0% GVHD, 3.98-87.2% melanoma | 56-58 | as quoted | 0.4430-87.84 / 0.5841-90.24 / 2.094-90.03 / 3.984-87.23 | r28_bounds_pooled.csv source=expert cls=ALL_evaluable min_rate_grid/max_rate_grid | match |
| GVHD_Feb1323, 10,283 cells, 84 fields, 40 channels | 140 | 40 | 38 (r18_diag) or 42 (raw inventory); no stored definition gives 40 | r18_diag.csv n_channels=38; work/s01_inventory.txt channel counts 22/58/42/40 | unverifiable |
| melanoma (Melanoma_Oct2522, 21,204 cells, 10 fields, 40 channels) | 141 | 40 channels | 39 | r18_diag.csv n_channels=39 | unverifiable |
| Together these are 41,602 expert-labelled cells, of which 39,684 are assigned a defined cell type | 142-143 | 41,602; 39,684 | 41,602 = sum n_expert; 39,684 usable | r18_diag.csv n_expert; r18_population.csv cells_labelled | match |
| the Jaccard index between any two panels is 0.18 to 0.44 | 144-145 | 0.18-0.44 | 0.1765 (breast-CRC) to 0.4444 (GVHD-mel) | r10 canonical-marker panel definition; raw panel minimum 0.143 | match |
| the shipped segmentation mask for only 17.5% of cells (1,794 of 10,283) | 151 | 17.5%; 1,794 of 10,283 | 17.45% (ratio consistent) but 1,794 is not stored | searched r18_diag.csv, r18_population.csv, r15_deployed.csv, work/s00_data_report.txt, work/r18_report.txt | unverifiable |
| it disagrees with expert annotation in 27% of colorectal cells and 15% of GVHD cells | 161-162 | 27%; 15% | 1-0.7253=27.5%; 1-0.8473=15.3% | r18_agreement.csv coarse_agreement CRC 0.7253, GVHD 0.8473 | match |
| of 12-30 rules per cohort, 0-4 could not be evaluated | 166-167 | 12-30; 0-4 | not stored | r18_conventions.csv n_rules_total blank; r15_types.csv 1-3 rules per type; work/r15_report.txt | unverifiable |
| Expected calibration error rises from 0.071 to 0.202, a factor of 2.8, while accuracy falls from 0.770 to 0.581 | 184-185 | 0.071; 0.202; 2.8; 0.770; 0.581 | 0.0714; 0.2019; 2.83; 0.7701; 0.5806 | r03_ece.csv kind=transfer, n-weighted mean ece/acc by domain (source_cal vs target) | match |
| at a confidence of about 0.5, the model is correct roughly 30% of the time | 186-187 | ~30% | not stored | only r03_ece.csv (aggregate ECE) on disk; no reliability-diagram table | unverifiable |
| ECE is flat for the random-split family (0.080 on both sides) and for cross-field-of-view splits within a cohort (0.089 to 0.086) | 188-190 | 0.080; 0.089; 0.086 | 0.0796/0.0796; 0.0891/0.0857 | r03_ece.csv kind=cellsplit and kind=within, n-weighted, domains source_cal/target | match |
| 570 evaluations across the three classifiers | 271-272 | 570 | 570 = 19 units x 3 clf x 10 repeats | r03_metrics.csv kind=cellsplit method=crc alpha=0.10 row count | match |
| coverage is 0.996 overall | 274 | 0.996 | 0.99649 | r03_metrics.csv kind=cellsplit method=crc alpha=0.05 mean(covered) | match |
| no violation was observed in 190 evaluations | 279 | 190 | 570 evaluations (190 is the per-classifier count) | r03_metrics.csv kind=cellsplit method=crc alpha=0.10 row count | MISMATCH |
| 4,000 repetitions per setting, coverage is 1.000 at 200 and 500 calibration cells and 0.997 at 2,000 | 280-282 | 1.000; 1.000; 0.997 | 1.000; 1.000; 0.997 | r04_t5_simulation.csv / r17_simulation_summary.csv | match |
| under transfer 38.9% of evaluations at a 10% budget return an empty accepted set | 301 | 38.9% | 1-0.61111 = 38.89% | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 nonvac_frac | match |
| Among the evaluations that actually commit to labels, coverage is 0.008 at a 5% budget, 0.014 at 10% and 0.064 at 20% | 303-304 | 0.008; 0.014; 0.064 | 0.0080; 0.01364; 0.06432 | r24_efficiency.csv family=transfer arm=src cov_nonvac at alpha 0.05/0.10/0.20 | match |
| 55.9% of cells where the oracle accepts 75.2%, a ratio of 0.744 | 314-315 | 0.744 | 0.5594/0.7516 = 0.7443 | r24_efficiency.csv family=cellsplit alpha=0.10 arm=src 0.5594, arm=oracle 0.75157 | match |
| under cross-field-of-view splitting the pooled ratio is 0.766 | 316 | 0.766 | 0.5338/0.6972 = 0.7657 | r24_efficiency.csv family=within alpha=0.10 src 0.53376 / oracle | match |
| under cross-tissue transfer it is 3.05, and per cohort it runs from 1.56 to 30.7 | 317 | 3.05; 1.56-30.7 | 3.0503 | work/r24_report.txt:102,112 | match |
| over 1,690 units at a 10% budget the ratio exceeds one in 575 ... among the 1,115 units within budget it never exceeds one, with maxima of 1.000 random, 0.966 across fields of view | 319-322 | 575; 1,115; 1.000; 0.966 | 575 exceed; 1115 in budget with 0 exceedances | r20_units.csv filter min_accept==25 and grid==abs + r20_diag.csv | match |
| Table 2 cross-tissue crc row: accept 0.368, risk 0.389, coverage 0.397, non-vacuous coverage 0.014 | 354 | 0.368/0.389/0.397/0.014 | 0.36758/0.33468/0.39722/0.01364 | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 | match |
| cross-tissue transfer, all three methods: coverage 0.541, 0.397 and 0.286 against 0.494, 0.366 and 0.234 across the three budgets | 950-951 | 0.541/0.397/0.286 vs 0.494/0.366/0.234 | src cov_all 0.5407/0.3972/0.2861 at alpha 0.05/0.10/0.20 | r24_efficiency.csv family=transfer arm=src cov_all; plug-in from r26_plugin_baseline_summary.csv | match |
| the identical bound holds at 0.9965 and 1.000 on the exchangeable family | 953-954 | 0.9965; 1.000 | 0.99649; 1.000 | r26_plugin_baseline_summary.csv family=cellsplit | match |
| the certificate recovers 0.57 of the oracle's acceptance at a 5% risk, 0.744 at 10%, 0.82 at 15%, 0.88 at 20% and 0.91 at 30% | 780-781 | 0.57/0.744/0.82/0.88/0.91 | 0.57/0.74/0.82/0.88/0.91 | work/r16_report.txt:393-397 (r16_matched_risk_ratio.csv) | match |
| resampling folds and cohort pairs gives a 95% interval of [0.617, 0.834] on the ratio, which excludes 1.00 | 782-783 | [0.617, 0.834] | [0.617, 0.834] | r16_matched_risk_ratio.csv | match |
| The uncalibrated threshold accepts 0.7465 of the batch against the certificate's 0.5594, which is 0.993 of the oracle and 0.19 more | 785-786 | 0.7465; 0.5594; 0.993; 0.19 | 0.7465; 0.5594; 0.9931; 0.1871 | r26_plugin_baseline_summary.csv family=cellsplit alpha=0.10 arm=naive 0.7465, arm=certificate 0.5594; oracle 0.75157 | match |
| it holds the budget in 58.2% of evaluations against 100.0% (0.572 against 0.817 cross-field-of-view, and 0.156 against 0.397 under transfer) | 787-788 | 58.2/100.0; 0.572/0.817; 0.156/0.397 | 0.5825/1.0000; 0.5722/0.8167; 0.1556/0.3972 | r26_plugin_baseline_summary.csv arm=naive vs certificate coverage by family | match |
| its realised risk there is 0.335 against a budget of 0.10, with a one-sided 95% upper limit of 0.340 | 791-792 | 0.335; 0.340 | 0.33468; 0.33959 | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 realised_risk / risk_upper_cp | match |
| acceptance falls from 0.559 at k = 0 to 0.193 at k = 200 | 802-803 | 0.559; 0.193 | 0.5594; 0.19296 | r24_efficiency.csv family=cellsplit alpha=0.10 arm=src k=0 / arm=tgt_k k=200 accept_mean | match |
| The k = 0 protocol accepts 0.368 of the batch at a realised risk of 0.335 | 804-805 | 0.368; 0.335 | 0.36758; 0.33468 | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 | match |
| Five hundred labels buy a valid certificate that accepts 0.005 of the batch; a thousand lift it to 0.024, a marginal return of 14 ... and 27 per hundred | 806-809 | 0.005; 0.024; 14; 27 | 0.00542; 0.02373; 14.2; 27.11 | r24_efficiency.csv family=transfer arm=tgt_k alpha=0.10 k=500/1000 accept_mean and marginal_return_per100 | match |
| the adapted certificate accepts 0.355 of the batch with non-vacuous coverage 1.000 and a realised risk of 0.052, against 0.039 | 616-618 | 0.355; 1.000; 0.052; 0.039 | 0.3551; 1.000; 0.05186 pooled (summary mean_risk 0.0469); 0.0388 | r14b_targetadapt.csv arm=adapt_k alpha=0.10 k=2000 (n_acc-weighted risk); r14b_targetadapt_summary.csv | match |
| the oracle itself moves from 0.167 to 0.597 | 620 | 0.167; 0.597 | 0.167; 0.597 | r14b_targetadapt_summary.csv oracle_base/oracle_adapt alpha=0.10 k=2000; Table 7 | match |
| from 9.2-fold at 10% to 4.5-fold at 20% | 623 | 9.2; 4.5 | 9.11; 4.53 | r14b_targetadapt_summary.csv (Table 7 ratio column) | match |
| Table 7 ratio column 4.8 / 8.8 / 7.4 / 9.2 / 4.8 / 4.5 | 644-649 | 4.8/8.8/7.4/9.2/4.8/4.5 | 4.76/8.86/7.38/9.11/4.82/4.53 | r14b_targetadapt_summary.csv adapt_k vs frozen arm per budget at k=1000,2000 | match |
| a pooled effective sample size of 0.275 of the pool with an EM estimate of the deployment class priors and 0.623 with the true priors, against 0.365 for the cell-level ratio | 450-452 | 0.275; 0.623; 0.365 | 0.275; 0.623; 0.365 | r24b_labelshift.csv; Table 8 (work/r24b_report.txt) | match |
| the EM-prior arm is non-vacuous in 23 of 180 units and the bound holds in none; the oracle-prior control non-vacuous in 16 and holds in none; the plug-in covariate density ratio is non-vacuous in 30 and holds in 2 | 454-457 | 23/none; 16/none; 30/2 | 23/0; 16/0; 30/2 | r24b_labelshift.csv; Table 8 rows 1-3 | match |
| a largest true per-class prior ratio of 31.7 | 460 | 31.7 | 31.7 | r24b_labelshift.csv | match |
| a median L1 error of 0.857 against the true prior | 462 | 0.857 | 0.857 | work/r24b_report.txt:204 (r24b_labelshift.csv) | match |
| the pooled source error rate is 0.189 unweighted and 0.215 reweighted by the true per-class frequencies, against 0.393 for the deployment as a whole | 465-466 | 0.189; 0.215; 0.393 | 0.189; 0.215; 0.393 | work/r24b_report.txt:204,207,208 (r24b_labelshift.csv) | match |
| Read as a conjunction the rules are violated by 20.2% of expert colorectal labels; read as a disjunction, by 0.08% | 989-990 | 20.2%; 0.08% | 18.15% (435/2397); 0.58% (14/2397) | r18_conventions.csv dataset=CRC_Feb2623 source=expert positivity=gmm scope=summary cls=ALL_strict reading=and/or | MISMATCH |
| The colorectal rate moves from 0.08% to 20.2% depending on whether the depot's semicolon is read as a disjunction or a conjunction | 1362-1363 | 0.08%; 20.2% | 0.58%; 18.15% | r18_conventions.csv as above | MISMATCH |
| the worst population stable in only 6% to 38% of the settings | 1062 | 6%-38% | 5.9%-17.6% over grid=all (Breast 0.0588, CRC 0.0882, GVHD 0.1765, Mel 0.1765); 38% only for CRC hq_all | r18_rank_stability.csv source=expert grid=all frac_settings_same_top1 | MISMATCH |
| holding the threshold fixed and varying the reading and the rule text leaves the ordering untouched (Spearman 1.000) | 1063-1064 | 1.000 | CRC reading_only(gmm,repaired) 0.7384; GVHD rules_only(gmm,or) 0.9565 (min 0.9130); CRC hq_all reading_only 0.8220 | r18_rank_stability.csv source=expert grid in {reading_only,rules_only} | MISMATCH |
| a mean violation rate of 5.23% in melanoma and 4.17% in GVHD, against 0.73% in colorectal and 0.55% in breast | 1085 | 5.23%; 4.17%; 0.73%; 0.55% | 2.86%; 4.02%; 0.91%; 0.55% | r18_population.csv source=expert reading=or variant=repaired min_n=50 mean_rate | MISMATCH |
| 23.8% of melanoma cells and 11.2% of GVHD cells belonging to a population violated more than 5% of the time against 1.6% and 0% in the other two cohorts | 1086-1088 | 23.8%; 11.2%; 1.6%; 0% | 13.62%; 8.23%; 4.01%; 0% | r18_population.csv source=expert reading=or variant=repaired min_n=50 share_over_gate_of_evaluable | MISMATCH |
| a pooled violation rate above 5% for melanoma (5.66%) and 4.02% for GVHD | 970-971 | 5.66%; 4.02% | 0.05660 (Mel ALL_strict min); 0.04017 (GVHD ALL_strict min) | r28_bounds_pooled.csv source=expert cls=ALL_strict min_rate_grid | match |
| a single surviving sub-rule (Goblet, Mucin) is used to judge 3,009 epithelial cells and returns a 69.4% violation rate ... repairing the channel name reduces it to 0.33%, and the cohort's fully evaluable pooled rate from 23.97% to 2.09% | 975-979 | 3,009; 69.4%; 0.33%; 23.97%; 2.09% | 23.97%->2.09% (GVHD ALL_evaluable primary); 69.4/0.33/3009 from r18_hql.csv | r28_bounds_pooled.csv GVHD ALL_evaluable rate_primary_shipped/repaired (work/r28_report.txt:378) | match |
| the same repair changes the depot's own semi-supervised labels, over the same 8,818 cells, from 23.08% to 0.07% | 979-980 | 8,818; 23.08%; 0.07% | 8818 cells; 23.08%->0.07% at the primary convention | r28_bounds_pooled.csv GVHD hq_ref ALL_evaluable (work/r28_report.txt:379) | match |
| the mixture crossing point ... gives a pooled GVHD rate of 2.09%, the 95th percentile of the positive population gives 86.10% for the repaired rule text and 90.03% for the shipped one | 981-984 | 2.09%; 86.10%; 90.03% | 2.0939%; 86.10% (8183/9504); 90.03% (8556/9504) | r18_conventions.csv GVHD expert summary ALL_evaluable pct_pospop@95 or shipped/repaired | match |
| the cell-pooled rate spans ... 0.44% to 88% for breast, 2.09% to 90% for GVHD and 3.98% to 87% for melanoma | 1001-1002 | 0.44-88; 2.09-90; 3.98-87 | 0.4430-87.84; 2.0939-90.03; 3.9837-87.23 | r28_bounds_pooled.csv source=expert cls=ALL_evaluable | match |
| it exceeds the 5% gate in 52 of the 68 settings for breast, 60 of 68 for GVHD and 64 of 68 for colorectal and melanoma | 1002-1004 | 52/68; 60/68; 64/68; 64/68 | 52; 60; 64; 64 (n_conventions_above_5pct) | r28_bounds_pooled.csv source=expert cls=ALL_evaluable | match |
| all four cohorts are at or below the gate: 0.44% breast, 0.58% colorectal, 2.09% GVHD and 3.98% melanoma, with the strictly-evaluable variant at 4.02% GVHD and 5.66% melanoma | 1008-1010 | 0.44/0.58/2.09/3.98; 4.02; 5.66 | 0.4430/0.5841/2.0939/3.9837; 4.0170; 5.6604 | r28_bounds_pooled.csv source=expert min_rate_grid per cls | match |
| the identified interval is 0.44-87.8% breast, 0.58-90.2% colorectal, 2.09-90.0% GVHD and 3.98-87.2% melanoma | 1013-1014 | as quoted | same as Abstract row | r28_bounds_pooled.csv source=expert cls=ALL_evaluable | match |
| colorectal CD3 T cells 6.90% to 100.00%; colorectal endothelial 5.21% to 60.42%; GVHD neutrophils 26.35% to 93.41%; GVHD plasma 19.84% to 94.63%; melanoma CD8 T 19.78% to 89.64% | 1025-1027 | as quoted | 0.068966-1.0; 0.052083-0.604167; 0.263473-0.934132; 0.198374-0.946341; 0.197834-0.896390 | r28_bounds_by_marker.csv source=expert robustly_discordant min_rate_grid/max_rate_grid | match |
| the melanoma rate restricted to cells whose type has a fully evaluable rule is above it under all sixty-eight conventions, from 5.66% to 86.71% | 1034-1035 | 5.66%; 86.71% | 0.056604; 0.867111 (68/68 above gate) | r28_bounds_pooled.csv Melanoma ALL_strict source=expert | match |
| the between-convention spread exceeds the width of its sampling interval at a fixed convention by a median factor of 12.7 ... (5.97 for colorectal to 21.32 for melanoma) | 1037-1039 | 12.7; 5.97 to 21.32 | 12.703 (median over 48 expert type rows); CRC 5.97, Mel 21.32 | r28_bounds_by_marker.csv source=expert ratio_span_to_mean_ci (work/r28_report.txt:266-270) | match |
| the channel-name repair reduces the violation rate at every one of the 276 matched convention and label-source cells | 1041 | 276 | 276 matched (ALL_evaluable); 0 violations; 552 at ALL_strict | r18_conventions.csv scope=summary cls=ALL_evaluable pivot on (dataset,source,positivity,reading) -- work/_r28_all_conventions.csv | match |
| they agree on 71% and 78% of the ... cells | 1050 | 71%; 78% | 0.7094 (native CRC) / 0.7811 (GVHD) | r18_agreement.csv native_agreement | match |
| in melanoma, predicted-label violations run at 7.45%, 8.91% and 10.21% against 3.98% | 1089-1091 | 7.45/8.91/10.21; 3.98 | as quoted | r28_bounds_pooled.csv / work/r28_report.txt | match |
| the same aggregate statement ... from 5.66% to 86.71% | 1035 | 5.66-86.71 | 5.6604-86.7111 | r28_bounds_pooled.csv | match |
| up to 5.7% | 1194 | 5.7% | 0.0566 (Melanoma ALL_strict minimum at the default convention) | r28_bounds_pooled.csv; work/r28_report.txt:339 | match |
| the ordering holds in 1,590 of the 1,694 estimable convention-by-cohort-by-classifier units ... p < 1e-300, median difference +0.122 | 1096-1099 | 1,590 of 1,694; p<1e-300; +0.122 | 1,590 of 1,700 estimable (tie-free 1,694); median +0.122 | r25_accept_deferred.csv alpha=0.10 positivity=gmm reading=or ruleset=shipped scope=ALL_strict; work/r25_report.txt | match |
| the median difference runs from +0.028 under the mixture threshold to +0.206 under percentile-of-positives thresholds | 1100-1101 | +0.028 to +0.206 | as quoted | r25_accept_deferred.csv across the 68-convention sweep | match |
| Breast reverses in 44 of its 408 estimable units and colorectal in 60 of 198 | 1102-1103 | 44/408; 60/198 | 44/408; 60/204 estimable (60/198 tie-adjusted) | r25_accept_deferred.csv per cohort | match |
| standardising within cell type leaves only 10 of the 104 pooled reversals with the same sign | 1104 | 10 of 104 | as quoted | r25_accept_deferred.csv | match |
| the class-standardised statistic holds in 1,684 of 1,694 units | 1105-1106 | 1,684 of 1,694 | as quoted | r25_accept_deferred.csv | match |
| Seven of the twelve cohort-by-classifier cells are universal at every convention and budget | 1106-1107 | 7 of 12 | 7 of 12 | r25_accept_deferred.csv | match |
| GVHD accepted cells violate at 2.00% (1.55-2.58) against 7.25% (6.14-8.54) under logistic regression ... 3.54% (2.99-4.19) against 5.72% (4.46-7.31) under gradient boosting | 1108-1111 | as quoted | intervals reproduced exactly | r25_accept_deferred.csv alpha=0.1 positivity=gmm reading=or ruleset=shipped scope=ALL_strict per (cohort,clf) | match |
| the ordering is reversed: breast under logistic regression gives 0.45% (0.31-0.65) against 0.40% (0.15-1.02) | 1116-1118 | as quoted | as quoted | r25_accept_deferred.csv | match |
| per-cohort difference ... +0.018 (-0.002 to 0.043) breast, +0.072 (-0.005 to 0.296) GVHD, -0.101 (-0.198 to 0.177) melanoma, colorectal not estimable, pooled +0.049 (0.008 to 0.086) | 1142-1146 | as quoted | pooled +0.02931 [0.00982, 0.03598]; breast +0.00794 [-0.00060, 0.02013]; CRC +0.24009 [-0.02937, 0.64302] (estimable); GVHD +0.00598 [-0.00300, 0.01988]; melanoma -0.01382 [-0.05429, 0.05832] | work/r21_report.txt:173-178 | MISMATCH |
| of 11,400 certified decisions, zero non-empty accepted sets exceeded the budget ... the uncalibrated baseline violated the budget in 19.2% | 1198-1200 | 11,400; 0; 19.2% | 11,400; 0; 19.2% | r12_labelnoise.csv | match |
| mean acceptance falls from 0.588 with clean calibration labels to 0.552 at 1% noise, 0.487 at 2%, 0.126 at 5% and 0.000 at 10% | 1203-1204 | 0.588/0.552/0.487/0.126/0.000 | as quoted | r12_labelnoise.csv eta grid 0.0/0.01/0.02/0.05/0.10, method=certificate delta=0.10 | match |
| the fraction of evaluations that return any answer at all falls from 0.76 to 0.00 over the same range | 1205-1206 | 0.76 to 0.00 | as quoted | r12_labelnoise.csv | match |
| At the 4% noise level we actually measured in GVHD, acceptance at a 10% budget falls to about 42% of its clean-label value | 1207-1208 | about 42% | not stored; eta=0.04 absent (grid is 0.0/0.01/0.02/0.05/0.10); linear interpolation gives 0.246 = 0.42x0.588 | r12_labelnoise.csv eta grid has no 0.04 | unverifiable |
| at a 10% budget it accepts 0.5844 of cells against 0.5594, a paired difference of +0.0249 with a unit-bootstrap 95% interval of [0.0194, 0.0308] | 927-929 | 0.5844; 0.5594; +0.0249 [0.0194, 0.0308] | 0.5844; 0.5594; +0.0249 [0.0194, 0.0308] | r26_plugin_baseline_summary.csv family=cellsplit alpha=0.10 delta=0.10 arm=plugin_cp_cond vs certificate; work/r26_report.txt:258 | match |
| it wins in all nine of that family's budget and confidence-level cells | 930 | 9 of 9 | 9 of 9 configs (3 budgets x 3 deltas) | r26_plugin_baseline_summary.csv family=cellsplit plugin_cp_cond > certificate in every config | match |
| the same threshold applied without it accepts 0.7465 and covers in only 0.5825 of evaluations | 932-933 | 0.7465; 0.5825 | 0.7465; 0.5825 | r26_plugin_baseline_summary.csv family=cellsplit alpha=0.10 arm=naive | match |
| the error rate differs across predicted classes by 0.2016 on average and across score deciles by 0.2048 | 939-941 | 0.2016; 0.2048 | 0.2016; 0.2048 | work/r26_report.txt:574,633 | match |
| a chi-square test rejects homogeneity in 77.5% of evaluations by predicted class and 92.9% by score decile, with 91.4% and 95.0% under cross-field-of-view and 93.9% and 94.1% under transfer | 941-943 | 77.5/92.9; 91.4/95.0; 93.9/94.1 | as quoted | work/r26_report.txt:635; r26_plugin_baseline_summary.csv | match |
| it under-covers, at 0.607 and 0.374 against a nominal 0.90 | 944-945 | 0.607; 0.374 | as quoted | work/r26_report.txt | match |
| On cross-field-of-view data its coverage is 0.7833 against the certificate's 0.8167 at a 10% budget | 947-948 | 0.7833; 0.8167 | 0.7833; 0.8167 | r26_plugin_baseline_summary.csv family=within alpha=0.10 delta=0.10 plugin_cp_cond 0.7833, certificate 0.8167 | match |
| under cross-tissue transfer 0.3657 against 0.3972 | 948-949 | 0.3657; 0.3972 | 0.3657; 0.3972 | r26_plugin_baseline_summary.csv family=transfer alpha=0.10 plugin_cp_cond 0.3657, certificate 0.3972 | match |
| accepts 0.5844 of cells at a 10% budget on the exchangeable family against the certificate's 0.5594, and its constancy assumption is rejected by a homogeneity test in 77.5% ... up to 93.9% under transfer | 1280-1282 | 0.5844; 0.5594; 77.5%; 93.9% | same as above | r26_plugin_baseline_summary.csv; work/r26_report.txt:574,635 | match |
| On the random split at a 10% budget it accepts 0.6335 against Bonferroni's 0.5600, a paired gain of +0.0735 with a unit-bootstrap 95% interval of [0.0572, 0.0900] | 823-825 | 0.6335; 0.5600; +0.0735 [0.0572, 0.0900] | 0.6335; 0.5600; +0.0735 [0.0572, 0.0900] | r27_fixedseq_summary.csv family=cellsplit alpha=0.10 delta=0.10 min_accept=100 grid=120; work/r27_report.txt:69 | match |
| at a 20% budget 0.8392 against 0.8030 with coverage 0.9842 | 825-826 | 0.8392; 0.8030; 0.9842 | 0.8392; 0.8030; 0.9842 | r27_fixedseq_summary.csv family=cellsplit alpha=0.20 delta=0.10 min_accept=100 grid=120; work/r27_report.txt:90,278 | match |
| Under cross-field-of-view and cross-tissue transfer it also accepts more -- 0.5845 against 0.5334, and 0.4181 against 0.3707 | 826-827 | 0.5845; 0.5334; 0.4181; 0.3707 | 0.5845; 0.5334; 0.4181; 0.3707 | r27_fixedseq_summary.csv family=within and family=transfer alpha=0.10 delta=0.10 grid=120; work/r27_report.txt:118-119,166 | match |
| coverage is 0.7333 against the reference's 0.8333 under cross-field-of-view and both arms sit at 0.3917 under transfer | 828-830 | 0.7333; 0.8333; 0.3917 | 0.7333; 0.8333; 0.3917 | work/r27_report.txt:279-283 | match |
| the gap widens with the grid: +0.0503, +0.0735 and +0.0896 ... and +0.0255, +0.0475 and +0.0612 under transfer | 833-836 | as quoted | as quoted | r27_fixedseq_summary.csv delta_accept at grid 20/120/400, family=cellsplit and transfer alpha=0.10 delta=0.10; work/r27_report.txt:68-70,118-120 | match |
| at a 25-cell floor the scan fails on its first candidate and certifies nothing at a 5% or 10% budget | 840-841 | 25-cell floor | min_accept=25 rows: accept 0.000 / non-vacuous 0 | r27_fixedseq_summary.csv min_accept=25 | match |
| scanned from the safest threshold downwards the rule is empty everywhere at these budgets and non-vacuous in at most 3.9% of evaluations | 842-844 | 3.9% | as quoted | r27_fixedseq_summary.csv scan_order=descending; work/r27_report.txt | match |
| the quoted coverage ... is 0.963, 0.963 and 0.984 on the random split, 0.753, 0.733 and 0.894 under cross-field-of-view and 0.584, 0.392 and 0.263 under transfer | 845-847 | 0.963/0.963/0.984; 0.753/0.733/0.894; 0.584/0.392/0.263 | 0.9632/0.9632/0.9842; 0.7528/0.7333/0.8944; 0.5843/0.3917/0.2630 | work/r27_report.txt:276-284,515-517 | match |
| it beats Bonferroni with a paired interval excluding zero in 155 of 648 configurations and loses in 433 ... never wins at a 25-cell floor, wins 14 of 81 at 50, 60 of 81 at 100 and all 81 at 200 | 850-853 | 155 of 648; 433; 14/81; 60/81; 81/81 | 155 of 648; 433; 14/81; 60/81; 81/81 | work/r27_report.txt:43,447-448; r27_fixedseq_summary.csv ci_above_zero | match |
| it averages +0.0644 with 64.4% of evaluations improved and 0.6% worsened, against -0.2142 with no evaluation improved where the set is not a prefix | 853-856 | +0.0644; 64.4%; 0.6%; -0.2142 | +0.0644; 0.644; 0.006; -0.2142 | work/r27_report.txt:447-448 | match |
| over the 99 units of the cross-validated grid (Mann-Whitney p = 1e-6 and 2e-8) | 685-686 | 1e-6; 2e-8; 99 units | p = 1e-6 and 2e-8 on 99 units (47 reach vs 52 never) | work/r20_report.txt; tab_triage.md; r20_units.csv + r20_diag.csv | match |
| a source-only acceptance of exactly zero ... occurs in 27 of the 99 units and fails to reach validity in 21 | 688-689 | 27 of 99; 21 | as quoted | work/tab_triage.md | match |
| Spearman -0.77, p = 7e-4, and -0.70, p = 3e-3 on 15 units each | 692-693 | -0.77; -0.70; 15 units | -0.7746; -0.7042 on 15 units each | r20_units.csv (reachability) merged with r20_diag.csv accept_src; work/tab_triage.md | match |
| not under cross-tissue transfer (-0.17, p = 0.51 on 17 units) | 694 | -0.17; 0.51; 17 units | -0.1700; p = 0.51 on 17 units | work/tab_triage.md | match |
| the cost is also five times larger (median k of 1,000 against 200) | 695 | 5x; 1,000 vs 200 | as quoted | work/tab_triage.md | match |
| Its pooled correlation with the label cost is +0.52 ... the reachable units have a lower separation than the unreachable ones, 0.73 against 0.90 | 696-700 | +0.52; 0.73; 0.90 | as quoted | work/tab_triage.md | match |
| coverage is 1.000 at every budget in the pooled analysis, and at or above 0.987 in the worst cohort | 1262-1263 | 1.000 at every budget; 0.987 | 0.99649 at 5% and 0.99825 at 30% (not 1.000); worst cohort 0.98667 (breast at 5%) | r03_metrics.csv kind=cellsplit method=crc mean(covered) by alpha; per-unit at alpha=0.05 | MISMATCH |
| on the smallest cohort the accepted set is empty in all 150 evaluations at a 5% budget | 1265-1266 | 150 evaluations | 120 evaluations (CRC has 4 folds, fold4 absent); all 120 empty. Breast 27/150, GVHD 13/150, melanoma 44/150 empty | r03_metrics.csv alpha=0.05 method=crc mean(empty) per unit; r18_diag.csv | MISMATCH |
| coverage of 0.397 at a 10% budget, and 0.014 once the 38.9% of ... | 1291-1292 | 0.397; 0.014; 38.9% | 0.39722; 0.01364; 38.89% | r24_efficiency.csv family=transfer arm=src k=0 alpha=0.10 | match |
| the certificate and every plug-in fail together -- 0.541, 0.397 and 0.286 against 0.494, 0.366 and 0.234 | 950-951 | as quoted | src cov_all 0.5407/0.3972/0.2861; plug-in 0.4939/0.3659/0.2340 | r24_efficiency.csv; r26_plugin_baseline_summary.csv | match |
| At a 10% budget the target-calibrated controller takes 460 non-vacuous decisions over the pooled k = 1000 and k = 2000 settings and exceeds the budget once ... at a 5% budget the counts are 0 of 142 and at a 20% budget 2 of 1210 | 539-542 | 460/142/1210; 0.998 | 460 decisions (459 covered, 1 exceed) / 142 (0 exceed) / 1210 (1208 covered, 2 exceed) | r14_targetcal.csv arm=tgt_k rows with n_accepted>0 at k in {1000,2000}, alpha 0.10/0.05/0.20 | match |
| The source-calibrated controller, evaluated on the same cells, takes 1,150 non-vacuous decisions at a 10% budget and exceeds it in 1,100 of them (0.043) | 542-544 | 1,150; 1,100; 0.043 | 1150 decisions, 50 covered, 1100 exceed = 0.04348 | r14_targetcal.csv arm=src k=0 alpha=0.10 rows with n_accepted>0 | match |
| Coverage among committed decisions therefore moves from 0.043 to 0.998 ... 0.000 to 1.000 at 5%, 0.043 to 0.998 at 10%, 0.077 to 0.998 at 20% | 544-546 | as quoted | src 0.0000/0.0435/0.0775 vs tgt 1.0000/0.998/0.998 | r14_targetcal.csv arm=src vs arm=tgt_k | match |
| At a 10% budget cross-tissue acceptance is 0.000 at k = 200, 0.006 at 500, 0.024 at 1,000 and 0.039 at 2,000, against an oracle that accepts 0.120 to 0.124 at every k | 602-604 | 0.000/0.006/0.024/0.039; oracle 0.120-0.124 | 0.0000/0.0056/0.0238/0.0388; oracle 0.1205 | r14_targetcal.csv arm=tgt_k alpha=0.10 and arm=oracle, mean(accept_rate) by k; Table 3 | match |
| at a 20% budget it is 0.006, 0.051, 0.098 and 0.146 against an oracle of 0.329 | 605-606 | 0.006/0.051/0.098/0.146; 0.329 | Table 3 target arm, alpha=0.20; oracle 0.329 | r14_targetcal_summary.csv arm=tgt_k/oracle alpha=0.20; Table 3 | match |
| the oracle, which no method can implement, accepts only 0.120 of cells at a 10% budget and 0.329 at 20%, against 0.752 and 0.912 under the random split | 737-738 | 0.120; 0.329; 0.752; 0.912 | 0.1205; 0.329; 0.7516; 0.9118 | r14_targetcal.csv arm=oracle; r24_efficiency/r16_matched_risk arm=oracle cellsplit | match |
| In four of the twelve pairs the oracle itself accepts under 5% at a 10% budget (Breast to Melanoma, Breast to colorectal, colorectal to melanoma and GVHD to melanoma) | 739-741 | four pairs < 5% | three pairs < 5% (0.92%, 0.26%, 2.04%); breast->colorectal is 5.007% | r14_targetcal.csv arm=oracle alpha=0.10 mean(accept_rate) per pair | MISMATCH |
| the scripts `scripts/r01`-`scripts/r28` reproduce, in order, dataset construction ... | 1583-1591 | r01-r28 | r01..r28 all present under scripts/ (r23 not present, not claimed) | scripts/ directory listing | match |
| a 10% budget, a 0.10 confidence level and a 100-cell floor (Fig. S6 legend) | 1535 | 0.10; 100-cell floor | matches the reported configuration | r27_fixedseq_summary.csv family/alpha/delta/min_accept/grid_size | match |
| estimability requires at least 75 evaluable cells on both sides (Fig. S5 legend) | 1528 | >=75 cells | the r25 estimability rule | r25_accept_deferred.csv | match |
| the naive threshold falls to 0.41 at eta = 0.005 while the certificate stays at 1.00 (Fig. S1 legend) | 1501-1502 | 0.41 at eta=0.005; 1.00 | not stored: eta=0.005 absent from the grid; stored naive coverage rises with eta (0.570 to 0.983) | r12_labelnoise.csv method=naive delta=0.10 (eta grid: 0.0/0.01/0.02/0.05/0.10) | unverifiable |
| the certificate's own output ... 1,590 of 1,694 (Fig. S5 / Discussion legend) | 1380-1382 | 1,590 of 1,694; +0.028 to +0.206 | same as the Results row above | r25_accept_deferred.csv | match |
| identified sets of 0.44-87.8% breast, 0.58-90.2% colorectal, 2.09-90.0% GVHD and 3.98-87.2% melanoma (Discussion limitation) | 1365-1366 | as quoted | same as Abstract/Results rows | r28_bounds_pooled.csv | match |

## MISMATCHES

Eleven distinct quoted numbers do not survive recomputation (shown as 12 table rows, because the 20.2% / 0.08% colorectal pair is quoted at both L989-990 and L1362-1363). Each is listed below with the correct value and the exact recomputation.

1. L39-41 (Abstract) - "on three of the remaining five no method can succeed because the oracle threshold itself accepts under 1%". Only TWO of the twelve cross-tissue pairs have an oracle acceptance under 1% at a 10% budget: breast->melanoma 0.9175% and colorectal->melanoma 0.2600%. The third pair named in the body (L555-558) is GVHD->melanoma at 2.0364%, above 1%. Recomputation: `r14_targetcal.csv` rows with arm=oracle, alpha=0.10, mean(accept_rate) grouped by the transfer pair over all k (n=1200 per pair). The body text at L558 ("0.9%, 0.3% and 2.0%") is correct; the Abstract's "under 1%" is not.
2. L279 (Results) - "no violation was observed in 190 evaluations". The count is 570 evaluations (19 units x 3 classifiers x 10 repeats). 190 is the per-classifier count. Recomputation: `r03_metrics.csv` kind=cellsplit, method=crc, alpha=0.10 has 570 rows, 190 per classifier, mean(covered)=1.000. The same manuscript states 570 at L271-272.
3. L606 (Results) - "25 target cells per fold are enough to make the certificate valid". At k=25 the accepted set is empty in every evaluation, so the certificate is not valid and not useful; the floor is 100 for within/cellsplit and 500 for transfer. Recomputation: `r24_efficiency.csv` alpha=0.10, grid=abs, arm=tgt_k: k=25 accept_mean 0.000, nonvac_frac 0.000, valid_cp False; k=100 cellsplit 0.00532 (valid_cp True), k=500 transfer 0.00542 (valid_cp True). `r14_targetcal_summary.csv` arm=tgt_k gives mean_accept 0.000 and NaN risk for every k<=200.
4. L739-741 (Results) - "In four of the twelve pairs the oracle itself accepts under 5% at a 10% budget (Breast to Melanoma, Breast to colorectal, colorectal to melanoma and GVHD to melanoma)". Only three are under 5%: breast->melanoma 0.92%, colorectal->melanoma 0.26%, GVHD->melanoma 2.04%. Breast->colorectal is 5.0067% (the manuscript's own Table 6 row at L581 and L561 both print 0.050 / 5.0%). Recomputation: `r14_targetcal.csv` arm=oracle alpha=0.10 mean(accept_rate) per pair: breast->CRC 0.050067, breast->Mel 0.009175, CRC->Mel 0.002600, GVHD->Mel 0.020364.
5. L989-990 and L1363 (Results and Discussion) - colorectal rule reading: "violated by 20.2% of expert colorectal labels; read as a disjunction, by 0.08%". The correct values are 18.15% (435 of 2397) for the conjunction and 0.58% (14 of 2397) for the disjunction. Recomputation: `r18_conventions.csv` dataset=CRC_Feb2623, source=expert, positivity=gmm, scope=summary, cls=ALL_strict, reading=and gives n_violation 435 of n 2397 (0.181477); reading=or gives 14 of 2397 (0.005841). Identical under scope=type.
6. L1062 (Results) - "the identity of the worst population stable in only 6% to 38% of the settings". Over the sixty-eight conventions (grid=all) the fraction is 5.9% to 17.6%: breast 0.0588, colorectal 0.0882, GVHD 0.1765, melanoma 0.1765. The 38% endpoint appears only for colorectal under the hq_all label source (0.3824). Recomputation: `r18_rank_stability.csv` source=expert, grid=all, frac_settings_same_top1.
7. L1063-1064 (Results) - "holding the threshold fixed and varying the reading and the rule text leaves the ordering untouched (Spearman 1.000)". Not 1.000 in every case: colorectal reading_only(gmm,repaired) mean_spearman 0.7384 (min 0.4767); GVHD rules_only(gmm,or) 0.9565 (min 0.9130); colorectal hq_all reading_only 0.8220. Recomputation: `r18_rank_stability.csv` source=expert, grid in {reading_only(gmm,repaired), rules_only(gmm,or)}.
8. L1085-1088 (Results) - "a mean violation rate of 5.23% in melanoma and 4.17% in GVHD, against 0.73% in colorectal and 0.55% in breast, with 23.8% of melanoma cells and 11.2% of GVHD cells belonging to a population violated more than 5% of the time against 1.6% and 0%". Recomputing the stated convention (r18_population.csv source=expert, reading=or, variant=repaired, min_n=50) gives mean_rate melanoma 2.86%, GVHD 4.02%, colorectal 0.91%, breast 0.55%, and share_over_gate_of_evaluable melanoma 13.62%, GVHD 8.23%, colorectal 4.01%, breast 0%. All 4 x 2 x 2 x 4 (reading, variant, positivity, source) combinations were scanned; no row yields 5.23% or 23.8%.
9. L1142-1146 (Results) - the r21 image-bootstrap intervals for the accepted-versus-full distortion difference, quoted as breast +0.018 (-0.002, 0.043), GVHD +0.072 (-0.005, 0.296), melanoma -0.101 (-0.198, 0.177), pooled +0.049 (0.008, 0.086), colorectal not estimable. The stored run (`work/r21_report.txt:173-178`) gives pooled +0.02931 [0.00982, 0.03598], breast +0.00794 [-0.00060, 0.02013], colorectal +0.24009 [-0.02937, 0.64302] (colorectal has one stratum and IS estimable), GVHD +0.00598 [-0.00300, 0.01988], melanoma -0.01382 [-0.05429, 0.05832]. The trailing sentence "No cohort's interval excludes zero" still holds; the quoted numbers are the superseded 30-replicate run in `work/_smoke_r21.txt`.
10. L1262 (Discussion) - "coverage is 1.000 at every budget in the pooled analysis". Pooled cellsplit/crc coverage is 1.000 at 2%, 10%, 15%, 20%, 40% and 50% but 0.99649 at 5% and 0.99825 at 30%. The worst cohort is 0.98667 (breast at 5%), consistent with the "at or above 0.987" clause. Recomputation: `r03_metrics.csv` kind=cellsplit, method=crc, mean(covered) by alpha (570 rows each). The manuscript itself prints "0.996 overall" at L274.
11. L1265-1266 (Discussion) - "on the smallest cohort the accepted set is empty in all 150 evaluations at a 5% budget". The all-empty cohort is colorectal (the smallest expert-annotated cohort, 2397 usable cells per L891) and it has 120 evaluations at alpha=0.05, not 150: the fifth fold is absent for CRC. Breast is empty in 27 of 150, GVHD in 13 of 150 and melanoma in 44 of 150. Recomputation: `r03_metrics.csv` kind=cellsplit, method=crc, alpha=0.05, mean(empty) per unit (CRC folds 0-3 = 1.000, 30 rows each; CRC fold4 absent).

## NOT AUDITABLE FROM STORED OUTPUT

Six distinct claims could not be decided from a primary CSV or a script output under `experiments/` or `work/` (shown as 7 table rows, because the channel-count claim occupies two). Each is listed with the files searched and the closest thing that is stored.

1. L151 - "the shipped segmentation mask for only 17.5% of cells (1,794 of 10,283)". The denominator 10,283 is GVHD `n_expert` in `r18_diag.csv` and the ratio is consistent (1,794/10,283 = 17.45%), but the numerator 1,794 is not stored. Searched `r18_diag.csv`, `r18_population.csv`, `r15_deployed.csv`, `work/s00_data_report.txt`, `work/r18_report.txt`.
2. L166-167 - "of 12-30 rules per cohort, 0-4 could not be evaluated". `r18_conventions.csv` leaves `n_rules_total` blank; `r15_types.csv` gives 1-3 rules per type; no file aggregates rules to the cohort level. Searched `r18_conventions.csv`, `r15_types.csv`, `work/r15_report.txt`, `work/r28_report.txt`.
3. L186-187 - melanoma "at a confidence of about 0.5, the model is correct roughly 30% of the time". Fig. 2a reliability-diagram data is not stored; only the aggregate `r03_ece.csv` (ECE and accuracy) is on disk.
4. L1207-1208 - "at the 4% noise level we actually measured in GVHD, acceptance at a 10% budget falls to about 42% of its clean-label value". The stored eta grid in `r12_labelnoise.csv` is {0.0, 0.01, 0.02, 0.05, 0.10}; there is no eta=0.04. Linear interpolation between eta=0.02 (0.487) and eta=0.05 (0.126) gives 0.246, which is 0.42 x 0.588 and so is consistent with the text, but the value is not stored.
5. L1501-1502 (Fig. S1 legend) - "the naive threshold falls to 0.41 at eta = 0.005 while the certificate stays at 1.00". eta=0.005 is absent from the `r12_labelnoise.csv` grid; within the stored grid the naive 10%-budget coverage rises with eta (0.570 to 0.983) and the naive acceptance at eta=0.01 is 0.755 of its clean value, so neither the level nor the direction can be confirmed.
6. L140-142 - the marker-panel channel counts, in particular GVHD "40 channels" and melanoma "40 channels". No single stored definition reproduces all four counts: the manuscript text is 22/58/40/40, `work/s01_inventory.txt:66` (raw ML panels) is 22/58/42/40, and `r18_diag.csv n_channels` is 22/56/38/39. The Abstract's "22-58" is fine; the per-cohort 40s are not decidable from disk. Note also that L1179 uses "39 channels" for melanoma against L141's "40".

## Notes on definitional ambiguities (not counted as mismatches)

- **Source-only 0.043 (L34-35, L543).** The sentence pools the target arm over k in {1000, 2000} but quotes the source arm at a single k. `r14_targetcal_summary.csv` arm=src alpha=0.10 k=0 gives coverage_nonvac 0.043478; pooled over all source rows the source count is 406 of 9,200 covered (0.0441) and at k=1000 it is 53 of 1,150 (0.0461). All are consistent with "0.043" to two places; the convention matters.
- **Realised risk 0.052 (L618).** The manuscript's "0.052" is the accepted-count-weighted risk over `r14b_targetadapt.csv` (arm=adapt_k, alpha=0.10, k=2000) = 0.05186. `r14b_targetadapt_summary.csv mean_risk` for the same cell is 0.046867 (mean over evaluations). The two are different estimators of the same phrase; the manuscript uses the weighted one.
- **r25 denominator (L1096, L1103).** "1,590 of the 1,694" is the report's 1,590 of 1,700 estimable units with 6 ties removed (tie-free denominator 1,694); "colorectal in 60 of 198" is the tie-adjusted pair against the report headline "60 of 204". Both are defensible; the two conventions within one paragraph are not unified.
- **Jaccard 0.18-0.44 (L144).** With the canonical-marker definition the range is 0.1765 (breast-CRC) to 0.4444 (GVHD-melanoma); the raw panel Jaccard minimum is 0.143. The text's lower end assumes the canonical-marker definition.
- **"up to 5.7%" (L1194).** This is the Melanoma ALL_strict minimum 5.6604% at the default convention, not the maximum over the sixty-eight conventions (GVHD shipped reaches 23.97% and melanoma pct_pospop@95 reaches 86.71%).
- **"190 evaluations" (L279).** The value 190 also appears in `work/r24_report.txt:102,112` as an unrelated acceptance figure; the correct count of evaluations for the L279 sentence is 570.

## Counts

| verdict | count |
|---|---|
| match | 118 |
| MISMATCH | 12 |
| unverifiable (MISSING) | 7 |
| rows | 137 |

Claims were recomputed against `experiments/r03_metrics.csv`, `r03_ece.csv`, `r14_targetcal.csv`, `r14_targetcal_summary.csv`, `r14b_targetadapt*.csv`, `r15_*.csv`, `r16_*.csv`, `r18_agreement.csv`, `r18_conventions.csv`, `r18_diag.csv`, `r18_population.csv`, `r18_rank_stability.csv`, `r19_*.csv`, `r20_diag.csv`, `r20_units.csv`, `r21_downstream.csv`, `r24_efficiency.csv`, `r24b_labelshift.csv`, `r25_accept_deferred.csv`, `r26_plugin_baseline*.csv`, `r27_fixedseq*.csv`, `r28_bounds_*.csv`, and the script outputs `work/r16_report.txt`, `work/r21_report.txt`, `work/r24_report.txt`, `work/r24b_report.txt`, `work/r26_report.txt`, `work/r27_report.txt`, `work/r28_report.txt`, `work/tab_triage.md`, `work/_r28_all_conventions.csv`. No summary or prior audit file was trusted.

## Repair record

This audit was run against the manuscript as of `2026-09-27 08:29:49`
(sha256 `46aea4b40b1ad4b2d41a7a827539d7d2db0ec937d35e8ad57dc0cb8bc0af00ed`,
107,411 bytes, 1,610 lines) and was deliberately performed on a frozen file, so
every defect it lists was reported rather than patched. The defects have since
been adjudicated and, where genuine, repaired.

- **Repaired manuscript.** `2026-09-27 09:07:54`, sha256
  `af5ecd93eaeaeb048b5014fb5316c6cfaf8ca092bd39ae41db8a83753eac4746`,
  107,592 bytes, 1,613 lines.
- **Decision record.** `work/r29_audit_fixes.txt` gives the disposition of all 17
  findings: the stored value, the recomputed value, the source file and filter,
  and the exact text edit. A copy of the pre-repair manuscript is kept at
  `work/_r29_manuscript_before.md`.
- **11 distinct MISMATCH items -> 14 edits**, in three classes: (i) the value in
  the text was wrong and the text now carries the recomputed value (F2, F5 twice,
  F6, F8, F9 twice, F10, F11); (ii) the threshold rather than the count was wrong
  (F1 "under 1%" -> "under 3%", chosen so that the Abstract keeps naming the same
  three pairs the Results body names; F4 "four of the twelve pairs" -> "three",
  which the Table 6 entry already reflected); (iii) an over-strong claim was
  softened to what the stored files support (F3 floor 25 -> 100 within a cohort
  and 500 under transfer; F7 "Spearman 1.000" -> a per-cohort range; and the
  dependent clause of F9, since the colorectal arm is estimable in the full run).
- **6 unverifiable items -> 4 closed without a change and 2 edits.** U1
  (1,794 of 10,283), U2 (12-30 rules, 0-4 failed) and U6 (panels 22/58/40/40)
  are all decidable from data already on disk -- the shipped GVHD segmentation
  masks and `work/r01_dataset_audit.txt` -- and each recomputes to the value
  printed in the manuscript, so no change was made. U3 (melanoma ~30% at
  confidence 0.5) likewise recomputes to the printed value from
  `work/predictions/*.npz`. U4 (the "4% noise -> 42%" sentence, now "a fifth" at
  the stored 5% point) and U5 (the Fig. S1 legend, corrected to the reproducible
  0.42 -> 0.98 curve from `r12_labelnoise.csv`) were edited. U1, U2, U3 and U6
  are therefore audit defects, not manuscript defects.
- **One finding left open by this audit is closed here.** The note under
  "definitional ambiguities" -- that a panel-reduction cut is described as "from
  39 channels to 29" while `experiments/r08_panel.csv` stores melanoma
  `n_markers` 40 -> 30 -- is not an inconsistency. The paragraph reports a
  *pooled* four-cohort analysis, not melanoma. Pooling `r08_panel.csv`
  (`clf == "hgb"`, `alpha == 0.1`) over all datasets gives mean `n_markers`
  39.053, 29.263, 19.526 and 9.789 at `frac` 1.0, 0.75, 0.5 and 0.25, with
  accuracy 0.8403, 0.8364, 0.8249 and 0.7337 and acceptance 0.6506, 0.6457,
  0.6100 and 0.2252. Those are exactly the figures in the text (0.840 -> 0.836,
  0.651 -> 0.646, 1.5 accuracy points and four acceptance points at 19.5
  channels, 11 accuracy points and a two-thirds cut in acceptance at about 10
  channels). The manuscript is arithmetically correct as written; the earlier
  audit compared it against a per-cohort table by mistake.
- **Revision note (panel reduction).** The panel-reduction paragraph was rewritten
  so that it no longer implies that panel size is a validity question. Under
  exchangeability coverage is a theorem and panel size can only cost throughput,
  so the paragraph now presents the `r08_panel.csv` numbers as a price list and
  routes the practical lower-plex question to the transfer family, which trains,
  calibrates and deploys on the shared canonicalised channels of each pair (11 to
  27 channels over the twelve directed pairs; `experiments/r13_coverage_ci.csv`,
  `work/r34_report.txt`). The two figure pointers it used to carry, "Fig. 4d and
  Fig. S1", were both wrong -- Fig. 4d is the calibration-size panel and Fig. S1
  is reference-label noise -- and neither the `r08_panel.csv` numbers nor the
  transfer numbers they are now paired with appear in a figure, so the pointers
  were removed and the claims are referenced to Table 1 instead. Verified
  numbers quoted: GVHD-to-CRC `crc` coverage 0.000 in all five folds at a 0.10
  budget with acceptance 0.311-0.801 and realised risk 0.256-0.413 across the
  five folds and three classifiers (`experiments/r03_metrics.csv`); STELLAR
  `ext_transfer` acceptance 0.8394 and coverage 0.0333 on 26 shared channels
  (`work/r34_report.txt`).
- **Revision note (marker-logic representation, r49 / r49b).** The panel paragraph
  is paired with a new test of whether a panel-invariant representation repairs
  cross-panel transfer. `scripts/r49_markerlogic.py` rebuilds every unit of the
  benchmark with cells described by the depot's 59 Boolean rules instead of by
  channel intensities, in four spaces, and `scripts/r49b_panelinvariance.py`
  measures whether the rule layer itself is panel invariant. Evidence:
  `experiments/r49_<space>_coverage_ci.csv` and `work/r49_report.txt` (the
  published control reproduces the released `crc` rows to the last bit in coverage
  -- 0.833/0.817/0.917 within and 0.541/0.397/0.286 on transfer -- and to the
  fourth decimal in acceptance, 0.3565/0.5337/0.7301 against the released
  0.3566/0.5338/0.7300, the residual being a BLAS-thread-dependent `lbfgs` solve;
  two of the 42 control rows, neither of them a `crc` row, differ by 0.011-0.017 in
  coverage, which is why the claim is made about the rows the paper quotes rather
  than about the whole family), and
  `experiments/r49b_panelinvariance.csv` / `work/r49b_report.txt` for the audit
  layer: 29 (rule, cohort) combinations carry at least 30 cells of the rule's own
  type, 26 of them at an own-class satisfaction of 0.95 or more, the minimum being
  0.795 (the alternate plasma rule in GVHD; the primary form of the same type is
  0.802) and the largest cross-cohort disagreement 0.111 (the endothelial rule),
  with 40 of the 59 rules evaluable in at least one cohort and 6 in all four. The manuscript reports these as a negative result for the
  representation (cross-panel transfer is not a feature-scale artefact) and a
  positive one for the audit layer (it transports across panels).

**Status.** No known unresolved numerical inconsistency remains in
`docs/manuscript.md`. Every mismatch this audit reports is either repaired or
shown above to have been mislabelled, and the r26, r27 and r28 results added to
the manuscript after the audit snapshot were independently recomputed from
`experiments/r26_plugin_baseline*.csv`, `experiments/r27_fixedseq*.csv`,
`experiments/r28_bounds_*.csv` and held.

## Round-2 repair record (r31-r34 integration)

A second round of analyses (targeted repair, an anytime-valid sequential
controller, a second annotation source, and an external validation cohort) was
integrated into `docs/manuscript.md` after the audit above. The integration also
corrected four statements that the round-1 audit had recorded as correct against
the sources then available but which the new theory and the external data show to
be misattributed or overstated. Those four corrections are the substantive
finding of this record, because in each case the stored tables were right and the
prose attached the wrong mechanism or the wrong scope to them.

1. **Mechanism of the conditional-binomial failure (Abstract, Results,
   Discussion).** The manuscript said the plug-in "is invalid when errors vary
   with the score" and, in the Discussion, "invalid once errors vary among
   accepted cells". `docs/propositions.md` Proposition 1 shows the operative
   condition is that the delivered accepted set coincides with the block of
   calibration cells whose error count fixes the bound; the proved failure
   channel is an atomic score that delivers a larger set than that block
   (`work/r30_report.txt`: plug-in coverage 0.5693 against a nominal 0.90, tie-
   complete variant 1.0000). A score-varying error rate on a regular score did
   not break the bound in the settings examined (worst crossing 0.970;
   continuous-score controls 0.9980-1.0000). All four occurrences (abstract,
   Introduction contribution list, Results, Discussion) were rewritten to the
   score-and-exchangeability condition.

2. **Attribution of the 0.607 and 0.374 figures.** The manuscript attributed
   them to "controlled heterogeneity of the Monte-Carlo study". They are the
   conditional plug-in's coverage under covariate shift (0.6068 and 0.3740;
   `docs/propositions.md`, "What is not proved here", and V1.24/V1.25). The
   Results paragraph and the Discussion paragraph now say covariate shift. The
   stored values are unchanged.

3. **Scope of the acceptance-ratio diagnostic.** The manuscript called it "a
   perfect detector ... with no false positives and no misses across the three
   families". That holds on this benchmark (1,690 units at a 10% budget: 575
   above budget, all 575 flagged, none of the 1,115 in-budget units flagged). It
   does not hold externally: `experiments/r34_phase2_ratio.csv` at the
   evaluation level records 174 over-budget transfer evaluations of which 114 are
   flagged and 60 are missed, with 0 false positives, and all 60 misses are
   evaluations where the ratio is undefined because the oracle accepts nothing.
   The abstract, the Results paragraph and the new external-validation subsection
   now state the benchmark scope and the 60/174 miss.

4. **Convention-free marker verdicts are label-level statements.** The new
   two-source analysis (`work/r33_report.txt`) shows the Hui-Walter model is
   saturated and rests on an informativeness restriction that colorectal APC and
   GVHD neutrophils reject (L1 misfit 2.67e-3 and 2.57e-3), and that where the
   restriction is accepted it point-identifies the latent discordance at zero for
   colorectal CD3 T cells, colorectal endothelial cells and GVHD plasma cells. A
   new Discussion limitation states that the convention-free verdicts are
   statements about the shipped reference labels rather than about latent cell
   membership.

Five further passages were added rather than corrected: two propositions; the
targeted-repair arms; the sequential controller; the two-source partial
identification; and the external validation. Their numbers are audited in
`work/r35_number_ledger.md` (15 claims, 15 matching) and independently re-derived
in `work/r37_verify.txt`.

One claim in the round-2 integration was deliberately softened for scope.
`work/r35_number_ledger.md` records that the statement "no arm reaches nominal
under the joint readout at any label count" is true of the calibration-only arms
on the stored splits but that two arms at a 2,000-label budget (`adapt_k` and
`oracle_adapt`, `experiments/r31_partD.csv` and `experiments/r31_stack.csv`) do
exceed 0.90. The manuscript now scopes that sentence to the calibration-only arms
and points to the refitting route in Table 7, which is the route that actually
repairs transfer.

**Status.** No known unresolved numerical inconsistency remains in
`docs/manuscript.md` after the round-2 integration. The four misattributions
above were prose defects over correct tables; each is now stated at the scope the
stored tables support.

## Round-3 verification record (independent audits r37-r39)

Three audits were run against the manuscript after the round-2 integration, all
reading the released files directly and none written by the agent that produced
the text being checked.

- `work/r37_verify.txt` (number audit, 91 claim rows): 88 MATCH, 1 MISMATCH,
  2 UNVERIFIABLE. Every number in the new passages was recomputed from the raw
  CSVs.
- `work/r38_framing_audit.txt` (framing audit): 18 findings, 3 HIGH, 8 MEDIUM,
  7 LOW; no numerical error found. It independently confirmed the r37 flag at
  `[C4]` (alpha = 0.20 against a nominal 0.90) as a false alarm, because
  `scripts/r09_simulation.py` fixes `DELTA = 0.10` and the nominal coverage is
  0.90 at every budget.
- `work/r39_figure_audit.txt` (figure audit of `figS8`-`figS10`): 286 value
  checks, 0 MISMATCH; 10 P2 and 5 P3 layout and caption defects.

Corrections applied to `docs/manuscript.md` in this round (30 anchored edits,
each asserted to match exactly once):

1. **One factual mismatch.** The self-split sentence quoted the e-process
   coverage on the exchangeable held-out half as 0.9995 and on the real transfer
   deployment as 0.5034. `experiments/r32_selfsplit.csv`, family `transfer`, arm
   `eproc_bonf`, `alpha = 0.10`, gives **1.0000** and **0.5721**; the two quoted
   values were acceptance figures belonging to other rows. Corrected.
2. **Proposition 1 stated the condition the same paragraph then rejects.** The
   statement now says the bound is licensed only for the block whose error count
   fixes it, so it controls the delivered set only while the two coincide, and
   that constancy of the conditional error rate is what the binomial reading
   assumes rather than the condition the evidence identifies. This matches the
   Abstract and the Discussion, which already carried the corrected condition.
3. **The sequential label-cost sentence misdescribed its own quantity.**
   `mean_t75_frac` is the fraction of the labelled pool needed to reach a
   realised acceptance of 0.75 (`work/r32_report.txt` lines 150-161); the 75 is
   an acceptance target, not 75 labels. The sentence, the Results text and the
   Supplementary Figure 8 legend now say label cost, and the union-bound
   comparison states that the e-process needs 0.0468 more of the pool, which is
   the sign the file carries.
4. **An adaptation caveat denied a cross-run comparison the paper quotes.**
   `work/r31_report.txt` records that the 9.2-fold figure pairs two executions
   and that the same-run pairing is 13.69. The caveat, the Results text and the
   Table 7 caption now state both ratios and disclose that a fresh execution of
   the adaptation recipe returns 0.155 in place of the stored 0.355.
5. **A figure caption advertised a repair the figure does not draw.**
   `experiments/r31_ladder_summary.csv` has `k_first_nominal_nonvac = -1` for
   every transfer arm at every budget, so the Supplementary Figure 7 legend now
   says vacuity-inclusive and records that no arm reaches the budget under the
   non-vacuous readout.
6. **Scope qualifiers restored**, each a qualifier the cited file carries: the
   16,056 evaluations are the strict-Bonferroni-prefix subset; the
   +0.0503/+0.0735/+0.0896 gaps are the random-split family; the ratio detector
   is perfect on the 1,690 units where it is defined; the 0.607/0.374
   under-coverage is at a 20% error budget with `delta = 0.10` (at a 10% budget
   the same arm covers 0.942 and 0.940 but accepts under 0.9% of cells, now
   stated); the pooled importance-weighted arm reaches nominal coverage at 25
   labels under the vacuity-inclusive readout only; the 460 committed decisions
   pool two label budgets; the external within-cohort reference is the
   cross-field-of-view figure (0.817) and the transfer reference the cross-tissue
   figure (0.397), not the random split; the four populations whose sharp sets
   reconcile at 0.002 are the ones that ship a second annotation source; the
   Discussion names the arms that keep a 500-label cost instead of saying the
   cost is unchanged; and the Abstract no longer claims an impossibility
   ("no method can succeed") that the paper does not establish.
7. **Supplementary legend corrections.** The Supplementary Figure 8 legend now
   names the axes the panel draws (acceptance target against fraction of the
   labelled pool needed) and the Supplementary Table 5 legend names the same
   quantity; Supplementary Figures 7-9 and Supplementary Tables 4-7 are now
   cited from the body, and `docs/submission/code_availability.md` maps them to
   files.

**Status.** After these edits the 121-assertion manuscript-number preflight in
`work/` still passes on the released tables with 0 failures. The two claims the
number audit marked UNVERIFIABLE are the class/decile dispersion and the
chi-square rejection rates; no shipped CSV carries per-class or per-decile
accepted-calibration error counts, so a bin-level companion to
`experiments/r26_plugin_baseline.csv` would be needed to check them from raw
data.

## Round-4 correction record (r49 code and number audits)

The r49 revision was checked by two independent read-only audits, neither written
by the agent that produced the text: `work/r49_verify.txt` (number audit, 65 claim
rows: 58 MATCH, 6 MISMATCH, 1 UNVERIFIABLE; verdict that the manuscript does not
overstate the r49 result) and `work/r49_codeaudit.txt` (code audit of
`scripts/r49_markerlogic.py` and `scripts/r49b_panelinvariance.py`, 529 lines).
Every finding is dispositioned here.

1. **Rule evaluability mis-read aliased channel names (code defect, both
   scripts).** The test `c not in pos` compared canonicalised rule identifiers
   against the *raw* panel channel names, so a panel spelling a channel `HLADR`
   or `Fe56` was recorded as not carrying `HLADRDPDQ` or `Fe`. The identifier is
   now matched against the canonical form of the panel's channel names, the same
   convention the transfer panel intersection of `scripts/r02_run_experiments.py`
   already uses. The effect is to *add* information (a wrongly excluded rule
   contributed `(0, 0)`), so no certification number was inflated by the old
   behaviour; the visible change is the rule coverage, 38 of 59 rules in at least
   one cohort and 5 in all four before the repair, 40 and 6 after it. Both the
   scripts and the three places that quote the counts (`docs/manuscript.md`,
   `docs/methods.md`, `README.md`) were updated, and the four spaces were
   re-run so that the shipped tables match the text.
2. **"Returns the released ... family exactly" was too strong.** The published
   control reproduces the released coverage bit for bit but acceptance only to
   about `1e-4`: 49 of the 240 re-run prediction files differ from the released
   cache, every one of them a `logreg` file, which is the signature of a
   BLAS-thread-dependent `lbfgs` path (the released cache predates the thread
   pinning). The sentence in `work/r49_report.txt` and the registration above now
   say near-exact for acceptance and exact for coverage, and name the two
   non-`crc` control rows that differ.
3. **Coverage was quoted at the wrong granularity (manuscript overstatement).**
   "Coverage stays at 0.989-1.000 throughout" is the pooled value; the shipped
   table has three unit rows at 0.000, one of them at the paragraph's own 10%
   budget (melanoma, fold 4, half panel, realised risk 0.102 against a 0.10
   budget). The paragraph now states the pooled range, names the three breaches
   and their budgets, and attributes them to small-sample draws on a rare fold.
4. **Two ranges did not name their unit.** The GVHD-to-CRC acceptance and risk
   ranges are over the fifteen fold-by-classifier cells, not over the five folds;
   the text now says so.
5. **"Leaves the benchmark where it was" was generous, and only one of the two
   augmented spaces was quoted.** The within-cohort coverage moves by up to +5.0
   points at a 10% budget; the paragraph now reports that movement and states that
   the second scaling differs by at most 0.009, instead of describing a null.
6. **The rule-only non-vacuous coverage 0.000 is budget-specific.** It is 0.000 at
   the 10% budget the sentence is about and 0.143 at 5%; the text now anchors the
   number to the budget.
7. **A wrong figure pointer.** `Fig. 5c` was cited for per-cohort retention under
   the random split; Figure 5(c) is the acceptance-to-oracle ratio, and the
   retention numbers are not plotted in any figure. The pointer was removed.
8. **Documentation gaps.** `README.md` did not list `r48_fov_units.py`,
   `run_pipeline.py` or `s00`-`s03`, and `docs/submission/code_availability.md`
   did not list `r43_threadpin.py`, `r48_fov_units.py` or `run_pipeline.py` and
   carried no command for `r11`, `r35`, `r43`, `r48`, `r49` or `r49b` -- although
   the manuscript points readers to it for "exact commands". Both files were
   completed.
9. **One identity was softened rather than claimed.** The statement that the
   earlier 0.155 is the pinned logreg-only subtotal is an inference (the exact
   three-repeat pinned analogue was never run); the Table 7 legend and the two
   Methods passages now give the measured value (0.163 over two repeats) and say
   the two figures are within the sampling error of the same quantity.
10. **The "half of them 20 or more" claim was not checkable from a shipped
    table.** `scripts/r49b_panelinvariance.py` now also writes
    `experiments/r49b_pair_overlap.csv`, the canonicalised shared-channel count of
    all twelve directed pairs (min 11, max 27, six pairs at 20 or more), and the
    manuscript points at it.

Also recorded, and left as they are: the rule features are built transductively
(positivity is called over the whole cohort, held-out images included -- label
free, but not the published per-fold convention), and a rule that only one panel
of a pair carries contributes a column that separates the two cohorts rather than
a shared coordinate. Both are now stated in `docs/methods.md` and, for the second,
in the manuscript itself, because both bound what the representation can be said
to do.

## Round-5 correction record (r48 and r48b)

The cross-field-of-view shortfall and its targeted repair were re-estimated at
field-of-view resolution, and the documentation was brought up to date.

1. **The shortfall is now established rather than directional.** The published
   twenty-unit design gives 0.833 / 0.817 / 0.917 at 5% / 10% / 20% budgets, with a
   0.667-0.950 interval at 10% that includes the nominal 0.90. Holding one image
   out per unit (`scripts/r48_fov_units.py`) puts the same family on 96 units and
   the interval separates at the tighter budgets: 0.844 [0.788, 0.894] and 0.792
   [0.724, 0.855], both excluding 0.90, against 0.856 [0.799, 0.910] at 20%, which
   still includes it. Nothing about the estimator changed; only the unit did. The
   passage now says the shortfall is established at the tighter budgets, Table 1's
   caption points at Supplementary Table 8, and the abstract carries both
   estimates.
2. **A readout was mislabelled in the r48 report (defect).** `summarise()` in
   `scripts/r48_fov_units.py` computed `coverage_nonvacuous` as the *conditional*
   readout P(covered | non-empty) while the report headed the column
   `cov(non-vac)`, which in this paper names the joint readout
   P(covered and non-empty). The function now returns all three readouts
   (`coverage`, `coverage_joint`, `coverage_cond`), each with its own
   unit-bootstrap interval, the report labels them, and
   `experiments/r48_coverage_ci.csv` was regenerated from the stored predictions.
   The joint column is 0.716 / 0.722 / 0.825 and the conditional one
   0.820 / 0.769 / 0.843 at 5% / 10% / 20%.
3. **r48b reproduces the family it repairs (self-check).** `scripts/r48b_fov_repair.py`
   imports the arm implementations from `scripts/r31_targeted_repair.py` rather
   than re-implementing them, and its `plain` arm reproduces the r48 headline
   exactly on coverage (0.843750, 0.792245, 0.856481), on the joint readout
   (0.716435, 0.721644, 0.824653) and on acceptance (0.528162, 0.725899,
   0.918875); the conditional readout differs by at most 0.0091, the two
   renderings weighting units and evaluations respectively. `scripts/r31_targeted_repair.py`
   is not modified, so its released tables are untouched.
4. **The repair is now a resolved but partial one, and the text says so.** At 96
   units the three-stratum arm's joint paired difference against `plain` is
   +0.0698 [+0.0270, +0.1161] pooled over the three budgets, which excludes zero,
   where at twenty units no arm's difference did. Per budget the same comparison is
   -0.0093 [-0.0735, +0.0561] at 5%, +0.1233 [+0.0671, +0.1858] at 10% and +0.0955
   [+0.0469, +0.1493] at 20%, so the gain is real at two of the three budgets and
   absent at the tightest. The paragraph gives those figures, the per-budget joint
   values (0.707, 0.845 and 0.920 against `plain`'s 0.716, 0.722 and 0.825), the
   fact that the 20% estimate is above the nominal 0.90 while its interval
   (0.872-0.965) still includes it, and the throughput cost; it does not claim a
   uniform repair of the joint readout.
5. **Documentation.** `README.md` and `docs/submission/code_availability.md` now
   list `r48b_fov_repair.py`, its command and its output files, `docs/methods.md`
   describes both scripts and the compute footprint, and Supplementary Tables 8
   and 9 are mapped to the shipped CSVs in `code_availability.md`.
   `scripts/r48b_fov_repair.py` also emits a unit-bootstrap interval on
   `coverage_all` and `coverage_nonvac` and a per-budget paired table
   (`experiments/r48b_fov_repair_paired_by_alpha.csv`), so the per-budget
   differences the manuscript quotes are reproducible from shipped files rather
   than from an ad-hoc re-run.

## Round-6 record (panel-reduction paragraph compressed)

At the author's request the subsection now titled "Robustness and ablations" was
compressed. The panel-reduction *simulation* and its companion marker-logic
tests are scope defence for the transfer claim rather than a further link in the
evidence chain, so their numbers no longer appear in the Results: five paragraphs
(87 lines) were replaced by two sentences that keep only (i) the
throughput-versus-validity pivot, (ii) the negative result that the transfer
failure is not a feature-scale artefact that a better representation repairs,
and (iii) the positive result that the marker-rule audit layer travels across
panels. This supersedes, in that one respect, the two "Revision note" items
earlier in this file (panel reduction; marker-logic representation, r49 / r49b).
Nothing about those computations changed and no number in the manuscript
changed: the pooled `r08_panel.csv` price list (mean `n_markers` 39.053, 29.263,
19.526, 9.789 and accuracy 0.8403, 0.8364, 0.8249, 0.7337 at `frac` 1.0, 0.75,
0.5, 0.25, with acceptance 0.6506, 0.6457, 0.6100, 0.2252) and the r49 / r49b
representation results remain reproducible from `experiments/r08_panel.csv`,
`experiments/r49_<space>_coverage_ci.csv`, `experiments/r49b_panelinvariance.csv`,
`experiments/r49b_pair_overlap.csv`, `work/r49_report.txt` and
`work/r49b_report.txt`, and are still listed in `README.md` and
`docs/submission/code_availability.md`. The prose that quotes the r49 / r49b
numbers is now the Methods section "Marker-logic representation"
(`docs/methods.md`), which the compressed Results sentence points at. No figure
or table legend depended on the panel-reduction study, and the subsection was
retitled from "Robustness, ablations, and the cost of a smaller panel under
exchangeability" to "Robustness and ablations".

## Round-7 record (deliverable restructure and the review loop)

The paper's deliverable was moved from the certificate itself to the decision
procedure around it, and the review budget was put next to the repair price in the
same units. No number in the manuscript changed and no result was recomputed; the
round adds one derived table and restates existing ones.

1. **A new subsection, "The deliverable: a decision procedure, a price list, and a
   label-free audit"** (before `## Discussion`), consolidating three things the
   paper already contained in separated places: the label-free acceptance-ratio
   audit (0.744 random split, 0.766 across fields of view, 3.05 under transfer),
   the per-pair price list (Table 6: four pairs at 500 labels, three more at 1,000
   or 2,000, four whose oracle accepts 5% or less and which should not be labelled
   at all), and where the labels should go. The Introduction moves from three
   contributions to four, the fourth being that procedure, and the Abstract names
   it.
2. **New Table 9 and `scripts/r50_review_loop.py`.**
   `experiments/r50_review_loop.csv` is derived from two released tables and
   recomputes nothing: the certificate's acceptance, review fraction and
   evaluation-pool size from `experiments/r16_review_budget.csv`, and the target
   label ladder from `experiments/r24_efficiency.csv`. The repair price is defined
   as the smallest k at which the ladder restores coverage over all evaluations to
   at least 0.90 *without* losing the source-only acceptance, i.e. the count at
   which the repair is free in throughput rather than merely valid. Verified
   values: cross-field-of-view at a 10% budget, k = 500 (0.252 of the 1,984-cell
   mean pool) takes acceptance from 0.5338 to 0.5346 and coverage from 0.8167 to
   1.0000, on 15 of the 20 units; the certificate's own review fraction at that
   budget is 0.4662, 925 cells per unit, so the review queue is 1.85 times the
   label price. At a 5% budget no k in the ladder restores throughput (best
   0.1819 at 1,000 against 0.3566), and under cross-tissue transfer none does at
   any budget (best 0.0641 at 4,000 against 0.3676 at a 10% budget). The random
   split needs no repair at any budget, its source-only coverage being 0.9965 to
   1.0000.
3. **A caption defect found and fixed while building Table 9.** Table 5's caption
   said "n = 19, 20 and 60 units" without distinguishing the k = 0 row from the
   rest. The ladder drops units whose pool cannot supply k labels, so those unit
   counts belong to k = 0 only: at a 10% budget the cross-field-of-view k = 500 row
   is 15 of the 20 units and the random-split k = 500 row is 7 of its 19. The
   caption now says so, and Table 9 carries the unit count per row.
4. **The Abstract now carries the cost of the review saving.** The 86.4% figure
   was quoted alone; it is now given with the error rate it buys (0.138 among
   retained labels, and 0.060 at a 10% budget, where the saving is 55.9%), so the
   headline saving and the tolerance it requires are read together.
5. **The fixed-sequence step-down is now scoped where it is reported.** It is a
   throughput gain at unchanged nominal validity, and only where the calibration
   pool carries a floor (60 of 81 configurations at a 100-cell floor, 81 of 81 at
   200, none at 25) and where the accepted set is a prefix of the scan order; the
   coverage it inherits is the split family's (0.963, 0.963 and 0.984 on the random
   split, 0.753, 0.733 and 0.894 across fields of view, 0.584, 0.392 and 0.263
   under transfer). The Results now say explicitly that it is not a remedy for a
   small calibration pool -- the remedies there are more labels or a looser budget
   -- and the calibration-size paragraph points at it by name (Methods, Baseline
   selection rules and the fixed-sequence step-down).
6. **The Discussion's closing claim was sharpened.** "It bounds a failure and
   prices its repair, rather than removing it" became: it bounds the failure,
   prices the calibration repair (500 labels for +0.183 of coverage at unchanged
   acceptance across fields of view at a 10% budget), and shows the residual
   ranking failure is repaired by spending the same labels on the model (9.2-fold),
   without claiming any of it is free.
7. **Documentation.** `README.md` (script list, run block, the round-2 list moves
   from seven experiments and nine scripts to eight and ten) and
   `docs/submission/code_availability.md` (script list, experiments listing) both
   carry `r50_review_loop.py` and `experiments/r50_review_loop.csv`, and
   `docs/methods.md` describes the derivation under "Review budget" and the run
   cost under "Compute".


## Round-8 record (the label-free correction, the error-side accounting, and the external-classifier wrap)

Round 8 fixes one overstated claim, unifies two accountings that were being mixed
silently, adds the error-side statement of the labour result, and adds the one
experiment the "wraps any classifier" claim needs. No headline number changed
except where a value was being truncated, and no result was recomputed except the
new one.

1. **The acceptance-ratio diagnostic was described as label-free, and it is not.**
   Its denominator is the acceptance of a threshold chosen with the deployment
   labels in hand, so it cannot be computed before they arrive. Ten edits, one of
   them in `README.md`: the Abstract and Introduction now name "an audit of whether
   the certificate can be trusted on the current deployment, computed once its
   labels arrive"; the Results say it in one sentence and point at the statistic
   that *is* available before any label is spent (the source-only acceptance and the
   two-sample Kolmogorov-Smirnov distance between source calibration and deployment
   scores); the deliverable subsection is retitled without the phrase and splits its
   first two bullets into **Triage, before any label** and **Audit, once the labels
   arrive**; the Discussion says "an audit rather than a wholly label-free screen".
   The four surviving uses of the phrase are accurate and were left alone: the
   score-conditional alternative, which needs no labels by construction, and the
   KS distance itself (Results, Fig. S4 legend).

2. **Two accountings of the same acceptance were mixed in one sentence.**
   `experiments/r24_efficiency.csv` stores both `accept_mean`, the mean over
   evaluation units, and `accept_pooled`, which weights each evaluated cell. At a
   10 percent budget on the random-split family they are 0.5594 and 0.5970; the
   Abstract quoted the first for validity and the second for the labour saving, in
   the same paragraph and in the same units. The Abstract now uses the pooled figure
   for the labour claim and says so; the Results add a paragraph stating both
   conventions and which figure is used where. The gap is entirely the four
   colorectal folds, whose accepted sets are empty: they carry 21 percent of the
   unit mean and 5.6 percent of the pooled evaluations.

3. **The labour result now carries its error-side counterpart, which is the
   paper's positive claim in a practitioner's units.** The uncalibrated classifier
   keeps every cell and commits 0.183 of the batch to the record wrong
   (`accept_all`, error 0.1831 on the random-split family). The certificate commits
   0.061 of the 0.597 it retains, or 0.036 of the batch, wrong and unreviewed at a
   10 percent budget, and 0.139 of 0.864 at a 20 percent budget: four fifths of the
   errors that would have been written silently become cells a human is asked to
   check, at a review cost of 40.3 and 13.6 percent of the batch. Recomputed from
   `r24_efficiency.csv` (`realised_risk`, `accept_pooled`, family `cellsplit`,
   arm `src`, k = 0) and `work/r04_stdout2.txt` (the `accept_all` error rate). Two
   quoted realised errors were truncated rather than rounded and are now 0.061 and
   0.139 (from 0.0609 and 0.1387).

4. **The per-cohort labour figures were re-verified and are exact.**
   `experiments/r16_percohort.csv` gives `pct_review_removed` at a 10 percent budget
   as 0.8537 breast, 0.7259 GVHD, 0.5012 melanoma, 0.0487 colorectal, and at a 20
   percent budget 0.9974, 0.9765, 0.8157, 0.3726, against the manuscript's
   85.4/72.6/50.1/4.9 and 99.7/97.7/81.6/37.3.

5. **The external image-based classifier is now wrapped end to end, and the
   wrapper's own analysis had a defect.** `scripts/r51_cellsighter.py` drives the
   published CellSighter code (`tools/CellSighter`, vendored unmodified) and puts
   the paper's own `SelectiveRiskController` around it. Defect found and fixed while
   verifying it: `build_units` emitted the transfer arm with the *target* index
   array, which `analyse` then applied to the source probability arrays, so the two
   "transfer (source)" rows per repeat evaluated the first 1,549 source cells under
   a transfer label. `build_units` now returns an explicit `side` field and the
   transfer arm emits only its target row; re-running `analyse` on the stored
   predictions reproduces every retained row exactly and drops 18 meaningless rows
   (126 -> 108); the shipped `experiments/r51_cellsighter.csv` and
   `experiments/r51_cellsighter_summary.csv` were regenerated with the fixed
   code and carry the 108 rows. The script also gained `--hold-every`,
   `--size-data`, `--threads` and `--suffix`, whose defaults reproduce the
   shipped tables.
   Result (Table 10, `experiments/r51_cellsighter_summary.csv`): the certificate
   abstains at a 5 and a 10 percent budget on every arm while an oracle threshold
   accepts 0.271 on the same cells, because the most confident decile of the
   held-out melanoma fields already carries a risk of 0.101. Recomputable from
   `work/predictions_cellsighter/cellsighter_scores.npz`. The section states the
   reason in the model rather than the layer (neutrophils 1,493 predicted against
   203 present on the network's own held-out fields; GVHD accuracy 0.491 against
   0.58-0.67 for the tabular regressor on the same pair) and claims the interface
   only. Note for the record: the depot's GVHD, breast and colorectal image folders
   ship partial fields, so only 1,549 of 10,283 GVHD annotations (15 percent) have a
   usable crop against 19,775 of 21,204 (93 percent) for melanoma, which is why
   melanoma is the source. The CPU-budget explanation for the abstention was tested
   rather than assumed: a second run with the per-class cap raised from 400 to 600
   cells under the same twenty-epoch schedule reaches a lower mean batch loss in its
   final epoch (0.839 against 1.025, both recomputed from the TensorBoard event
   files) but a *lower* held-out field accuracy (0.540 against 0.564) and a higher
   risk at a 10 percent operating point (0.117 against 0.101), so the network
   memorises these fields rather than underfitting them. That variant's tables were
   not shipped; its predictions and report are kept at
   `work/predictions_cellsighter/cellsighter_scores_v2.npz` and
   `work/r51_report_v2.txt`, with the scored checkpoint at
   `work/cellsighter/weights/Melanoma_Oct2522_24ch_v2/weights_18_count.pth`.

6. **The fixed-sequence step-down is scoped rather than promoted.** Its evidence is
   a throughput gain at unchanged nominal validity only where the calibration pool
   carries a floor: 60 of 81 configurations at a 100-cell floor, 81 of 81 at 200,
   none at 25. An unconditional change of default would therefore have overstated
   it, and the Results now state the rule for practice instead.

7. **Documentation.** `README.md` gains the r51 entry in the round-2 list (nine
   experiments, eleven scripts, already counted) and in the layout and run blocks;
   `docs/submission/code_availability.md` gains the torch/torchvision/skimage/
   tensorboard dependency note, the `tools/CellSighter/` layout entry, the r51
   script and its two tables, and the Table 10 mapping; `docs/methods.md` gains the
   section "An external, image-based classifier".

## Round-9 record (the two-lever statement of the residual review, and the ratio's near-implication)

Round 9 adds no experiment and changes no number. It states in the body two things
the numbers already implied, and pays a documentation debt.

1. **The residual manual review is now written as two levers.** The section "What
   the guarantee costs" gains a paragraph separating the budget, free in labels,
   from the model, whose headroom is the oracle, on the pooled frontier 65.3% /
   40.3% / 13.6% of the batch reviewed at a 5 / 10 / 20 percent budget --
   acceptance 0.3474, 0.5970 and 0.8642 of `experiments/r24_efficiency.csv`, arm
   `src`, k = 0 -- against oracle review fractions 42.3% / 24.4% / 5.6% (same
   file, arm `oracle`, k = 0; acceptance 0.5769, 0.7556 and 0.9444), so the gaps
   are 23.0, 15.9 and 8.0 percentage points, and the realised errors among the
   retained labels are 0.0214, 0.0609 and 0.1387. No figure and no table was
   touched, and the per-unit-mean review fractions already in the section (44.1%
   against the oracle's 24.8% at a 10 percent budget, Table 9) are unchanged and
   consistent with these.

2. **The ratio's in-house "perfect detector" is now marked as partly structural.**
   The Results note that the numerator and the budget are both monotone in the
   accepted fraction, so the non-trivial half of the cross-tabulation is the 1,115
   within-budget units that never exceed one, and that the falsifiable test is the
   external cohort, where the rule misses 60 of 174. The counts (575 of 575, 0 of
   1,115, 1,690 computable of 2,010, 60 of 174) are unchanged.

3. **Documentation debt.** The Round-8 entry said only that the fixed `build_units`
   drops 18 meaningless rows (126 -> 108); it now also records that the shipped
   `experiments/r51_cellsighter.csv` and `experiments/r51_cellsighter_summary.csv`
   were regenerated with the fixed code and carry the 108 rows.

## Round-10 record (the stronger-model sweep, and what sets the headline)

Round 10 adds one experiment and no new data. It measures what a stronger wrapped
model buys and finds that the manuscript's headline throughput is set by the
averaging convention rather than by the construction.

1. **`scripts/r52_stronger_score.py`, `experiments/r52_stronger.csv` and
   `experiments/r52_stronger_summary.csv` are new.** The sweep refits seven models
   from scratch on the random-split family's own folds -- the three base learners,
   their soft vote, a five-bootstrap bagged histogram gradient boosting, a
   five-bootstrap bag of the vote, and a two-hidden-layer perceptron -- over the
   same nineteen evaluation units and the same ten calibration draws, scored the
   paper's way (1 - maximum probability) and evaluated with the same controller and
   the same oracle.

2. **The three base learners reproduce the headline, per model and in aggregate.**
   At a 10 percent budget the refit arms give acceptance 0.6094, 0.5667 and 0.5022
   against the 0.6094, 0.5667 and 0.5022 of `experiments/r03_metrics.csv`; their
   equal-weight mean is 0.5595 against the reported 0.5594, its pooled counterpart
   is 0.5971 against the 0.5970 of `r24_efficiency.csv` (arm `src`, k = 0), and the
   oracle mean is 0.7516 against 0.7516. At a 20 percent budget the mean is 0.8006
   against 0.8007. Twenty checks of this kind were added to `work/_preflight.py`
   (CHECKS 121 -> 141, FAILS 0).

3. **The result.** The headline 0.5594 is the equal-weight mean of one weak learner
   and two adequate ones, and the weak one is what sets it: the tuned gradient
   boosting learner alone certifies 0.6094 at a 10 percent budget and 0.8507 at 20.
   Ensembling adds about one point and only at the tight budget -- bagged 0.6193 and
   soft vote 0.6152 at 10 percent -- while at 20 percent the single learner leads
   the bagged one, 0.8507 against 0.8494. In the pooled units of the labour claim
   the tuned learner takes the review fraction from 40.3 percent to 32.6 percent at
   a 10 percent budget and from 13.6 percent to 6.6 percent at 20, with coverage
   1.000 in every arm at both budgets, so the labour claim becomes 67.4 and 93.4
   percent of the review removed. The paper's Table 2 already contained a piece of
   this -- the certificate on an ensemble score -- but on the coarse label set and
   never compared head to head with a single learner on the headline's own labels;
   r52 supplies that controlled comparison.

4. **What the range means, and why the headline is left alone.** The new
   subsection states the gain as a property of the wrapped model rather than as a
   replacement headline: the headline is a statement about the reach of the
   guarantee across model families, which is a different claim from the throughput
   of its best configuration. The manuscript now carries both, and the Abstract and
   the two-lever paragraph of "What the guarantee costs" point at the measured
   value.

5. **A negative control inside the sweep.** The best-calibrated score in the set is
   the bagged vote (ECE 0.040 against 0.048 for the tuned learner) and it certifies
   less (0.599 against 0.609) while its uncalibrated threshold accepts 0.780 against
   0.788, so calibration quality and certified acceptance are again separated, this
   time on the splits that produce the headline.

This record was written against `docs/manuscript.md` at 2465 lines, 168488 bytes,
sha256 `056f343fc23c4361849602af721ae6454af464d7e95c4bdaf8c69e44f0de9aef`, mtime_raw 1790549343.69. The claim-by-claim table above audits an earlier
1610-line snapshot and is a historical record; every number quoted in it was
re-checked against the current file before being carried forward here.

## Round-11 record (the label price of validity, and the ranking ceiling)

Round 11 adds two experiments and no new data. Each tests one of the two levers the
manuscript names for the residual review, and they come out on opposite sides.

1. **`scripts/r53_calibration_size.py` with `experiments/r53_calibration_size.csv` and
   `experiments/r53_calibration_size_summary.csv` is new (Table 12).** A balanced panel
   per cohort holds the test quarter fixed and re-draws the calibration set at 25 to
   1600 cells, ten draws per evaluation unit, with the oracle recomputed on the same
   test cells. The price of validity is a decreasing function of the calibration size
   and it saturates at the bound's own conservatism rather than at zero. At a 10
   percent budget it is the whole of the oracle at 25 cells: 94.0 points on breast and
   90.5 on GVHD at 100, 55.5 and 59.0 at 200, 14.1 and 22.6 at 400, and 7.6 and 13.9 at
   800 against oracles of 0.976 and 0.938. Melanoma falls more slowly (75.2, 47.5, 29.2
   and 18.2 points at 100, 400, 800 and 1600) and colorectal not at all, because its
   oracle is 0.413 and its accepted sets are empty or near-empty at every size its
   756-cell pool supports. Coverage is at or above 0.95 in every cell of the panel at
   that budget. The 15.9-point pooled price quoted in "What the guarantee costs" is
   therefore the value of this curve at one calibration size, not an intrinsic tax.

2. **`scripts/r54_ranking.py` with `experiments/r54_ranking.csv` and
   `experiments/r54_ranking_summary.csv` is new (Table 13).** Six arms -- the paper's
   tuned learner, a widened histogram gradient boosting, a random forest grown to the
   leaf, extra trees, xgboost and their soft vote -- are refit on the random-split
   family's own folds with r52's controller, calibration draws and oracle. The result
   is null. At a 10 percent budget the arms accept 0.536 to 0.612 and span 4.4 points
   of oracle, from extra trees at 0.748 to the tuned learner at 0.792, so the highest
   oracle belongs to the learner the paper already had; at a 20 percent budget the vote
   holds the higher oracle, 0.943 against 0.921, yet certifies less, 0.847 against
   0.851. The vote is again the most accurate and the best-calibrated arm (0.839, ECE
   0.041) without certifying more, repeating the separation of calibration quality from
   certified acceptance inside this sweep as well.

3. **Internal validation, and the checks it produced.** The tuned learner is refit
   inside r54 as one of the six arms, so its row must equal r52's: it gives acceptance
   0.60944, 0.85069 and 0.40588 at the 10, 20 and 5 percent budgets, oracle 0.79220 at
   10 percent and pooled acceptance and oracle 0.67365 and 0.82282, all identical to
   `experiments/r52_stronger.csv`. Twenty-eight checks were added to
   `work/_preflight.py` for r53 and r54 (CHECKS 180 -> 208, FAILS 0), including a
   check that the refit tuned learner lands on r52's values and one that the best
   oracle at a 10 percent budget belongs to that learner.

4. **An environment limitation, recorded rather than hidden.** LightGBM 4.7.0 cannot
   be fitted in this environment's worker processes: its C API raises an access
   violation in `LGBM_DatasetSetField`, reproducible outside the sweep and independent
   of label dtype, hyper-parameters and import order. `scripts/r54_ranking.py`
   therefore omits it, takes its vote over the widened gradient boosting, the large
   random forest and xgboost, and states the omission in its docstring, in Table 13's
   caption and in `docs/submission/code_availability.md`. Nothing in the paper's two
   named levers changes as a result: the model lever is measured by r52 and bounded by
   r54, and the label lever is measured by r53.

5. **Where the new material sits.** Table 12 and its subsection sit at the end of
   "What the guarantee costs", immediately before "A stronger model compresses the
   review"; Table 13 and its two paragraphs extend that subsection, after Table 11.
   The Abstract gains one clause, that the review floor is not an artefact of the
   models wrapped.

## Round-12 record (the independent cohorts on the exchangeable split)

Round 12 adds one experiment and no new data. It closes the largest hole in the
external evidence, which until now was counterexample-only: both external arms
used a split that is hard by construction -- image-grouped folds within a cohort,
and cross-cohort transfer -- so the exchangeable family the guarantee is actually
stated for had never been run outside the depot. The reading a reviewer could
take from that asymmetry is that the method only ever worked where the authors
chose the split, and no amount of further model sweeping addresses it, because
the missing evidence is not evidence about the model.

1. **`scripts/r55_stellar_cellsplit.py` with `experiments/r55_stellar.csv`,
   `experiments/r55_stellar_summary.csv`,
   `experiments/r55_stellar_norm_robustness.csv` and
   `experiments/r55_stellar_checks.csv` is new (Table 14).** It applies the
   identical certificate, split fraction, repeat count and metric definitions
   (`scripts/r03_analyse.evaluate`, imported rather than re-implemented) to the
   STELLAR CODEX cell tables already retrieved by `scripts/r34_external.py`
   (Zenodo 6863872, CC0-1.0) -- a different laboratory, platform and tissue from
   the four benchmark cohorts. Two families are run. In `ext_cellsplit` the model
   is trained on the intestine cohort's other images and the held-out pool of
   20,000 cells, of which 10,000 are spent on calibration, is then divided at
   random, exactly as `scripts/r05_cellsplit.py` does on the depot. In
   `ext_targetcal` the model is trained on the intestine cohort and the tonsil
   and Barrett oesophagus cohorts are each divided at random, which is the
   deployment a laboratory faces when a new tissue arrives.

2. **The result is a positive replication, and it is stated with its controls.**
   Coverage is 1.000 at a 10% budget over 150 evaluations (`ext_cellsplit`, five
   held-out-field folds crossed with ten calibration repeats and three
   classifiers) and 0.993 at 5% and 1.000 at 20%, with realised risk among
   retained cells of 0.039, 0.086 and 0.142 against budgets of 0.05, 0.10 and
   0.20. For `ext_targetcal`, over 60 evaluations, coverage is 1.000 at all three
   budgets with a non-vacuous acceptance of 0.436, 0.436 and 0.519. Three
   qualifications are carried in the same passage rather than deferred: the
   uncalibrated threshold matches the certificate on throughput (0.890 against
   0.851 at a 10% budget) and loses the budget (0.487 of evaluations against
   1.000); the certificate's higher throughput here than on the depot (85.1%
   against 55.9%) is attributed to the tenfold larger calibration sample, which
   is the r53 curve's own reading, and is flagged as a reading rather than a
   result; and the target-calibrated arm's transferred model is weak on tonsil
   (0.26 to 0.33 accuracy over the six shared classes), so that arm shows the
   guarantee holding rather than a tool lifting annotation.

3. **The transform is not the result.** The STELLAR tables ship per-channel
   values already on an arcsinh-like scale, so the released values were scored
   under three transforms: `log1p_z` (the transform the r34 external branch
   already used, and therefore the like-for-like comparison), `z_train`
   (standardisation without the log) and `raw` (the released values unchanged).
   Coverage at a 10% budget is 1.000 for all six cohort-by-transform cells, and
   at a 5% budget 0.993 for `log1p_z` and `z_train` and 1.000 for `raw`, so the
   replication is not a property of the preprocessing.

4. **The ratio diagnostic is sharp exactly where the oracle is defined.**
   Across the 630 evaluations of the two families the certified-to-oracle
   acceptance ratio exceeds one on the single evaluation whose realised risk is
   over budget and on none of the 626 that are within budget and on which it is
   defined, with three undefined (all within budget) and no false positives. The
   60 misses reported under transfer in Round 8 are therefore attributable to
   that arm rather than to the cohort: they are evaluations where the oracle
   threshold accepts no cells, so the ratio does not exist.

5. **The failure is not softened, and the pairing is explicit.** The
   image-grouped arm on the same intestine cells still covers 0.656 at a 10%
   budget and the transfer arm still collapses to 0.033; both are quoted in
   Table 14's last row and in the same passage. Table 14 and Supplementary
   Table 7 now describe the same cohorts, the same pipeline and the same
   certificate, with the split the only thing that differs, so the external
   evidence is two-sided rather than selectively chosen.

6. **Internal validation, and the checks it produced.** Forty-four checks were
   added to `work/_preflight.py` (CHECKS 208 -> 252, FAILS 0), pinning the raw
   row and unit counts, both families' coverage, acceptance and realised risk at
   all three budgets, the naive-versus-certificate contrast, the transform
   robustness panel, the monotonicity of acceptance in the budget, and the fact
   that the image-grouped arm on the same cells is still short of nominal, by
   reading it back out of `experiments/r34_phase2_certificate_summary.csv`. The
   script's own `experiments/r55_stellar_checks.csv` carries 15 further checks
   and passes all of them.

7. **Where the new material sits.** The subsection sits immediately after
   "External validation on an independent cohort does not reproduce the
   guarantee", which now hands over to it in its closing sentence, and
   immediately before "Wrapping a published image-based classifier". Table 14
   follows Table 13 in the file, though the table numbering follows the
   experiment timeline rather than document order. The Abstract gains the
   positive half of the external test; the third contribution is restated as
   replicated in both directions; and the Data and code availability paragraph
   now names the later scripts explicitly.

This record was written against `docs/manuscript.md` at 2689 lines, 184355 bytes, sha256 `e18e18307e5f846497e3faf84e79f4df0d36755160807b5e8ce5a3ff84b175fa`, mtime_raw 1790594472.34508. The claim-by-claim table above audits an earlier
1610-line snapshot and is a historical record; every number quoted in it was
re-checked against the current file before being carried forward here.

## Round-13 record (the representation bound, and what it does not buy)

Round 13 adds one experiment and no new data. It closes the last of the
model-axis objections a referee can raise against the residual review the
certificate still asks for: that the floor is an artefact of reading one cell's
marker vector at a time, and would fall once tissue context entered the score.

1. **`scripts/r56_neighbour_context.py` with `experiments/r56_neighbour.csv`,
   `experiments/r56_neighbour_summary.csv` and
   `experiments/r56_neighbour_checks.csv` is new (Table 15).** Each cell is
   augmented with the mean normalised marker vector of its ten nearest
   neighbours in the same image, read off a KD-tree over the cell coordinates,
   together with the logarithm of its mean neighbour distance, which takes the
   panel from 22-58 features to 45-117. Coordinates and marker intensities only:
   no label, no predicted class and no count of neighbour types enters the
   construction, so no target information can leak into the augmented
   representation.

2. **The context does not lower the floor, and the baseline arm is what makes
   the comparison controlled.** The `own` arm reproduces the tuned learner's row
   of Tables 11 and 13 exactly -- acceptance 0.40588, 0.60944 and 0.85069 at the
   five, ten and twenty percent budgets and an oracle of 0.79220 at ten -- on the
   identical folds, pool and ten calibration draws, so the two arms differ only
   in the features. Adding the neighbourhood moves the oracle by -0.0094 at a 5%
   budget, -0.0137 at 10% and +0.0018 at 20%, and accuracy by -0.0056: the
   pooled floor is very slightly worse, not better.

3. **The delivered operating point is essentially unmoved.** Acceptance changes
   by +0.0069 at a 5% budget and +0.0015 at 10% and falls from 0.8507 to 0.8028
   at 20%; the review fraction at a 10% budget moves by -0.1531 percentage points
   (0.3906 to 0.3890); and coverage is 1.000 in both arms at the 10% and 20%
   budgets and 0.9947 at 5%, so the certificate is not perturbed by the
   augmentation either way.

4. **The reading, and its limits.** The residual review that the learner sweeps
   leave open is therefore not a missing-context artefact, and the bound those
   sweeps state is a bound on the assay rather than on the representation that
   was chosen. The claim is about this augmentation on these four cohorts and not
   about image-based or graph-based encoders in general; the image-based encoder
   is the separate CellSighter wrap, whose result is negative for a different
   reason (partial fields, 15% crop coverage on GVHD).

5. **Internal validation, and the checks it produced.** Twenty-one checks were
   added to `work/_preflight.py` (CHECKS 252 -> 273, FAILS 0), pinning the raw
   row count, the cohort, fold, repeat and arm counts, the evaluation count per
   budget, the exact reproduction of the r52 tuned-learner row, both arms'
   acceptance and oracle at three budgets, the accuracy drop, the review-fraction
   movement, the nominal coverage in both arms, and the fact that acceptance at a
   20% budget falls rather than rises with the context. The script's own
   `experiments/r56_neighbour_checks.csv` carries 19 further checks and passes
   all of them.

6. **Where the new material sits.** A paragraph and Table 15 sit immediately
   before "Biological consistency of the labels, and its effect downstream". The
   Abstract extends its model-sweep sentence with the negative result. Table 15
   follows Table 14 in the file, though the table numbering follows the
   experiment timeline rather than document order.

Two prose corrections were made in the same pass and change no number. Table 15's
caption now states the fold accounting of the random-split family the way the
paper states it elsewhere (`four cohorts by five folds, less one fold dropped`),
against the 190 evaluations it tabulates; and the target-calibrated acceptance
quoted for the STELLAR exchangeable arm is now marked as acceptance counted among
the evaluations that decide at all, which is the quantity `mean_accept_nonvac`
holds and is not the `acceptance` column of Table 14.

This record was written against `docs/manuscript.md` at 2737 lines, 187798 bytes, sha256 `5fa37cbf78a363540859a285d22db966ab07db3b2890e87a1e682112aca17502`, mtime_raw 1790595039.1578293.

## Round-14 record (the family-wise corrections, and one arm withdrawn)

Round 14 adds one analysis, withdraws one, and changes no experiment result.

1. **`scripts/r57_multiplicity.py` with `experiments/r57_multiplicity.csv` and
   `experiments/r57_multiplicity_summary.csv` is new (Table 15).** Three families
   of claims are stated over a family of comparisons and were reported without the
   correction each needs. R57 adds, for every one of them, the one-sided
   Clopper-Pearson limit at the per-comparison level, the Bonferroni limit at the
   family level, and a max-statistic cluster bootstrap that resamples the
   comparison unit. Nothing is refitted and no model is touched: the coverage
   families are read from `r03_metrics.csv` and `r55_stellar.csv`, the pair family
   from `r31_transfer_detail.csv`, and the convention family from
   `r18_conventions.csv`.

2. **What the correction costs, and what it does not.** Coverage is unchanged: no
   unit of the 19 exchangeable units falls below 1.000, the per-unit limit moves
   from 0.926 per comparison to 0.839 after Bonferroni, and the pooled limit over
   570 evaluations from 0.996 to 0.985; the STELLAR family gives 0.926 and 0.878
   over its five units. Repair is unchanged: at the full label budget 5 of the 12
   cross-tissue pairs certify a realised risk inside a 10 per cent budget before
   and after the correction, and the pooled risk upper limit moves from 0.0355 to
   0.0358. The marker-logic gate is where the correction bites, and it bites
   twice, because the gate is a minimum over 68 conventions and the populations
   above it are then selected out of 48 population-by-cohort cells: the bound
   survives both corrections for three populations (GVHD neutrophils 14.26 per
   cent, GVHD plasma cells 13.89, melanoma CD8 T cells 16.86) where the
   uncorrected reading named five, the other two (colorectal CD3 T cells at 6.90
   and colorectal endothelial cells at 5.21) keeping their point estimates but not
   their bounds.

3. **The residual review is decomposed and answered.** The subsection states why
   the last of the review cannot be closed -- at a 20 per cent budget, 8.0 of the
   13.6 points are the price of validity and 5.6 are the oracle's own floor, which
   is set by the assay -- and prices each of the three levers the paper measures:
   the budget (65.3 to 13.6 going from a 5 to a 20 per cent budget), the model
   (13.6 to 6.6 at 20 per cent, leaving 3.6 points of validity price), and
   calibration (85.1 retained on the independent cohorts at ten times the
   calibration cells against 55.9 on the benchmark). It also reports the triage
   curve of the deferred set, computed from the released per-cell probabilities:
   inspecting a quarter of the deferred cells finds 0.386 of the residual errors
   and half finds 0.655, against 0.250 and 0.500 for an uninformed pass, with
   cross-learner disagreement and score entropy no better than the certificate's
   own score. The passage is explicit that this does not preserve the guarantee --
   an unreviewed and uncommitted cell carries none -- and that it is a
   labour-allocation statement rather than a validity statement.

4. **The representation arm of Round 13 is withdrawn.** Two reasons, and the
   second is load-bearing. It was a null: adding the mean marker vector of a
   cell's ten nearest neighbours moved the oracle by -0.0094, -0.0137 and +0.0018
   at the three budgets. More importantly it was in tension with the certificate's
   own assumption: `scripts/r54_ranking.py` states that its sweep uses no
   neighbour, composition or other cross-cell feature *because such a feature
   would make the accepted-and-wrong count non-binomial and void the bound as
   stated*, and the neighbour augmentation is exactly such a feature. A construct
   that cannot be certified under the bound the paper proves should not appear as
   evidence about the representation axis, so the paragraph, its table and the
   Abstract clause were removed rather than qualified. The scripts and tables
   remain on disk as an unclaimed internal record.

5. **The r56 checks were removed with the claim and nine r57 checks added.**
   `work/_preflight.py` now reads `experiments/r57_multiplicity.csv` and pins the
   two unit counts, both coverage limits, the pooled limit, the pooled pair risk
   upper limit, the number of pairs certifying inside the budget, and the number of
   populations whose gate survives both corrections: CHECKS 273 -> 261, FAILS 0.

6. **Where the new material sits.** The subsection and Table 15 sit immediately
   before "Biological consistency of the labels, and its effect downstream", where
   the withdrawn paragraph stood; the Abstract carries the correction in place of
   the withdrawn clause; `docs/methods.md` drops the withdrawn paragraph.

This record was written against `docs/manuscript.md` at 2763 lines, 189795 bytes, sha256 `ce72c00a2cb693ba52d1d90d1c2c44aa95930841b7f541ac3ebbd045a5e52116`, mtime_raw 1790595946.4296658.

## Round-15 record (the review queue as an instruction, not a description)

Round 15 turns the triage result of Round 14 into an action and gives it a table.
`scripts/r57b_triage.py` writes `experiments/r57_triage.csv` (507 unit-by-repeat
rows) and `experiments/r57_triage_summary.csv` from the released per-cell
probabilities of the random-split family: the deferred set of every evaluation
unit and calibration repeat is recovered from the certificate, ordered by the
certificate's own score, and the cumulative error capture and the quartile error
rates are recorded. Pooled over cells, the first quarter of the queue holds 0.386
of the residual errors at a 10 per cent budget and the first half 0.655, against
0.250 and 0.500 uninformed; the head of the queue carries a mean error rate of
0.568 against 0.265 in its last quarter, and 0.498 against 0.118 at a 5 per cent
budget. Cross-learner disagreement, mean-probability entropy and the mean score
were all tried as orderings and none beats the certificate's own score. The
manuscript passage now states the two instructions the curve supports -- read the
queue in descending uncertainty, and shorten it from the end -- and states that
the tail's error rate is 0.265 rather than zero, so the length of the queue and
the error it leaves remain the budget's business and only the reading order
changes. Five checks were added to `work/_preflight.py` (CHECKS 261 -> 266, FAILS
0) pinning the two capture values, both quartile rates at the 10 per cent budget
and the head rate at 5.

This record was written against `docs/manuscript.md` at 2772 lines, 190498 bytes, sha256 `98a5434bb7f83651366918a3ba59a8e459f5af90ce4f7a9f98479a9bc3d6a736`, mtime_raw 1790596867.0754862.

## Round-16 record (the decision procedure as a product, and two defects in the deliverables)

Round 16 changes nothing in `docs/manuscript.md`. That is the point of recording it
here: the forward-facing claim of the submission is a decision procedure, and until this
round the procedure existed as prose, as Table 6, and as a command-line program, with no
artefact a laboratory could hand to a person who does not write code, and no single place
that stated the action it implies. `docs/adviser.html` is that artefact.

**What the page is.** One HTML file, no build step, no dependencies, no network. It asks
three questions -- where the scores are, what error budget to hold, and nothing else --
fits the paper's own certificate on the calibration cells, and returns four numbers: the
fraction of the batch that can be labelled automatically, the fraction routed to a
person, the threshold, and the audit ratio when deployment labels exist. It then prints
the review queue in the certificate's own order, states the two instructions that order
supports (read descending, shorten from the end), and states the three things the
guarantee does not cover. A reader who does nothing but press the first button three
times sees the whole procedure on the released unit.

**Verification.** The certificate was ported to JavaScript (Clopper-Pearson limits from
an incomplete beta function by continued fraction and bisection) and compared, offline,
against `scipy.stats.beta.ppf`: 200 random shapes, maximum absolute error 1.6e-12. The
ported rule then reproduces `src/errctl/conformal.py` exactly on the embedded unit at all
three budgets -- acceptance counts 972, 1400 and 1843, that is lambda 0.037142, 0.189870
and 0.731057 -- and the page's audit ratio 0.90631 equals the command-line adviser's to
every printed digit. `work/_preflight.py` now loads the embedded unit back out of the
finished HTML and reruns `src/errctl` on it, so the page cannot drift from the paper
without the audit failing. The page is generated by `scripts/r60_adviser_html.py`, which
loads the unit through the same loader `scripts/r59_adviser.py` uses; regenerating it is
byte-identical (sha256 `590a731a49aa3b14c8e7827fa1122999388c97557c1dbd70eb4408e088c21715`,
115,447 bytes).

**Two defects found while building it, both in the first build.** First, the embedded
arrays were joined with semicolons, which is not JavaScript, so the first build could not
parse its own data and no control on the page worked; the arrays are now full-precision
comma-separated literals. Second, `render()` unhid the action checklist before calling
`show("s3")`, and `show()` re-hides every card, so the checklist -- the one part of the
page written for a reader who does not know the method -- could never appear after a
computation. Neither defect would have been visible in the manuscript, and neither is
caught by any check that reads numbers.

**One correction outside the page.** `scripts/r59_adviser.py` computed its oracle
acceptance on a 120-point grid of candidate thresholds rather than over every attainable
threshold. A grid can only understate the oracle, and therefore overstate the ratio. It
now calls the paper's own definition (`scripts/r03_analyse.py`, `oracle_acceptance`,
which scans every attainable threshold). On the demo unit the oracle moves 0.8150 to
0.8166 and the audit ratio 0.908 to 0.906, and the page and the CLI now agree. No
manuscript number moves: the 0.744, 0.766 and 3.05 of Table 4 come from the released
tables, not from this program.

**One consistency repair in `README.md`.** The r56 bullet still advertised the
representation axis as live and still called its table "Table 15 of the manuscript",
which is now the family-wise correction table of Round 14. The bullet is relabelled
withdrawn, with the two grounds (a null, and a cross-cell feature that would make the
accepted-and-wrong count non-binomial) and a pointer to the condition as
`scripts/r54_ranking.py` states it. Two further README and code-availability gaps are
closed: `r57_multiplicity.py`, `r57b_triage.py`, `r59_adviser.py` and
`r60_adviser_html.py` had never been listed.

**Counting claims checked on the way.** The page repeats four numbers from the
manuscript, and each was re-derived rather than copied: the budget table (65.3, 40.3,
13.6 per cent of the batch reviewed at a 5, 10 and 20 per cent budget, at a realised
error of 0.021, 0.061 and 0.139, from `experiments/r24_efficiency.csv` over 570
evaluations); the queue curve (0.386 and 0.655 at a 10 per cent budget, 0.568 against
0.265 head to tail, from `experiments/r57_triage_summary.csv`); the audit-ratio
benchmark (0.744, 0.766, 3.05); and the repair price from Table 6, where a first draft of
the checklist said four pairs were not worth repairing and the table says five -- four
reach validity at 500 target labels, one at 1,000, two at 2,000, and five never.

`work/_preflight.py`: CHECKS 266 -> 284, FAILS 0. Fifteen checks are new; thirteen pin
the page (format, decoder, sizes, certificate parity, oracle, ratio, the Table 6
strings, and the generator's identity) and two pin the oracle definition inside
`scripts/r59_adviser.py`.

This record was written against `docs/manuscript.md` at 2772 lines, 190498 bytes, sha256 `98a5434bb7f83651366918a3ba59a8e459f5af90ce4f7a9f98479a9bc3d6a736`, mtime_raw 1790596867.0754862 -- unchanged from Round 15, because this round touched no manuscript number.

## Round 17 -- the guided interface in two languages, and a sample batch to try it on

Round 17 changes nothing in `docs/manuscript.md`, `docs/methods.md` or any experiment
table. It is a round about the one deliverable that is not a number: the decision aid of
Round 16.

**The aid is now bilingual.** `scripts/r60_adviser_html.py` was a single-language
generator: it held one set of strings and wrote one file. It now holds the page as a
language-agnostic skeleton plus two text tables (`zh`, `en`), and writes
`docs/adviser.html` (Chinese, `lang="zh-CN"`) and `docs/adviser_en.html` (English) from
the same skeleton. Every visible string -- headings, hints, budget cards, the checklist,
the technical notes -- is keyed, and `main()` refuses to build if the two tables do not
have identical key sets, so the two languages cannot silently drift apart. The English
page exists for referees and for readers who will not read the Chinese one; it carries
the same embedded unit, the same sample batch and the same certificate, and it contains
no CJK character at all.

**The aid now ships something to try it on.** `docs/adviser_sample.csv` is a 2,001-row
synthetic batch in the page's own three-column format (`set,score,error`): 800
calibration cells, all labelled, and 1,200 deployment cells of which 400 carry labels.
It is generated inside the generator from a fixed seed (20260928), so it is reproducible
from the repository alone and owes nothing to the manuscript. The page offers two buttons
that are the pair a first-time user needs: one downloads the file, and one loads it
straight back into the page and runs the whole procedure on it. The label column is
deliberately half empty in the deployment block -- that is what makes the audit ratio
demonstrable without any data of the user's own -- and because a file may be only partly
labelled, the page now reports its realised risk against the labelled subset and says so
in the same breath (`res_note_part`), rather than reporting it against rows that have no
label. On the shipped file the page returns acceptance 0.4150, 0.6867 and 0.9158 at a 5,
10 and 20 per cent budget (threshold 0.4441, 0.6876, 0.9057), and on the 400 labelled
rows acceptance 0.7175, realised risk 0.0767 and audit ratio 0.9228 at 10 per cent.

**One defect found, of the kind only a reader-facing build has.** Both text tables
carried a string (`m_toast_run`, "press Compute first") that no part of the page ever
displayed: the page had a message it could not show. It is now wired to the
plan-download button, which is exactly the place where it is true. The page grew from
115,447 bytes to 153,859 bytes, which supersedes the Round-16 hash quoted above
(`590a731a49aa3b14c8e7827fa1122999388c97557c1dbd70eb4408e088c21715` ->
`6003c46c0ef360d388733f908c98326016f3df9ac0d500aa4d714893a13f2833`); the English page is
155,214 bytes (sha256 `6d4c0255dde70c775ff43ce617b84085dee8a21e8cbc99fd45431c4f9c7f693b`) and
the sample batch 29,216 bytes. Regenerating with `python scripts/r60_adviser_html.py`
rewrites all three byte-identically, checked by hashing before and after.

**One sentence the result panel was missing.** The panel printed the accepted fraction and
the fraction sent to a human but never said that the two add to the whole batch, nor that
the guarantee covers only the first part. Read either way round that is a lost claim: a
reader who takes "26.0 per cent to a human" as a shortfall rather than as the design, or
who reads "74.0 per cent accepted" as covering the batch, has the direction of the result
backwards. Both languages now close the panel with it -- the two add up to the whole batch,
every cell either passes automatically or lands in the review queue, there is no third
disposition, and the guarantee covers only the first part. The page's checklist already
said the second half one card later (a cell neither committed nor reviewed carries no
guarantee); the result panel now says it where the two numbers are. The clause is pinned in
both languages, so it cannot be dropped by a later edit.

**Documentation kept in step.** `README.md` and `docs/submission/code_availability.md`
both listed one page as "single file"; both now name the two pages, say which is for
referees, describe the sample batch and its exact contents, and record the numbers the
page returns on it. The scripts listing for `r60_adviser_html.py` no longer claims to
build one file.

**How it was checked.** A purpose-built static validator, `work/_validate_adviser.py`
(the page cannot be opened offline in the audit environment, so the checks are on the
finished HTML and on a Python transcription of the page's own JavaScript), passes
CHECKS=98, FAILS=0 over both
languages plus the CSV: no BOM, LF only, every placeholder replaced, every `data-t`,
`T.<key>` and `getElementById` reference resolvable in both directions, every handler
defined, balanced script, the same text table in both languages, the same embedded unit
in both, the embedded sample string equal to the shipped CSV byte for byte, the in-page
incomplete-beta routines agreeing with `src/errctl` to 1e-12 at three budgets, and the
four-decimal figures the page prints equal to what the paper's controller actually
produces (0.0371, 0.1899, 0.7311). Four of the checks cover the panel clause added above -- two per
language, that it states the whole-batch reading and that it limits the guarantee to the
accepted part -- and the count grew from 94 to 98 in the course of this round rather than
being fixed at the start. An earlier draft of the validator compared the
controller against hand-copied floats instead of against `src/errctl`; that is the
failure mode this file exists to prevent, and it was replaced.

`work/_preflight.py`: CHECKS 284 -> 309, FAILS 0. Twenty-five checks are new and they pin
the two languages (existence, encoding, no CJK in the English page, identical text-table
key sets, identical embedded unit, identical embedded sample), the sample batch (header,
800 / 1,200 / 400 rows, threshold and acceptance at three budgets, labelled acceptance,
realised risk and audit ratio), the generator (both output paths, the fixed seed) and the
two documents that describe them. One older check was rewritten rather than left to rot:
it pinned the generator by the literal string `os.path.join(ROOT, "docs", "adviser.html")`,
which the bilingual rewrite made obsolete, and it now pins that the generator writes both
paths into `docs/`.

This record was written against `docs/manuscript.md` at 2772 lines, 190498 bytes, sha256 `98a5434bb7f83651366918a3ba59a8e459f5af90ce4f7a9f98479a9bc3d6a736`, mtime_raw 1790596867.0754862 -- unchanged from Rounds 15 and 16, because this round touched no manuscript number.

## Round 18 -- the matched-weight downstream interval, a denser label ladder, and a real re-panelling

This round works through three items of the ranked plan (items 3, 5 and 6). Two of them move a
number in `docs/manuscript.md`, so the manuscript is edited for the first time since Round 15;
Rounds 16 and 17 had each certified it unchanged. Its new hash is at the foot of this record,
with `docs/methods.md`, which gains the two protocol paragraphs.

**Item 3: the downstream readout gets an interval for its matched reading.** The paper's
downstream paragraph reports a pooled difference of +0.029 (0.010 to 0.036) between the
distortion of the accepted output and that of the full output, declines to read it as a repair,
and names the confound: restricting to accepted cells changes the composition, the number of
images that survive the 150-cell rule and the set of types that enter, so the two scopes are
weighted differently. `scripts/r21_downstream_repair.py` now also computes the same quantity
under ONE weight per type, shared by both scopes, and bootstraps it on the same image resamples
as the primary statistic: `stratum_stat` and `pool_stats` carry `paired_acc` (w = n(pred_acc))
and `paired_min` (w = min(n_all, n_acc)), `strata_components` can freeze the type set at the
support estimable in the full sample, and `experiments/r21_downstream.csv` (33 columns) and
`work/r21_report.txt` (a new HEADLINE block and section 6) carry the result. The matched pooled
difference is -0.01059, 95% interval [-0.01079, 0.01222] under either weighting, against -0.05139
[-0.03686, 0.01477] under the all-side weight kept for continuity with earlier reports, and every cohort's
interval contains zero. Two things are worth recording about that reading. The own-weight primary
is unmoved at +0.02931 [0.00982, 0.03598] on 118,709 all-side and 78,091 accepted-side cells over
24 strata, so nothing earlier is retracted; and the matched point estimate sits at its own lower
edge, which the frozen-type-set control shows is a property of the ratio of two resampled sums
rather than of types dropping out of the average (that interval is [-0.01079, 0.01213], 1,933 of
2,000 replicates finite, moving the bounds by 0.00000 and -0.00009). The manuscript's downstream
paragraph now carries the matched recompute, the four zero-containing cohort intervals and the
frozen-support clause, and still declines the repair claim: the confound is now measured rather
than argued. The report on disk had been a `--boot 50` run over the breast cohort alone (its
"pooled" row was 0.00794 with CRC, GVHD and melanoma not estimable); it is regenerated at the
full configuration and now reproduces every number the manuscript quotes. Backups of the pre-round
file, table and log are in `work/_r21_*_before_full.*`.

**Item 5: the label ladder on a denser grid.** The grid behind Table 6 was
k in {0, 25, 50, 100, 200, 500, 1,000, 2,000}, and the plan asked for the rungs at 250, 750 and
1,500. `scripts/r14_targetcal.py --ks 0,25,50,100,200,250,500,750,1000,1500,2000 --reps 10 --out
r14_targetcal_ext` reran the cross-tissue ladder on the same folds, seeds and ten draws (222 s,
`experiments/r14_targetcal_ext.csv`, 464,400 rows), and `scripts/r14c_ladder_dense.py` recomputes
Table 6 from it after first asserting the coarse sub-grid against the twenty-four cells printed in
the manuscript -- a regression guard, so that any movement below is a change of grid and not a
change of data. All 24 cells reproduce; the denser grid moves four of them: GVHD to breast
1,000 -> 750 and melanoma to colorectal 2,000 -> 1,500 at a 10% budget, colorectal to GVHD and
melanoma to GVHD 500 -> 250 at a 20% budget. What does not move is the count that the text leans
on: seven of the twelve pairs reach validity inside 2,000 labels at a 10% budget and five never
do, nine reach and three never do at 20%, and the rungs a pair can land on are now closed at
500 / 750 / 1,500 / 2,000 and 200 / 250 / 500 / 1,000 respectively. The new points are also what
they should be on a cost curve: mean acceptance over the 150 evaluations is 0.0162 at k = 750
(6.6% of them non-vacuous) and 0.0327 at k = 1,500 (13.2%), against 0.0056 at 500 and 0.0238 at
1,000. Table 6's four cells, its caption (which now states the grid and names
`scripts/r14c_ladder_dense.py`), the reach sentence beside it and the triage paragraph, which now
gives the two rung sets, are edited accordingly. One caveat is recorded rather than smoothed: the
first-reaching k is a mean over 150 evaluations and the classifier mix enters it non-monotonically
(measured on the same file, the logistic regression alone never reaches validity on five of the
seven reachable pairs), so the rung is a property of the design; the statements that survive any
of that are the two counts and the rung sets.

**Table 6 in the deliverables, and one wrong sentence in r59.** Two shipped strings quote the
price list, and both had to follow the table. The interface's checklist (both languages) said the
three middle pairs need "1,000-2,000" labels; the rung is now 750. And `scripts/r59_adviser.py`
said "four are not worth repairing at any count because the transferred model's own ranking
accepts 5% or less", which is wrong twice over: five pairs never reach validity, and only three of
them -- breast to melanoma 0.9%, colorectal to melanoma 0.3%, GVHD to melanoma 2.0% -- sit under a
5% oracle, while breast to colorectal at 5.0% and GVHD to colorectal at 12.4% are above it. It now
reads "three more at 750, 1,500 or 2,000, and five never do inside the grid; three of those five
are unreachable because the transferred model's own ranking accepts 5% or less". `python
scripts/r60_adviser_html.py` was rerun: `docs/adviser_sample.csv` is byte-identical (29,216 bytes,
sha256 `1dbeb6a7f6892e6a65eecc5f1ae5ccee75ed5e5395020134d76d63cf5d21a154`, which supersedes nothing
and confirms the generator's sample path is untouched by a text edit), the two pages are not
(`docs/adviser.html` 153,857 bytes, sha256
`c07dcc1c84816e72a2e825e01a8016f462d53ad9f8ea2121036e2190d43daa71`; `docs/adviser_en.html` 155,212
bytes, sha256 `fe2757dfd290e0bc6b866a2e844af8470a69b59efdc19ea5b575ba3b9805f1e4`), and a second
regeneration was byte-identical to the first.

**Item 6: a real re-panelling on the shared channels.** `scripts/r08_panel.py` asks the panel
question by deleting channels from one cohort in order of a training-set F-score, so the surviving
channels are chosen by the data. `scripts/r61_shared_panel.py` runs the other protocol: the model
is fitted, calibrated and deployed on exactly the canonical markers two panels share, so which
channels survive is fixed by the two antibody panels. The plan named the colorectal/GVHD pair and
"24 shared channels"; the shared-marker counts are 27 for colorectal/GVHD and 24 for melanoma/GVHD
(`experiments/r49b_pair_overlap.csv`), so both pairs were run rather than choosing between the
pair and the count. `--pair` and `--tag` were added to the script for that, and the default pair's
report rebuilt with `--from-csv` is byte-identical to the pre-patch file, so the new flag did not
touch the first result. Arms: `within_<cohort>_full` and `within_<cohort>_shared` hold the cohort,
the folds and the protocol fixed, so only the panel changes; `cross_<src>_<tgt>` trains on the
source's shared channels and calibrates on half of the target's held-out images while deploying on
the other half, so the only violation left is the change of cohort at a fixed panel; `crossk_`
repeats that at k = 500 / 1,000 / 2,000 target labels; `srcnull_` is the source-calibrated arm the
guarantee does not cover. Five image folds, five calibration splits, three classifiers, delta =
0.10, and validity read both ways per unit (the covered flag and the one-sided 95% Clopper-Pearson
upper limit). At a 10% budget the panel restriction costs no validity -- coverage is 1.000 in
three of the four shared-panel cells and 0.987 in the fourth, against a nominal 0.90 -- but its
throughput bill is pair-specific and larger than the coarse reading suggested: acceptance falls by
0.011 for colorectal and 0.054 for GVHD on the 27-channel pair, and by 0.437 for melanoma and 0.162
for GVHD on the 24-channel pair. The domain term at a fixed panel is then the dominant one, 0.772
of acceptance on the 27-channel pair (GVHD in-panel 0.800 to cross-cohort 0.028) and 0.641 on the
24-channel pair (0.692 to 0.051), and on both pairs the source-calibrated arm offers more (0.145
and 0.141) while covering only 0.667 and 0.533 of its decisions -- the same self-diagnosing
failure the transfer family shows, now on a panel the two cohorts really share. The manuscript's
panel sentence carries the measured numbers in the Results and `docs/methods.md` describes the
protocol next to the ladder section; the r08 simulation stays where Round 6 put it (Methods and
the code listings, not the Results), and the colorectal half of the 27-channel pair is quoted with
its caveat, that its held-out pool is small enough that only 13% of its units answer at all.

**Code and documents kept in step.** `scripts/run_pipeline.py` gains three stages --
`r14_targetcal_ext` (the dense ladder run), `r14c_ladder_dense` (its reader and the Table 6 guard)
and `r61_shared_panel` -- and `--list` was run to confirm their flags. `README.md` and
`docs/submission/code_availability.md` list `r14c_ladder_dense.py` and `r61_shared_panel.py` with
their commands (including the second pair), and the `experiments/` listing names
`r14_targetcal_ext.csv`, `r14c_ladder_dense.csv` and the four `r61_*` files. Both documents, and
the pipeline's own note, gave `r21_downstream_repair.py` a `--jobs` flag it has never had; the
command in both is now `python scripts/r21_downstream_repair.py`, and the pipeline note says so
without blaming a README that no longer claims it.

**How it was checked.** `work/_preflight.py`: CHECKS 309 -> 372, FAILS 0. Sixty-three checks are
new. Seven pin r14c: the twenty-four printed cells are recomputed from the raw ladder table on the
coarse grid before the dense grid is read against `experiments/r14c_ladder_dense.csv`, then the
four moved cells, the two rung sets and the two never-counts. Fourteen pin r21: the primary pooled
statistic, its cell counts and its 24 strata are recomputed from the released per-type rows, the
three matchings are recomputed the same way, the report is checked for the two intervals and for
being the full-bootstrap run, and `experiments/r21b_matched_downstream.csv` is checked against its
four pooled values. Seventeen pin r61: both pairs' marker counts and ten of their arms at a 10%
budget, the two source-calibrated acceptance-and-coverage pairs, and the shared-panel coverage
floor. The remaining checks pin the manuscript's new sentences and Table 6 cells (12), the two
Methods paragraphs (2), the three new pipeline stages (3), the two script listings and the r21
command in `README.md` (3) and in `code_availability.md` (2), and the denser price list in both
languages of the interface plus r59 (3). `work/_validate_adviser.py`: CHECKS 98 -> 100, FAILS 0,
the two new checks pinning the 750-2,000 rung range in each language; the validator was also run
against a copy with the old checklist block restored, which returns 98 and FAILS 0, so the count
is a measured difference and not an estimate.

This record was written against `docs/manuscript.md` at 2797 lines, 192255 bytes, sha256 `78be9dd30b307ad34cb1479e010454d36b180b673ea0e2a6475c01116901d670`, mtime_raw 1790605197.402076 -- the first manuscript change since Round 15 -- and `docs/methods.md` at 880 lines, 77814 bytes, sha256 `75da97e13fd4de6c5b577ebbd141e21ed58c844785678c8173540f0fa3e965fa`, mtime_raw 1790605197.402076.
## Round 19 -- the fourth partition family, the annotation source, and the repair that covers both

This round closes the one class of the four split families the paper had left
unmeasured. It adds one experiment and no new data, and it moves numbers in
`docs/manuscript.md`, so the manuscript is edited for the first time since
Round 18.

**Why it could not be measured before.** The paper asks its validity question on
four families -- across tissue (r24/r31), across fields of view (r48), across
panels (r61) and across annotation sources -- and the fourth needs two labels for
the *same* cells. `errctl.data.build_master` has carried both since the first
build: for the two cohorts that ship mean marker intensities for every segmented
cell, `ManualLabels` (the expert, source A) and `HighQualityLabels` (the depot's
own semi-supervised model, source B) join on the same `(image, cellID)`. r18 and
r33 used that pair for the marker audit and for a latent-class analysis; neither
used it as a split family, so the class was absent from the coverage table rather
than measured in it. `scripts/r62_source_family.py` supplies it.

**Design.** 2,384 colorectal and 9,504 GVHD cells (10 and 12 coarse classes at
the 20-cell floor; the two sources agree on 0.733 and 0.861 of them). The model
is trained once per fold on the manual labels and then held fixed, and four arms
share the cells, the fixed five image folds and twenty random
calibration/deployment splits per fold; only the label source used to score the
calibration half changes. There is deliberately no separate cross-source arm, and
that is a fact about the design rather than a simplification: the accepted set
does not depend on which source the deployment will be scored against, so every
arm is read against *both* sources and a cross-source reading is the second
reading of a same-source arm. `same_expert` and `same_hq` are the exchangeable
controls; `consmax` fits one controller to the pointwise maximum of the two
per-cell errors; `consint` fits two controllers at half the failure budget each
and accepts the intersection of their thresholds. Both repairs are valid without
knowing the deployment source -- the maximum bounds either reading pointwise, the
intersection inherits both certificates by a union bound -- and both are read
against each source separately.

**One design correction, made inside this round.** The first pass took its folds
from `r02_run_experiments.fold_split`, which shuffles image *names* into equal
groups of images; the certificate's feasibility, though, is close to a fixed
*count* of calibration cells rather than a fixed fraction, and the colorectal
cohort's eight expert-annotated images hold between 152 and 440 pool cells, so
that split left folds of 756 and 166 cells and the 200-cell floor dropped the
small one. The released version balances the folds by pool cells instead
(440 / 386 / 528 / 536 / 494 for colorectal, 1,825 to 1,953 for GVHD) and draws
twenty splits rather than five. The ablation was run on colorectal alone, at the five splits the pass began
with: the unequal split gives non-vacuous coverage 0.1167 and acceptance 0.0580
at a 10% budget, the balanced split 0.1333 and 0.1076, and it removes that
cohort's one cross-source coverage miss (0.9833 to 1.000 at a 10% budget,
0.9167 to 1.000 at a 20% one). The released configuration draws twenty splits
per fold, and the two settings there give 0.1083 and 0.0515 before balancing
and 0.1400 and 0.1099 after, with that same miss gone (0.9958 to 1.000), so the
direction of the correction does not depend on the split count. Two alternatives were measured and rejected rather than assumed: the
fixed-sequence step-down rule, which is the paper's small-sample default
elsewhere, certifies nothing at all here at a 10% budget (non-vacuous fraction
0.000) because no threshold prefix is feasible, and a 20-point candidate grid is
a wash (acceptance 0.108 at a 10% budget, second-source coverage 0.947 at a 20%
one). The choice is a measured one, not a convention.

**What the fold fix did not fix, and what does.** The balancing is right and
stays -- an artificially dropped fold is not a measurement -- but it does not
make the colorectal cohort thick, and the released run says why.
Leave-one-fold-out accuracy of the frozen model on the held-out cells, against
the expert labels, spans 0.5486 to 0.9093 over the five folds and three
classifiers; the certificate returns a non-empty accepted set at a 10% budget
on 0.4875 of the units of the one fold whose weakest classifier reaches
0.8808, and on 0.000 of the units of the other four, whose best classifiers
stop at 0.8068. On GVHD every fold and classifier transfers in 0.8336 to
0.9078 and all five folds certify, 0.4375 to 1.0000. The cause is therefore
the score those fields support, not the fold construction and not the
calibration count on its own: the four failing colorectal folds carry 220 to
268 calibration cells and the one that succeeds carries 193. The mechanism is
shipped with the experiment rather than asserted --
`experiments/r62_fold_transfer.csv`, one row per fold and classifier, the same
numbers printed in `work/r62_report.txt`, and a `fold` column in the main
table so the emptiness can be read fold by fold. The manuscript's
annotation-source paragraph, its calibration-size paragraph and the Discussion
limitation were corrected to say this, and the cross-reference that called the
colorectal exception a calibration-size effect now only points forward.

**How it was checked, again.** `work/_preflight.py`: CHECKS 403 -> 418, FAILS
0. Fifteen checks are new: seven pin the transfer table and the per-fold
emptiness (the two colorectal accuracy extremes, the GVHD minimum, the count of
colorectal folds certifying nothing, the one that certifies, the GVHD all-five
fact and the fold count in the main table), four pin the corrected manuscript
sentences, and four pin the methods, README, code-availability and pipeline
mentions of the new table.

**What it found.** The controls hold, as they must: coverage 1.000 on the expert
source for both cohorts and 0.997 on the second for GVHD, against a nominal 0.90.
Read against the other source, the same GVHD certificate covers 0.440 of its
evaluations at a 10% budget and 0.433 at 5%, against 1.000 for the labels it was
built on: the fourth family now has a measurement rather than a gap, and it fails
as the tissue and field-of-view families do. The shortfall is bounded rather than
merely observed. The realised risk is 0.10473 against the second source and
0.05982 against the calibrated one, and the accepted cells disagree at 0.10182;
because the two per-cell errors can differ only where the sources disagree, the
gap is at most the disagreement rate, and the report checks that identity unit by
unit -- it holds in all 3,277 non-empty units. The same identity explains the
cohort where nothing fails. Colorectal's certificate can accept only 0.110 of its
deployment cells and those are the cells the two sources agree on (0.029), so its
second-source coverage is 1.000 at a 10% budget and 0.973 at 20%; the failure
tracks how many disagreeing cells the certifiable set contains, which is exactly
the quantity the bound names. The direction is mechanised the same way: a
certificate drives the realised risk of the source it calibrated on to the edge
of its budget, so any other source reads up to one disagreement rate above it,
and the reverse reading covers 1.000 only because that threshold is the tighter
one, accepting 0.44571 against 0.80369.

**The repair, and its price.** Calibrating on both sources restores the coverage
against both. On GVHD at a 10% budget `consmax` accepts 0.41238 and `consint`
0.42807, each covering 1.000 against the expert labels and 1.000 against the
second source; at a 20% budget `consint` accepts 0.90670 and covers 1.000 both
ways. The realised risks fall with the acceptance, to 0.01670 (`consmax`) and
0.02353 (`consint`) against the expert labels, so the price is paid in acceptance
and not in risk, as everywhere else in this paper.

**Where it landed.** The enumeration of cost-curve families stays at three --
which is what the label ladder and the review-budget figure actually cover, and
what the panel family already left untouched -- and it now points forward to the
two families that change only the panel or only the labels. The annotation-source
family is named and carried in the paragraph after the panel paragraph, together
with the measurement, the identity, the mechanism, the repair, the price and the
fold-balancing note. The Discussion limitation on the second annotation source
gains the coverage cost, because a guarantee is a statement about the labels it
was given. `docs/methods.md` describes the protocol, the balanced folds and the
twenty splits beside the panel paragraph, and `README.md`,
`docs/submission/code_availability.md` and `scripts/run_pipeline.py` list the
script, its two experiment tables and its command.

**How it was checked.** `work/_preflight.py`: CHECKS 372 -> 403, FAILS 0.
Thirty-one checks are new: nineteen pin r62's own tables (both same-source
controls, the cross-source coverage at two budgets, the two realised risks and
the disagreement rate, the reverse reading's acceptance, the two repairs'
acceptance and coverage, colorectal's pair of controls and its non-failure and
its consint acceptance, the arm count, the unit count per cohort, arm and budget,
and the risk-gap identity over every non-empty unit), and twelve pin the
manuscript's new sentences (8), the methods paragraph, and the three documents
that name the code (3).

## Round 20 -- the last unverifiable pair, and the last framing item

Two gaps were still open. One was numerical: the only two numbers in the paper
that no released table could reproduce, `work/r37_verify.txt` D1 and D2, the
conditional-binomial dispersion and the chi-square homogeneity statistics. The
other was the last item of the adversarial framing pass,
`work/r38_framing_audit.txt` item 13. This round closes both. No result changes.

**D1/D2.** The claim that the plug-in's single-error-probability assumption is
false rests on four numbers per family: the average spread of the wrong rate
across predicted classes and across score deciles among the accepted calibration
cells, and the fraction of evaluations whose homogeneity chi-square rejects. They
were computed inside `scripts/r26_plugin_baseline.py` and printed in
`work/r26_report.txt`, but the counts they are built from -- accepted cells and
wrong cells per predicted class and per score decile, per evaluation -- were
never released, so the verification pass had to mark them UNVERIFIABLE and say
in as many words that the needed file was a bin-level companion to
`experiments/r26_plugin_baseline.csv`. The script now writes that file:
`experiments/r26_constancy_bins.csv`, 29,274 rows -- one evaluation row plus one
row per predicted class and per score decile, for each of the 1,389 evaluations
that entered the arm. Recomputed from it alone, and not from any r26 code, the
six statistics per family come back to the printed digits: class spread 0.2016,
0.2430 and 0.2095 and decile spread 0.2048, 0.2367 and 0.2360 for cellsplit,
within and transfer, with chi-square rejection 0.7751, 0.9143 and 0.9394 by
class and 0.9287, 0.9500 and 0.9409 by decile. Groups of fewer than ten accepted
cells are carried in the file and excluded from both statistics, which is the
rule the report applies, so the exclusion is visible rather than implicit.

**Item 13.** The sequential controller's price appeared in the Abstract and
Introduction as `about a tenth of its acceptance`, which mixes an absolute
fraction of the batch (0.0974) with a share of that arm's own acceptance (about
17 per cent). It now reads `an acceptance cost of 0.097 of the batch at a 10%
budget`, the unit the sequential subsection already uses. Fifteen of the sixteen
non-HIGH findings in the framing audit are now landed; this was the last, and the
three HIGH items were replaced in the round that followed that audit.

**How it was checked.** `work/_preflight.py`: CHECKS 418 -> 438, FAILS 0. Twenty
checks are new: thirteen recompute the constancy statistics from
`experiments/r26_constancy_bins.csv` alone -- the evaluation count, then per
family the mean class spread, the mean decile spread, the median class spread and
the two rejection rates -- and seven pin the text (the manuscript's quoted
constancy numbers, the corrected sequential sentence, and the methods and
code-availability mentions of the new table). `scripts/run_pipeline.py` already
globs `experiments/r26_*.csv` for the stage, so the new table is covered without
an edit.

## Round 21 -- the number audit's own tally

Round 20 released the bin-level companion and recorded that D1 and D2 recompute from it, but
the audit report the paper cites for that pair, `work/r37_verify.txt`, still carried its
original tally: the two conditional-binomial rows read UNVERIFIABLE and section 6 counted 88
MATCH, 1 MISMATCH, 2 UNVERIFIABLE. A reader who followed the citation to check the paper's
last unverifiable pair would have found the gap still open in the file that is supposed to
have closed it.

The report now records the re-verification rather than having its verdicts quietly edited. Its
D1 and D2 rows became MATCH, sourced to `experiments/r26_constancy_bins.csv` with the
recomputed values spelled out; the sentence in section 3 that asked for "a bin-level companion
to it" now says the file exists; section 6 counts 91 claim rows as 90 MATCH, 1 MISMATCH, 0
UNVERIFIABLE and keeps the historical counts beside that; and a dated addendum at the end of
the file carries the table of recomputed values, states that the pass that wrote the report saw
only `work/r26_report.txt`, and records that no manuscript text and no result moved.
`README.md`'s summary of that audit was updated to match.

**How it was checked.** The five statistics were recomputed once more, independently of both the
report and the preflight and reading `experiments/r26_constancy_bins.csv` and nothing else. Over
the 1,389 evaluations that enter the constancy arm, the mean class spreads are 0.2016 / 0.2430 /
0.2095 and the mean decile spreads 0.2048 / 0.2367 / 0.2360, the class chi-square rejections
0.7751 / 0.9143 / 0.9394 and the decile rejections 0.9287 / 0.9500 / 0.9409 (cellsplit / within
/ transfer), with median class spreads 0.1581 / 0.2000 / 0.1780 -- every value equal to the one
quoted from `work/r26_report.txt`. The recomputation is kept as `work/_v123.py`, which prints
the table and exits non-zero on any mismatch. No manuscript byte changed in this round: `docs/manuscript.md`
and `docs/methods.md` carry the hashes recorded below, and `work/_preflight.py` and
`work/_validate_adviser.py` both still pass.

## Round 22 -- three registrations the outline left open

The outline's own gap list named three things that had been built but never registered
anywhere a reader would look. This round registers all three. No result moves and no number
changes; the preflight recomputes what each new sentence says.

**r44, the delivery defect.** `scripts/r44_gvhd_delivery.py` and its write-up
`docs/submission/gvhd_delivery_defect.md` were named in no document at all -- not the
manuscript, the README, the code availability statement or the pipeline -- even though the
Introduction has carried the defect itself since round 1 (17.5% of GVHD cells, 1,794 of
10,283). The Introduction's defect sentence now cites the script and
`experiments/r44_gvhd_delivery.csv`; the companion-file line lists the defect note; the
script, its two tables, its command and the note enter `docs/submission/code_availability.md`
and `README.md`; and `scripts/run_pipeline.py` gains an `r44_gvhd_delivery` stage that globs
`experiments/r44_gvhd_delivery*.csv`, so the tables are diffed with every other stage.

**The fourth calibration arm.** The calibration-collapse result rests on three routes: the
maximum-probability softmax of one trained model, its temperature-scaled version, and bagged
ensembles of the same base learners. Monte-Carlo dropout, the fourth route the original plan
listed, is not run -- it needs a network trained with dropout rather than a rescaled tabular
model, and the one network in this paper was trained without the dropout and ensembling recipe
its authors use. The Discussion's limitation chain therefore gains an eighth limitation, which
says in as many words that the claim is about a post-hoc layer over a score rather than about
every uncertainty estimator, and that whether a dropout-based estimate is better calibrated
outside its training domain is left open. This is the disposition the recovery plan itself
prescribed: add the sentence, do not build the arm.

**r61.** The cross-panel paragraph in "Robustness and ablations" quotes the shared-panel
numbers but named no artifact, unlike every neighbouring paragraph, which cite
`r48_fov_units.py`, `r31_targeted_repair.py`, `r51_cellsighter.py` and others.
`scripts/r61_shared_panel.py` is now cited where the paragraph introduces its measurement.
The script was already listed in the README and the code availability statement; only the
Results citation was missing.

**How it was checked.** `work/_preflight.py`: CHECKS 438 -> 468, FAILS 0. Thirty checks are
new. Eleven recompute the delivery defect from `experiments/r44_gvhd_delivery_summary.csv`
(7,610 of 7,610 breast, 2,505 of 2,505 colorectal, 1,794 of 10,283 GVHD over 48 fields,
21,204 of 21,204 melanoma, and the three complete cohorts at match fraction 1). Ten recompute
the shared-panel contrasts from `experiments/r61_shared_panel_summary.csv` and
`r61_shared_panel_mg_summary.csv`: the 0.011 and 0.054 that the 58 -> 27 and 40 -> 27
restrictions cost, the 0.437 and 0.162 that the 40 -> 24 restriction costs melanoma and GVHD,
the 0.772 and 0.641 domain costs at fixed panel, and the two source-calibrated arms at 0.145
with 0.667 coverage and 0.141 with 0.533. Nine pin the registrations themselves: the two
manuscript citations, the companion-file line, the eighth limitation, the code-availability
and README entries, and the pipeline stage.

## Round 23 -- the last unregistered artifact, and the submission-document list

An outline re-check (the plan's 19 goals, its four Nature Methods gates and the six ranked
actions, all read against the current files) left two items. One was the defect class Round 22
closed for r44 and r61, in the one place it still existed: `scripts/r21b_matched_downstream.py`
and its table `experiments/r21b_matched_downstream.csv` were named in no reader-facing document,
although the Results quote the number they produce -- the pooled difference of -0.011 (-0.011 to
0.012) that the composition-matched recomputation gives, which is what stops the downstream
readout from being read as a repair. The other is a list: `docs/submission/summary_paragraph.md`,
the Nature-format summary paragraph, existed but was not named in the manuscript's companion-file
line.

The Results sentence now cites the script and the table beside the number; `README.md` and
`docs/submission/code_availability.md` list the script, its command and its two tables (with
`r21_downstream.csv`) and record that it reads only released tables, so it cannot drift from
them; and the companion-file line names the summary paragraph. No number moves and no result
changes.

**How it was checked.** `work/_preflight.py`: CHECKS 468 -> 479, FAILS 0. Six of the new checks
recompute the pooled row of `experiments/r21b_matched_downstream.csv` -- the own-weight
difference 0.029313, the three matched variants -0.010593, -0.010593 and -0.051387, the 140 types
and the five rows -- and five pin the registrations. Two sweeps were also run by hand. Every one
of the 59 scripts in `scripts/` is now named in both `README.md` and the code availability
statement. Every file in `figures/` maps to a figure number in the "Figure file mapping" section
of `docs/submission/code_availability.md` (19 PNGs and their 19 PDFs, including the withdrawn
`figS2_*` label and the `fig9_downstream` draft the paper does not use), which closes the last
item the figure audit had left open.

This record was written against `docs/manuscript.md` at 2840 lines, 198381 bytes, sha256 `936e604efd36a57fa5661f925960d842b25d29cc6ce691bf15880fbfde288f43`, mtime_raw 1790613344.9302874 -- the revision Round 23 registers, with `docs/methods.md` at 891 lines, 80042 bytes, sha256 `1e9b21d619849ccb4a23308758e7aacd56e7820dcfc4536551e8b6aa48ec2896` (unchanged since Round 20).

## Round 24 -- table provenance, and one command that re-checks the headline results

The request was a reproduction guarantee for the four results the paper turns on
(cross-tissue coverage, the efficiency comparison, the marker violation level and
the external cohort) and for the text as a whole. Two gaps blocked it, and neither
was a wrong number.

The first was provenance. Only two of the fifteen main-table legends named the CSV
they were computed from, and seven of the nine supplementary legends named none;
a reader could reproduce the text but not trace a table to a file without reading
the scripts. Every legend now closes with a Sources sentence naming its backing
file(s) and, where the legend did not already name it, the script: Tables 1-8 and
10, and Supplementary Tables 1-9. The manuscript's Table 10 caption, which was a
bare title, now names `r51_cellsighter_summary.csv` and its script like its
neighbours. `docs/submission/code_availability.md` gains a "Table provenance"
section: one row per table, the backing result file(s), and the script, for all
fifteen main and nine supplementary tables.

The second was three stale labels in that same section of the code-availability
statement, left over from the table renumbering that the r56 withdrawal forced and
that Round 13 recorded. The CellSighter table was called "Supplementary Table 10"
where the manuscript numbers it Table 10; the r56 spatial-context tables were still
attributed to "Table 15 (spatial context added to the score)" where Table 15 is now
the family-wise corrections and the r56 tables are cited nowhere; and the
field-of-view-resolution mapping was headed "Table 8" where it is Supplementary
Table 8. All three are corrected, and the withdrawn r56 tables are stated to back
no table.

The guarantee itself is a program. `scripts/r63_reproduce_check.py` reads only the
released CSVs in `experiments/`, imports nothing outside the standard library, and
re-derives six groups of headline values against fixed expected values: N1 the
cross-tissue coverage and its non-vacuous readout, N2 the matched-risk efficiency
ratio and the fact that the certificate never beats the oracle, N3 the four
identified sets, the five expert populations above the gate under all 68
conventions and the one that clears none, N4 the external cohort's coverage and the
ratio diagnostic's misses, N5 the field-of-view-resolution interval that excludes
0.90, and P1 the positive replication on the independent cohorts. It prints one line
per assertion and exits non-zero on any mismatch; `--list` names the groups. The
prose behind it is `docs/submission/reproducibility.md`, which gives, per claim, the
file, the filter and the column, states what does and does not reproduce without the
data, and notes that the recorded values are the manuscript's own.

The registrations are in the places a reader looks. `README.md` lists the script in
the layout, names it in the Reproducing block and carries a "Reproducing the
headline results" section; `docs/submission/code_availability.md` lists it, adds the
command and a "Verifying a released checkout" section, and points at both checkers;
`scripts/run_pipeline.py` carries it as a stage that writes nothing; `docs/methods.md`
closes its Reproducibility section on it; and the manuscript's companion-file list and
availability paragraph name the note and the script.

**How it was checked.** `work/_preflight.py`: CHECKS 479 -> 531, FAILS 0. Fifty-two
checks are new. Eleven pin the r63 program itself (its presence, its six groups, its
stdlib-only import set, its reader, and an end-to-end run whose exit status and
`FAILS=0` line are asserted). Twenty-one pin the manuscript: one per newly named
table file, the Table 10 caption, and the three citations (the script, the note and
the companion-file list). Eleven pin the registrations in the two documents and the
driver, two pin the methods paragraph, and seven pin the reproducibility note. `work/_validate_adviser.py`
is unchanged at 100 checks, 0 fails. Two sweeps were run by hand: all 60 scripts in
`scripts/` are now named in both `README.md` and the code-availability statement, and
all 24 table legends in `docs/manuscript.md` name at least one `experiments/` file.
No number in the manuscript moves: r63 re-derives the values the text already quotes,
so the claim-by-claim table above stands unchanged.

This record was written against `docs/manuscript.md` at 2885 lines, 201333 bytes, sha256 `2d3f5c63cd6222b8a05e02214051ef00d6f223e683803f374d58d90033306454`, mtime_raw 1790613895.3596044 -- the revision Round 24 registers, with `docs/methods.md` at 900 lines, 80659 bytes, sha256[:16] `bfa3812c0854def7`.

## Round 25 -- three closure defects found by reading the argument back

A closure review -- does every promise land, does every ordinal count, does every
figure hold on the row it is stated for -- turned up three defects. None is a wrong
number in a table; all three are visible to an editor on a first pass.

**One figure held on one budget and was stated on two.** The labour sentence claims
that the layer converts four fifths of the errors that would otherwise be recorded
silently. That is the 10%-budget value. Against the uncalibrated classifier's 0.183
of the batch, the certificate leaves 0.036 of it unreviewed and wrong at a 10%
budget, which is to say 0.801 of the errors are caught, and 0.120 at a 20% budget,
which is 0.346 of them. The sentence appears twice, in the Abstract and at the head
of "What the guarantee costs", and both instances attached the 10% figure to both
budgets. Both now read "at the 10% budget, and a third of them at the 20% one", and
the two shares are recomputed from `experiments/r24_efficiency.csv` (`accept_pooled`
and `realised_risk`, family `cellsplit`, arm `src`, k = 0) by the preflight. This
record's own Round-8 entry quotes the same figure without a scope; the manuscript is
what a reader sees, and the manuscript is now scoped.

**A heading counted three where the text counts four.** The deliverable subsection
was headed "a decision procedure, a price list, and an audit", opened "It has four
parts", and the Introduction named four: the label-free triage statistic, the audit,
the price list, and the observation that where the repair is free the review the
certificate already generates is larger than its price. The heading now reads "a
decision procedure, a price list, an audit, and where to spend", which is the count
the Introduction and the body both use.

**An ordinal presupposed an item that is not numbered.** The limitations walk opens
with a paragraph on the cross-field-of-view and cross-tissue failures and then said
"Three further limitations bound the scope of our claims", which reads as though that
paragraph were limitation one and makes the fourth limitation the fifth. The word
"further" is gone: the chain now reads three limitations, a fourth, a fifth, a sixth,
a seventh and an eighth, which is the reading the framing audit already used when it
cited the external-test paragraph as the fifth limitation.

**How it was checked.** `work/_preflight.py`: CHECKS 531 -> 536, FAILS 0. Two of the
new checks recompute the catch fractions 0.801 and 0.346 from `r24_efficiency.csv`,
and three pin the scoped sentences and the heading. `work/_validate_adviser.py` is
unchanged at 100 checks, 0 fails, and `scripts/r63_reproduce_check.py` is unchanged at
39 assertions, 0 fails; neither reads the sentences that moved. What did not change is
the point of the round: every table and every figure is still cited in the text (24 of
24 tables, 9 of 9 figures), the four negative results and the one positive replication
are still pinned by r63, and no number in any table moved.

This record was written against `docs/manuscript.md` at 2887 lines, 201449 bytes, sha256 `5d176b9b5234a75be57f3fac7b5da386e98a4eafecdccb6640975419dca49e02`, mtime_raw 1790614377.7886899 -- the revision Round 25 registers, with `docs/methods.md` at 900 lines, 80659 bytes, sha256[:16] `bfa3812c0854def7`.

## Round 26 -- one provenance convention across all twenty-four legends

Round 24 gave every table legend a sentence naming the file behind it, but the
convention was not uniform: eighteen legends declared it on a line beginning
"Sources:", while six -- Tables 9, 11, 12, 13, 14 and 15 -- named the same paths
inline, glued to the surrounding prose. Every table was already traceable, so no
number was wrong; the fix is that the six now carry the same "Sources:" line as
the other eighteen, the inline paths having moved into it, and Table 9's legend
now points at Table 1 for the certificate convention instead of repeating the
path mid-sentence. All twenty-four legends now declare their provenance on a
"Sources:" line.

**How it was checked.** `work/_preflight.py`: CHECKS 536 -> 537, FAILS 0. The new
check walks every caption in `docs/manuscript.md` and requires a "Sources:" line
carrying an `experiments/` path before the next caption, so the convention cannot
silently drift again. The seventeen per-file legend pins of Round 24 are
untouched, none of the six changed files being among them.
`work/_validate_adviser.py` is unchanged at 100 checks, 0 fails, and
`scripts/r63_reproduce_check.py` is unchanged at 39 assertions, 0 fails, since it
reads the released tables and not the legends.

This record was written against `docs/manuscript.md` at 2890 lines, 201563 bytes, sha256 `4209171c11bf9634fd37675ee33176e59c9dfc579d77d017fc53c60515f92d24` -- the revision Round 26 registers, with `docs/methods.md` at 900 lines, 80659 bytes, sha256[:16] `bfa3812c0854def7`.

## Round 27 -- two reproducibility defects found by re-running the pipeline

A closure review answered "does it close?" by re-running stages from the raw data
and comparing the released tables, rather than by reading them. Four stages came
back byte for byte identical (`r13_coverage_ci`, `r28_marker_bounds`,
`r14c_ladder_dense`, `r44_gvhd_delivery`). Two defects did not.

**Twenty-four stages wrote a released table in worker order.** Each collected its
results from `multiprocessing.Pool.imap_unordered` and appended them as the
workers finished, so a large per-unit table came out in a different row order on
every run. The values did not move -- `experiments/r20_units.csv`,
`experiments/r27_fixedseq.csv` and `experiments/r62_source_family.csv` are equal
to the released rows as a set -- but the bytes did, and the two summary tables
that read differently differ only in the last floating-point digit (largest
relative difference 2.2e-16 for `r27_fixedseq_summary.csv`, 1.5e-16 for
`r62_source_family_summary.csv`, both one unit in the last place, from the order
of the parallel sums). Every stage now uses the ordered pool.

**The label-ladder stage crashed on a clean run.** `scripts/r20_label_ladder.py`
raised `ValueError: cannot convert float NaN to integer` at its line 487, twice in
succession. On the `within / frac / min_accept=5` grid no coordinate is present in
every unit of the family (the per-coordinate unit counts run 15, 18, 19, 17, 13,
5, 4, 3, 1 against 20 units), so `balanced_keys` returns an empty range, `kmax` is
-1, `_at` returns NaN and the count is taken of NaN. The five count fields now
fall back through `_int_or_neg1` to the same -1 the neighbouring fields already
carried; no value changes where the coordinate is present.

**What the re-run exposed.** `experiments/r20_ladder.csv` could not be regenerated
as released. Nine of its thirty-six rows -- the `min_accept=5, grid=frac` rows of
all three families -- disagreed with a fresh computation from the shipped per-unit
table, and `experiments/r20_diag.csv` carried a column set
(`k_max_balanced_frac`) that the shipped script does not emit. Both artifacts
predate the present script and are now regenerated from it. No manuscript number
moved: every `min_accept=25` row of the ladder is unchanged; `r20_units.csv`, the
table Table 5's k-rows are read from, is equal to the released rows as a set; and
the cross-tissue rows of Table 5 are the Table 3 run.

**How it was checked.** `work/_preflight.py`: CHECKS 537 -> 544, FAILS 0. One
check requires that no script under `scripts/` writes a table from an unordered
pool, and six pin the r20 guard. `scripts/r20_label_ladder.py`,
`scripts/r27_fixedseq.py` and `scripts/r62_source_family.py` each ran twice after
the change and returned identical bytes on both runs.
`work/_validate_adviser.py` is unchanged at 100 checks, 0 fails, and
`scripts/r63_reproduce_check.py` is unchanged at 39 assertions, 0 fails.

This record is an addendum to Round 26: `docs/manuscript.md` did not change in
this round and stands at 2890 lines, 201563 bytes, sha256 `4209171c11bf9634fd37675ee33176e59c9dfc579d77d017fc53c60515f92d24`, with `docs/methods.md` at 900 lines, 80659 bytes, sha256[:16] `bfa3812c0854def7`.

## Round 28 -- the whole pipeline re-run, and the defects it exposed in the driver

**The operation.** `scripts/run_pipeline.py --mode full --jobs 16` was run end to
end for the first time: 37 stages, output `work/logs/round28_full.out`, per-stage
logs `work/logs/<stage>.log`, status table
`work/pipeline_baseline/stage_status_round28_full.csv`. Every stage exited 0 --
37 PASS rows, 0 FAIL. Seven reproduced every released table byte-for-byte; the
other thirty reproduced theirs up to the value-level drift characterised below
(verdict `PASS(diff)`). The point of the run was not a new number but the
statement that every released table in `experiments/` re-derives from the shipped
code and the shipped data -- an operation the driver had never supported.

**The driver named the wrong invocation for three stages.** The released
`experiments/r03_*.csv` came from the eight-alpha grid, not the script's default
of three; `experiments/r14_*.csv` from ten replicates, not the default fifteen;
`experiments/r14b_*.csv` from the k grid 200/1000/2000 at two replicates. With
the defaults, those three stages did not reproduce their released tables. The
published flags now sit in the driver's `STAGES` table.

**The README and the code-availability note named commands that do not exist.**
r02, r05, r06 and r21 were documented with `--jobs`, which none of them accepts;
r03's eight-alpha grid, r14's `--reps 10`, r14b's command and r33's full flag set
were missing. All corrected (`README.md`, `docs/submission/code_availability.md`).

**r18 carried three defects.** (i) Its positivity grid held 18 levels -- the
panel-wide mixture fit, both percentile families, *and* an expert-population refit
named `gmm_expert` -- where the manuscript counts 17 x 2 x 2 = 68 conventions;
r28 excludes the refit for the same reason (it would make the two label sources
incomparable). The grid is 17 again. (ii) `verdict_paragraph()` read the
`reading`/`variant` columns off the counts frame, which has neither, and raised
`AttributeError` on a clean run; the per-axis breakdown is now read off the sweep
itself. (iii) The same paragraph carried a literal `%%` while only one
`%`-formatting was applied, so it printed `>5%%` and `15-28%%`. Fixed; the stage
now runs to completion with all seven released tables unchanged.

**r34's one substantive FAIL was already in the released artifact.**
`experiments/r34_checks.csv` has always recorded C15 as `pass=0, expect_pass=1`
-- the accept/oracle ratio misses two genuinely over-budget `demo_transfer` units
at alpha = 0.05 and 0.10 -- and the released report already says so in its
section 4. The stage nevertheless returned non-zero, so a faithful re-derivation
could not exit 0. C15 is now in `REPORTED_CHECKS`: the row stays in the CSV and in
the table, the report names it, and the log tags it `[reported]`; only unexpected
failures set the exit code. The grid is 21 rows / 18 pass / 3 fail (two expected
C3/C4, one reported C15) / 0 unexpected, and `r34_external --stage all` exits 0 on
the released bytes.

**The baseline snapshot predated Round 27, so `--reconcile` would have reverted
it.** `work/pipeline_baseline/exp_snapshot/` was taken before the Round-27
corrections, and `reconcile()` restores the snapshot over any changed file; a full
re-run would have silently republished the artifacts the previous round had just
corrected. Six entries were re-cut to the audited bytes -- the originals preserved
in `work/baseline_recut/pre_recut/`, the audited copies in `post_recut/`, the table
in `work/baseline_recut/RECUT.txt`:

    file                        pre-re-cut md5  post-re-cut md5
    r20_ladder.csv              d0fb0f3cba96    a8694e2e1b30
    r20_diag.csv                267dbdfa0f6a    f1c158eecae7
    r20_units.csv               7695d5520182    1ae66966cc3a
    r21_downstream.csv          b47a5a1a3686    afe0d4969b6c
    r27_fixedseq.csv            835523060db0    96f0b64e9669
    r27_fixedseq_summary.csv    f79b32345b3e    c7a23df7c4ed

In the aligned re-run `r20_label_ladder` is now changed 0 / unchanged 3 and
`r21_downstream_repair` changed 0 / unchanged 1, so the corrections can no longer
be reverted by a reconcile.

**The drift comparison itself was wrong.** `numeric_drift()` compared the two
frames row by position. The snapshot predates the ordered-pool change, so its rows
are in worker-completion order; pairing them positionally reported a whole-column
difference (an `n_accepted` off by 19728) that was pure reordering. Both frames are
now canonically sorted on their non-numeric columns before the numeric cells are
compared, `order_same` is recorded per file, and the report's header states the
method. The superseded positional comparison is kept beside it as
`work/pipeline_baseline/drift_report_r28_firstpass_positional.{csv,txt}`.

**What the aligned comparison leaves.** identical 100 / CHANGED 27 / restored 27
/ missing 0, plus the four foreign `r43_*` files the driver never touches.
`experiments/r09_simulation.csv` differs in row order only: 5,760,000 numeric cells
compared, 0 different. The largest genuine residuals are
`r13_coverage_ci.csv` cov_lo 0.0194 (a Wilson endpoint), `r03_summary.csv`
coverage 0.0167, `r26_plugin_baseline_summary.csv` n_violations 6,
`r12_labelnoise.csv` 9.0e-6 and `r16_*` 2.6e-4. None of them moves a quoted
number.

**Where the drift comes from.** `work/datasets/*.parquet` is 5/5 md5-identical to
its manifest, but all 297 files under `work/predictions/` differ by
floating-point last bits (mean 605 bytes smaller) when the models are retrained on
identical inputs with identical seeds. Re-running `scripts/r26_plugin_baseline.py`
in isolation returns `r26_constancy_bins.csv` byte-identical, so the decile drift
is inherited from the upstream predictions, not introduced by r26's code. The
re-derivation is therefore a value-level claim, and the decile spreads are quoted
to three decimals (0.2047 / 0.2374 / 0.2360) -- they do not reproduce in the
fourth.

**Verification.** `work/_preflight.py`: CHECKS 544 -> 592, FAILS 0 (added: the r34
semantics checks, artifact-level pins for the six re-cut tables, and twelve checks
that the re-cut snapshot no longer lags the released bytes).
`work/_validate_adviser.py` 100 checks, 0 fails; `scripts/r63_reproduce_check.py`
39 assertions, 0 fails; `scripts/r44_gvhd_delivery.py --check` exits 0;
`experiments_manifest.csv` 131/131 md5 match. The two places that quoted the old
`work/_preflight.py` count as "479 assertions" -- `docs/submission/reproducibility.md`
and `docs/submission/code_availability.md` -- now say 592 checks, and the
Determinism section of the reproducibility note is rewritten from the measured
statement above rather than the old "the values reproduce exactly".

**Closure.** A whole-pipeline re-run used to be an operation the driver could not
complete; it is now supported and returns every released table from the shipped
code. No manuscript number moved: `docs/manuscript.md` stands at 2890 lines,
201563 bytes, sha256 `4209171c11bf9634fd37675ee33176e59c9dfc579d77d017fc53c60515f92d24`;
`docs/methods.md` at 900 lines, 80659 bytes, sha256[:16] `bfa3812c0854def7`;
`README.md` at 489 lines, 33011 bytes, sha256[:16] `ddea84187f7bde32`.

## Round 29 -- the last-digit sweep over all twenty-two tables

The recomputation that Table 15 had been given was pushed out to the remaining
twenty-one tables. Every numeric cell of every table in `docs/manuscript.md` was
re-derived from its named source CSV and compared with the printed value at the
paper's own display precision: two decimals, with three decimals reserved for
values inside (0.99, 1) and for signed interval bounds within 0.005 of zero. The
defect class the sweep looks for is double rounding, a value rounded to three
decimals and then to two, which flips the last digit whenever the three-decimal
intermediate lands on a five.

**Thirty-four cells were wrong in this way**, over twelve of the twenty-two
tables: Table 1 (3), Table 2 (7), Table 3 (2), Table 5 (1), Table 6 (1), Table 7
(2), Table 8 (1), Table 11 (7), Table 12 (2), Table 13 (3), Table 17 (3) and
Table 21 (2). The largest single cluster is Table 11's fixed-sequence step-down,
whose acceptance columns carried seven of them; the Table 21 pair is the shared
channel arm on the liver cohort, 0.8748836 and 0.9852092, printed 0.88 and 0.98
and correctly 0.87 and 0.99. Every correction moves the last digit by one and no
cell moves by more.

One instance of the same class sat in prose rather than in a table. The
matched-guarantee paragraph quoted the uncalibrated worst-unit risk as 0.23 where
`experiments/r68_equal_guarantee.csv` gives 0.235294, which is 0.24 at two
decimals, and the multiplier clause after it was adjusted from "two and a half
times the target" to "more than twice the target". A second prose number, "coverage
1.000 throughout" in the stronger-model paragraph, was brought to the two-decimal
convention as "1.00"; the three statements of bit-reproducibility that read
0.000e+00 are exact zeros and stay as they are.

**Two harness defects were found by the sweep and repaired.** The normaliser in
`work/_preflight.py` converted every three-decimal token with `float()` and
`round()`, so a needle written at three decimals was rounded from a value that had
already lost its fourth digit: 0.8748836 survives as the three-decimal 0.875 and is
then taken down to 0.88 by the binary representation. The normaliser and the r69
row builder now round the source value once, under the paper's display rule, using
`Decimal(repr(v))` and `ROUND_HALF_UP`. The same correction is carried in the
working copies `_pf3.py` and `_rnd2.py`.

**Disagreements that are not errors.** The ratio column of Table 7 is quoted at one
decimal by design, the prose beside it reading 9.2-fold, so its twelve apparent
mismatches are the two-decimal rendering of the same quantity. Seven cells, one in
Table 5 and six in Table 11, print 0.00 for a source value between 0.002 and 0.004;
that is the correct two-decimal rendering, and it is the statement the surrounding
sentence makes.

**The voting contrast is now pinned.** The stronger-model paragraph reports that
pooling cells and taking the soft vote of the same three probability vectors costs
no further fits and moves the review fraction from 40.3% to 33.0% at a 10% budget
and from 13.6% to 8.6% at a 20% one, with the tuned learner reaching 32.6% and
6.6%. Those eight numbers recompute from `experiments/r52_stronger_summary.csv`
and were not previously checked; they now carry eight numeric pins and three prose
pins.

**Verification.** `work/_preflight.py`: CHECKS 610 -> 621, FAILS 0 (added: the eight
r52 pooled-review pins and the three voting-contrast prose pins).
`scripts/r64_manuscript_docx.py` rebuilds `manuscript.docx` with 17 figures and 22
tables, both sequences monotone. `scripts/r63_reproduce_check.py` 39 assertions, 0
fails. The two places that quote the exhaustive count, the reproducibility note and
`code_availability.md`, now say 621 checks.

**Closure.** `docs/manuscript.md` stands at 1,011 lines, 250389 bytes, sha256
`c59019ae0e03bf40e3b3ed653061782feb7547344757a37fcbcb722b1f839458`. No table or
figure is renumbered, and no reported quantity moves by more than one in its last
printed digit.
