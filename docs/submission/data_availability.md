# Data availability

## Source data

All primary data analysed in this work come from **CellTuneDepot** (Cell Label Database, Keren Lab, Weizmann Institute of Science), a curated collection of multiplexed-imaging datasets with expert-validated cell labels, together with the companion **CellTune demo dataset**.

- Download: **https://celltune.org/celltunedepot**
- Files as retrieved: `CellTuneDepot-v0.2.zip` and `CellTune_Demo_Data.zip`
- Citation: Bussi et al., *CellTune: An integrative software to expedite accurate cell classification in spatial proteomics*, bioRxiv (2025).

The companion **CellTune demo dataset is not an independent study**: its
40-marker panel is identical, name for name, to the GVHD_Feb1323 panel; 38 of its
75 fields of view also carry benchmark expert labels; and 3,420 of its 3,460
expert-labelled cells match the depot's GVHD HighQualityLabels table by
`(image, cellID)`. It is therefore a third annotation pass over the same
material, not an external cohort. (Sources: `experiments/r34_panel_overlap.csv`,
`experiments/r34_overlap.csv`, `experiments/r34_md5_identity.csv`.)

Four cohorts were used, comprising 41,602 manually annotated cells (39,684 usable after excluding `Unidentified` and non-cell labels) across marker panels of 22, 58, 40 and 40 channels:

| Cohort | Segmented cells | Images (FOVs) | Expert-labelled cells | Usable | Panel (channels) |
|---|---|---|---|---|---|
| Breast_Jun0524 | 7,610 | 30 | 7,610 | 7,450 | 22 |
| CRC_Feb2623 | 94,389 | 35 | 2,505 | 2,397 | 58 |
| GVHD_Feb1323 | 175,548 | 84 | 10,283 | 9,504 | 40 |
| Melanoma_Oct2522 | 21,204 | 10 | 21,204 | 20,333 | 40 |

(Source: `work/r01_summary.csv`; panel sizes confirmed in `work/s01_inventory.txt`.)

## External validation data (STELLAR CODEX)

An independent external check uses the STELLAR / Hickey CODEX cell tables:

- Record: **Zenodo record 6863872** (DOI `10.5281/zenodo.6863872`)
- Data DOI: `10.5061/dryad.g4f4qrfrc`
- URL: **https://zenodo.org/records/6863872**
- Licence: **CC0-1.0** (Zenodo licence id `cc-zero`)
- Files used, downloaded and verified against the published Zenodo checksums:
  - `B004_training_dryad.csv` - md5 `c52671c4bf1edd1357709aec83cfb836`,
    248,285 usable cells, 48 markers, human intestine (small intestine and colon,
    donor B004); expert annotation released as the STELLAR training set.
  - `BE_Tonsil_l3_dryad.csv` - md5 `a274fe7dab20e0e4a41339962cabee26`,
    219,926 usable cells, 44 markers, human tonsil and Barrett's esophagus;
    expert annotation released for the STELLAR validation set.
- Both files are cached locally under `data/external/`, and each cohort is
  subsampled once to 20,000 cells with seed **20240618** (`scripts/r34_external.py`).
- Only two of the three deposited files are used: `B0056_unnanotated` carries no
  cell-type labels, so no realised risk can be computed on it and it is excluded
  rather than used as an unlabelled stand-in.
- The external class vocabulary is used exactly as deposited and is not mapped
  onto `errctl.labels.COARSE_MAP`, so no external number is comparable class by
  class with a benchmark statistic. The B004 cohort has 8 image-disjoint regions
  and supports the paper's cross-field-of-view family; the tonsil and Barrett's
  esophagus samples are single images and are used only as transfer targets.

## Further external validation data (MIBI-TOF and IMC)

Two further independent cohorts are used in the same external-validation design,
neither sharing a laboratory or a platform with the depot or with STELLAR.
Both are released as flat per-cell tables carrying marker intensities and the
depositors' own cell type labels, both under **CC-BY-4.0**, and both are public
and redistributable.

- **Multi-tissue tissue microarray, multiplexed ion beam imaging (MIBI-TOF)**
  - Record: **Zenodo record 5945388** (DOI `10.5281/zenodo.5945388`)
  - URL: **https://zenodo.org/records/5945388**
  - Companion publication: Liu C C, Bosse M, Kong A, et al., *Laboratory
    Investigation* (2022), DOI `10.1038/s41374-022-00778-8`
  - File used: `MIBITOF_Angelo_DataTables.zip`, 65,495,275 bytes, md5
    `2e2670deb84225889b730bbbe2b5cc88`, member
    `cell_table_size_normalized_clusters.csv`
  - 345,490 cells, 165 fields of view, 15 markers, 8 phenotype labels
- **Human liver allograft biopsies, imaging mass cytometry (Hyperion)**
  - Record: **Zenodo record 7570242** (record URL
    **https://zenodo.org/records/7570242**; the record carries the study DOI
    `10.3389/fimmu.2022.831103` rather than a separate Zenodo DOI)
  - URL: **https://zenodo.org/records/7570242**
  - Companion publication: Ung N, Goldbeck C, Man C, et al., *Frontiers in
    Immunology* (2022), DOI `10.3389/fimmu.2022.831103`
  - Files used: `LiverTx_IMC_processed_data.csv`, 70,003,245 bytes, md5
    `16e3981cc7036615a499f622e263c9aa`, and `LiverTx_IMC_metadata_samples.csv`,
    751 bytes, md5 `026bb6751a49ba9e38717c87e7aed594`
  - 109,245 cells, 32 images over 23 tissue blocks, 11 channels of which nine
    carry an antibody and two are the iridium DNA channels, 12 phenotype labels

Both are cached locally under `data/external2/` and driven by
`scripts/r69_external2.py`, which reads them in place and writes
`experiments/r69_external2_log1p_z.csv`, `..._summary.csv`,
`experiments/r69_external2_panel_overlap.csv` and
`experiments/r69_external2_checks.csv` (report in `work/r69_report.txt`).

Four further candidates were retrieved or inspected and **not** used, and the
report records why: a Dryad CODEX colon table whose depositor is the same as
STELLAR's, so it would not be independent; a Bodenmiller breast IMC record whose
per-cell intensity table ships only inside a 36.8 GB archive; an IMMUcan record
whose labelled cells ship only as R objects; and a Manchester glioblastoma IMC
deposit with 40 channels and 273,562 cells but no cell type label column.

## Real antibody-panel change (adult duodenum CODEX)

The cross-panel replication replaces the antibody panel rather than dropping
channels. It trains on the STELLAR small-intestine table and deploys to an
independent adult human proximal-intestine CODEX atlas.

- Record: **Zenodo record 10724499** (DOI `10.5281/zenodo.10724499`)
- URL: **https://zenodo.org/records/10724499**
- Licence: **CC-BY-4.0**
- Files used, read in place from the record:
  - `raw_mat.csv`, 72,258,998 bytes - the per-cell marker matrix
  - `cells.csv`, 21,649,466 bytes - the depositors' cell type labels and
    ROI identifiers
- 211,424 labelled cells, 42 channels, 3 donors, 107 ROIs
- Fourteen markers are shared with the 48-marker STELLAR panel: CD31, CD4, CD8,
  CD44, CD3, CD90, aSMA, CD206, CD138, CD19, CD163, HLA-DR, PDPN and VIM
  (`work/duodenum_shared_markers.csv`). Marker rows were fixed from the
  published panel and checked against one-versus-rest cell-type signatures
  before any error endpoint was computed.
- Target labels enter the calibration half only; the classifier is trained on
  STELLAR and never sees a target label. The run is
  `scripts/r85_duodenum_crosspanel.py`, and the released summary is re-derived
  by `scripts/r86_verify_duodenum.py`.
- Only the two flat CSV members above were downloaded. The 25 GB
  `zenodo_duodenum.zip` project archive is not needed and was not used.

## Licence and redistribution

CellTuneDepot is licensed, not sold, by the Weizmann Institute of Science under a **non-exclusive, non-transferable licence for non-commercial academic research and educational use at a degree-granting institution**. The licence explicitly prohibits commercial use and prohibits redistributing, sublicensing, publishing, renting, leasing or selling the database or any portion of it.

**Consequently this work does not redistribute the data, in whole or in part, and no data copy is included in the code repository.** Users must obtain CellTuneDepot directly from the official URL above and accept its licence. Note also that datasets associated with unpublished studies are released only upon publication of their respective manuscripts.

## What this work ships instead of data

The repository ships the **retrieval and preprocessing path**, not the data:

- `scripts/s00_inspect_data.py`, `scripts/s01_inventory.py`, `scripts/s02_probe_demo_and_join.py`, `scripts/s03_probe_images.py` — inventory and integrity audit of a locally obtained depot (writes `work/s00_data_report.txt`, `work/s01_inventory.txt`, `work/s02_probe.txt`, `work/s03_images.txt`).
- `scripts/r01_build_datasets.py` — builds the per-cohort cell tables from the depot tables and caches them to `work/datasets/*.parquet`, emitting the data-audit numbers (`work/r01_dataset_audit.txt`).
- `scripts/r34_external.py` - downloads the two CC0-1.0 STELLAR CODEX tables when absent, verifies their md5 against the published checksums, and builds the external cell tables (report in `work/r34_report.txt`).
- `src/errctl/data.py` and `src/errctl/preprocess.py` — depot table loading, expert/HQ label joining, TIFF decoding, segmentation-mask aggregation and marker-intensity normalisation.

Placement expected by the scripts (all paths configurable in `src/errctl/paths.py`):

```
data/depot/CellTuneDepot/{ManualLabels,HighQualityLabels}/<cohort>/{Images,Tables}/
data/demo/CellTune_Store/...
```

## Reproducibility

With the depot placed as above and the environment installed, the full analysis is reproduced by running, in order, `scripts/r01` through `scripts/r11`. The depot itself is retrieved by hand from the official URL, because its licence has to be accepted at the point of download and nothing in this repository may redistribute it; `scripts/r34_external.py` downloads the two CC0-1.0 STELLAR CODEX tables automatically and verifies each against its published md5 before use.

## Derived data generated by this work

The following derived artefacts are released with the code, since they contain no redistributable source data beyond label/score summaries:

- `experiments/*.csv` — coverage, efficiency, calibration, marker-audit, ablation, simulation and panel-reduction result tables.
- `work/predictions/*.npz` — per-fold model scores and predictions keyed by cohort/fold/classifier.
- `figures/*` — figure source panels.

`work/datasets/*.parquet` holds per-cell marker intensities derived from the depot and is **not** released: it is a copy of licensed material in another format, so the licence that forbids redistribution applies to it unchanged. It is regenerated locally in minutes by `scripts/r01_build_datasets.py` once the depot is in place.

## Clinical use

The CellTuneDepot licence additionally prohibits use in clinical, diagnostic or patient-care settings without express permission from the licensor. Nothing in this work constitutes a clinical or diagnostic claim.
