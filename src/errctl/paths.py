"""Central path definitions so scripts never hard-code locations twice."""

import os

ROOT = r"D:\TT paper\0926Distribution-free error"
DATA = os.path.join(ROOT, "data")
DEPOT = os.path.join(DATA, "depot", "CellTuneDepot")
DEMO = os.path.join(DATA, "demo", "CellTune_Store", "CellTune_Data", "Demo_Data")

WORK = os.path.join(ROOT, "work")
DATASETS = os.path.join(WORK, "datasets")
RESULTS = os.path.join(ROOT, "experiments")
FIGURES = os.path.join(ROOT, "figures")
CACHE = os.path.join(WORK, "cache")

MANUAL_DATASETS = ["Breast_Jun0524", "CRC_Feb2623", "GVHD_Feb1323", "Melanoma_Oct2522"]
HQ_DATASETS = ["CRC_Feb2623", "GVHD_Feb1323"]

for _d in (DATASETS, RESULTS, FIGURES, CACHE):
    os.makedirs(_d, exist_ok=True)


def manual_dir(ds):
    return os.path.join(DEPOT, "ManualLabels", ds)


def hq_dir(ds):
    return os.path.join(DEPOT, "HighQualityLabels", ds)
