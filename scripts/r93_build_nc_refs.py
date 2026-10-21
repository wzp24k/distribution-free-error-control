# -*- coding: utf-8 -*-
"""Build the 35 Nature-style references from verified DOI metadata.

The metadata in ``work/ref_metadata_1020.json`` was retrieved from Crossref for
journal articles and DataCite for arXiv DOIs. Every DOI resolved with HTTP 200.
"""

import csv
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "work", "ref_metadata_1020.json")
OUT_JSON = os.path.join(ROOT, "work", "references_nc_1020.json")
OUT_CSV = os.path.join(ROOT, "work", "ref_doi_audit_1020.csv")

TITLES = {
    1: "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front",
    2: "Spatial mapping of protein composition and tissue organization: a primer for multiplexed antibody-based imaging",
    3: "CellSighter: a neural network to classify cells in highly multiplexed images",
    4: "CellTune: an integrative software for accurate cell classification in spatial proteomics",
    5: "Calibration: the Achilles heel of predictive analytics",
    6: "The myth of generalisability in clinical research and machine learning in health care",
    7: "CODEX multiplexed tissue imaging with DNA-conjugated antibodies",
    8: "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study",
    9: "On optimum recognition error and reject tradeoff",
    10: "Algorithmic learning in a random world",
    11: "Distribution-free predictive inference for regression",
    12: "A gentle introduction to conformal prediction and distribution-free uncertainty quantification",
    13: "Selective classification for deep neural networks",
    14: "Single-cell map of diverse immune phenotypes in the breast tumor microenvironment",
    15: "Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning",
    16: "Spatially organized multicellular immune hubs in human colorectal cancer",
    17: "Molecular phenotyping reveals the identity of Barrett's esophagus and its malignant transition",
    18: "Single-cell analysis of human B cell maturation predicts how antibody class switching shapes selection dynamics",
    19: "Reproducible, high-dimensional imaging in archival human tissue by multiplexed ion beam imaging by time-of-flight (MIBI-TOF)",
    20: "Adaptation of imaging mass cytometry to explore the single cell alloimmune landscape of liver transplant rejection",
    21: "A spatial expression atlas of the adult human proximal small intestine",
    22: "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure",
    23: "Cross-tissue immune cell analysis reveals tissue-specific features in humans",
    24: "Direct importance estimation for covariate shift adaptation",
    25: "Detecting and correcting for label shift with black box predictors",
    26: "A theory of learning from different domains",
    27: "Mapping single-cell data to reference atlases by transfer learning",
    28: "Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage",
    29: "A comparison of automatic cell identification methods for single-cell RNA sequencing data",
    30: "A structured tumor-immune microenvironment in triple negative breast cancer revealed by multiplexed ion beam imaging",
    31: "Deep profiling of mouse splenic architecture with CODEX multiplexed imaging",
    32: "Mass cytometry imaging for the study of human diseases: applications and data analysis strategies",
    33: "On calibration of modern neural networks",
    34: "Dropout as a Bayesian approximation: representing model uncertainty in deep learning",
    35: "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift",
}

JOURNALS = {
    1: "Cell", 2: "Nature Methods", 3: "Nature Communications",
    4: "Nature Methods", 5: "BMC Medicine", 6: "The Lancet Digital Health",
    7: "Nature Protocols", 8: "PLOS Medicine",
    9: "IEEE Transactions on Information Theory",
    11: "Journal of the American Statistical Association",
    14: "Cell", 15: "Nature Biotechnology", 16: "Cell", 17: "Science",
    18: "Science Immunology", 19: "Laboratory Investigation",
    20: "Frontiers in Immunology", 21: "Nature", 22: "Ecography",
    23: "Science", 24: "Annals of the Institute of Statistical Mathematics",
    26: "Machine Learning", 27: "Nature Biotechnology",
    28: "Nature Immunology", 29: "Genome Biology", 30: "Cell",
    31: "Cell", 32: "Frontiers in Immunology",
}

PAGES = {
    1: "838", 2: "284-295", 3: "4302", 4: "1574-1587", 5: "230",
    6: "e489-e492", 7: "3802-3835", 8: "e1002683", 9: "41-46",
    11: "1094-1111", 14: "1293-1308.e36", 15: "555-565",
    16: "4734-4752.e20", 17: "760-767", 18: "eabe6291",
    19: "762-770", 20: "831103", 21: "1101-1109", 22: "913-929",
    23: "eabl5197", 24: "699-746", 26: "151-175", 27: "121-130",
    28: "163-172", 29: "194", 30: "1373-1387.e19",
    31: "968-981.e15", 32: "2657",
}

YEARS = {
    1: 2020, 2: 2022, 3: 2023, 4: 2026, 5: 2019, 6: 2020, 7: 2021,
    8: 2018, 9: 1970, 10: 2005, 11: 2018, 12: 2021, 13: 2017,
    14: 2018, 15: 2022, 16: 2021, 17: 2021, 18: 2021, 19: 2022,
    20: 2022, 21: 2024, 22: 2017, 23: 2022, 24: 2008, 25: 2018,
    26: 2010, 27: 2022, 28: 2019, 29: 2019, 30: 2018, 31: 2018,
    32: 2019, 33: 2017, 34: 2015, 35: 2019,
}

PREPRINTS = {12, 13, 25, 33, 34, 35}
BOOKS = {10}


def initials(given):
    if not given:
        return ""
    parts = re.split(r"[\s]+", given.strip())
    out = []
    for part in parts:
        if not part:
            continue
        bits = part.split("-")
        out.append("-".join(b[0].upper() + "." for b in bits if b))
    return " ".join(out)


def author_string(authors):
    vals = []
    for fam, given in authors:
        vals.append("%s, %s" % (fam, initials(given)))
    if len(vals) > 5:
        return vals[0] + " et al."
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return "%s & %s" % (vals[0], vals[1])
    return ", ".join(vals[:-1]) + " & " + vals[-1]


def format_ref(row):
    n = int(row["n"])
    auth = author_string(row["authors"])
    title = TITLES[n]
    doi = row["doi"]
    year = YEARS[n]
    if n in BOOKS:
        return "%s %s. Springer, %d. https://doi.org/%s" % (
            auth, title, year, doi)
    if n in PREPRINTS:
        return "%s %s. Preprint at https://doi.org/%s (%d)." % (
            auth, title, doi, year)
    vol = row["volume"]
    page = PAGES[n].replace("-", "\u2013")
    if not vol or not page:
        raise AssertionError("reference %d has no volume or page" % n)
    return "%s %s. %s %s, %s (%d). https://doi.org/%s" % (
        auth, title, JOURNALS[n], vol, page, year, doi)


def main():
    data = json.load(io.open(META, encoding="utf-8"))
    if len(data) != 35:
        raise AssertionError("expected 35 references, got %d" % len(data))
    assert [int(x["n"]) for x in data] == list(range(1, 36))
    data[9]["authors"] = [["Vovk", "V."], ["Gammerman", "A."],
                          ["Shafer", "G."]]
    refs = []
    for row in data:
        if int(row["status"]) != 200:
            raise AssertionError("DOI %s did not resolve" % row["doi"])
        refs.append({"n": int(row["n"]), "doi": row["doi"],
                     "text": format_ref(row)})
        print("%2d. %s" % (row["n"], refs[-1]["text"]))
    json.dump(refs, io.open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "doi", "http_status", "metadata_source",
                    "resolved_title", "formatted_reference"])
        for row, ref in zip(data, refs):
            w.writerow([row["n"], row["doi"], row["status"], row["source"],
                        row["title"], ref["text"]])
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_CSV)


if __name__ == "__main__":
    main()
