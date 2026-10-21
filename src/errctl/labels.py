"""Harmonise the four CellTuneDepot label vocabularies onto one coarse space.

Every dataset ships its own vocabulary, so any cross-tissue experiment needs an
explicit mapping. The coarse space keeps biologically distinct populations
separate while still allowing the four panels to be compared.
"""

COARSE_CLASSES = [
    "CD4T", "CD8T", "CD3T", "Treg", "Bcell", "Plasma",
    "Macrophage", "Monocyte", "APC", "Neutrophil", "Mast", "NK",
    "Endothelial", "Fibroblast", "TumorEpithelial", "Unidentified",
]

COARSE_MAP = {
    # Breast_Jun0524
    "TumorEpithelial": "TumorEpithelial",
    "Macrophage": "Macrophage",
    "Fibroblast": "Fibroblast",
    "CD4T": "CD4T",
    "CD8T": "CD8T",
    "Endothelial": "Endothelial",
    "Monocyte": "Monocyte",
    "Bcell": "Bcell",
    "Plasma": "Plasma",
    "ImmuneOther": "Unidentified",
    "Unidentified": "Unidentified",
    "Neutrophil": "Neutrophil",
    "Mast": "Mast",
    "CD3T": "CD3T",
    "APC": "APC",
    # CRC_Feb2623
    "Tumor": "TumorEpithelial",
    "Treg": "Treg",
    "Neuron": "Unidentified",
    "SMV": "Fibroblast",
    "Stroma": "Fibroblast",
    "Lymphatic": "Endothelial",
    "DC": "APC",
    # GVHD_Feb1323
    "Epithelial": "TumorEpithelial",
    "Goblet": "TumorEpithelial",
    "Paneth": "TumorEpithelial",
    "Endocrine": "Unidentified",
    # Melanoma_Oct2522
    "Myeloid": "Macrophage",
    "Tumor_BCAT": "TumorEpithelial",
    "NKT": "NK",
    "HLM": "Macrophage",
    "ActivatedFibro": "Fibroblast",
    "MyoFibro": "Fibroblast",
    "Tcell": "CD3T",
    "Garbage": "Unidentified",
}


def to_coarse(labels):
    """Vectorised coarse mapping; unknown labels fall back to Unidentified."""
    s = labels.astype(str)
    return s.map(COARSE_MAP).fillna("Unidentified")
