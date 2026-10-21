suppressPackageStartupMessages(library(SingleCellExperiment))
x <- readRDS("data/external3/sce_labelled_V3.rds")
cd <- as.data.frame(colData(x))
rd <- as.data.frame(rowData(x))
keep <- as.logical(rd$use_channel)
X <- t(as.matrix(assay(x, "exprs")[keep, , drop = FALSE]))
colnames(X) <- rownames(rd)[keep]
df <- data.frame(
  image = as.character(cd$image),
  sample_id = as.character(cd$sample_id),
  label = as.character(cd$cell_labels),
  X,
  check.names = FALSE
)
df <- df[!is.na(df$label) & df$label != "" & df$label != "unlabelled", , drop = FALSE]
write.csv(df, "data/external3/immucan_v3_cells.csv", row.names = FALSE)
cat("wrote", nrow(df), "labelled cells and", ncol(df) - 3, "markers\n")
