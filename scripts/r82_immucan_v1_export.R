suppressPackageStartupMessages(library(SingleCellExperiment))

v1 <- readRDS("data/external3/sce_labelled_V1.rds")
v3 <- readRDS("data/external3/sce_labelled_V3.rds")

cd1 <- as.data.frame(colData(v1))
rd1 <- as.data.frame(rowData(v1))
cd3 <- as.data.frame(colData(v3))

keep <- as.logical(rd1$use_channel)
X <- t(as.matrix(assay(v1, "exprs")[keep, , drop = FALSE]))
colnames(X) <- rownames(rd1)[keep]

df <- data.frame(
  image = as.character(cd1$image),
  sample_id = as.character(cd1$sample_id),
  label = as.character(cd1$cell_labels),
  X,
  check.names = FALSE
)
df <- df[!is.na(df$label) & df$label != "" & df$label != "unlabelled", , drop = FALSE]

samples_v3 <- unique(as.character(cd3$sample_id))
overlap <- intersect(unique(df$sample_id), samples_v3)

cat("V1 labelled cells", nrow(df), "samples", length(unique(df$sample_id)),
    "markers", ncol(X), "\n")
cat("V3 overlap samples", length(overlap), "\n")
if (length(overlap)) {
  cat(paste(overlap, collapse = "\n"), "\n")
}

df <- df[!(df$sample_id %in% overlap), , drop = FALSE]
write.csv(df, "data/external3/immucan_v1_cells.csv", row.names = FALSE)

cat("exported", nrow(df), "cells from", length(unique(df$sample_id)),
    "samples and", ncol(X), "markers\n")
print(sort(table(df$label), decreasing = TRUE))
