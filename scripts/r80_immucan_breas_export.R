suppressPackageStartupMessages(library(SingleCellExperiment))

x <- readRDS("data/external3/sce_labelled_V3.rds")
cd <- as.data.frame(colData(x))
rd <- as.data.frame(rowData(x))

markers <- c("CD3", "CD4", "CD8a", "CD20", "CD38", "CD14", "CD15",
             "CD68", "Ecad", "HLADR", "Ki67", "SMA")
if (!all(markers %in% rownames(rd))) {
  stop("missing marker channels: ",
       paste(setdiff(markers, rownames(rd)), collapse = ", "))
}
if (!all(as.logical(rd[markers, "use_channel"]))) {
  stop("one or more requested markers are not marked use_channel")
}

keep <- as.logical(rd$use_channel)
X <- t(as.matrix(assay(x, "exprs")[keep, , drop = FALSE]))
colnames(X) <- rownames(rd)[keep]

label <- as.character(cd$cell_labels)
indication <- as.character(cd$Indication)
map <- c(
  Tumor = "Tumor",
  CD4 = "CD4",
  CD8 = "CD8",
  B = "B",
  plasma = "plasma",
  MacCD163 = "MacCD163"
)
ok <- indication == "BREAS" & label %in% names(map)
df <- data.frame(
  sample_id = as.character(cd$sample_id)[ok],
  label = unname(map[label[ok]]),
  X[ok, markers, drop = FALSE],
  check.names = FALSE
)
rownames(df) <- NULL

out <- "data/external3/immucan_breas_repair.csv"
write.csv(df, out, row.names = FALSE)

cat("wrote", nrow(df), "BREAS labelled cells and", length(markers),
    "markers to", out, "\n")
cat("samples:", length(unique(df$sample_id)), "\n")
cat("class counts:\n")
print(sort(table(df$label), decreasing = TRUE))
cat("1000-label fraction of this analysis set:",
    sprintf("%.4f", 1000 / nrow(df)), "\n")
