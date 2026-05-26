#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(MARVEL))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: run_external_plate_r.R <output_dir>")
}

command_args <- commandArgs()
script_arg <- grep("^--file=", command_args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1]))
repo_root <- dirname(dirname(dirname(script_path)))

data_root <- file.path(repo_root, "external_plate_data", "unpacked", "Data")
outdir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_tsv <- function(df, path) {
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
}

read_tsv <- function(path, dtype = NULL) {
  read.delim(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
}

compute_variable_splicing_table <- function(MarvelObject, sample.ids=NULL,
                                            cell.group.column, cell.group.order,
                                            min.cells=25) {
  df <- do.call(rbind.data.frame, MarvelObject$PSI)
  df.pheno <- MarvelObject$SplicePheno

  row.names(df) <- df$tran_id
  df$tran_id <- NULL
  df.pheno$pca.cell.group.label <- df.pheno[[cell.group.column]]

  if (!is.null(sample.ids[1])) {
    df.pheno <- df.pheno[which(df.pheno$sample.id %in% sample.ids), ]
  }
  if (is.null(cell.group.order[1])) {
    cell.group.order <- unique(df.pheno$pca.cell.group.label)
  }
  index <- which(df.pheno$pca.cell.group.label %in% cell.group.order)
  df.pheno <- df.pheno[index, ]
  df <- df[, df.pheno$sample.id]

  index.keep <- which(apply(df, 1, function(x) {sum(!is.na(x))}) >= min.cells)
  df <- df[index.keep, ]

  results <- data.frame(
    "tran_id" = row.names(df),
    "mean" = apply(df, 1, function(x) {mean(x, na.rm=TRUE)}),
    "sd" = apply(df, 1, function(x) {sd(x, na.rm=TRUE)}),
    stringsAsFactors = FALSE
  )
  model <- mgcv::gam(sd ~ s(mean, bs="ps", sp=0.6), data=results)
  pred <- predict(model, results, type="link", se.fit=TRUE)
  results$sd_pred <- pred$fit
  results$sd_pred_ci_lower <- pred$fit - (2 * pred$se.fit)
  results$sd_pred_ci_upper <- pred$fit + (2 * pred$se.fit)
  results$variable <- ifelse(results$sd > results$sd_pred, "Yes", "No")
  results
}

splice_feature <- list(
  SE = read_tsv(file.path(data_root, "rMATS", "SE", "SE_featureData.txt")),
  MXE = read_tsv(file.path(data_root, "rMATS", "MXE", "MXE_featureData.txt")),
  RI = read_tsv(file.path(data_root, "rMATS", "RI", "RI_featureData.txt")),
  A5SS = read_tsv(file.path(data_root, "rMATS", "A5SS", "A5SS_featureData.txt")),
  A3SS = read_tsv(file.path(data_root, "rMATS", "A3SS", "A3SS_featureData.txt"))
)

marvel <- CreateMarvelObject(
  SplicePheno = read_tsv(file.path(data_root, "SJ", "SJ_phenoData.txt")),
  SpliceJunction = read_tsv(file.path(data_root, "SJ", "SJ.txt")),
  IntronCounts = read_tsv(file.path(data_root, "MARVEL", "PSI", "RI", "Counts_by_Region.txt")),
  SpliceFeature = splice_feature,
  GeneFeature = read_tsv(file.path(data_root, "RSEM", "TPM_featureData.txt")),
  Exp = read_tsv(file.path(data_root, "RSEM", "TPM.txt"))
)

marvel <- CheckAlignment(MarvelObject = marvel, level = "SJ")
for (event_type in c("SE", "MXE", "RI", "A5SS", "A3SS")) {
  marvel <- ComputePSI(
    MarvelObject = marvel,
    CoverageThreshold = 10,
    EventType = event_type,
    UnevenCoverageMultiplier = 10,
    thread = 2,
    read.length = 1
  )
}

pass_ids <- marvel$SplicePheno[
  marvel$SplicePheno$cell.type %in% c("iPSC", "Endoderm") & marvel$SplicePheno$qc.seq == "pass",
  "sample.id"
]
marvel <- SubsetSamples(MarvelObject = marvel, sample.ids = pass_ids)

marvel <- TransformExpValues(
  MarvelObject = marvel,
  offset = 1,
  transformation = "log2",
  threshold.lower = 1
)
marvel <- CheckAlignment(MarvelObject = marvel, level = "splicing")
marvel <- CheckAlignment(MarvelObject = marvel, level = "gene")
marvel <- CheckAlignment(MarvelObject = marvel, level = "splicing and gene")

cell_group_g1 <- marvel$SplicePheno[marvel$SplicePheno$cell.type == "iPSC", "sample.id"]
cell_group_g2 <- marvel$SplicePheno[marvel$SplicePheno$cell.type == "Endoderm", "sample.id"]

marvel_iPSC <- CountEvents(MarvelObject = marvel, sample.ids = cell_group_g1, min.cells = 25)
write_tsv(marvel_iPSC$N.Events$Table, file.path(outdir, "n_events_min_cells_iPSC_min_cells_25.tsv"))

marvel_endoderm <- CountEvents(MarvelObject = marvel, sample.ids = cell_group_g2, min.cells = 25)
write_tsv(marvel_endoderm$N.Events$Table, file.path(outdir, "n_events_min_cells_Endoderm_min_cells_25.tsv"))

marvel_de <- CompareValues(
  MarvelObject = marvel,
  cell.group.g1 = cell_group_g1,
  cell.group.g2 = cell_group_g2,
  min.cells = 3,
  method = "wilcox",
  method.adjust = "fdr",
  level = "gene",
  show.progress = FALSE
)

variable_start <- proc.time()[["elapsed"]]
marvel_variable <- IdentifyVariableEvents(
  MarvelObject = marvel,
  cell.group.column = "cell.type",
  cell.group.order = c("iPSC", "Endoderm"),
  min.cells = 25
)
variable_runtime_seconds <- proc.time()[["elapsed"]] - variable_start
variable_table <- compute_variable_splicing_table(
  MarvelObject = marvel,
  cell.group.column = "cell.type",
  cell.group.order = c("iPSC", "Endoderm"),
  min.cells = 25
)

if (!is.null(marvel$PSI$SE)) {
  write_tsv(marvel$PSI$SE, file.path(outdir, "psi_se.tsv"))
}
if (!is.null(marvel$PSI$RI)) {
  write_tsv(marvel$PSI$RI, file.path(outdir, "psi_ri.tsv"))
}
write_tsv(marvel_de$DE$Exp$Table, file.path(outdir, "de_gene.tsv"))
write_tsv(variable_table, file.path(outdir, "variable_splicing_table.tsv"))
write_tsv(data.frame(tran_id = marvel_variable$VariableSplicing$tran_ids), file.path(outdir, "variable_splicing_tran_ids.tsv"))

summary <- list(
  source_mode = "flat_files_compute_psi",
  n_samples = nrow(marvel$SplicePheno),
  group1 = list(name = "iPSC", n = length(cell_group_g1)),
  group2 = list(name = "Endoderm", n = length(cell_group_g2)),
  psi_events_computed = c("SE", "MXE", "RI", "A5SS", "A3SS"),
  variable_splicing = list(
    min_cells = 25,
    n_events_retained = nrow(variable_table),
    n_variable_events = length(marvel_variable$VariableSplicing$tran_ids),
    runtime_seconds = variable_runtime_seconds
  ),
  note = "Plate R replay builds a MARVEL object from external flat files and recomputes PSI for SE/MXE/RI/A5SS/A3SS. AFE/ALE are skipped because external_plate_data does not provide flat feature tables for them."
)

write_json(summary, file.path(outdir, "summary.json"), auto_unbox = TRUE, pretty = TRUE)
