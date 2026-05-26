#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(MARVEL))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: run_external_droplet_r.R <output_dir>")
}

command_args <- commandArgs()
script_arg <- grep("^--file=", command_args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1]))
repo_root <- dirname(dirname(dirname(script_path)))

data_root <- file.path(repo_root, "external_droplet_data", "unpacked", "Data")
outdir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_tsv <- function(df, path) {
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
}

write_vector_tsv <- function(values, column, path) {
  df <- data.frame(values, stringsAsFactors = FALSE)
  names(df) <- column
  write_tsv(df, path)
}

load(file.path(data_root, "R Object", "MARVEL.rdata"))
if (!exists("marvel")) {
  stop("Expected object named 'marvel' in MARVEL.rdata")
}

marvel <- AnnotateGenes.10x(MarvelObject = marvel)
marvel <- AnnotateSJ.10x(MarvelObject = marvel)
marvel <- ValidateSJ.10x(MarvelObject = marvel, keep.novel.sj = FALSE)
marvel <- FilterGenes.10x(MarvelObject = marvel, gene.type = "protein_coding")
marvel <- CheckAlignment.10x(MarvelObject = marvel)

sample_metadata <- marvel$sample.metadata
cell_ids_1 <- sample_metadata[sample_metadata$cell.type == "iPSC", "cell.id"]
cell_ids_2 <- sample_metadata[sample_metadata$cell.type == "Cardio day 10", "cell.id"]

set.seed(1)
pct_sj_size <- round(nrow(marvel$sj.count.matrix) * (10 / 100), digits = 0)
pct_sj_coord_introns <- sample(rownames(marvel$sj.count.matrix), size = pct_sj_size, replace = FALSE)
write_vector_tsv(
  pct_sj_coord_introns,
  "coord.intron",
  file.path(outdir, "pct_expr_sj_downsample_coord_introns.tsv")
)

set.seed(1)
de_n_cells_downsample <- min(length(cell_ids_1), length(cell_ids_2))
de_cell_ids_1 <- sample(cell_ids_1, size = de_n_cells_downsample, replace = FALSE)
de_cell_ids_2 <- sample(cell_ids_2, size = de_n_cells_downsample, replace = FALSE)
write_vector_tsv(de_cell_ids_1, "cell.id", file.path(outdir, "de_sj_downsample_cells_g1.tsv"))
write_vector_tsv(de_cell_ids_2, "cell.id", file.path(outdir, "de_sj_downsample_cells_g2.tsv"))

set.seed(1)
de_combined_cell_ids <- c(de_cell_ids_1, de_cell_ids_2)
permutation_records <- do.call(
  rbind.data.frame,
  lapply(seq_len(10), function(i) {
    shuffled <- sample(de_combined_cell_ids, size = length(de_combined_cell_ids), replace = FALSE)
    data.frame(
      iteration = i,
      position = seq_along(shuffled),
      cell.id = shuffled,
      stringsAsFactors = FALSE
    )
  })
)
write_tsv(permutation_records, file.path(outdir, "de_sj_permutation_cell_ids.tsv"))

marvel_pct_gene <- PlotPctExprCells.Genes.10x(
  marvel,
  cell.group.g1 = cell_ids_1,
  cell.group.g2 = cell_ids_2,
  min.pct.cells = 5
)
write_tsv(marvel_pct_gene$pct.cells.expr$Gene$Data, file.path(outdir, "pct_expr_gene.tsv"))

marvel_pct_sj <- PlotPctExprCells.SJ.10x(
  marvel,
  cell.group.g1 = cell_ids_1,
  cell.group.g2 = cell_ids_2,
  min.pct.cells.genes = 5,
  min.pct.cells.sj = 5,
  downsample = TRUE,
  downsample.pct.sj = 10
)
write_tsv(marvel_pct_sj$pct.cells.expr$SJ$Data, file.path(outdir, "pct_expr_sj.tsv"))

marvel_de_sj <- CompareValues.SJ.10x(
  marvel,
  cell.group.g1 = cell_ids_1,
  cell.group.g2 = cell_ids_2,
  min.pct.cells.genes = 10,
  min.pct.cells.sj = 10,
  min.gene.norm = 1.0,
  seed = 1,
  n.iterations = 10,
  downsample = TRUE,
  show.progress = FALSE
)
write_tsv(marvel_de_sj$DE$SJ$Table, file.path(outdir, "de_sj.tsv"))

marvel_de_gene <- CompareValues.Genes.10x(
  marvel_de_sj,
  log2.transform = TRUE,
  show.progress = FALSE,
  method = "wilcox",
  mast.method = "bayesglm",
  mast.ebayes = TRUE
)

gene_table_columns <- c(
  "gene_short_name",
  "n.cells.total.norm.g1",
  "n.cells.expr.gene.norm.g1",
  "pct.cells.expr.gene.norm.g1",
  "mean.expr.gene.norm.g1",
  "n.cells.total.norm.g2",
  "n.cells.expr.gene.norm.g2",
  "pct.cells.expr.gene.norm.g2",
  "mean.expr.gene.norm.g2",
  "log2fc.gene.norm",
  "pval.gene.norm",
  "pval.adj.gene.norm"
)
gene_table <- unique(marvel_de_gene$DE$SJ$Table[, gene_table_columns])
write_tsv(gene_table, file.path(outdir, "de_gene.tsv"))

write_tsv(marvel$gene.metadata, file.path(outdir, "preprocessed_gene_metadata.tsv"))
write_tsv(marvel$sj.metadata, file.path(outdir, "preprocessed_sj_metadata.tsv"))

summary <- list(
  group_column = "cell.type",
  group1 = "iPSC",
  group2 = "Cardio day 10",
  group1_size = length(cell_ids_1),
  group2_size = length(cell_ids_2),
  gene_count_after_preprocess = nrow(marvel$gene.metadata),
  sj_count_after_preprocess = nrow(marvel$sj.metadata),
  de_iterations = 10,
  note = "External droplet benchmark uses 10 permutations for tractable replay on the full tutorial dataset."
)

write_json(summary, file.path(outdir, "summary.json"), auto_unbox = TRUE, pretty = TRUE)
