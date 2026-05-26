#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(MARVEL))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || !nzchar(args[[1]])) {
  cat("Usage: export_phase3_plate_reference.R <output_dir>\n", file = stderr())
  quit(status = 1, save = "no")
}

outdir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_tsv <- function(df, path) {
  utils::write.table(
    df,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )
}

build_shared_inputs <- function() {
  splice_pheno <- data.frame(
    sample.id = c("s1", "s2", "s3", "s4"),
    cell.type = c("iPSC", "iPSC", "Endoderm", "Endoderm"),
    stringsAsFactors = FALSE
  )

  splice_junction <- data.frame(
    coord.intron = c(
      "chr1:120:199",
      "chr1:120:299",
      "chr1:220:299",
      "chr2:320:399",
      "chr2:320:499",
      "chr2:420:499"
    ),
    s1 = c(8, 2, 8, 2, 8, 2),
    s2 = c(7, 3, 7, 3, 7, 3),
    s3 = c(9, 1, 9, 1, 9, 1),
    s4 = c(8, 2, 8, 2, 8, 2),
    stringsAsFactors = FALSE
  )

  splice_feature_se <- data.frame(
    tran_id = c(
      "chr1:100:119:+@chr1:200:219:+@chr1:300:319",
      "chr2:500:519:-@chr2:400:419:-@chr2:300:319"
    ),
    gene_id = c("GENE1", "GENE2"),
    gene_short_name = c("GENE1", "GENE2"),
    gene_type = c("protein_coding", "protein_coding"),
    stringsAsFactors = FALSE
  )

  gene_feature <- data.frame(
    gene_id = c("GENE1", "GENE2"),
    gene_short_name = c("GENE1", "GENE2"),
    gene_type = c("protein_coding", "protein_coding"),
    stringsAsFactors = FALSE
  )

  exp <- data.frame(
    gene_id = c("GENE1", "GENE2"),
    s1 = c(100, 80),
    s2 = c(120, 70),
    s3 = c(20, 85),
    s4 = c(30, 95),
    stringsAsFactors = FALSE
  )

  list(
    splice_pheno = splice_pheno,
    splice_junction = splice_junction,
    splice_feature = list(
      SE = splice_feature_se,
      MXE = NULL,
      RI = NULL,
      A5SS = NULL,
      A3SS = NULL,
      ALE = NULL,
      AFE = NULL
    ),
    splice_feature_validated = list(
      SE = NULL,
      MXE = NULL,
      RI = NULL,
      A5SS = NULL,
      A3SS = NULL,
      ALE = NULL,
      AFE = NULL
    ),
    psi = list(
      SE = NULL,
      MXE = NULL,
      RI = NULL,
      A5SS = NULL,
      A3SS = NULL,
      ALE = NULL,
      AFE = NULL
    ),
    gene_feature = gene_feature,
    exp = exp
  )
}

create_plate_object <- function(shared) {
  CreateMarvelObject(
    SplicePheno = shared$splice_pheno,
    SpliceJunction = shared$splice_junction,
    IntronCounts = NULL,
    SpliceFeature = shared$splice_feature,
    SpliceFeatureValidated = shared$splice_feature_validated,
    PSI = shared$psi,
    GeneFeature = shared$gene_feature,
    Exp = shared$exp,
    GTF = NULL
  )
}

build_plot_values_gene <- function(marvel_object, feature, cell_group_list) {
  exp <- marvel_object$Exp
  row <- exp[which(exp$gene_id == feature), , drop = FALSE]
  records <- list()
  index <- 1
  for (group_name in names(cell_group_list)) {
    sample_ids <- cell_group_list[[group_name]]
    for (sample_id in sample_ids) {
      records[[index]] <- data.frame(
        cell_group = group_name,
        sample_id = sample_id,
        feature = feature,
        value = as.numeric(row[[sample_id]][[1]]),
        stringsAsFactors = FALSE
      )
      index <- index + 1
    }
  }
  do.call(rbind.data.frame, records)
}

write_pca_outputs <- function(marvel_object, slot, coords_path, explained_path) {
  pca_coords <- as.data.frame(marvel_object$PCA[[slot]]$Results$ind$coord)
  pca_coords$sample.id <- row.names(pca_coords)
  pca_coords <- pca_coords[, c("sample.id", "Dim.1", "Dim.2")]
  names(pca_coords) <- c("sample_id", "PC1", "PC2")
  pca_coords <- merge(
    shared$splice_pheno[, c("sample.id", "cell.type")],
    pca_coords,
    by.x = "sample.id",
    by.y = "sample_id",
    sort = FALSE
  )
  names(pca_coords)[names(pca_coords) == "sample.id"] <- "sample_id"
  write_tsv(pca_coords, file.path(outdir, coords_path))

  pca_eig <- as.data.frame(marvel_object$PCA[[slot]]$Results$eig)
  pca_eig$component <- paste0("PC", seq_len(nrow(pca_eig)))
  pca_eig <- pca_eig[, c("component", "percentage of variance")]
  names(pca_eig) <- c("component", "explained_variance_ratio")
  pca_eig$explained_variance_ratio <- pca_eig$explained_variance_ratio / 100
  write_tsv(pca_eig, file.path(outdir, explained_path))
}

label_gene_de <- function(df, pval, log2fc) {
  df$sig <- "n.s."
  df$sig[df$p.val.adj < pval & df$log2FC < -log2fc] <- "down"
  df$sig[df$p.val.adj < pval & df$log2FC > log2fc] <- "up"
  df
}

write_manifest <- function() {
  manifest <- list(
    phase = "phase3_plate",
    functions = c(
      "RunPCA",
      "PlotValues",
      "PlotDEValues",
      "PropModality",
      "ModalityChange",
      "IsoSwitch",
      "IsoSwitch.PlotExpr"
    ),
    outputs = c(
      "run_pca_gene_coords.tsv",
      "run_pca_gene_explained.tsv",
      "run_pca_splicing_coords.tsv",
      "run_pca_splicing_explained.tsv",
      "plot_values_gene.tsv",
      "plot_de_gene_global.tsv",
      "prop_modality.tsv",
      "modality_change.tsv",
      "iso_switch.tsv",
      "iso_switch_plot_expr.tsv"
    ),
    comparison_status = list(
      RunPCA = "exact",
      PlotValues = "exact",
      PlotDEValues = "exact",
      PropModality = "partial",
      ModalityChange = "partial",
      IsoSwitch = "partial",
      `IsoSwitch.PlotExpr` = "partial"
    ),
    notes = list(
      RunPCA = paste(
        "Shared plate inputs are exported as gene-level and splicing-level PCA coordinates and explained variance.",
        "Exact explained-variance parity is expected; coordinate sign can flip per component."
      ),
      PlotValues = "Exact parity is exported as a stable tidy value table for level='gene'.",
      PlotDEValues = "Exact parity is exported as the labeled gene.global DE table.",
      PropModality = paste(
        "Tiny shared inputs produce R Missing/NA modality labels while Python computes concrete labels;",
        "the committed reference captures the upstream R table for audit."
      ),
      ModalityChange = paste(
        "Exported from marvel.demo.rds because the tiny shared fixture does not carry rich differential splicing state."
      ),
      IsoSwitch = paste(
        "Exported from marvel.demo.rds because iso-switch classification depends on richer DE state than the tiny shared fixture."
      ),
      `IsoSwitch.PlotExpr` = "Stored as the stable plotting table derived from the upstream IsoSwitch raw table."
    )
  )

  jsonlite::write_json(
    manifest,
    file.path(outdir, "manifest.json"),
    auto_unbox = TRUE,
    pretty = TRUE
  )
}

shared <- build_shared_inputs()

marvel_shared <- create_plate_object(shared)
marvel_shared <- TransformExpValues(
  MarvelObject = marvel_shared,
  offset = 1,
  transformation = "log2",
  threshold.lower = 1
)
marvel_shared <- ComputePSI(
  MarvelObject = marvel_shared,
  CoverageThreshold = 1,
  EventType = "SE"
)
marvel_shared <- AssignModality(
  MarvelObject = marvel_shared,
  sample.ids = c("s1", "s2", "s3", "s4"),
  min.cells = 1,
  seed = 1
)

marvel_pca <- RunPCA(
  MarvelObject = marvel_shared,
  sample.ids = c("s1", "s2", "s3", "s4"),
  cell.group.column = "cell.type",
  min.cells = 1,
  features = c("GENE1", "GENE2"),
  level = "gene",
  mode = "pca"
)
write_pca_outputs(
  marvel_object = marvel_pca,
  slot = "Exp",
  coords_path = "run_pca_gene_coords.tsv",
  explained_path = "run_pca_gene_explained.tsv"
)

splicing_features <- c(
  "chr1:100:119:+@chr1:200:219:+@chr1:300:319",
  "chr2:500:519:-@chr2:400:419:-@chr2:300:319"
)
marvel_pca_splicing <- RunPCA(
  MarvelObject = marvel_shared,
  sample.ids = c("s1", "s2", "s3", "s4"),
  cell.group.column = "cell.type",
  min.cells = 1,
  features = splicing_features,
  level = "splicing",
  mode = "pca",
  method.impute = "random",
  seed = 1
)
write_pca_outputs(
  marvel_object = marvel_pca_splicing,
  slot = "PSI",
  coords_path = "run_pca_splicing_coords.tsv",
  explained_path = "run_pca_splicing_explained.tsv"
)

plot_values_gene <- build_plot_values_gene(
  marvel_object = marvel_shared,
  feature = "GENE1",
  cell_group_list = list(
    iPSC = c("s1", "s2"),
    Endoderm = c("s3", "s4")
  )
)
write_tsv(plot_values_gene, file.path(outdir, "plot_values_gene.tsv"))

marvel_gene_de <- CompareValues(
  MarvelObject = marvel_shared,
  cell.group.g1 = c("s1", "s2"),
  cell.group.g2 = c("s3", "s4"),
  min.cells = 1,
  method = "wilcox",
  method.adjust = "fdr",
  level = "gene",
  show.progress = FALSE
)
plot_de_gene <- label_gene_de(marvel_gene_de$DE$Exp$Table, pval = 0.1, log2fc = 0.5)
write_tsv(plot_de_gene, file.path(outdir, "plot_de_gene_global.tsv"))

prop_modality <- PropModality(
  MarvelObject = marvel_shared,
  modality.column = "modality.bimodal.adj",
  modality.type = "extended",
  event.type = c("SE"),
  across.event.type = FALSE
)
write_tsv(
  prop_modality$Modality$Prop$DoughnutChart$Table,
  file.path(outdir, "prop_modality.tsv")
)

marvel_demo_path <- system.file("extdata/data", "marvel.demo.rds", package = "MARVEL")
marvel_demo <- readRDS(marvel_demo_path)

modality_demo <- ModalityChange(
  MarvelObject = marvel_demo,
  method = "ad",
  psi.pval = 0.1,
  psi.delta = 0
)
write_tsv(modality_demo$DE$Modality$Table, file.path(outdir, "modality_change.tsv"))

iso_demo <- IsoSwitch(
  MarvelObject = marvel_demo,
  method = "ad",
  psi.pval = 0.1,
  psi.delta = 0,
  gene.pval = 0.1,
  gene.log2fc = 0.5
)
write_tsv(iso_demo$DE$Cor$Table, file.path(outdir, "iso_switch.tsv"))
write_tsv(
  iso_demo$DE$Cor$Table_Raw[, c("gene_short_name", "mean.diff", "log2fc.gene", "cor")],
  file.path(outdir, "iso_switch_plot_expr.tsv")
)

write_manifest()
