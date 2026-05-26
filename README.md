# marvel-py

A **pure-Python reimplementation of MARVEL** for single-cell alternative splicing analysis across plate-based and 10x droplet workflows.

- AnnData-native workflow controller for Scanpy-style pipelines, with MARVEL inputs in `adata.uns` and results written back to `adata.uns` / `adata.obsm`
- No `rpy2`; flat TSV / Matrix Market inputs remain supported as the backend compatibility layer
- Implements PSI quantification, modality summaries, differential gene / splicing analysis, variable splicing event selection, PCA helpers, isoform switching, and rMATS / cryptic-splice-site utilities
- R MARVEL parity is tracked with committed reference fixtures and full external replay benchmarks

> This package is a standalone Python port of the public user-facing MARVEL workflow surface. The R package remains the reference implementation; this repo keeps Python behavior close to R MARVEL through R-vs-Python tests, audit fixtures, and external benchmark reports.

## Install

Install the released package from PyPI:

```bash
pip install marvel-python
```

For local development from this repository:

```bash
pip install -e .
```

For benchmark report generation, also install `matplotlib`, or use the `uv run --with ...` commands shown below.

The bundled tutorial notebooks are expected to run in the `ove` micromamba environment and import the package normally:

```python
import marvel_py as mp
```

They do not modify `sys.path`.

## Quick Start

### AnnData-native plate workflow

```python
import marvel_py as mp

adata = mp.setup_plate_anndata(
    splice_pheno="splice_pheno.tsv",
    splice_junction="splice_junction.tsv",
    intron_counts="intron_counts.tsv",
    gene_feature="gene_feature.tsv",
    exp="exp.tsv",
    gtf="annotation.gtf",
    splice_feature={
        "SE": "SE_feature.tsv",
        "MXE": "MXE_feature.tsv",
        "RI": "RI_feature.tsv",
        "A5SS": "A5SS_feature.tsv",
        "A3SS": "A3SS_feature.tsv",
    },
)

adata = mp.check_alignment(adata, level="SJ")
adata = mp.compute_psi(adata, event_type="SE", coverage_threshold=10.0)

# Results stay with the AnnData object.
adata.uns["marvel"]["tables"]["psi"]["SE"]
adata.obsm["X_marvel_psi_se"]

pass_ids = adata.obs.loc[adata.obs["qc.seq"] == "pass", "sample.id"]
adata = mp.subset_samples(adata, sample_ids=pass_ids.tolist())
adata = mp.transform_exp_values(adata, offset=1.0, transformation="log2", threshold_lower=1.0)

g1 = adata.obs.loc[adata.obs["cell.type"] == "iPSC", "sample.id"].tolist()
g2 = adata.obs.loc[adata.obs["cell.type"] == "Endoderm", "sample.id"].tolist()
adata = mp.compare_values(adata, cell_group_g1=g1, cell_group_g2=g2, level="gene", method="wilcox")
```

The old function names are AnnData-aware. Passing a `MarvelPlate` / `Marvel10x` object preserves the legacy behavior; passing an `AnnData` object updates that `AnnData` in place and returns it.

### AnnData-native 10x droplet workflow

```python
import marvel_py as mp

adata = mp.setup_10x_anndata(
    gene_norm_matrix="gene_norm_matrix.mtx",
    gene_norm_pheno="gene_norm_pheno.tsv",
    gene_norm_feature="gene_norm_feature.tsv",
    gene_count_matrix="gene_count_matrix.mtx",
    gene_count_pheno="gene_count_pheno.tsv",
    gene_count_feature="gene_count_feature.tsv",
    sj_count_matrix="sj_count_matrix.mtx",
    sj_count_pheno="sj_count_pheno.tsv",
    sj_count_feature="sj_count_feature.tsv",
    pca="pca.tsv",
    gtf="annotation.gtf",
)

# Large 10x Matrix Market inputs are lazy by default. Pass
# load_matrices=True when you want matrices loaded into adata.X/layers
# during setup instead of during MARVEL processing.
adata = mp.annotate_genes_10x(adata)
adata = mp.annotate_sj_10x(adata)
adata = mp.validate_sj_10x(adata)
adata = mp.filter_genes_10x(adata)
adata = mp.check_alignment_10x(adata)

g1 = adata.obs.loc[adata.obs["cell.type"] == "iPSC", "sample.id"].tolist()
g2 = adata.obs.loc[adata.obs["cell.type"] == "Cardio day 10", "sample.id"].tolist()
adata = mp.plot_pct_expr_cells_genes_10x(adata, cell_group_g1=g1, cell_group_g2=g2)
adata = mp.plot_pct_expr_cells_sj_10x(adata, cell_group_g1=g1, cell_group_g2=g2)
adata = mp.compare_values_sj_10x(adata, cell_group_g1=g1, cell_group_g2=g2, n_iterations=10)
adata = mp.compare_values_genes_10x(adata)
```

Tabular outputs are mirrored under `adata.uns["marvel"]["tables"]`, and plate PSI matrices are exposed in `adata.obsm`. A runtime backend object is cached internally so consecutive function calls on the same `AnnData` keep MARVEL state without placing a non-serializable object in `adata.uns`.

Both AnnData function calls and low-level object helpers (`MarvelPlate`, `Marvel10x`) are supported. The preferred public API is `import marvel_py as mp` followed by AnnData setup plus the existing MARVEL function names; existing flat-file functions such as `mp.create_marvel_object(...)` and `mp.create_marvel_object_10x(...)` remain available for benchmarks and legacy scripts. The old nested `api` facade and legacy `py_marvel` mirror are not the supported interface.

---

## AnnData-native vs Original MARVEL-style API

py-MARVEL supports two equivalent execution styles. The AnnData-native style is the recommended interface for new Python workflows; the original MARVEL-style object API is kept for R MARVEL parity, tests, and users migrating existing MARVEL scripts.

| Topic | AnnData-native py-MARVEL | Original MARVEL-style py-MARVEL / R MARVEL |
|---|---|---|
| Primary object | `AnnData` | `MarvelPlate`, `Marvel10x`, or R S3 `MarvelObject` |
| Plate setup | `adata = mp.setup_plate_anndata(...)` | `marvel = mp.create_marvel_object(...)` / `CreateMarvelObject(...)` |
| 10x setup | `adata = mp.setup_10x_anndata(...)` | `marvel = mp.create_marvel_object_10x(...)` / `CreateMarvelObject.10x(...)` |
| Workflow calls | Same function names, e.g. `mp.compute_psi(adata, ...)` | Same function names on MARVEL objects, e.g. `mp.compute_psi(marvel, ...)` |
| Cell metadata | `adata.obs` | `marvel.splice_pheno`, `marvel.sample_metadata`, or R object slots |
| Gene metadata | `adata.var` plus MARVEL tables in `adata.uns` | `marvel.gene_feature`, `marvel.gene_metadata`, or R object slots |
| Results | `adata.uns["marvel"]["tables"]`, `adata.obsm[...]` | Attributes on `MarvelPlate` / `Marvel10x` or R slots such as `$PSI`, `$DE`, `$N.Events` |
| Scanpy integration | Native: can share `obs`, `var`, `obsm`, `layers`, and downstream Scanpy tooling | Not native; data are stored in MARVEL-specific tables |
| Serialization | AnnData can be written with standard `.h5ad` workflows after removing or avoiding unsupported table objects as needed | Pickle / TSV export in Python, R serialization in R |
| Best use case | New Python and omicverse workflows | R parity benchmarks, direct porting of MARVEL tutorials, debugging algorithm-level behavior |

The downstream function names are intentionally shared. This means a migration from original MARVEL-style Python to AnnData usually changes the setup step and the way results are retrieved, not the analysis verbs.

Plate example:

```python
# AnnData-native
adata = mp.setup_plate_anndata(...)
adata = mp.check_alignment(adata, level="SJ")
adata = mp.compute_psi(adata, event_type="SE", coverage_threshold=10)
psi_se = adata.uns["marvel"]["tables"]["psi"]["SE"]

# Original MARVEL-style
marvel = mp.create_marvel_object(...)
marvel = mp.check_alignment(marvel, level="SJ")
marvel = mp.compute_psi(marvel, event_type="SE", coverage_threshold=10)
psi_se = marvel.psi["SE"]
```

10x droplet example:

```python
# AnnData-native
adata = mp.setup_10x_anndata(...)
adata = mp.annotate_genes_10x(adata)
adata = mp.compare_values_genes_10x(adata)
de_gene = adata.uns["marvel"]["tables"]["de_gene"]

# Original MARVEL-style
marvel = mp.create_marvel_object_10x(...)
marvel = mp.annotate_genes_10x(marvel)
marvel = mp.compare_values_genes_10x(marvel)
de_gene = marvel.de_gene
```

---

## R MARVEL API Mapping

The Python API follows the R MARVEL workflow names but uses Python `snake_case`. The preferred Python entry point is AnnData-native: setup functions create an `AnnData` object, and downstream functions update that object in place while returning it. Low-level `MarvelPlate` and `Marvel10x` objects remain available for parity tests and legacy scripts.

### Core object model

| R MARVEL | py-MARVEL | Notes |
|---|---|---|
| `CreateMarvelObject(...)` | `mp.setup_plate_anndata(...)` | Preferred plate entry point; stores inputs in `adata.uns["marvel_input"]`. |
| `CreateMarvelObject(...)` | `mp.create_marvel_object(...)` | Legacy low-level `MarvelPlate` constructor, kept for R parity and benchmarks. |
| `CreateMarvelObject.10x(...)` | `mp.setup_10x_anndata(...)` | Preferred 10x entry point; Matrix Market inputs are lazy by default. |
| `CreateMarvelObject.10x(...)` | `mp.create_marvel_object_10x(...)` | Legacy low-level `Marvel10x` constructor. |
| R S3 `MarvelObject` slots | `adata.uns["marvel"]["tables"]`, `adata.obsm[...]` | Tabular results are mirrored into AnnData; the runtime backend object is cached outside `adata.uns`. |

### Plate workflow

| R MARVEL | py-MARVEL |
|---|---|
| `CheckAlignment(...)` | `mp.check_alignment(...)` |
| `SubsetSamples(...)` | `mp.subset_samples(...)` |
| `TransformExpValues(...)` | `mp.transform_exp_values(...)` |
| `DetectEvents(...)`, `DetectEvents.AFE(...)`, `DetectEvents.ALE(...)` | `mp.detect_events(...)` |
| `ComputePSI(...)`, `ComputePSI.SE(...)`, `ComputePSI.MXE(...)`, `ComputePSI.RI(...)`, `ComputePSI.A5SS(...)`, `ComputePSI.A3SS(...)`, `ComputePSI.AFE(...)`, `ComputePSI.ALE(...)` | `mp.compute_psi(..., event_type="SE")` |
| `ComputePSI.Posterior(...)` | `mp.compute_psi_posterior(...)` |
| `AssignModality(...)` | `mp.assign_modality(...)` |
| `CountEvents(...)` | `mp.count_events(...)` |
| `PropModality(...)`, `PropModality.Bar(...)`, `PropModality.Doughnut(...)` | `mp.prop_modality(...)`, `mp.prop_modality_bar(...)`, `mp.prop_modality_doughnut(...)` |
| `CompareValues(...)`, `CompareValues.PSI(...)`, `CompareValues.Exp(...)`, `CompareValues.Exp.Spliced(...)` | `mp.compare_values(...)` |
| `RunPCA(...)` | `mp.run_pca(...)` |
| `PlotValues(...)` | `mp.plot_values(...)` |
| `PlotDEValues(...)` | `mp.plot_de_values(...)` |
| `ModalityChange(...)` | `mp.modality_change(...)` |
| `IsoSwitch(...)`, `IsoSwitch.PlotExpr(...)` | `mp.iso_switch(...)`, `mp.iso_switch_plot_expr(...)` |
| `IdentifyVariableEvents(...)` | `mp.identify_variable_events(...)` |
| `PctASE(...)` | `mp.pct_ase(...)` |
| `ParseGTF(...)` | `mp.parse_gtf(...)` |
| `PrepareBedFile.RI(...)` | `mp.prepare_bed_file_ri(...)` |
| `Preprocess_rMATS(...)`, `Preprocess_rMATS.SE(...)`, `Preprocess_rMATS.MXE(...)`, `Preprocess_rMATS.RI(...)`, `Preprocess_rMATS.A5SS(...)`, `Preprocess_rMATS.A3SS(...)` | `mp.preprocess_rmats(...)`, `mp.preprocess_rmats_se(...)`, `mp.preprocess_rmats_mxe(...)`, `mp.preprocess_rmats_ri(...)`, `mp.preprocess_rmats_a5ss(...)`, `mp.preprocess_rmats_a3ss(...)` |
| `RemoveCrypticSS(...)`, `RemoveCrypticSS.AFE(...)`, `RemoveCrypticSS.ALE(...)` | `mp.remove_cryptic_ss(...)`, `mp.remove_cryptic_ss_afe(...)`, `mp.remove_cryptic_ss_ale(...)` |
| `SubsetCrypticSS(...)`, `SubsetCrypticSS.A5SS(...)`, `SubsetCrypticSS.A3SS(...)`, `SubsetCrypticA3SS(...)` | `mp.subset_cryptic_ss(...)`, `mp.subset_cryptic_ss_a5ss(...)`, `mp.subset_cryptic_ss_a3ss(...)`, `mp.subset_cryptic_a3ss(...)` |

### 10x droplet workflow

| R MARVEL | py-MARVEL |
|---|---|
| `AnnotateGenes.10x(...)` | `mp.annotate_genes_10x(...)` |
| `AnnotateSJ.10x(...)` | `mp.annotate_sj_10x(...)` |
| `ValidateSJ.10x(...)` | `mp.validate_sj_10x(...)` |
| `FilterGenes.10x(...)` | `mp.filter_genes_10x(...)` |
| `CheckAlignment.10x(...)` | `mp.check_alignment_10x(...)` |
| `PlotPctExprCells.Genes.10x(...)` | `mp.plot_pct_expr_cells_genes_10x(...)` |
| `PlotPctExprCells.SJ.10x(...)` | `mp.plot_pct_expr_cells_sj_10x(...)` |
| `CompareValues.SJ.10x(...)` | `mp.compare_values_sj_10x(...)` |
| `CompareValues.Genes.10x(...)` | `mp.compare_values_genes_10x(...)` |
| `CompareValues.SJ.DonorLevel.10x(...)` | `mp.compare_values_sj_donor_level_10x(...)` |
| `PlotDEValues.SJ.10x(...)` | `mp.plot_de_values_sj_10x(...)` |
| `PlotDEValues.Genes.10x(...)` | `mp.plot_de_values_genes_10x(...)` |
| `PlotValues.Gene.Pseudobulk.10x(...)` | `mp.plot_values_gene_pseudobulk_10x(...)` |
| `PlotValues.Gene.SingleCell.10x(...)` | `mp.plot_values_gene_single_cell_10x(...)` |
| `PlotValues.PCA.CellGroup.10x(...)` | `mp.plot_values_pca_cell_group_10x(...)` |
| `PlotValues.PCA.Gene.10x(...)` | `mp.plot_values_pca_gene_10x(...)` |
| `PlotValues.PCA.PSI.10x(...)` | `mp.plot_values_pca_psi_10x(...)` |
| `PlotValues.PSI.Pseudobulk.10x(...)` | `mp.plot_values_psi_pseudobulk_10x(...)` |
| `PlotValues.PSI.Pseudobulk.Heatmap.10x(...)` | `mp.plot_values_psi_pseudobulk_heatmap_10x(...)` |
| `IsoSwitch.10x(...)` | `mp.iso_switch_10x(...)` |
| `adhocGene.TabulateExpression.Gene.10x(...)` | `mp.adhoc_gene_tabulate_expression_gene_10x(...)` |
| `adhocGene.TabulateExpression.PSI.10x(...)` | `mp.adhoc_gene_tabulate_expression_psi_10x(...)` |
| `adhocGene.DE.Gene.10x(...)` | `mp.adhoc_gene_de_gene_10x(...)` |
| `adhocGene.DE.PSI.10x(...)` | `mp.adhoc_gene_de_psi_10x(...)` |
| `adhocGene.PlotDEValues.10x(...)` | `mp.adhoc_gene_plot_de_values_10x(...)` |
| `adhocGene.PlotSJPosition.10x(...)` | `mp.adhoc_gene_plot_sj_position_10x(...)` |

### Not yet ported from R MARVEL

Some R exports are intentionally not part of the current Python public surface: `BioPathways*`, `FindPTC*`, `PropPTC`, `CompareExpr`, and `AnnoVolcanoPlot`. Use R MARVEL directly for those workflows until they are ported.

---

## Workflow Coverage

| Area | Plate | 10x droplet |
|---|---:|---:|
| Object creation and alignment | yes | yes |
| Gene and splice-junction annotation | via input feature tables / GTF helpers | yes |
| PSI / splice-junction expression summaries | yes | yes |
| Differential gene and splicing analysis | yes | yes |
| Variable splicing event selection | yes | no |
| PCA / plotting helper tables | yes | yes |
| Isoform switching helpers | yes | yes |
| rMATS and cryptic splice-site utilities | yes | no |

The implementation targets the public `MARVEL/man` workflow surface rather than private R internals.

## Benchmarks

The external benchmark replays R MARVEL and `marvel_py` on the full external tutorial datasets:

- `external_plate_data`: `iPSC` vs `Endoderm`, `qc.seq == pass`, with `SE/MXE/RI/A5SS/A3SS` inputs available
- `external_droplet_data`: `iPSC` vs `Cardio day 10`, using 10 permutations for tractable full-data replay
- Plate R replay rebuilds from flat files and recomputes PSI. Droplet R replay uses the bundled R MARVEL object and recomputes downstream summaries / DE; Python replay loads the flat Matrix Market / TSV inputs.

Latest validated run:

| Run | Artifacts | Numeric metrics | Nonzero row deltas | Max absolute difference | Minimum Pearson r |
|---|---:|---:|---:|---:|---:|
| `benchmark/runs/20260525T070728Z` | 10 | 323 | 0 | `8.91e-4` | `0.9999961188133987` |

Runtime from the latest run:

| Section | R | Python | Speedup |
|---|---:|---:|---:|
| Plate full replay | `1116.16 s` | `345.39 s` | `3.23x` |
| Plate variable splicing only | `2.71 s` | `0.86 s` | `3.15x` |
| Droplet replay | `1412.29 s` | `477.30 s` | `2.96x` |

Artifact-level agreement:

| Section | Artifact | Rows R | Rows Python | Max abs diff | Min Pearson r |
|---|---|---:|---:|---:|---:|
| Plate | `psi_se` | 20509 | 20509 | `6.66e-16` | `0.9999999999999976` |
| Plate | `psi_ri` | 8295 | 8295 | `6.66e-16` | `0.9999999999999988` |
| Plate | `de_gene` | 21059 | 21059 | `5.15e-14` | `0.9999999999999998` |
| Plate | `count_events_iPSC` | 5 | 5 | `0` | `1.0` |
| Plate | `count_events_endoderm` | 5 | 5 | `0` | `0.9999999999999999` |
| Plate | `variable_splicing` | 13318 | 13318 | `8.91e-4` | `0.9999961188133987` |
| Droplet | `pct_expr_gene` | 20187 | 20187 | `0` | `1.0` |
| Droplet | `pct_expr_sj` | 1861 | 1861 | `0` | `0.9999999999999998` |
| Droplet | `de_sj` | 2937 | 2937 | `8.88e-15` | `0.9999999999999999` |
| Droplet | `de_gene` | 831 | 831 | `7.11e-15` | `0.9999999999999999` |

The retained summary figures are in [`benchmark/results/figures`](benchmark/results/figures):

- `benchmark_summary_figure.png`
- `artifact_row_delta.png`
- `top20_metric_max_abs_diff.png`
- `lowest20_metric_pearson_r.png`
- `plate_metric_heatmap.png`
- `droplet_metric_heatmap.png`
- `runtime_r_vs_python.png`
- per-artifact scatter plots such as `plate__psi_se__scatter.png`, `plate__variable_splicing__scatter.png`, and `droplet__de_sj__scatter.png`

To rerun the external benchmark and archive the results:

```bash
# Optional: only needed when Rscript is not already on PATH.
export MARVEL_RSCRIPT=/path/to/Rscript

uv run --with matplotlib \
  python benchmark/scripts/run_external_benchmark_archive.py

uv run --with matplotlib \
  python benchmark/scripts/plot_benchmark_summary_figure.py \
  --run-dir benchmark/runs/<run_id>
```

R benchmarks require an R environment with MARVEL and its dependencies installed. `MARVEL_RSCRIPT`
can point to any compatible R installation; it is not tied to a local micromamba path.

The archive contains copied results, code snapshots, command logs, `metric_summary.tsv`, `artifact_summary.tsv`, and report figures.

---

## Examples

| Notebook | What it covers |
|---|---|
| [`examples/plate_data.ipynb`](examples/plate_data.ipynb) | Plate-based MARVEL workflow using `import marvel_py as mp` |
| [`examples/Droplet_data.ipynb`](examples/Droplet_data.ipynb) | 10x droplet MARVEL workflow using `import marvel_py as mp` |

Scripted demos are also available:

| Script | Purpose |
|---|---|
| [`scripts/run_plate_ref_python.py`](scripts/run_plate_ref_python.py) | Run the core plate tutorial workflow from exported flat files |
| [`scripts/run_ref1_python.py`](scripts/run_ref1_python.py) | Run the core droplet ref1 workflow from Matrix Market / TSV inputs |
| [`scripts/export_plate_demo_inputs.R`](scripts/export_plate_demo_inputs.R) | Export R MARVEL plate demo data into Python-friendly inputs |
| [`scripts/export_marvel_demo_inputs.R`](scripts/export_marvel_demo_inputs.R) | Export R MARVEL droplet demo data into Python-friendly inputs |

## Relationship to R MARVEL

R MARVEL is the canonical package:

- Reference package: [`MARVEL`](../MARVEL)
- Python package: this repo, importing as `marvel_py`
- Benchmark code: [`benchmark/scripts`](benchmark/scripts)


If you need exact R package behavior for unsupported private internals, use R MARVEL directly. If you need the implemented public workflows from Python, use `marvel_py`.

## Relationship to omicverse
Developed following the [omicverse-to-developer](https://github.com/omicverse/omicverse-to-developer) py-<Name> conventions (pure-Python, no rpy2 in production code, AnnData-native I/O, Numba only on hot kernels). Upstream integration plan:

Canonical implementation: omicverse.external.copykat_py (pending)
Standalone mirror (this repo): same code, same API, without the full omicverse packaging

## Citation

If you use this package, please cite the original MARVEL paper:

> Wei Xiong Wen, Adam J Mead, Supat Thongjuea, MARVEL: an integrated alternative splicing analysis platform for single-cell RNA sequencing data, Nucleic Acids Research, Volume 51, Issue 5, 21 March 2023, Page e29, https://doi.org/10.1093/nar/gkac1260

and acknowledge omicverse / this repo for the Python port and optimisations.

## License

GNU GPLv3.
