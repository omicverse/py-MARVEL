from __future__ import annotations

from pathlib import Path
import sys
import time

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_external_benchmark as base
import marvel_py as mp


def step(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}\t{elapsed:.3f}s", flush=True)
    return result


def main() -> None:
    root = base.EXTERNAL_DROPLET_ROOT
    adata = step(
        "setup_10x_anndata",
        lambda: mp.setup_10x_anndata(
            gene_norm_matrix=root / "Gene_SingCellaR" / "matrix_normalised.mtx",
            gene_norm_pheno=root / "Gene_SingCellaR" / "phenoData.txt",
            gene_norm_feature=root / "Gene_SingCellaR" / "featureData.txt",
            gene_count_matrix=root / "Gene_STARsolo" / "matrix_counts.mtx",
            gene_count_pheno=root / "Gene_STARsolo" / "phenoData.txt",
            gene_count_feature=root / "Gene_STARsolo" / "featureData.txt",
            sj_count_matrix=root / "SJ_STARsolo" / "matrix_counts.mtx",
            sj_count_pheno=root / "SJ_STARsolo" / "phenoData.txt",
            sj_count_feature=root / "SJ_STARsolo" / "featureData.txt",
            pca=root / "Gene_SingCellaR" / "dim_red_coordinates_iPSC_CardioDay10.txt",
            gtf=root / "GTF" / "refdata-cellranger-GRCh38-3.0.0.gtf",
        ),
    )
    adata = step("annotate_genes_10x", lambda: mp.annotate_genes_10x(adata))
    adata = step("annotate_sj_10x", lambda: mp.annotate_sj_10x(adata))
    adata = step("validate_sj_10x", lambda: mp.validate_sj_10x(adata))
    adata = step("filter_genes_10x", lambda: mp.filter_genes_10x(adata))
    adata = step("check_alignment_10x", lambda: mp.check_alignment_10x(adata))

    group1 = adata.obs.loc[adata.obs["cell.type"] == "iPSC", "sample.id"].astype(str).tolist()
    group2 = adata.obs.loc[adata.obs["cell.type"] == "Cardio day 10", "sample.id"].astype(str).tolist()
    controls = base._load_droplet_replay_controls(base.R_RUNS_ROOT / "droplet")

    adata = step(
        "plot_pct_expr_cells_genes_10x",
        lambda: mp.plot_pct_expr_cells_genes_10x(adata, cell_group_g1=group1, cell_group_g2=group2, min_pct_cells=5.0),
    )
    adata = step(
        "plot_pct_expr_cells_sj_10x",
        lambda: mp.plot_pct_expr_cells_sj_10x(
            adata,
            cell_group_g1=group1,
            cell_group_g2=group2,
            min_pct_cells_genes=5.0,
            min_pct_cells_sj=5.0,
            downsample=True,
            downsample_pct_sj=10.0,
            seed=1,
            downsample_coord_introns=controls.pct_expr_sj_coord_introns,
        ),
    )
    de_group1 = controls.de_cell_group_g1 or group1
    de_group2 = controls.de_cell_group_g2 or group2
    permutations = controls.permutation_cell_ids[:1] if controls.permutation_cell_ids is not None else None
    adata = step(
        "compare_values_sj_10x_iter1",
        lambda: mp.compare_values_sj_10x(
            adata,
            cell_group_g1=de_group1,
            cell_group_g2=de_group2,
            min_pct_cells_genes=10.0,
            min_pct_cells_sj=10.0,
            min_gene_norm=1.0,
            seed=1,
            n_iterations=1,
            downsample=controls.de_cell_group_g1 is None,
            permutation_cell_ids=permutations,
            bounded_pval=controls.permutation_cell_ids is None,
        ),
    )
    adata = step("compare_values_genes_10x", lambda: mp.compare_values_genes_10x(adata))
    backend = mp._controller_from_anndata(adata).object
    print(
        f"rows\tgenes={len(backend.gene_metadata)}\tsj={len(backend.sj_metadata)}"
        f"\tpct_gene={len(backend.pct_expr_gene)}\tpct_sj={len(backend.pct_expr_sj)}"
        f"\tde_sj={len(backend.de_sj)}\tde_gene={len(backend.de_gene)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
