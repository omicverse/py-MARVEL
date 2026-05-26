from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_external_benchmark as base
import marvel_py as mp


OUTPUT_ROOT = REPO_ROOT / "benchmark" / "results" / "python_anndata"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _compare_tables(
    *,
    artifact: str,
    baseline_path: Path,
    anndata_path: Path,
    key_columns: list[str],
) -> dict[str, object]:
    baseline = _read_tsv(baseline_path)
    anndata = _read_tsv(anndata_path)
    if key_columns:
        merged = baseline.merge(anndata, on=key_columns, how="outer", suffixes=("_baseline", "_anndata"))
    else:
        merged = baseline.reset_index().merge(
            anndata.reset_index(),
            on="index",
            how="outer",
            suffixes=("_baseline", "_anndata"),
        )

    metrics = []
    for column in sorted(set(baseline.columns).intersection(anndata.columns).difference(key_columns)):
        left = f"{column}_baseline"
        right = f"{column}_anndata"
        if left not in merged or right not in merged:
            continue
        left_values = pd.to_numeric(merged[left], errors="coerce")
        right_values = pd.to_numeric(merged[right], errors="coerce")
        mask = left_values.notna() & right_values.notna()
        if not mask.any():
            continue
        diff = (left_values[mask] - right_values[mask]).abs()
        metrics.append(
            {
                "metric": column,
                "n": int(mask.sum()),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        )

    max_abs_diff = max((metric["max_abs_diff"] for metric in metrics), default=None)
    return {
        "artifact": artifact,
        "baseline_path": str(baseline_path),
        "anndata_path": str(anndata_path),
        "rows_baseline": int(len(baseline)),
        "rows_anndata": int(len(anndata)),
        "row_delta": int(len(anndata) - len(baseline)),
        "max_abs_diff": max_abs_diff,
        "metrics": metrics,
    }


def _run_plate_anndata(output_dir: Path) -> dict[str, object]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    variable_output_dir = output_dir.parent / "plate_variable_splicing"
    variable_output_dir.mkdir(parents=True, exist_ok=True)
    root = base.EXTERNAL_PLATE_ROOT

    adata = mp.setup_plate_anndata(
        splice_pheno=root / "SJ" / "SJ_phenoData.txt",
        splice_junction=root / "SJ" / "SJ.txt",
        intron_counts=root / "MARVEL" / "PSI" / "RI" / "Counts_by_Region.txt",
        gene_feature=root / "RSEM" / "TPM_featureData.txt",
        exp=root / "RSEM" / "TPM.txt",
        gtf=root / "GTF" / "gencode.v31.annotation.gtf",
        splice_feature={
            "SE": root / "rMATS" / "SE" / "SE_featureData.txt",
            "MXE": root / "rMATS" / "MXE" / "MXE_featureData.txt",
            "RI": root / "rMATS" / "RI" / "RI_featureData.txt",
            "A5SS": root / "rMATS" / "A5SS" / "A5SS_featureData.txt",
            "A3SS": root / "rMATS" / "A3SS" / "A3SS_featureData.txt",
        },
    )
    adata = mp.check_alignment(adata, level="SJ")
    for event_type in ["SE", "MXE", "RI", "A5SS", "A3SS"]:
        adata = mp.compute_psi(
            adata,
            event_type=event_type,
            coverage_threshold=10.0,
            uneven_coverage_multiplier=10.0,
            read_length=1.0,
        )

    pass_ids = adata.obs.loc[
        adata.obs["cell.type"].isin(["iPSC", "Endoderm"]) & (adata.obs["qc.seq"] == "pass"),
        "sample.id",
    ].astype(str).tolist()
    adata = mp.subset_samples(adata, sample_ids=pass_ids)
    adata = mp.transform_exp_values(adata, offset=1.0, transformation="log2", threshold_lower=1.0)
    adata = mp.check_alignment(adata, level="splicing")
    adata = mp.check_alignment(adata, level="gene")
    adata = mp.check_alignment(adata, level="splicing and gene")

    cell_group_g1 = adata.obs.loc[adata.obs["cell.type"] == "iPSC", "sample.id"].astype(str).tolist()
    cell_group_g2 = adata.obs.loc[adata.obs["cell.type"] == "Endoderm", "sample.id"].astype(str).tolist()

    adata = mp.count_events(adata, sample_ids=cell_group_g1, min_cells=25, label="iPSC_min_cells_25")
    adata = mp.count_events(adata, sample_ids=cell_group_g2, min_cells=25, label="Endoderm_min_cells_25")
    adata = mp.compare_values(
        adata,
        cell_group_g1=cell_group_g1,
        cell_group_g2=cell_group_g2,
        level="gene",
        method="wilcox",
        min_cells=3,
    )
    variable_start = time.perf_counter()
    adata = mp.identify_variable_events(
        adata,
        cell_group_column="cell.type",
        cell_group_order=["iPSC", "Endoderm"],
        min_cells=25,
    )
    variable_runtime_seconds = time.perf_counter() - variable_start

    plate = mp._controller_from_anndata(adata).object
    summary = {
        "api": "anndata",
        "n_samples": int(len(adata.obs)),
        "n_genes": int(adata.n_vars),
        "group1": {"name": "iPSC", "n": int(len(cell_group_g1))},
        "group2": {"name": "Endoderm", "n": int(len(cell_group_g2))},
        "psi_events_computed": ["SE", "MXE", "RI", "A5SS", "A3SS"],
        "runtime_seconds": time.perf_counter() - start,
        "variable_splicing": {
            "min_cells": 25,
            "n_events_retained": int(len(plate.variable_splicing["table"])),
            "n_variable_events": int(len(plate.variable_splicing["tran_ids"])),
            "runtime_seconds": float(variable_runtime_seconds),
        },
    }
    base._write_plate_benchmark_outputs(output_dir, plate, summary)
    plate.variable_splicing["table"].to_csv(variable_output_dir / "variable_splicing_table.tsv", sep="\t", index=False)
    pd.DataFrame({"tran_id": plate.variable_splicing["tran_ids"]}).to_csv(
        variable_output_dir / "variable_splicing_tran_ids.tsv",
        sep="\t",
        index=False,
    )
    _write_json(variable_output_dir / "summary.json", summary["variable_splicing"])
    return summary


def _run_droplet_anndata(output_dir: Path, *, n_iterations: int = 1) -> dict[str, object]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = base.EXTERNAL_DROPLET_ROOT

    adata = mp.setup_10x_anndata(
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
    )
    adata = mp.annotate_genes_10x(adata)
    adata = mp.annotate_sj_10x(adata)
    adata = mp.validate_sj_10x(adata)
    adata = mp.filter_genes_10x(adata)
    adata = mp.check_alignment_10x(adata)

    group1 = adata.obs.loc[adata.obs["cell.type"] == "iPSC", "sample.id"].astype(str).tolist()
    group2 = adata.obs.loc[adata.obs["cell.type"] == "Cardio day 10", "sample.id"].astype(str).tolist()
    replay_controls = base._load_droplet_replay_controls(base.R_RUNS_ROOT / "droplet")

    adata = mp.plot_pct_expr_cells_genes_10x(adata, cell_group_g1=group1, cell_group_g2=group2, min_pct_cells=5.0)
    adata = mp.plot_pct_expr_cells_sj_10x(
        adata,
        cell_group_g1=group1,
        cell_group_g2=group2,
        min_pct_cells_genes=5.0,
        min_pct_cells_sj=5.0,
        downsample=True,
        downsample_pct_sj=10.0,
        seed=1,
        downsample_coord_introns=replay_controls.pct_expr_sj_coord_introns,
    )
    de_group1 = replay_controls.de_cell_group_g1 or group1
    de_group2 = replay_controls.de_cell_group_g2 or group2
    uses_r_de_replay = replay_controls.de_cell_group_g1 is not None and replay_controls.de_cell_group_g2 is not None
    permutation_cell_ids = replay_controls.permutation_cell_ids
    if permutation_cell_ids is not None:
        permutation_cell_ids = permutation_cell_ids[:n_iterations]
    adata = mp.compare_values_sj_10x(
        adata,
        cell_group_g1=de_group1,
        cell_group_g2=de_group2,
        min_pct_cells_genes=10.0,
        min_pct_cells_sj=10.0,
        min_gene_norm=1.0,
        seed=1,
        n_iterations=n_iterations,
        downsample=not uses_r_de_replay,
        permutation_cell_ids=permutation_cell_ids,
        bounded_pval=replay_controls.permutation_cell_ids is None,
    )
    adata = mp.compare_values_genes_10x(adata)

    marvel = mp._controller_from_anndata(adata).object
    summary = {
        "api": "anndata",
        "group_column": "cell.type",
        "group1": "iPSC",
        "group2": "Cardio day 10",
        "group1_size": len(group1),
        "group2_size": len(group2),
        "gene_count_after_preprocess": int(len(marvel.gene_metadata)),
        "sj_count_after_preprocess": int(len(marvel.sj_metadata)),
        "pct_expr_gene_rows": int(len(marvel.pct_expr_gene)),
        "pct_expr_sj_rows": int(len(marvel.pct_expr_sj)),
        "de_sj_rows": int(len(marvel.de_sj)),
        "de_gene_rows": int(len(marvel.de_gene)),
        "de_iterations": n_iterations,
        "seed": 1,
        "runtime_seconds": time.perf_counter() - start,
        "python_replay_controls": {
            "pct_expr_sj_downsample_coord_introns": replay_controls.pct_expr_sj_coord_introns is not None,
            "de_sj_downsample_cells": uses_r_de_replay,
            "de_sj_permutation_cell_ids": permutation_cell_ids is not None,
        },
    }
    marvel.save_outputs(output_dir, summary)
    return summary


def _compare_against_python_baseline(run_dir: Path, *, droplet_iterations: int) -> list[dict[str, object]]:
    baseline_root = REPO_ROOT / "benchmark" / "external" / "python_runs"
    specs = [
        ("plate.psi_se", baseline_root / "plate" / "psi_se.tsv", run_dir / "plate" / "psi_se.tsv", ["tran_id"]),
        ("plate.psi_ri", baseline_root / "plate" / "psi_ri.tsv", run_dir / "plate" / "psi_ri.tsv", ["tran_id"]),
        ("plate.de_gene", baseline_root / "plate" / "de_gene.tsv", run_dir / "plate" / "de_gene.tsv", ["gene_id"]),
        (
            "plate.count_events_iPSC",
            baseline_root / "plate" / "n_events_min_cells_iPSC_min_cells_25.tsv",
            run_dir / "plate" / "n_events_min_cells_iPSC_min_cells_25.tsv",
            ["event_type"],
        ),
        (
            "plate.count_events_endoderm",
            baseline_root / "plate" / "n_events_min_cells_Endoderm_min_cells_25.tsv",
            run_dir / "plate" / "n_events_min_cells_Endoderm_min_cells_25.tsv",
            ["event_type"],
        ),
        (
            "plate.variable_splicing",
            baseline_root / "plate" / "variable_splicing_table.tsv",
            run_dir / "plate" / "variable_splicing_table.tsv",
            ["tran_id"],
        ),
        (
            "droplet.pct_expr_gene",
            baseline_root / "droplet" / "pct_expr_gene.tsv",
            run_dir / "droplet" / "pct_expr_gene.tsv",
            ["cell.group", "gene_short_name"],
        ),
        (
            "droplet.pct_expr_sj",
            baseline_root / "droplet" / "pct_expr_sj.tsv",
            run_dir / "droplet" / "pct_expr_sj.tsv",
            ["cell.group", "coord.intron"],
        ),
        (
            "droplet.de_gene",
            baseline_root / "droplet" / "de_gene.tsv",
            run_dir / "droplet" / "de_gene.tsv",
            ["gene_short_name"],
        ),
    ]
    if droplet_iterations == 10:
        specs.append(
            (
                "droplet.de_sj",
                baseline_root / "droplet" / "de_sj.tsv",
                run_dir / "droplet" / "de_sj.tsv",
                ["coord.intron"],
            )
        )
    rows = []
    for artifact, baseline_path, anndata_path, key_columns in specs:
        if baseline_path.exists() and anndata_path.exists():
            rows.append(
                _compare_tables(
                    artifact=artifact,
                    baseline_path=baseline_path,
                    anndata_path=anndata_path,
                    key_columns=key_columns,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python-only AnnData API benchmark against Python baseline outputs.")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--skip-plate", action="store_true")
    parser.add_argument("--skip-droplet", action="store_true")
    parser.add_argument("--droplet-iterations", type=int, default=1)
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"run_id": args.run_id, "run_dir": str(run_dir), "sections": {}}

    if not args.skip_plate:
        summary["sections"]["plate"] = _run_plate_anndata(run_dir / "plate")
    if not args.skip_droplet:
        summary["sections"]["droplet"] = _run_droplet_anndata(
            run_dir / "droplet",
            n_iterations=args.droplet_iterations,
        )

    comparisons = _compare_against_python_baseline(run_dir, droplet_iterations=args.droplet_iterations)
    summary["comparisons"] = comparisons
    _write_json(run_dir / "summary.json", summary)
    if comparisons:
        pd.DataFrame(
            [
                {
                    "artifact": row["artifact"],
                    "rows_baseline": row["rows_baseline"],
                    "rows_anndata": row["rows_anndata"],
                    "row_delta": row["row_delta"],
                    "max_abs_diff": row["max_abs_diff"],
                }
                for row in comparisons
            ]
        ).to_csv(run_dir / "artifact_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
