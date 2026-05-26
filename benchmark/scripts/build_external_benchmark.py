from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marvel_py import (
    check_alignment,
    compare_values,
    compare_values_genes_10x,
    compare_values_sj_10x,
    compute_psi,
    count_events,
    create_marvel_object,
    create_marvel_object_10x,
    identify_variable_events,
    plot_pct_expr_cells_genes_10x,
    plot_pct_expr_cells_sj_10x,
    subset_samples,
    transform_exp_values,
)

BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "external"
R_RUNS_ROOT = BENCHMARK_ROOT / "r_runs"
PY_RUNS_ROOT = BENCHMARK_ROOT / "python_runs"

EXTERNAL_PLATE_ROOT = REPO_ROOT / "external_plate_data" / "unpacked" / "Data"
EXTERNAL_DROPLET_ROOT = REPO_ROOT / "external_droplet_data" / "unpacked" / "Data"


class DropletReplayControls:
    def __init__(
        self,
        *,
        pct_expr_sj_coord_introns: list[str] | None = None,
        de_cell_group_g1: list[str] | None = None,
        de_cell_group_g2: list[str] | None = None,
        permutation_cell_ids: list[list[str]] | None = None,
    ) -> None:
        self.pct_expr_sj_coord_introns = pct_expr_sj_coord_introns
        self.de_cell_group_g1 = de_cell_group_g1
        self.de_cell_group_g2 = de_cell_group_g2
        self.permutation_cell_ids = permutation_cell_ids


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_rscript() -> str:
    configured = os.environ.get("MARVEL_RSCRIPT") or os.environ.get("RSCRIPT")
    if configured:
        rscript = Path(configured).expanduser()
        if not rscript.exists():
            raise FileNotFoundError(f"Configured Rscript does not exist: {configured}")
        return str(rscript)

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FileNotFoundError(
            "Rscript was not found. Put Rscript on PATH or set "
            "MARVEL_RSCRIPT=/path/to/Rscript before running R benchmarks."
        )
    return rscript


def _run_command(args: list[str], *, cwd: Path | None = None) -> None:
    log_dir = _ensure_dir(BENCHMARK_ROOT / "command_logs")
    script_name = Path(args[1]).stem if len(args) > 1 else Path(args[0]).stem
    log_path = log_dir / f"{script_name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(args, check=True, cwd=cwd or REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)


def _run_command_timed(args: list[str], *, cwd: Path | None = None) -> float:
    start = time.perf_counter()
    _run_command(args, cwd=cwd)
    return time.perf_counter() - start


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_tsv(path: Path, *, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=dtype)


def _read_optional_column(path: Path, column: str) -> list[str] | None:
    if not path.exists():
        return None
    df = _read_tsv(path, dtype=str)
    if column not in df.columns:
        raise ValueError(f"{path} must contain column {column!r}")
    return df[column].astype(str).tolist()


def _load_droplet_replay_controls(r_droplet_dir: Path) -> DropletReplayControls:
    permutation_cell_ids = None
    permutation_path = r_droplet_dir / "de_sj_permutation_cell_ids.tsv"
    if permutation_path.exists():
        permutation_df = _read_tsv(permutation_path, dtype=str)
        required = {"iteration", "position", "cell.id"}
        missing = required.difference(permutation_df.columns)
        if missing:
            raise ValueError(f"{permutation_path} is missing columns: {sorted(missing)}")
        permutation_df["iteration"] = permutation_df["iteration"].astype(int)
        permutation_df["position"] = permutation_df["position"].astype(int)
        permutation_df = permutation_df.sort_values(["iteration", "position"])
        permutation_cell_ids = [
            group["cell.id"].astype(str).tolist()
            for _, group in permutation_df.groupby("iteration", sort=True)
        ]

    return DropletReplayControls(
        pct_expr_sj_coord_introns=_read_optional_column(
            r_droplet_dir / "pct_expr_sj_downsample_coord_introns.tsv",
            "coord.intron",
        ),
        de_cell_group_g1=_read_optional_column(r_droplet_dir / "de_sj_downsample_cells_g1.tsv", "cell.id"),
        de_cell_group_g2=_read_optional_column(r_droplet_dir / "de_sj_downsample_cells_g2.tsv", "cell.id"),
        permutation_cell_ids=permutation_cell_ids,
    )


def _flatten_numeric_pairs(
    r_df: pd.DataFrame,
    py_df: pd.DataFrame,
    *,
    key_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    merged = r_df.merge(py_df, on=key_columns, suffixes=("_r", "_py"), how="inner")
    metrics: dict[str, dict[str, float]] = {}
    long_frames: list[pd.DataFrame] = []
    shared_columns = [column for column in r_df.columns if column in py_df.columns and column not in key_columns]

    for column in shared_columns:
        if not pd.api.types.is_numeric_dtype(r_df[column]) or not pd.api.types.is_numeric_dtype(py_df[column]):
            continue
        pair = merged[key_columns + [f"{column}_r", f"{column}_py"]].copy()
        pair = pair.rename(columns={f"{column}_r": "r_value", f"{column}_py": "py_value"})
        pair["metric"] = column
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_value", "py_value"])
        if pair.empty:
            continue
        diff = pair["py_value"] - pair["r_value"]
        corr = pair["r_value"].corr(pair["py_value"])
        metrics[column] = {
            "n": int(len(pair)),
            "max_abs_diff": float(diff.abs().max()),
            "mean_abs_diff": float(diff.abs().mean()),
            "rmse": float(math.sqrt(np.mean(np.square(diff.to_numpy(dtype=float))))),
            "pearson_r": None if pd.isna(corr) else float(corr),
        }
        long_frames.append(pair)

    if not long_frames:
        return pd.DataFrame(columns=[*key_columns, "r_value", "py_value", "metric"]), metrics
    return pd.concat(long_frames, ignore_index=True), metrics


def _plot_scatter(long_df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    if long_df.empty:
        ax.text(0.5, 0.5, "No numeric overlap", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    plot_df = long_df
    if len(plot_df) > 100000:
        plot_df = plot_df.sample(100000, random_state=1)

    metric_names = list(dict.fromkeys(plot_df["metric"].astype(str).tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, metric in enumerate(metric_names):
        subset = plot_df[plot_df["metric"] == metric]
        ax.scatter(subset["r_value"], subset["py_value"], s=10, alpha=0.45, label=metric, color=cmap(idx % 10))

    min_value = float(min(plot_df["r_value"].min(), plot_df["py_value"].min()))
    max_value = float(max(plot_df["r_value"].max(), plot_df["py_value"].max()))
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("R")
    ax.set_ylabel("Python")
    ax.set_title(title)
    if len(metric_names) <= 10:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _benchmark_artifact(
    *,
    title: str,
    artifact_dir: Path,
    r_path: Path,
    python_path: Path,
    key_columns: list[str],
    preprocess=None,
) -> dict[str, object]:
    _ensure_dir(artifact_dir)
    r_df = _read_tsv(r_path)
    py_df = _read_tsv(python_path)
    if preprocess is not None:
        r_df, py_df = preprocess(r_df, py_df)
    merged_numeric, metrics = _flatten_numeric_pairs(r_df, py_df, key_columns=key_columns)

    r_copy = artifact_dir / "r.tsv"
    py_copy = artifact_dir / "python.tsv"
    merged_path = artifact_dir / "merged_numeric.tsv"
    metrics_path = artifact_dir / "metrics.json"
    scatter_path = artifact_dir / "scatter.png"

    r_df.to_csv(r_copy, sep="\t", index=False)
    py_df.to_csv(py_copy, sep="\t", index=False)
    merged_numeric.to_csv(merged_path, sep="\t", index=False)
    _write_json(metrics_path, metrics)
    _plot_scatter(merged_numeric, title=title, output_path=scatter_path)

    return {
        "artifact": artifact_dir.name,
        "title": title,
        "rows_r": int(len(r_df)),
        "rows_python": int(len(py_df)),
        "key_columns": key_columns,
        "numeric_metrics": metrics,
    }


def _run_plate_python() -> Path:
    start = time.perf_counter()
    output_dir = _reset_dir(PY_RUNS_ROOT / "plate")
    variable_output_dir = _reset_dir(PY_RUNS_ROOT / "plate_variable_splicing")

    plate = create_marvel_object(
        splice_pheno=EXTERNAL_PLATE_ROOT / "SJ" / "SJ_phenoData.txt",
        splice_junction=EXTERNAL_PLATE_ROOT / "SJ" / "SJ.txt",
        intron_counts=EXTERNAL_PLATE_ROOT / "MARVEL" / "PSI" / "RI" / "Counts_by_Region.txt",
        gene_feature=EXTERNAL_PLATE_ROOT / "RSEM" / "TPM_featureData.txt",
        exp=EXTERNAL_PLATE_ROOT / "RSEM" / "TPM.txt",
        gtf=EXTERNAL_PLATE_ROOT / "GTF" / "gencode.v31.annotation.gtf",
        splice_feature={
            "SE": EXTERNAL_PLATE_ROOT / "rMATS" / "SE" / "SE_featureData.txt",
            "MXE": EXTERNAL_PLATE_ROOT / "rMATS" / "MXE" / "MXE_featureData.txt",
            "RI": EXTERNAL_PLATE_ROOT / "rMATS" / "RI" / "RI_featureData.txt",
            "A5SS": EXTERNAL_PLATE_ROOT / "rMATS" / "A5SS" / "A5SS_featureData.txt",
            "A3SS": EXTERNAL_PLATE_ROOT / "rMATS" / "A3SS" / "A3SS_featureData.txt",
        },
    )
    plate = check_alignment(plate, level="SJ")

    for event_type in ["SE", "MXE", "RI", "A5SS", "A3SS"]:
        plate = compute_psi(
            plate,
            event_type=event_type,
            coverage_threshold=10.0,
            uneven_coverage_multiplier=10.0,
            read_length=1.0,
        )

    pass_ids = plate.splice_pheno.loc[
        plate.splice_pheno["cell.type"].isin(["iPSC", "Endoderm"]) & (plate.splice_pheno["qc.seq"] == "pass"),
        "sample.id",
    ].astype(str).tolist()
    plate = subset_samples(plate, sample_ids=pass_ids)
    plate = transform_exp_values(plate, offset=1.0, transformation="log2", threshold_lower=1.0)
    plate = check_alignment(plate, level="splicing")
    plate = check_alignment(plate, level="gene")
    plate = check_alignment(plate, level="splicing and gene")

    cell_group_g1 = plate.get_sample_ids("cell.type", ["iPSC"])
    cell_group_g2 = plate.get_sample_ids("cell.type", ["Endoderm"])

    plate.count_events(sample_ids=cell_group_g1, min_cells=25, label="iPSC_min_cells_25")
    plate.count_events(sample_ids=cell_group_g2, min_cells=25, label="Endoderm_min_cells_25")
    plate = compare_values(
        plate,
        cell_group_g1=cell_group_g1,
        cell_group_g2=cell_group_g2,
        level="gene",
        method="wilcox",
        min_cells=3,
    )
    variable_start = time.perf_counter()
    plate = identify_variable_events(
        plate,
        cell_group_column="cell.type",
        cell_group_order=["iPSC", "Endoderm"],
        min_cells=25,
    )
    variable_runtime_seconds = time.perf_counter() - variable_start
    variable_table = plate.variable_splicing["table"].copy()
    variable_tran_ids = plate.variable_splicing["tran_ids"]
    variable_table.to_csv(variable_output_dir / "variable_splicing_table.tsv", sep="\t", index=False)
    pd.DataFrame({"tran_id": variable_tran_ids}).to_csv(
        variable_output_dir / "variable_splicing_tran_ids.tsv",
        sep="\t",
        index=False,
    )

    summary = {
        "n_samples": int(len(plate.splice_pheno)),
        "group1": {"name": "iPSC", "n": int(len(cell_group_g1))},
        "group2": {"name": "Endoderm", "n": int(len(cell_group_g2))},
        "psi_events_computed": ["SE", "MXE", "RI", "A5SS", "A3SS"],
        "runtime_seconds": time.perf_counter() - start,
        "variable_splicing": {
            "min_cells": 25,
            "n_events_retained": int(len(variable_table)),
            "n_variable_events": int(len(variable_tran_ids)),
            "runtime_seconds": float(variable_runtime_seconds),
        },
        "note": "AFE/ALE are skipped because external_plate_data does not provide flat feature tables for them.",
    }
    _write_json(
        variable_output_dir / "summary.json",
        {
            "dataset": "external_plate_data",
            "workflow": "identify_variable_events",
            "n_samples": int(len(plate.splice_pheno)),
            "cell_group_column": "cell.type",
            "cell_group_order": ["iPSC", "Endoderm"],
            "min_cells": 25,
            "n_events_retained": int(len(variable_table)),
            "n_variable_events": int(len(variable_tran_ids)),
            "runtime_seconds": float(variable_runtime_seconds),
        },
    )
    _write_plate_benchmark_outputs(output_dir, plate, summary)
    return output_dir


def _write_plate_benchmark_outputs(output_dir: Path, plate, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for event_type in ("SE", "RI"):
        psi_df = plate.psi.get(event_type)
        if psi_df is not None and not psi_df.empty:
            psi_df.to_csv(output_dir / f"psi_{event_type.lower()}.tsv", sep="\t", index=False)

    if plate.de_gene is not None:
        plate.de_gene.to_csv(output_dir / "de_gene.tsv", sep="\t", index=False)
    if plate.variable_splicing is not None:
        plate.variable_splicing["table"].to_csv(output_dir / "variable_splicing_table.tsv", sep="\t", index=False)
        pd.DataFrame({"tran_id": plate.variable_splicing["tran_ids"]}).to_csv(
            output_dir / "variable_splicing_tran_ids.tsv",
            sep="\t",
            index=False,
        )
    for label, table in plate.n_events.items():
        table.to_csv(output_dir / f"n_events_min_cells_{label}.tsv", sep="\t", index=False)

    _write_json(output_dir / "summary.json", summary)


def _run_droplet_python(replay_controls: DropletReplayControls | None = None) -> Path:
    start = time.perf_counter()
    output_dir = _reset_dir(PY_RUNS_ROOT / "droplet")
    marvel = create_marvel_object_10x(
        gene_norm_matrix=EXTERNAL_DROPLET_ROOT / "Gene_SingCellaR" / "matrix_normalised.mtx",
        gene_norm_pheno=EXTERNAL_DROPLET_ROOT / "Gene_SingCellaR" / "phenoData.txt",
        gene_norm_feature=EXTERNAL_DROPLET_ROOT / "Gene_SingCellaR" / "featureData.txt",
        gene_count_matrix=EXTERNAL_DROPLET_ROOT / "Gene_STARsolo" / "matrix_counts.mtx",
        gene_count_pheno=EXTERNAL_DROPLET_ROOT / "Gene_STARsolo" / "phenoData.txt",
        gene_count_feature=EXTERNAL_DROPLET_ROOT / "Gene_STARsolo" / "featureData.txt",
        sj_count_matrix=EXTERNAL_DROPLET_ROOT / "SJ_STARsolo" / "matrix_counts.mtx",
        sj_count_pheno=EXTERNAL_DROPLET_ROOT / "SJ_STARsolo" / "phenoData.txt",
        sj_count_feature=EXTERNAL_DROPLET_ROOT / "SJ_STARsolo" / "featureData.txt",
        pca=EXTERNAL_DROPLET_ROOT / "Gene_SingCellaR" / "dim_red_coordinates_iPSC_CardioDay10.txt",
        gtf=EXTERNAL_DROPLET_ROOT / "GTF" / "refdata-cellranger-GRCh38-3.0.0.gtf",
    )
    marvel.annotate_genes()
    marvel.annotate_sj()
    marvel.validate_sj()
    marvel.filter_genes()
    marvel.check_alignment()

    group1, group2 = marvel.get_cell_groups("cell.type", "iPSC", "Cardio day 10")
    replay_controls = replay_controls or DropletReplayControls()

    marvel = plot_pct_expr_cells_genes_10x(marvel, cell_group_g1=group1, cell_group_g2=group2, min_pct_cells=5.0)
    marvel = plot_pct_expr_cells_sj_10x(
        marvel,
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
    marvel = compare_values_sj_10x(
        marvel,
        cell_group_g1=de_group1,
        cell_group_g2=de_group2,
        min_pct_cells_genes=10.0,
        min_pct_cells_sj=10.0,
        min_gene_norm=1.0,
        seed=1,
        n_iterations=10,
        downsample=not uses_r_de_replay,
        permutation_cell_ids=replay_controls.permutation_cell_ids,
        bounded_pval=replay_controls.permutation_cell_ids is None,
    )
    marvel = compare_values_genes_10x(marvel)

    summary = {
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
        "de_iterations": 10,
        "seed": 1,
        "runtime_seconds": time.perf_counter() - start,
        "r_replay_controls": {
            "pct_expr_sj_downsample_coord_introns": replay_controls.pct_expr_sj_coord_introns is not None,
            "de_sj_downsample_cells": uses_r_de_replay,
            "de_sj_permutation_cell_ids": replay_controls.permutation_cell_ids is not None,
        },
        "note": "External droplet benchmark uses 10 permutations for tractable replay on the full tutorial dataset.",
    }
    marvel.save_outputs(output_dir, summary)
    return output_dir


def _run_r_workflows() -> dict[str, Path]:
    plate_dir = R_RUNS_ROOT / "plate"
    droplet_dir = R_RUNS_ROOT / "droplet"
    if (
        (plate_dir / "summary.json").exists()
        and (droplet_dir / "summary.json").exists()
        and _summary_has_runtime(plate_dir / "summary.json")
        and _summary_has_runtime(droplet_dir / "summary.json")
        and _plate_summary_uses_flat_compute_psi(plate_dir / "summary.json")
    ):
        return {"plate": plate_dir, "droplet": droplet_dir}

    _reset_dir(R_RUNS_ROOT)
    plate_dir = R_RUNS_ROOT / "plate"
    droplet_dir = R_RUNS_ROOT / "droplet"
    rscript = _resolve_rscript()
    plate_seconds = _run_command_timed([rscript, str(SCRIPT_ROOT / "run_external_plate_r.R"), str(plate_dir)])
    droplet_seconds = _run_command_timed([rscript, str(SCRIPT_ROOT / "run_external_droplet_r.R"), str(droplet_dir)])
    _annotate_runtime(plate_dir / "summary.json", plate_seconds)
    _annotate_runtime(droplet_dir / "summary.json", droplet_seconds)
    return {"plate": plate_dir, "droplet": droplet_dir}


def _annotate_runtime(summary_path: Path, runtime_seconds: float) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_seconds"] = float(runtime_seconds)
    _write_json(summary_path, summary)


def _summary_has_runtime(summary_path: Path) -> bool:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("runtime_seconds") is not None


def _plate_summary_uses_flat_compute_psi(summary_path: Path) -> bool:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("source_mode") == "flat_files_compute_psi"


def _normalize_count_events(r_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_df = r_df.copy()
    py_df = py_df.copy()
    for df in (r_df, py_df):
        if "event_type" in df.columns:
            df["event_type"] = df["event_type"].astype(str)
    common_events = sorted(set(r_df["event_type"]).intersection(set(py_df["event_type"])))
    r_df = r_df[r_df["event_type"].isin(common_events)].copy()
    py_df = py_df[py_df["event_type"].isin(common_events)].copy()
    for df in (r_df, py_df):
        total = float(pd.to_numeric(df["freq"], errors="coerce").sum())
        if "pct" in df.columns and total > 0:
            df["pct"] = pd.to_numeric(df["freq"], errors="coerce") / total * 100.0
    return r_df.sort_values("event_type").reset_index(drop=True), py_df.sort_values("event_type").reset_index(drop=True)


def _normalize_droplet_pct_expr_gene(r_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename_map = {"group": "cell.group", "n.cells.expr.gene": "n.cells.expr"}
    r_df = r_df.rename(columns=rename_map)
    py_df = py_df.rename(columns=rename_map)
    key = ["cell.group", "gene_short_name"]
    return r_df.sort_values(key).reset_index(drop=True), py_df.sort_values(key).reset_index(drop=True)


def _normalize_droplet_pct_expr_sj(r_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename_map = {"group": "cell.group", "n.cells.expr.sj": "n.cells.expr"}
    r_df = r_df.rename(columns=rename_map)
    py_df = py_df.rename(columns=rename_map)
    key = ["cell.group", "coord.intron"]
    return r_df.sort_values(key).reset_index(drop=True), py_df.sort_values(key).reset_index(drop=True)


def _normalize_droplet_de_gene(r_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_df = r_df.copy()
    py_df = py_df.copy()
    if "diff.mean.log2.gene.norm" in py_df.columns and "log2fc.gene.norm" in py_df.columns:
        py_df["true.log2fc.gene.norm"] = py_df["log2fc.gene.norm"]
        py_df["log2fc.gene.norm"] = py_df["diff.mean.log2.gene.norm"]
    key = ["gene_short_name"]
    return r_df.sort_values(key).reset_index(drop=True), py_df.sort_values(key).reset_index(drop=True)


def _normalize_variable_splicing(r_df: pd.DataFrame, py_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_columns = ["tran_id", "mean", "sd", "sd_pred", "variable"]
    r_df = r_df.loc[:, [column for column in keep_columns if column in r_df.columns]].copy()
    py_df = py_df.loc[:, [column for column in keep_columns if column in py_df.columns]].copy()
    return r_df.sort_values("tran_id").reset_index(drop=True), py_df.sort_values("tran_id").reset_index(drop=True)


def _build_readme(summary: dict[str, object]) -> None:
    lines = [
        "# External Data Benchmark",
        "",
        "This directory stores fresh R and Python replay outputs on the full external tutorial datasets:",
        "- `external_plate_data`",
        "- `external_droplet_data`",
        "",
        "Notes:",
        "- Plate replay subsets `qc.seq == pass` and `cell.type in {iPSC, Endoderm}` as in the tutorial.",
        "- Plate external flat inputs do not include AFE/ALE feature tables, so the replay covers `SE/MXE/RI/A5SS/A3SS`.",
        "- Droplet replay uses `iPSC` vs `Cardio day 10` with `10` permutations for tractable benchmarking on the full dataset.",
        "",
        "Sections:",
    ]
    for section in summary["sections"]:
        lines.append(f"## {section['section'].capitalize()}")
        for artifact in section["artifacts"]:
            lines.append(f"- `{artifact['artifact']}`: rows R={artifact['rows_r']}, Python={artifact['rows_python']}")
    (BENCHMARK_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dir(BENCHMARK_ROOT)
    r_outputs = _run_r_workflows()
    plate_py = _run_plate_python()
    droplet_replay_controls = _load_droplet_replay_controls(r_outputs["droplet"])
    droplet_py = _run_droplet_python(droplet_replay_controls)

    plate_dir = _reset_dir(BENCHMARK_ROOT / "plate")
    droplet_dir = _reset_dir(BENCHMARK_ROOT / "droplet")

    summary = {
        "benchmark_root": str(BENCHMARK_ROOT),
        "r_runs_root": str(R_RUNS_ROOT),
        "python_runs_root": str(PY_RUNS_ROOT),
        "sections": [],
    }

    plate_artifacts = [
        _benchmark_artifact(
            title="External Plate CompareValues Gene",
            artifact_dir=plate_dir / "de_gene",
            r_path=r_outputs["plate"] / "de_gene.tsv",
            python_path=plate_py / "de_gene.tsv",
            key_columns=["gene_id"],
        ),
        _benchmark_artifact(
            title="External Plate CountEvents iPSC",
            artifact_dir=plate_dir / "count_events_iPSC",
            r_path=r_outputs["plate"] / "n_events_min_cells_iPSC_min_cells_25.tsv",
            python_path=plate_py / "n_events_min_cells_iPSC_min_cells_25.tsv",
            key_columns=["event_type"],
            preprocess=_normalize_count_events,
        ),
        _benchmark_artifact(
            title="External Plate CountEvents Endoderm",
            artifact_dir=plate_dir / "count_events_endoderm",
            r_path=r_outputs["plate"] / "n_events_min_cells_Endoderm_min_cells_25.tsv",
            python_path=plate_py / "n_events_min_cells_Endoderm_min_cells_25.tsv",
            key_columns=["event_type"],
            preprocess=_normalize_count_events,
        ),
        _benchmark_artifact(
            title="External Plate IdentifyVariableEvents",
            artifact_dir=plate_dir / "variable_splicing",
            r_path=r_outputs["plate"] / "variable_splicing_table.tsv",
            python_path=plate_py / "variable_splicing_table.tsv",
            key_columns=["tran_id"],
            preprocess=_normalize_variable_splicing,
        ),
    ]
    if (r_outputs["plate"] / "psi_se.tsv").exists() and (plate_py / "psi_se.tsv").exists():
        plate_artifacts.insert(
            0,
            _benchmark_artifact(
                title="External Plate ComputePSI SE",
                artifact_dir=plate_dir / "psi_se",
                r_path=r_outputs["plate"] / "psi_se.tsv",
                python_path=plate_py / "psi_se.tsv",
                key_columns=["tran_id"],
            ),
        )
    if (r_outputs["plate"] / "psi_ri.tsv").exists() and (plate_py / "psi_ri.tsv").exists():
        plate_artifacts.insert(
            1,
            _benchmark_artifact(
                title="External Plate ComputePSI RI",
                artifact_dir=plate_dir / "psi_ri",
                r_path=r_outputs["plate"] / "psi_ri.tsv",
                python_path=plate_py / "psi_ri.tsv",
                key_columns=["tran_id"],
            ),
        )

    droplet_artifacts = [
        _benchmark_artifact(
            title="External Droplet PlotPctExprCells Genes",
            artifact_dir=droplet_dir / "pct_expr_gene",
            r_path=r_outputs["droplet"] / "pct_expr_gene.tsv",
            python_path=droplet_py / "pct_expr_gene.tsv",
            key_columns=["cell.group", "gene_short_name"],
            preprocess=_normalize_droplet_pct_expr_gene,
        ),
        _benchmark_artifact(
            title="External Droplet PlotPctExprCells SJ",
            artifact_dir=droplet_dir / "pct_expr_sj",
            r_path=r_outputs["droplet"] / "pct_expr_sj.tsv",
            python_path=droplet_py / "pct_expr_sj.tsv",
            key_columns=["cell.group", "coord.intron"],
            preprocess=_normalize_droplet_pct_expr_sj,
        ),
        _benchmark_artifact(
            title="External Droplet CompareValues SJ 10x",
            artifact_dir=droplet_dir / "de_sj",
            r_path=r_outputs["droplet"] / "de_sj.tsv",
            python_path=droplet_py / "de_sj.tsv",
            key_columns=["coord.intron"],
        ),
        _benchmark_artifact(
            title="External Droplet CompareValues Genes 10x",
            artifact_dir=droplet_dir / "de_gene",
            r_path=r_outputs["droplet"] / "de_gene.tsv",
            python_path=droplet_py / "de_gene.tsv",
            key_columns=["gene_short_name"],
            preprocess=_normalize_droplet_de_gene,
        ),
    ]

    shutil.copy2(plate_py / "summary.json", plate_dir / "python_summary.json")
    shutil.copy2(r_outputs["plate"] / "summary.json", plate_dir / "r_summary.json")
    shutil.copy2(droplet_py / "summary.json", droplet_dir / "python_summary.json")
    shutil.copy2(r_outputs["droplet"] / "summary.json", droplet_dir / "r_summary.json")

    plate_r_summary = json.loads((r_outputs["plate"] / "summary.json").read_text(encoding="utf-8"))
    plate_py_summary = json.loads((plate_py / "summary.json").read_text(encoding="utf-8"))
    plate_variable_py_summary = json.loads((PY_RUNS_ROOT / "plate_variable_splicing" / "summary.json").read_text(encoding="utf-8"))
    droplet_r_summary = json.loads((r_outputs["droplet"] / "summary.json").read_text(encoding="utf-8"))
    droplet_py_summary = json.loads((droplet_py / "summary.json").read_text(encoding="utf-8"))

    summary["sections"].append(
        {
            "section": "plate",
            "artifacts": plate_artifacts,
            "runtime_seconds": {
                "r": plate_r_summary.get("runtime_seconds"),
                "python": plate_py_summary.get("runtime_seconds"),
            },
        }
    )
    summary["sections"].append(
        {
            "section": "plate_variable_splicing",
            "artifacts": [],
            "runtime_seconds": {
                "r": (plate_r_summary.get("variable_splicing") or {}).get("runtime_seconds"),
                "python": plate_variable_py_summary.get("runtime_seconds"),
            },
        }
    )
    summary["sections"].append(
        {
            "section": "droplet",
            "artifacts": droplet_artifacts,
            "runtime_seconds": {
                "r": droplet_r_summary.get("runtime_seconds"),
                "python": droplet_py_summary.get("runtime_seconds"),
            },
        }
    )

    _write_json(BENCHMARK_ROOT / "summary.json", summary)
    _build_readme(summary)


if __name__ == "__main__":
    main()
