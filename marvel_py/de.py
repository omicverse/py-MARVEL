from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .adhoc import adhoc_gene_de_gene_10x, adhoc_gene_de_psi_10x
from .matrix import Marvel10x
from .models import MarvelPlate
from .modality import annotate_splicing_outliers
from .psi import PLATE_EVENT_TYPES

__all__ = [
    "_normalize_methods",
    "adhoc_gene_de_gene_10x",
    "adhoc_gene_de_psi_10x",
    "compare_values",
    "compare_values_genes_10x",
    "compare_values_sj_10x",
    "compare_values_sj_donor_level_10x",
]


def _normalize_method_adjust(method_adjust: str) -> str:
    return "fdr_bh" if method_adjust == "fdr" else method_adjust


def _normalize_methods(method, *, level: str) -> list[str] | str:
    if isinstance(method, str):
        method = method.strip()
        if not method:
            raise ValueError("method must be non-empty")
        return method if level == "gene" else [method]

    try:
        methods = list(method)
    except TypeError as exc:
        raise ValueError("method must be non-empty") from exc
    if not methods:
        raise ValueError("method must be non-empty")

    normalized_methods: list[str] = []
    for item in methods:
        if item is None:
            raise ValueError("method entries must be non-empty")
        normalized_item = str(item).strip()
        if not normalized_item:
            raise ValueError("method entries must be non-empty")
        normalized_methods.append(normalized_item)

    if level == "gene":
        if len(normalized_methods) != 1:
            raise ValueError("compare_values(level='gene') accepts exactly one method")
        return normalized_methods[0]
    return normalized_methods


def _normalize_event_types(event_type) -> list[str]:
    if isinstance(event_type, str):
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must be non-empty for level='splicing'")
        return [event_type]

    try:
        event_types = list(event_type)
    except TypeError as exc:
        raise ValueError("event_type must be non-empty for level='splicing'") from exc

    if not event_types:
        raise ValueError("event_type must be non-empty for level='splicing'")

    normalized_event_types: list[str] = []
    for item in event_types:
        if item is None:
            raise ValueError("event_type entries must be non-empty")
        normalized_item = str(item).strip()
        if not normalized_item:
            raise ValueError("event_type entries must be non-empty")
        normalized_event_types.append(normalized_item)
    return normalized_event_types


def _plate_compare_values_genes_inplace(
    marvel_object: MarvelPlate,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    min_cells: int = 25,
    pct_cells: float | None = None,
    method: str = "wilcox",
    method_adjust: str = "fdr_bh",
    custom_gene_ids: list[str] | None = None,
) -> pd.DataFrame:
    method_adjust = _normalize_method_adjust(method_adjust)
    group1 = [str(sample_id) for sample_id in cell_group_g1]
    group2 = [str(sample_id) for sample_id in cell_group_g2]
    exp = marvel_object.exp.copy()
    exp.index = exp["gene_id"].astype(str)
    values = exp[group1 + group2].apply(pd.to_numeric, errors="coerce")

    if custom_gene_ids is None:
        expr_g1 = (values[group1] > 0).sum(axis=1)
        expr_g2 = (values[group2] > 0).sum(axis=1)
        if pct_cells is None:
            keep = (expr_g1 >= min_cells) | (expr_g2 >= min_cells)
        else:
            pct_g1 = expr_g1 / len(group1) * 100.0
            pct_g2 = expr_g2 / len(group2) * 100.0
            keep = (pct_g1 >= pct_cells) | (pct_g2 >= pct_cells)
        gene_ids = values.index[keep].tolist()
    else:
        gene_ids = [gene_id for gene_id in custom_gene_ids if gene_id in values.index]

    values = values.loc[gene_ids]
    variability = values.nunique(axis=1, dropna=False) > 1
    values = values.loc[variability]
    gene_ids = values.index.tolist()

    feature = marvel_object.gene_feature.copy()
    feature = feature[feature["gene_id"].astype(str).isin(gene_ids)].copy()
    feature = feature.set_index("gene_id").loc[gene_ids].reset_index()

    records = []
    for gene_id in gene_ids:
        x = values.loc[gene_id, group1].to_numpy(dtype=float)
        y = values.loc[gene_id, group2].to_numpy(dtype=float)
        statistic, pvalue = marvel_object._run_de_test(x, y, method)
        records.append(
            {
                "gene_id": gene_id,
                "n.cells.g1": int(np.sum(x > 0)),
                "n.cells.g2": int(np.sum(y > 0)),
                "mean.g1": float(np.nanmean(x)),
                "mean.g2": float(np.nanmean(y)),
                "log2fc": float(np.nanmean(y) - np.nanmean(x)),
                "statistic": statistic,
                "p.val": pvalue,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        result = feature.iloc[0:0].copy()
        result["n.cells.g1"] = pd.Series(dtype=int)
        result["n.cells.g2"] = pd.Series(dtype=int)
        result["mean.g1"] = pd.Series(dtype=float)
        result["mean.g2"] = pd.Series(dtype=float)
        result["log2fc"] = pd.Series(dtype=float)
        result["statistic"] = pd.Series(dtype=float)
        result["p.val"] = pd.Series(dtype=float)
        result["p.val.adj"] = pd.Series(dtype=float)
        marvel_object.de_gene = result
        return result
    result = result.dropna(subset=["p.val"]).sort_values("p.val").reset_index(drop=True)
    result["p.val.adj"] = multipletests(result["p.val"], method=method_adjust)[1] if not result.empty else []
    result = feature.merge(result, on="gene_id", how="inner")
    marvel_object.de_gene = result
    return result


def _plate_compare_values_splicing_dts_inplace(
    marvel_object: MarvelPlate,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    method_adjust: str,
    min_cells: int,
    pct_cells: float | None,
    event_types: list[str] | None,
    assign_modality: bool,
    annotate_outliers: bool,
    n_cells_outliers: int,
    sigma_sq: float,
    bimodal_adjust: bool,
    bimodal_adjust_fc: float,
    bimodal_adjust_diff: float,
    seed: int,
    nboots: int,
    seed_dts: int | None,
) -> pd.DataFrame:
    method_adjust = _normalize_method_adjust(method_adjust)
    rng = np.random.default_rng(seed if seed_dts is None else seed_dts)
    event_types = PLATE_EVENT_TYPES if event_types is None else [event.upper() for event in event_types]
    group1 = [str(sample_id) for sample_id in cell_group_g1]
    group2 = [str(sample_id) for sample_id in cell_group_g2]

    psi_tables = []
    feature_tables = []
    for event_type in event_types:
        psi_df = marvel_object.psi.get(event_type)
        feature_df = marvel_object.splice_feature_validated.get(event_type)
        if psi_df is None or feature_df is None or psi_df.empty or feature_df.empty:
            continue
        psi_tables.append(psi_df)
        feature_tables.append(feature_df)
    if not psi_tables:
        raise ValueError("No PSI tables available for the requested event types")

    psi_all = pd.concat(psi_tables, ignore_index=True).set_index("tran_id")
    feature_all = pd.concat(feature_tables, ignore_index=True)
    values = psi_all[group1 + group2].apply(pd.to_numeric, errors="coerce")
    n_g1 = values[group1].notna().sum(axis=1)
    n_g2 = values[group2].notna().sum(axis=1)
    if pct_cells is None:
        keep = (n_g1 >= min_cells) & (n_g2 >= min_cells)
    else:
        pct_g1 = n_g1 / len(group1) * 100.0
        pct_g2 = n_g2 / len(group2) * 100.0
        keep = (pct_g1 >= pct_cells) & (pct_g2 >= pct_cells)
    values = values.loc[keep]

    records = []
    for tran_id in values.index.tolist():
        x = values.loc[tran_id, group1].dropna().to_numpy(dtype=float)
        y = values.loc[tran_id, group2].dropna().to_numpy(dtype=float)
        if len(x) == 0 or len(y) == 0:
            continue
        observed = float(np.abs(np.nanmean(y) - np.nanmean(x)))
        pvalue = _bootstrap_abs_mean_diff_pvalue_blocked(x, y, rng=rng, nboots=nboots, observed=observed)
        records.append(
            {
                "tran_id": tran_id,
                "n.cells.g1": int(len(x)),
                "n.cells.g2": int(len(y)),
                "mean.g1": float(np.nanmean(x)),
                "mean.g2": float(np.nanmean(y)),
                "mean.diff": float(np.nanmean(y) - np.nanmean(x)),
                "statistic": observed,
                "p.val": pvalue,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        result = feature_all.iloc[0:0].copy()
        result["n.cells.g1"] = pd.Series(dtype=int)
        result["n.cells.g2"] = pd.Series(dtype=int)
        result["mean.g1"] = pd.Series(dtype=float)
        result["mean.g2"] = pd.Series(dtype=float)
        result["mean.diff"] = pd.Series(dtype=float)
        result["statistic"] = pd.Series(dtype=float)
        result["p.val"] = pd.Series(dtype=float)
        result["p.val.adj"] = pd.Series(dtype=float)
        result["modality.bimodal.adj.g1"] = pd.Series(dtype=str)
        result["modality.bimodal.adj.g2"] = pd.Series(dtype=str)
        result["n.cells.outliers.g1"] = pd.Series(dtype=int)
        result["n.cells.outliers.g2"] = pd.Series(dtype=int)
        result["outliers"] = pd.Series(dtype=bool)
        result["outlier"] = pd.Series(dtype=bool)
        marvel_object.de_splicing["dts"] = result
        return result

    result = result.dropna(subset=["p.val"]).sort_values("p.val").reset_index(drop=True)
    result["mean.g1"] = result["mean.g1"] * 100.0
    result["mean.g2"] = result["mean.g2"] * 100.0
    result["mean.diff"] = result["mean.diff"] * 100.0
    result["statistic"] = result["statistic"] * 100.0
    result["p.val.adj"] = multipletests(result["p.val"], method=method_adjust)[1]
    result = feature_all.merge(result, on="tran_id", how="inner")
    if assign_modality:
        modality_g1 = marvel_object.assign_modality(
            sample_ids=group1,
            min_cells=min_cells,
            sigma_sq=sigma_sq,
            bimodal_adjust=bimodal_adjust,
            bimodal_adjust_fc=bimodal_adjust_fc,
            bimodal_adjust_diff=bimodal_adjust_diff,
            seed=seed,
            tran_ids=result["tran_id"].astype(str).tolist(),
            update_store=False,
        )[["tran_id", "modality.bimodal.adj"]].rename(columns={"modality.bimodal.adj": "modality.bimodal.adj.g1"})
        modality_g2 = marvel_object.assign_modality(
            sample_ids=group2,
            min_cells=min_cells,
            sigma_sq=sigma_sq,
            bimodal_adjust=bimodal_adjust,
            bimodal_adjust_fc=bimodal_adjust_fc,
            bimodal_adjust_diff=bimodal_adjust_diff,
            seed=seed,
            tran_ids=result["tran_id"].astype(str).tolist(),
            update_store=False,
        )[["tran_id", "modality.bimodal.adj"]].rename(columns={"modality.bimodal.adj": "modality.bimodal.adj.g2"})
        result = result.merge(modality_g1, on="tran_id", how="left")
        result = result.merge(modality_g2, on="tran_id", how="left")
    else:
        result["modality.bimodal.adj.g1"] = pd.NA
        result["modality.bimodal.adj.g2"] = pd.NA
    if annotate_outliers and assign_modality:
        result = annotate_splicing_outliers(
            result=result,
            values=values,
            group1=group1,
            group2=group2,
            n_cells_outliers=n_cells_outliers,
        )
    else:
        result["n.cells.outliers.g1"] = 0
        result["n.cells.outliers.g2"] = 0
        result["outliers"] = False
    result["outlier"] = result["outliers"].astype(bool)
    result = result[
        [
            "tran_id",
            "event_type",
            "gene_id",
            "gene_short_name",
            "gene_type",
            "n.cells.g1",
            "n.cells.g2",
            "mean.g1",
            "mean.g2",
            "mean.diff",
            "statistic",
            "p.val",
            "p.val.adj",
            "modality.bimodal.adj.g1",
            "modality.bimodal.adj.g2",
            "n.cells.outliers.g1",
            "n.cells.outliers.g2",
            "outliers",
            "outlier",
        ]
    ]
    marvel_object.de_splicing["dts"] = result
    return result


def _bootstrap_abs_mean_diff_pvalue_blocked(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    nboots: int,
    observed: float | None = None,
    block_size: int = 256,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    if observed is None:
        observed = float(np.abs(np.nanmean(y) - np.nanmean(x)))
    block_size = max(1, int(block_size))
    tail_count = 0
    for start in range(0, int(nboots), block_size):
        size = min(block_size, int(nboots) - start)
        resampled = rng.choice(pooled, size=(size, pooled.size), replace=True)
        boot_x = resampled[:, : len(x)]
        boot_y = resampled[:, len(x) :]
        boot_stat = np.abs(np.nanmean(boot_y, axis=1) - np.nanmean(boot_x, axis=1))
        tail_count += int(np.sum(boot_stat >= observed))
    return float((tail_count + 1.0) / (float(nboots) + 1.0))


def _plate_compare_values_splicing_inplace(
    marvel_object: MarvelPlate,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    method: str = "ad",
    method_adjust: str = "fdr_bh",
    min_cells: int = 25,
    pct_cells: float | None = None,
    event_types: list[str] | None = None,
    assign_modality: bool = True,
    annotate_outliers: bool = True,
    n_cells_outliers: int = 10,
    sigma_sq: float = 0.001,
    bimodal_adjust: bool = True,
    bimodal_adjust_fc: float = 3.0,
    bimodal_adjust_diff: float = 50.0,
    seed: int = 1,
    nboots: int = 1000,
    seed_dts: int | None = None,
) -> pd.DataFrame:
    method_adjust = _normalize_method_adjust(method_adjust)
    method = method.lower()
    if method == "dts":
        return _plate_compare_values_splicing_dts_inplace(
            marvel_object,
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            method_adjust=method_adjust,
            min_cells=min_cells,
            pct_cells=pct_cells,
            event_types=event_types,
            assign_modality=assign_modality,
            annotate_outliers=annotate_outliers,
            n_cells_outliers=n_cells_outliers,
            sigma_sq=sigma_sq,
            bimodal_adjust=bimodal_adjust,
            bimodal_adjust_fc=bimodal_adjust_fc,
            bimodal_adjust_diff=bimodal_adjust_diff,
            seed=seed,
            nboots=nboots,
            seed_dts=seed_dts,
        )
    event_types = PLATE_EVENT_TYPES if event_types is None else [event.upper() for event in event_types]
    group1 = [str(sample_id) for sample_id in cell_group_g1]
    group2 = [str(sample_id) for sample_id in cell_group_g2]
    psi_tables = []
    feature_tables = []
    for event_type in event_types:
        psi_df = marvel_object.psi.get(event_type)
        feature_df = marvel_object.splice_feature_validated.get(event_type)
        if psi_df is None or feature_df is None or psi_df.empty or feature_df.empty:
            continue
        psi_tables.append(psi_df)
        feature_tables.append(feature_df)
    if not psi_tables:
        raise ValueError("No PSI tables available for the requested event types")
    psi_all = pd.concat(psi_tables, ignore_index=True).set_index("tran_id")
    feature_all = pd.concat(feature_tables, ignore_index=True)
    values = psi_all[group1 + group2].apply(pd.to_numeric, errors="coerce")
    n_g1 = values[group1].notna().sum(axis=1)
    n_g2 = values[group2].notna().sum(axis=1)
    if pct_cells is None:
        keep = (n_g1 >= min_cells) & (n_g2 >= min_cells)
    else:
        pct_g1 = n_g1 / len(group1) * 100.0
        pct_g2 = n_g2 / len(group2) * 100.0
        keep = (pct_g1 >= pct_cells) & (pct_g2 >= pct_cells)
    values = values.loc[keep]
    if method == "permutation":
        values = values.loc[values.apply(lambda row: row.dropna().nunique() > 1, axis=1)]
    records = []
    for tran_id in values.index.tolist():
        x = values.loc[tran_id, group1].dropna().to_numpy(dtype=float)
        y = values.loc[tran_id, group2].dropna().to_numpy(dtype=float)
        statistic, pvalue = marvel_object._run_de_test(x, y, method)
        records.append(
            {
                "tran_id": tran_id,
                "n.cells.g1": int(len(x)),
                "n.cells.g2": int(len(y)),
                "mean.g1": float(np.mean(x)),
                "mean.g2": float(np.mean(y)),
                "mean.diff": float(np.mean(y) - np.mean(x)),
                "statistic": statistic,
                "p.val": pvalue,
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        result = feature_all.iloc[0:0].copy()
        result["n.cells.g1"] = pd.Series(dtype=int)
        result["n.cells.g2"] = pd.Series(dtype=int)
        result["mean.g1"] = pd.Series(dtype=float)
        result["mean.g2"] = pd.Series(dtype=float)
        result["mean.diff"] = pd.Series(dtype=float)
        result["statistic"] = pd.Series(dtype=float)
        result["p.val"] = pd.Series(dtype=float)
        result["p.val.adj"] = pd.Series(dtype=float)
        result["modality.bimodal.adj.g1"] = pd.Series(dtype=str)
        result["modality.bimodal.adj.g2"] = pd.Series(dtype=str)
        result["n.cells.outliers.g1"] = pd.Series(dtype=int)
        result["n.cells.outliers.g2"] = pd.Series(dtype=int)
        result["outliers"] = pd.Series(dtype=bool)
        result["outlier"] = pd.Series(dtype=bool)
        marvel_object.de_splicing[method] = result
        return result
    result = result.dropna(subset=["p.val"]).sort_values("p.val").reset_index(drop=True)
    result["mean.g1"] = result["mean.g1"] * 100.0
    result["mean.g2"] = result["mean.g2"] * 100.0
    result["mean.diff"] = result["mean.diff"] * 100.0
    result["p.val.adj"] = result["p.val"] if method in {"dts", "permutation"} else multipletests(result["p.val"], method=method_adjust)[1]
    result = feature_all.merge(result, on="tran_id", how="inner")
    if assign_modality:
        modality_g1 = marvel_object.assign_modality(sample_ids=group1, min_cells=min_cells, sigma_sq=sigma_sq, bimodal_adjust=bimodal_adjust, bimodal_adjust_fc=bimodal_adjust_fc, bimodal_adjust_diff=bimodal_adjust_diff, seed=seed, tran_ids=result["tran_id"].astype(str).tolist(), update_store=False)[["tran_id", "modality.bimodal.adj"]].rename(columns={"modality.bimodal.adj": "modality.bimodal.adj.g1"})
        modality_g2 = marvel_object.assign_modality(sample_ids=group2, min_cells=min_cells, sigma_sq=sigma_sq, bimodal_adjust=bimodal_adjust, bimodal_adjust_fc=bimodal_adjust_fc, bimodal_adjust_diff=bimodal_adjust_diff, seed=seed, tran_ids=result["tran_id"].astype(str).tolist(), update_store=False)[["tran_id", "modality.bimodal.adj"]].rename(columns={"modality.bimodal.adj": "modality.bimodal.adj.g2"})
        result = result.merge(modality_g1, on="tran_id", how="left").merge(modality_g2, on="tran_id", how="left")
    else:
        result["modality.bimodal.adj.g1"] = pd.NA
        result["modality.bimodal.adj.g2"] = pd.NA
    if annotate_outliers and assign_modality:
        result = annotate_splicing_outliers(result=result, values=values, group1=group1, group2=group2, n_cells_outliers=n_cells_outliers)
    else:
        result["n.cells.outliers.g1"] = 0
        result["n.cells.outliers.g2"] = 0
        result["outliers"] = False
    result["outlier"] = result["outliers"].astype(bool)
    result = result[["tran_id","event_type","gene_id","gene_short_name","gene_type","n.cells.g1","n.cells.g2","mean.g1","mean.g2","mean.diff","statistic","p.val","p.val.adj","modality.bimodal.adj.g1","modality.bimodal.adj.g2","n.cells.outliers.g1","n.cells.outliers.g2","outliers","outlier"]]
    marvel_object.de_splicing[method] = result
    return result


def _droplet_compare_values_sj_inplace(
    marvel_object: Marvel10x,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    coord_introns: list[str] | None = None,
    min_pct_cells_genes: float = 10,
    min_pct_cells_sj: float = 10,
    min_gene_norm: float = 1.0,
    seed: int = 1,
    n_iterations: int = 100,
    downsample: bool = False,
    permutation_cell_ids: list[list[str]] | None = None,
    bounded_pval: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g1 = list(cell_group_g1)
    g2 = list(cell_group_g2)
    if downsample:
        n_cells = min(len(g1), len(g2))
        g1 = rng.choice(np.asarray(g1), size=n_cells, replace=False).tolist()
        g2 = rng.choice(np.asarray(g2), size=n_cells, replace=False).tolist()
    from .utils import ordered_intersection
    gene_expr_g1 = marvel_object._expression_rate_table(marvel_object.gene_norm_matrix, g1, "gene_short_name", "cell.group.g1")
    gene_expr_g2 = marvel_object._expression_rate_table(marvel_object.gene_norm_matrix, g2, "gene_short_name", "cell.group.g2")
    genes_1 = gene_expr_g1.loc[gene_expr_g1["pct.cells.expr"] > min_pct_cells_genes, "gene_short_name"].tolist()
    genes_2 = gene_expr_g2.loc[gene_expr_g2["pct.cells.expr"] > min_pct_cells_genes, "gene_short_name"].tolist()
    gene_short_names = ordered_intersection(genes_1, genes_2)
    gene_norm_small = marvel_object.gene_norm_matrix.subset_rows(gene_short_names).subset_cols(g1 + g2)
    gene_norm_log = gene_norm_small.matrix.copy().tocsr()
    gene_norm_log.data = np.log2(gene_norm_log.data + 1.0)
    mean_combined = np.asarray(gene_norm_log.sum(axis=1)).ravel() / gene_norm_log.shape[1]
    mean_combined_df = pd.DataFrame({"gene_short_name": gene_norm_small.row_ids, "mean.expr.gene.norm.g1.g2": mean_combined})
    gene_short_names = mean_combined_df.loc[mean_combined_df["mean.expr.gene.norm.g1.g2"] > min_gene_norm, "gene_short_name"].tolist()
    requested_coord_introns = None if coord_introns is None else {str(coord) for coord in coord_introns}
    coord_introns = list(dict.fromkeys(marvel_object._sj_above_threshold(g1, gene_short_names, min_pct_cells_sj) + marvel_object._sj_above_threshold(g2, gene_short_names, min_pct_cells_sj)))
    if requested_coord_introns is not None:
        coord_introns = [coord for coord in coord_introns if coord in requested_coord_introns]
    observed = marvel_object._build_sj_results(coord_introns, g1, g2)
    observed["log2fc"] = np.log2((observed["psi.g2"] + 1.0) / (observed["psi.g1"] + 1.0))
    observed["delta"] = observed["psi.g2"] - observed["psi.g1"]
    sj_matrix = marvel_object.sj_count_matrix.subset_rows(coord_introns).subset_cols(g1 + g2)
    sj_meta = marvel_object.sj_metadata.set_index("coord.intron").loc[coord_introns].reset_index()
    sj_gene_names = sj_meta["gene_short_name.start"].tolist()
    gene_ids = list(dict.fromkeys(sj_gene_names))
    gene_matrix = marvel_object.gene_count_matrix.subset_rows(gene_ids).subset_cols(g1 + g2)
    gene_index = {gene: i for i, gene in enumerate(gene_ids)}
    sj_gene_idx = np.array([gene_index[gene] for gene in sj_gene_names])
    permutation_indices: list[tuple[np.ndarray, np.ndarray]] = []
    if permutation_cell_ids is not None:
        if len(permutation_cell_ids) != n_iterations:
            raise ValueError("permutation_cell_ids must contain one shuffled cell-id list per iteration")
        g1_set = set(g1)
        g2_set = set(g2)
        expected_cells = set(g1 + g2)
        for shuffled_cell_ids in permutation_cell_ids:
            shuffled = [str(cell_id) for cell_id in shuffled_cell_ids]
            if len(shuffled) != len(g1) + len(g2) or set(shuffled) != expected_cells:
                raise ValueError("each permutation_cell_ids entry must contain the same cells as cell_group_g1 + cell_group_g2")
            g1_idx = np.array([idx for idx, cell_id in enumerate(shuffled) if cell_id in g1_set])
            g2_idx = np.array([idx for idx, cell_id in enumerate(shuffled) if cell_id in g2_set])
            permutation_indices.append((g1_idx, g2_idx))
    else:
        for _ in range(n_iterations):
            perm = rng.permutation(len(g1) + len(g2))
            permutation_indices.append((perm[: len(g1)], perm[len(g1):]))
    perm_deltas = _compute_permutation_deltas_blocked(
        sj_matrix.matrix,
        gene_matrix.matrix,
        sj_gene_idx,
        permutation_indices,
    )
    delta_obs = observed["delta"].to_numpy()
    permutation_tail_counts = (np.abs(perm_deltas) > np.abs(delta_obs[:, None])).sum(axis=1)
    if bounded_pval:
        observed["pval"] = (permutation_tail_counts + 1.0) / float(n_iterations + 1)
    else:
        observed["pval"] = permutation_tail_counts / float(n_iterations)
    observed = observed.sort_values("pval").reset_index(drop=True).merge(mean_combined_df, on="gene_short_name", how="left")
    observed.attrs["cell.group.g1"] = g1
    observed.attrs["cell.group.g2"] = g2
    marvel_object.de_sj = observed
    marvel_object.de_sj_cell_group_g1 = list(g1)
    marvel_object.de_sj_cell_group_g2 = list(g2)
    return observed


def _compute_permutation_deltas_blocked(
    sj_matrix,
    gene_matrix,
    sj_gene_idx: np.ndarray,
    permutation_indices: list[tuple[np.ndarray, np.ndarray]],
    *,
    block_size: int = 64,
) -> np.ndarray:
    n_permutations = len(permutation_indices)
    n_sj, n_cells = sj_matrix.shape
    perm_deltas = np.empty((n_sj, n_permutations), dtype=float)
    block_size = max(1, int(block_size))

    for start in range(0, n_permutations, block_size):
        block = permutation_indices[start : start + block_size]
        masks_g1 = np.zeros((n_cells, len(block)), dtype=float)
        masks_g2 = np.zeros((n_cells, len(block)), dtype=float)
        for column, (group1_idx, group2_idx) in enumerate(block):
            masks_g1[np.asarray(group1_idx, dtype=int), column] = 1.0
            masks_g2[np.asarray(group2_idx, dtype=int), column] = 1.0

        sj_g1 = np.asarray(sj_matrix @ masks_g1)
        sj_g2 = np.asarray(sj_matrix @ masks_g2)
        gene_g1 = np.asarray(gene_matrix @ masks_g1)[sj_gene_idx, :]
        gene_g2 = np.asarray(gene_matrix @ masks_g2)[sj_gene_idx, :]
        psi_g1 = np.round(
            np.divide(sj_g1, gene_g1, out=np.zeros_like(sj_g1, dtype=float), where=gene_g1 != 0) * 100.0,
            2,
        )
        psi_g2 = np.round(
            np.divide(sj_g2, gene_g2, out=np.zeros_like(sj_g2, dtype=float), where=gene_g2 != 0) * 100.0,
            2,
        )
        perm_deltas[:, start : start + len(block)] = psi_g2 - psi_g1

    return perm_deltas


def _summarize_donor_sparse_counts(
    *,
    sj_count_matrix,
    gene_count_matrix,
    coord_introns: list[str],
    gene_short_names: list[str],
    donor_maps: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    if len(coord_introns) != len(gene_short_names):
        raise ValueError("coord_introns and gene_short_names must have the same length")
    if not coord_introns:
        return pd.DataFrame(
            columns=[
                "cell.group",
                "sample.id",
                "coord.intron",
                "gene_short_name",
                "n.cells.total",
                "sj.counts.total",
                "n.cells.sj.expr",
                "gene.counts.total",
                "n.cells.gene.expr",
                "psi",
            ]
        )

    donor_records: list[tuple[str, str, list[str]]] = []
    all_cells: list[str] = []
    for group_name, donor_map in donor_maps.items():
        for donor_id, cell_ids in donor_map.items():
            cell_ids = [str(cell_id) for cell_id in cell_ids]
            donor_records.append((str(group_name), str(donor_id), cell_ids))
            all_cells.extend(cell_ids)
    all_cells = list(dict.fromkeys(all_cells))

    sj_subset = sj_count_matrix.subset_rows(coord_introns).subset_cols(all_cells).matrix.tocsr()
    unique_genes = list(dict.fromkeys(gene_short_names))
    gene_subset = gene_count_matrix.subset_rows(unique_genes).subset_cols(all_cells).matrix.tocsr()
    gene_index = {gene: idx for idx, gene in enumerate(unique_genes)}
    gene_row_idx = np.asarray([gene_index[gene] for gene in gene_short_names], dtype=int)

    cell_index = {cell_id: idx for idx, cell_id in enumerate(all_cells)}
    donor_mask = np.zeros((len(all_cells), len(donor_records)), dtype=float)
    donor_sizes = np.zeros(len(donor_records), dtype=int)
    for donor_idx, (_, _, cell_ids) in enumerate(donor_records):
        donor_sizes[donor_idx] = len(cell_ids)
        for cell_id in cell_ids:
            donor_mask[cell_index[str(cell_id)], donor_idx] = 1.0

    sj_sums = np.asarray(sj_subset @ donor_mask)
    gene_sums_all = np.asarray(gene_subset @ donor_mask)
    gene_sums = gene_sums_all[gene_row_idx, :]
    sj_expr = np.asarray((sj_subset >= 1.0).astype(float) @ donor_mask).astype(int)
    gene_expr_all = np.asarray((gene_subset >= 1.0).astype(float) @ donor_mask).astype(int)
    gene_expr = gene_expr_all[gene_row_idx, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (sj_sums / gene_sums) * 100.0
    psi = np.where(gene_sums == 0.0, np.nan, psi)

    rows = []
    for coord_idx, (coord_intron, gene_short_name) in enumerate(zip(coord_introns, gene_short_names)):
        for donor_idx, (group_name, donor_id, _) in enumerate(donor_records):
            rows.append(
                {
                    "cell.group": group_name,
                    "sample.id": donor_id,
                    "coord.intron": coord_intron,
                    "gene_short_name": gene_short_name,
                    "n.cells.total": int(donor_sizes[donor_idx]),
                    "sj.counts.total": float(sj_sums[coord_idx, donor_idx]),
                    "n.cells.sj.expr": int(sj_expr[coord_idx, donor_idx]),
                    "gene.counts.total": float(gene_sums[coord_idx, donor_idx]),
                    "n.cells.gene.expr": int(gene_expr[coord_idx, donor_idx]),
                    "psi": float(psi[coord_idx, donor_idx]) if not np.isnan(psi[coord_idx, donor_idx]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _droplet_compare_values_sj_donor_level_inplace(
    marvel_object: Marvel10x,
    *,
    cell_group_list: dict[str, dict[str, list[str]]],
    coord_introns: list[str] | None = None,
    min_pct_cells_gene_expr: float = 10,
    min_n_cells_gene_expr: float = 10,
    min_gene_counts_total: float = 3,
    min_sj_count: float = 1,
    min_donor_gene_expr: int = 3,
    min_donor_sj_expr: int = 3,
) -> pd.DataFrame:
    from .matrix import _normalize_donor_cell_group_list
    from .utils import ordered_intersection
    normalized_groups = _normalize_donor_cell_group_list(cell_group_list)
    if len(normalized_groups) != 2:
        raise ValueError("cell_group_list must contain exactly two cell groups")
    group1_name, group2_name = list(normalized_groups)
    group1_donors = normalized_groups[group1_name]
    group2_donors = normalized_groups[group2_name]
    group1_cells = [cell_id for donor_cells in group1_donors.values() for cell_id in donor_cells]
    group2_cells = [cell_id for donor_cells in group2_donors.values() for cell_id in donor_cells]
    gene_expr_g1 = marvel_object._expression_rate_table(marvel_object.gene_norm_matrix, group1_cells, "gene_short_name", group1_name)
    gene_expr_g2 = marvel_object._expression_rate_table(marvel_object.gene_norm_matrix, group2_cells, "gene_short_name", group2_name)
    genes_1 = gene_expr_g1.loc[(gene_expr_g1["pct.cells.expr"] >= float(min_pct_cells_gene_expr)) & ((gene_expr_g1["pct.cells.expr"] * len(group1_cells) / 100.0) >= float(min_n_cells_gene_expr)), "gene_short_name"].tolist()
    genes_2 = gene_expr_g2.loc[(gene_expr_g2["pct.cells.expr"] >= float(min_pct_cells_gene_expr)) & ((gene_expr_g2["pct.cells.expr"] * len(group2_cells) / 100.0) >= float(min_n_cells_gene_expr)), "gene_short_name"].tolist()
    candidate_genes = ordered_intersection(genes_1, genes_2)
    gene_count_subset = marvel_object.gene_count_matrix.subset_rows(candidate_genes)
    total_gene_counts = np.asarray(gene_count_subset.matrix.sum(axis=1)).ravel()
    candidate_genes = pd.DataFrame({"gene_short_name": candidate_genes, "gene.counts.total": total_gene_counts}).loc[lambda df: df["gene.counts.total"] >= float(min_gene_counts_total), "gene_short_name"].tolist()
    sj_metadata = marvel_object.sj_metadata[marvel_object.sj_metadata["gene_short_name.start"].isin(candidate_genes)].copy()
    sj_metadata["coord.intron"] = sj_metadata["coord.intron"].astype(str)
    if coord_introns is not None:
        requested_coord_introns = set(dict.fromkeys(str(coord) for coord in coord_introns))
        sj_metadata = sj_metadata[sj_metadata["coord.intron"].isin(requested_coord_introns)].copy()
    coord_list = sj_metadata["coord.intron"].tolist()
    gene_list = sj_metadata["gene_short_name.start"].astype(str).tolist()
    donor_summary = _summarize_donor_sparse_counts(
        sj_count_matrix=marvel_object.sj_count_matrix,
        gene_count_matrix=marvel_object.gene_count_matrix,
        coord_introns=coord_list,
        gene_short_names=gene_list,
        donor_maps={group1_name: group1_donors, group2_name: group2_donors},
    )
    records = []
    for coord_intron, gene_short_name in zip(coord_list, gene_list):
        coord_summary = donor_summary[donor_summary["coord.intron"] == coord_intron]
        donor_psi_by_group = {group1_name: [], group2_name: []}
        for _, row in coord_summary.iterrows():
            group_name = str(row["cell.group"])
            gene_sum = float(row["gene.counts.total"])
            sj_sum = float(row["sj.counts.total"])
            psi = float(row["psi"]) if not pd.isna(row["psi"]) else np.nan
            if gene_sum >= float(min_gene_counts_total) and int(row["n.cells.gene.expr"]) >= int(min_donor_gene_expr) and sj_sum >= float(min_sj_count) and not np.isnan(psi):
                donor_psi_by_group[group_name].append(round(float(psi), 2))
        psi_g1 = donor_psi_by_group[group1_name]
        psi_g2 = donor_psi_by_group[group2_name]
        if not psi_g1 or not psi_g2:
            continue
        try:
            pval = float(mannwhitneyu(psi_g1, psi_g2, alternative="two-sided").pvalue)
        except ValueError:
            pval = 1.0
        mean_g1 = float(np.mean(psi_g1))
        mean_g2 = float(np.mean(psi_g2))
        records.append({"coord.intron": coord_intron, "gene_short_name": gene_short_name, "mean.psi.g1": round(mean_g1, 3), "mean.psi.g2": round(mean_g2, 3), "log2fc": float(np.log2(mean_g2 / mean_g1)) if mean_g1 > 0.0 and mean_g2 > 0.0 else np.nan, "delta": round(mean_g2 - mean_g1, 3), "pval": pval})
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("pval").reset_index(drop=True)
    marvel_object.de_sj_donor = result
    marvel_object.de_sj_donor_cell_group_list = normalized_groups
    return result


def _droplet_compare_values_genes_inplace(
    marvel_object: Marvel10x,
    *,
    log2_transform: bool = True,
    method: str = "wilcox",
) -> pd.DataFrame:
    if marvel_object.de_sj is None:
        raise ValueError("compare_values_sj must run before compare_values_genes")
    if method != "wilcox":
        raise ValueError("Only wilcox is implemented in the Python version")
    genes = pd.Index(marvel_object.de_sj["gene_short_name"]).unique().tolist()
    group1 = marvel_object.de_sj_cell_group_g1 or marvel_object.de_sj.attrs.get("cell.group.g1")
    group2 = marvel_object.de_sj_cell_group_g2 or marvel_object.de_sj.attrs.get("cell.group.g2")
    if group1 is None or group2 is None:
        raise ValueError("Missing cell group metadata on de_sj table")
    gene_norm_g1_raw = marvel_object.gene_norm_matrix.subset_rows(genes).subset_cols(group1).matrix.tocsr()
    gene_norm_g2_raw = marvel_object.gene_norm_matrix.subset_rows(genes).subset_cols(group2).matrix.tocsr()
    raw_mean_g1 = np.asarray(gene_norm_g1_raw.sum(axis=1)).ravel() / len(group1)
    raw_mean_g2 = np.asarray(gene_norm_g2_raw.sum(axis=1)).ravel() / len(group2)
    gene_norm_g1 = gene_norm_g1_raw
    gene_norm_g2 = gene_norm_g2_raw
    if log2_transform:
        gene_norm_g1 = gene_norm_g1.copy()
        gene_norm_g2 = gene_norm_g2.copy()
        gene_norm_g1.data = np.log2(gene_norm_g1.data + 1.0)
        gene_norm_g2.data = np.log2(gene_norm_g2.data + 1.0)
    n_g1 = len(group1)
    n_g2 = len(group2)
    n_expr_g1 = np.asarray(gene_norm_g1.getnnz(axis=1)).ravel()
    n_expr_g2 = np.asarray(gene_norm_g2.getnnz(axis=1)).ravel()
    mean_g1 = np.asarray(gene_norm_g1.sum(axis=1)).ravel() / n_g1
    mean_g2 = np.asarray(gene_norm_g2.sum(axis=1)).ravel() / n_g2
    pvals = []
    for i in range(len(genes)):
        values_g1 = gene_norm_g1.getrow(i).toarray().ravel()
        values_g2 = gene_norm_g2.getrow(i).toarray().ravel()
        try:
            pvals.append(mannwhitneyu(values_g1, values_g2, alternative="two-sided").pvalue)
        except ValueError:
            pvals.append(1.0)
    results = pd.DataFrame(
        {
            "gene_short_name": genes,
            "n.cells.total.norm.g1": n_g1,
            "n.cells.expr.gene.norm.g1": n_expr_g1,
            "pct.cells.expr.gene.norm.g1": np.round(n_expr_g1 / n_g1 * 100.0, 2),
            "mean.expr.gene.norm.g1": mean_g1,
            "n.cells.total.norm.g2": n_g2,
            "n.cells.expr.gene.norm.g2": n_expr_g2,
            "pct.cells.expr.gene.norm.g2": np.round(n_expr_g2 / n_g2 * 100.0, 2),
            "mean.expr.gene.norm.g2": mean_g2,
            "log2fc.gene.norm": np.log2((raw_mean_g2 + 1.0) / (raw_mean_g1 + 1.0)),
            "diff.mean.log2.gene.norm": mean_g2 - mean_g1 if log2_transform else np.nan,
            "pval.gene.norm": pvals,
        }
    )
    results["pval.adj.gene.norm"] = multipletests(results["pval.gene.norm"], method="fdr_bh")[1]
    marvel_object.de_gene = results
    marvel_object.de_sj = marvel_object.de_sj.merge(results, on="gene_short_name", how="left")
    marvel_object.de_sj.attrs["cell.group.g1"] = group1
    marvel_object.de_sj.attrs["cell.group.g2"] = group2
    marvel_object.de_sj_cell_group_g1 = list(group1)
    marvel_object.de_sj_cell_group_g2 = list(group2)
    return results


def compare_values(
    marvel_object: MarvelPlate,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    level: str,
    method,
    method_adjust: str = "fdr",
    min_cells: int,
    event_type: str | list[str] | None = None,
    show_progress: bool = False,
) -> MarvelPlate:
    _ = show_progress
    method_adjust = _normalize_method_adjust(method_adjust)

    if level == "gene":
        method_name = _normalize_methods(method, level=level)
        marvel_object.compare_values_genes(
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            min_cells=min_cells,
            method=str(method_name),
            method_adjust=method_adjust,
        )
        return marvel_object

    if level == "splicing":
        methods = _normalize_methods(method, level=level)
        if event_type is None:
            raise ValueError("event_type must be non-empty for level='splicing'")
        event_types = _normalize_event_types(event_type)
        for method_name in methods:
            marvel_object.compare_values_splicing(
                cell_group_g1=cell_group_g1,
                cell_group_g2=cell_group_g2,
                method=str(method_name),
                method_adjust=method_adjust,
                min_cells=min_cells,
                event_types=event_types,
            )
        return marvel_object

    raise ValueError(f"Unsupported compare_values level: {level}")


def compare_values_sj_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    coord_introns: list[str] | None = None,
    min_pct_cells_genes: float,
    min_pct_cells_sj: float,
    min_gene_norm: float,
    seed: int = 1,
    n_iterations: int = 100,
    downsample: bool = False,
    show_progress: bool = False,
    permutation_cell_ids: list[list[str]] | None = None,
    bounded_pval: bool = True,
) -> Marvel10x:
    _ = show_progress
    marvel_object.compare_values_sj(
        coord_introns=coord_introns,
        cell_group_g1=cell_group_g1,
        cell_group_g2=cell_group_g2,
        min_pct_cells_genes=min_pct_cells_genes,
        min_pct_cells_sj=min_pct_cells_sj,
        min_gene_norm=min_gene_norm,
        seed=seed,
        n_iterations=n_iterations,
        downsample=downsample,
        permutation_cell_ids=permutation_cell_ids,
        bounded_pval=bounded_pval,
    )
    return marvel_object


def compare_values_genes_10x(
    marvel_object: Marvel10x,
    *,
    log2_transform: bool = True,
    show_progress: bool = False,
    method: str = "wilcox",
    mast_method: str = "bayesglm",
    mast_ebayes: bool = True,
) -> Marvel10x:
    _ = show_progress
    _ = mast_method
    _ = mast_ebayes
    if method != "wilcox":
        raise NotImplementedError("Python compare_values_genes_10x currently supports only method='wilcox'")
    marvel_object.compare_values_genes(log2_transform=log2_transform, method=method)
    return marvel_object


def compare_values_sj_donor_level_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list: dict[str, dict[str, list[str]]],
    coord_introns: list[str] | None = None,
    min_pct_cells_gene_expr: float = 10,
    min_n_cells_gene_expr: float = 10,
    min_gene_counts_total: float = 3,
    min_sj_count: float = 1,
    min_donor_gene_expr: int = 3,
    min_donor_sj_expr: int = 3,
    show_progress: bool = False,
) -> Marvel10x:
    _ = show_progress
    marvel_object.compare_values_sj_donor_level(
        cell_group_list=cell_group_list,
        coord_introns=coord_introns,
        min_pct_cells_gene_expr=min_pct_cells_gene_expr,
        min_n_cells_gene_expr=min_n_cells_gene_expr,
        min_gene_counts_total=min_gene_counts_total,
        min_sj_count=min_sj_count,
        min_donor_gene_expr=min_donor_gene_expr,
        min_donor_sj_expr=min_donor_sj_expr,
    )
    return marvel_object
