from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .adhoc import adhoc_gene_plot_de_values_10x, adhoc_gene_plot_sj_position_10x
from .matrix import Marvel10x
from .iso import iso_switch_plot_expr, label_droplet_sj
from .modality import modality_change, prop_modality_bar, prop_modality_doughnut
from .models import MarvelPlate
from .psi import PLATE_EVENT_TYPES

__all__ = [
    "adhoc_gene_plot_de_values_10x",
    "adhoc_gene_plot_sj_position_10x",
    "iso_switch_plot_expr",
    "modality_change",
    "plot_de_values",
    "plot_de_values_genes_10x",
    "plot_de_values_sj_10x",
    "plot_pct_expr_cells_genes_10x",
    "plot_pct_expr_cells_sj_10x",
    "plot_values",
    "plot_values_gene_pseudobulk_10x",
    "plot_values_gene_single_cell_10x",
    "plot_values_pca_cell_group_10x",
    "plot_values_pca_gene_10x",
    "plot_values_pca_psi_10x",
    "plot_values_psi_pseudobulk_10x",
    "plot_values_psi_pseudobulk_heatmap_10x",
    "prop_modality_bar",
    "prop_modality_doughnut",
    "run_pca",
]


def _subset_gene_matrix(marvel_object: MarvelPlate, features: list[str]) -> pd.DataFrame:
    gene_ids = [str(feature) for feature in features]
    exp = marvel_object.exp.copy()
    exp["gene_id"] = exp["gene_id"].astype(str)
    matched = exp[exp["gene_id"].isin(gene_ids)].copy()
    if matched.empty:
        raise ValueError("No requested gene features were found in MarvelObject.exp")
    matched = matched.drop_duplicates("gene_id").set_index("gene_id")
    return matched.loc[[gene_id for gene_id in gene_ids if gene_id in matched.index]]


def _subset_splicing_matrix(marvel_object: MarvelPlate, features: list[str], sample_ids: list[str], min_cells: int) -> pd.DataFrame:
    tran_ids = [str(feature) for feature in features]
    rows = []
    seen = set()
    for event_type in PLATE_EVENT_TYPES:
        psi = marvel_object.psi.get(event_type)
        if psi is None or psi.empty:
            continue
        missing_samples = [sample_id for sample_id in sample_ids if sample_id not in psi.columns]
        if missing_samples:
            continue
        table = psi.copy()
        table["tran_id"] = table["tran_id"].astype(str)
        matched = table[table["tran_id"].isin(tran_ids)].drop_duplicates("tran_id")
        for _, row in matched.iterrows():
            tran_id = str(row["tran_id"])
            if tran_id in seen:
                continue
            values = pd.to_numeric(row[sample_ids], errors="coerce")
            if int(values.notna().sum()) >= int(min_cells):
                rows.append(pd.Series(values.to_numpy(dtype=float), index=sample_ids, name=tran_id))
                seen.add(tran_id)
    if not rows:
        raise ValueError("No requested splicing features were found with enough non-missing PSI values")
    matrix = pd.DataFrame(rows)
    retained = [tran_id for tran_id in tran_ids if tran_id in matrix.index]
    return matrix.loc[retained]


def _impute_splicing_matrix(
    matrix: pd.DataFrame,
    *,
    sample_metadata: pd.DataFrame,
    cell_group_column: str,
    method_impute: str,
    seed: int,
) -> pd.DataFrame:
    method_key = str(method_impute).lower()
    imputed = matrix.copy()
    if method_key == "random":
        max_value = float(np.nanmax(imputed.to_numpy(dtype=float))) if imputed.notna().any().any() else 1.0
        upper = 100.0 if max_value > 1.0 else 1.0
        rng = np.random.default_rng(seed)
        mask = imputed.isna().to_numpy()
        if mask.any():
            random_values = rng.uniform(0.0, upper, size=mask.sum())
            values = imputed.to_numpy(dtype=float, copy=True)
            values[mask] = random_values
            imputed = pd.DataFrame(values, index=imputed.index, columns=imputed.columns)
    elif method_key == "population.mean":
        group_by_sample = sample_metadata.set_index("sample.id")[cell_group_column]
        for group_name, group_samples in group_by_sample.groupby(group_by_sample).groups.items():
            _ = group_name
            columns = [sample_id for sample_id in group_samples if sample_id in imputed.columns]
            if not columns:
                continue
            means = imputed[columns].mean(axis=1, skipna=True)
            imputed.loc[:, columns] = imputed[columns].T.fillna(means).T
        imputed = imputed.T.fillna(imputed.mean(axis=1, skipna=True)).T
    else:
        raise ValueError("method_impute must be either 'random' or 'population.mean'")
    return imputed.fillna(0.0)


def _run_gene_pca(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [str(column) for column in matrix.columns]
    transposed = matrix.to_numpy(dtype=float).T
    scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(transposed)
    n_components = max(1, min(scaled.shape[0], scaled.shape[1]))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    coord_df = pd.DataFrame(
        {
            "sample.id": sample_ids,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1] if coords.shape[1] >= 2 else 0.0,
        }
    )
    explained_df = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return coord_df, explained_df


def label_de(
    df: pd.DataFrame,
    *,
    pval: float,
    log2fc: float | None,
    delta: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.copy()
    labeled["sig"] = "n.s."
    if delta is not None and "mean.diff" in labeled.columns:
        labeled.loc[(labeled["p.val.adj"] < pval) & (labeled["mean.diff"] > delta), "sig"] = "up"
        labeled.loc[(labeled["p.val.adj"] < pval) & (labeled["mean.diff"] < -delta), "sig"] = "down"
    if log2fc is not None and "log2fc" in labeled.columns:
        labeled.loc[(labeled["p.val.adj"] < pval) & (labeled["log2fc"] > log2fc), "sig"] = "up"
        labeled.loc[(labeled["p.val.adj"] < pval) & (labeled["log2fc"] < -log2fc), "sig"] = "down"
    summary = labeled["sig"].value_counts(dropna=False).rename_axis("sig").reset_index(name="freq")
    return labeled, summary


def run_pca(
    marvel_object: MarvelPlate,
    *,
    cell_group_column: str,
    features: list[str],
    level: str,
    min_cells: int,
    pcs: tuple[int, int] = (1, 2),
    mode: str = "pca",
    method_impute: str = "random",
    seed: int = 1,
) -> MarvelPlate:
    _ = pcs
    if cell_group_column not in marvel_object.splice_pheno.columns:
        raise KeyError(f"Missing cell_group_column in splice_pheno: {cell_group_column}")
    if mode != "pca":
        raise NotImplementedError("Phase 3 currently supports only mode='pca'")

    sample_metadata = marvel_object.splice_pheno.copy()
    sample_metadata["sample.id"] = sample_metadata["sample.id"].astype(str)
    if level == "gene":
        matrix = _subset_gene_matrix(marvel_object, features)
    elif level == "splicing":
        sample_ids = sample_metadata["sample.id"].tolist()
        matrix = _subset_splicing_matrix(marvel_object, features, sample_ids, min_cells)
        matrix = _impute_splicing_matrix(
            matrix,
            sample_metadata=sample_metadata,
            cell_group_column=cell_group_column,
            method_impute=method_impute,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported run_pca level: {level}")

    coord_df, explained_df = _run_gene_pca(matrix)
    sample_metadata = sample_metadata.set_index("sample.id").loc[coord_df["sample.id"]].reset_index()
    coords = sample_metadata[["sample.id", cell_group_column]].copy()
    coords["PC1"] = coord_df["PC1"].to_numpy()
    coords["PC2"] = coord_df["PC2"].to_numpy()

    marvel_object.pca_results[level] = {
        "coords": coords,
        "explained_variance": explained_df,
        "features": list(matrix.index.astype(str)),
        "mode": mode,
        "plot": None,
    }
    if level == "splicing":
        marvel_object.pca_results[level]["method_impute"] = method_impute
    return marvel_object


def _build_gene_value_table(
    marvel_object: MarvelPlate,
    *,
    cell_group_list: dict[str, list[str]],
    feature: str,
) -> pd.DataFrame:
    exp = marvel_object.exp.copy()
    exp["gene_id"] = exp["gene_id"].astype(str)
    match = exp[exp["gene_id"] == str(feature)]
    if match.empty:
        raise ValueError(f"Feature not found in exp table: {feature}")

    row = match.iloc[0]
    records: list[dict[str, object]] = []
    for group_name, sample_ids in cell_group_list.items():
        for sample_id in sample_ids:
            sample_id = str(sample_id)
            if sample_id not in row.index:
                raise KeyError(f"Missing sample id in exp table: {sample_id}")
            records.append(
                {
                    "cell_group": str(group_name),
                    "sample_id": sample_id,
                    "feature": str(feature),
                    "value": float(row[sample_id]),
                }
            )
    return pd.DataFrame(records)


def plot_values(
    marvel_object: MarvelPlate,
    *,
    cell_group_list: dict[str, list[str]],
    feature: str,
    level: str,
) -> MarvelPlate:
    if level != "gene":
        raise NotImplementedError("Phase 3 Task 3 currently supports only level='gene'")

    table = _build_gene_value_table(
        marvel_object,
        cell_group_list=cell_group_list,
        feature=feature,
    )
    marvel_object.value_plots[level] = {
        "table": table,
        "plot": None,
        "level": level,
        "feature": str(feature),
    }
    return marvel_object


def _label_splicing_table(
    df: pd.DataFrame,
    *,
    pval: float,
    delta: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_delta = 0.0 if delta is None else float(delta)
    return label_de(df, pval=pval, log2fc=None, delta=threshold_delta)


def plot_de_values(
    marvel_object: MarvelPlate,
    *,
    level: str,
    method: str | None = None,
    pval: float,
    log2fc: float | None = None,
    delta: float | None = None,
    point_size: float = 0.5,
    xlabel_size: float | None = None,
    anno: bool = False,
    anno_gene_short_name: list[str] | None = None,
) -> MarvelPlate:
    _ = point_size, xlabel_size, anno, anno_gene_short_name
    level_key = str(level)

    if level_key == "gene.global":
        if marvel_object.de_gene is None:
            raise ValueError("compare_values(level='gene') must run before plot_de_values(level='gene.global')")
        labeled, summary = label_de(marvel_object.de_gene, pval=pval, log2fc=log2fc, delta=None)
    elif level_key == "gene.spliced":
        if marvel_object.de_spliced_gene is None:
            raise ValueError("compare_values(level='gene') must run before plot_de_values(level='gene.spliced')")
        labeled, summary = label_de(marvel_object.de_spliced_gene, pval=pval, log2fc=log2fc, delta=None)
    elif level_key in {"splicing", "splicing.mean", "splicing.distance", "splicing.mean.g2vsg1"}:
        if method is None:
            raise ValueError(f"plot_de_values(level={level!r}) requires method")
        method_key = str(method).lower()
        if method_key not in marvel_object.de_splicing:
            raise ValueError(f"Missing splicing DE results for method={method}")
        labeled, summary = _label_splicing_table(marvel_object.de_splicing[method_key], pval=pval, delta=delta)
        marvel_object.de_plots.setdefault("splicing_methods", {})[method_key] = {
            "table": labeled,
            "summary": summary,
            "plot": None,
        }
    else:
        raise ValueError(f"Unsupported plot_de_values level: {level}")

    marvel_object.de_plots[level_key] = {
        "table": labeled,
        "summary": summary,
        "plot": None,
    }
    if level_key == "splicing":
        marvel_object.de_plots["splicing_methods"][method_key] = marvel_object.de_plots["splicing"]
    return marvel_object


def _label_droplet_genes(
    df: pd.DataFrame,
    *,
    pval: float,
    log2fc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.copy()
    labeled["sig"] = "n.s."
    labeled.loc[
        (labeled["pval.adj.gene.norm"] < pval) & (labeled["log2fc.gene.norm"] > log2fc),
        "sig",
    ] = "up"
    labeled.loc[
        (labeled["pval.adj.gene.norm"] < pval) & (labeled["log2fc.gene.norm"] < -log2fc),
        "sig",
    ] = "down"
    if {"mean.expr.gene.norm.g1", "mean.expr.gene.norm.g2"}.issubset(labeled.columns):
        labeled["mean.expr.gene.norm.g1.g2"] = (
            labeled["mean.expr.gene.norm.g1"] + labeled["mean.expr.gene.norm.g2"]
        ) / 2.0
    else:
        labeled["mean.expr.gene.norm.g1.g2"] = pd.NA
    for column in [
        "n.cells.expr.gene.norm.g1",
        "pct.cells.expr.gene.norm.g1",
        "mean.expr.gene.norm.g1",
        "n.cells.expr.gene.norm.g2",
        "pct.cells.expr.gene.norm.g2",
        "mean.expr.gene.norm.g2",
        "diff.mean.log2.gene.norm",
        "pval.gene.norm",
    ]:
        if column not in labeled.columns:
            labeled[column] = pd.NA
    labeled["label"] = pd.NA
    labeled = labeled[
        [
            "gene_short_name",
            "mean.expr.gene.norm.g1.g2",
            "n.cells.expr.gene.norm.g1",
            "pct.cells.expr.gene.norm.g1",
            "mean.expr.gene.norm.g1",
            "n.cells.expr.gene.norm.g2",
            "pct.cells.expr.gene.norm.g2",
            "mean.expr.gene.norm.g2",
            "log2fc.gene.norm",
            "diff.mean.log2.gene.norm",
            "pval.gene.norm",
            "pval.adj.gene.norm",
            "sig",
            "label",
        ]
    ].copy()
    summary = labeled["sig"].value_counts(dropna=False).rename_axis("sig").reset_index(name="freq")
    return labeled, summary


def plot_de_values_sj_10x(
    marvel_object: Marvel10x,
    *,
    pval: float,
    delta: float | None,
    min_gene_norm: float,
    log2fc: float | None = None,
    anno: bool = False,
) -> Marvel10x:
    _ = anno, log2fc
    if marvel_object.de_sj is None:
        raise ValueError("compare_values_sj_10x or compare_values_sj must run before plot_de_values_sj_10x")
    labeled, summary = label_droplet_sj(
        marvel_object.de_sj,
        pval=pval,
        delta=0.0 if delta is None else delta,
        min_gene_norm=min_gene_norm,
    )
    marvel_object.de_plots["sj"] = {"table": labeled, "summary": summary, "plot": None}
    return marvel_object


def plot_de_values_genes_10x(
    marvel_object: Marvel10x,
    *,
    pval: float,
    log2fc: float,
) -> Marvel10x:
    if marvel_object.de_gene is None:
        raise ValueError("compare_values_genes_10x or compare_values_genes must run before plot_de_values_genes_10x")
    labeled, summary = _label_droplet_genes(
        marvel_object.de_gene,
        pval=pval,
        log2fc=log2fc,
    )
    marvel_object.de_plots["genes"] = {"table": labeled, "summary": summary, "plot": None}
    return marvel_object


def plot_pct_expr_cells_genes_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    min_pct_cells: float,
) -> Marvel10x:
    table = marvel_object.plot_pct_expr_cells_genes(
        cell_group_g1=cell_group_g1,
        cell_group_g2=cell_group_g2,
        min_pct_cells=min_pct_cells,
    )
    marvel_object.value_plots["pct_expr_genes"] = {"table": table, "plot": None}
    return marvel_object


def plot_pct_expr_cells_sj_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_g1: list[str],
    cell_group_g2: list[str],
    min_pct_cells_genes: float,
    min_pct_cells_sj: float,
    downsample: bool = True,
    downsample_pct_sj: float = 10.0,
    seed: int = 1,
    downsample_coord_introns: list[str] | None = None,
) -> Marvel10x:
    table = marvel_object.plot_pct_expr_cells_sj(
        cell_group_g1=cell_group_g1,
        cell_group_g2=cell_group_g2,
        min_pct_cells_genes=min_pct_cells_genes,
        min_pct_cells_sj=min_pct_cells_sj,
        downsample=downsample,
        downsample_pct_sj=downsample_pct_sj,
        seed=seed,
        downsample_coord_introns=downsample_coord_introns,
    )
    marvel_object.value_plots["pct_expr_sj"] = {"table": table, "plot": None}
    return marvel_object


def _copy_pca_table(marvel_object: Marvel10x, *, value_column: str, values: pd.Series) -> pd.DataFrame:
    pca = marvel_object.pca.copy()
    cell_col = "cell.id" if "cell.id" in pca.columns else pca.columns[0]
    pca[cell_col] = pca[cell_col].astype(str)
    table = pca.copy()
    table[value_column] = table[cell_col].map(values.to_dict())
    return table


def plot_values_pca_cell_group_10x(
    marvel_object: Marvel10x,
    *,
    cell_ids: list[str] | None = None,
    cell_group_column: str = "cell.type",
    type: str,
) -> Marvel10x:
    _ = type
    table = marvel_object.pca.copy()
    cell_col = "cell.id" if "cell.id" in table.columns else table.columns[0]
    if cell_ids is not None:
        allowed = {str(cell_id) for cell_id in cell_ids}
        table = table[table[cell_col].astype(str).isin(allowed)].copy()
    table["group"] = table[cell_col].astype(str).map(
        marvel_object.sample_metadata.set_index("cell.id")[cell_group_column].astype(str).to_dict()
    )
    marvel_object.value_plots["pca_cell_group"] = {"table": table, "plot": None}
    return marvel_object


def plot_values_pca_gene_10x(
    marvel_object: Marvel10x,
    *,
    gene_short_name: str,
    cell_ids: list[str] | None = None,
    log2_transform: bool = True,
    type: str,
) -> Marvel10x:
    _ = type
    matrix = marvel_object.gene_norm_matrix.subset_rows([gene_short_name])
    values = pd.Series(matrix.matrix.toarray().ravel(), index=matrix.col_ids.astype(str))
    if log2_transform:
        values = np.log2(values + 1.0)
    table = _copy_pca_table(marvel_object, value_column="gene_value", values=values)
    cell_col = "cell.id" if "cell.id" in table.columns else table.columns[0]
    if cell_ids is not None:
        allowed = {str(cell_id) for cell_id in cell_ids}
        table = table[table[cell_col].astype(str).isin(allowed)].copy()
    marvel_object.value_plots["pca_gene"] = {"table": table, "plot": None, "gene_short_name": gene_short_name}
    return marvel_object


def plot_values_pca_psi_10x(
    marvel_object: Marvel10x,
    *,
    coord_intron: str,
    cell_ids: list[str] | None = None,
    min_gene_count: float = 3,
    log2_transform: bool = False,
    type: str,
) -> Marvel10x:
    _ = type
    sj_counts = marvel_object.sj_count_matrix.subset_rows([coord_intron])
    gene_name = marvel_object.sj_metadata.set_index("coord.intron").loc[coord_intron, "gene_short_name.start"]
    gene_counts = marvel_object.gene_count_matrix.subset_rows([gene_name])
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (sj_counts.matrix.toarray().ravel() / gene_counts.matrix.toarray().ravel()) * 100.0
    psi = pd.Series(psi, index=sj_counts.col_ids.astype(str))
    gene_expr = pd.Series(gene_counts.matrix.toarray().ravel(), index=gene_counts.col_ids.astype(str))
    psi = psi.where(gene_expr >= min_gene_count)
    if log2_transform:
        psi = np.log2(psi + 1.0)
    table = _copy_pca_table(marvel_object, value_column="psi", values=psi)
    cell_col = "cell.id" if "cell.id" in table.columns else table.columns[0]
    if cell_ids is not None:
        allowed = {str(cell_id) for cell_id in cell_ids}
        table = table[table[cell_col].astype(str).isin(allowed)].copy()
    marvel_object.value_plots["pca_psi"] = {"table": table, "plot": None, "coord_intron": coord_intron}
    return marvel_object


def plot_values_gene_single_cell_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list: dict[str, list[str]],
    gene_short_name: str,
    log2_transform: bool = True,
) -> Marvel10x:
    matrix = marvel_object.gene_norm_matrix.subset_rows([gene_short_name])
    values = pd.Series(matrix.matrix.toarray().ravel(), index=matrix.col_ids.astype(str))
    if log2_transform:
        values = np.log2(values + 1.0)
    rows = []
    for group_name, cell_ids in cell_group_list.items():
        for cell_id in cell_ids:
            rows.append({"cell.group": group_name, "cell.id": cell_id, "exp": float(values[str(cell_id)])})
    marvel_object.value_plots["gene_single_cell"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def _normalize_pseudobulk_cell_groups(
    cell_group_list: Mapping[str, object],
) -> list[tuple[str, str, list[str]]]:
    records: list[tuple[str, str, list[str]]] = []
    for group_name, samples in cell_group_list.items():
        if isinstance(samples, Mapping):
            sample_iter = samples.items()
        else:
            sample_iter = [(group_name, samples)]
        for sample_id, cell_ids in sample_iter:
            if isinstance(cell_ids, str):
                raise TypeError("Pseudobulk cell groups must contain iterables of cell IDs, not a single string")
            records.append((str(group_name), str(sample_id), [str(cell_id) for cell_id in cell_ids]))
    return records


def _normalize_pseudobulk_cell_group_map(cell_group_list: Mapping[str, object]) -> dict[str, dict[str, list[str]]]:
    sample_map: dict[str, dict[str, list[str]]] = {}
    for group_name, sample_id, cell_ids in _normalize_pseudobulk_cell_groups(cell_group_list):
        sample_map.setdefault(group_name, {})[sample_id] = cell_ids
    return sample_map


def plot_values_gene_pseudobulk_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list: Mapping[str, object],
    gene_short_name: str,
    log2_transform: bool = True,
) -> Marvel10x:
    matrix = marvel_object.gene_norm_matrix.subset_rows([gene_short_name])
    sample_records = _normalize_pseudobulk_cell_groups(cell_group_list)
    all_cells: list[str] = []
    for _, _, cell_ids in sample_records:
        all_cells.extend(cell_ids)
    all_cells = list(dict.fromkeys(all_cells))
    matrix = matrix.subset_cols(all_cells)
    cell_index = {cell_id: idx for idx, cell_id in enumerate(all_cells)}
    sample_mask = np.zeros((len(all_cells), len(sample_records)), dtype=float)
    sample_sizes = np.zeros(len(sample_records), dtype=int)
    for sample_idx, (_, _, cell_ids) in enumerate(sample_records):
        sample_sizes[sample_idx] = len(cell_ids)
        for cell_id in cell_ids:
            sample_mask[cell_index[cell_id], sample_idx] = 1.0
    sample_sums = np.asarray(matrix.matrix @ sample_mask).ravel()
    rows = []
    for sample_idx, (group_name, sample_id, _) in enumerate(sample_records):
        mean_expr = float(sample_sums[sample_idx] / sample_sizes[sample_idx]) if sample_sizes[sample_idx] else float("nan")
        if log2_transform:
            mean_expr = float(np.log2(mean_expr + 1.0))
        rows.append(
            {
                "cell.group": group_name,
                "sample.id": sample_id,
                "n.cells.total": int(sample_sizes[sample_idx]),
                "mean.expr.gene.norm": mean_expr,
            }
        )
    marvel_object.value_plots["gene_pseudobulk"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def plot_values_psi_pseudobulk_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list: Mapping[str, object],
    coord_intron: str,
) -> Marvel10x:
    gene_name = marvel_object.sj_metadata.set_index("coord.intron").loc[coord_intron, "gene_short_name.start"]
    from .de import _summarize_donor_sparse_counts

    summary = _summarize_donor_sparse_counts(
        sj_count_matrix=marvel_object.sj_count_matrix,
        gene_count_matrix=marvel_object.gene_count_matrix,
        coord_introns=[coord_intron],
        gene_short_names=[gene_name],
        donor_maps=_normalize_pseudobulk_cell_group_map(cell_group_list),
    )
    summary["pct.cells.sj.expr"] = np.where(
        summary["n.cells.total"] > 0,
        np.round(summary["n.cells.sj.expr"] / summary["n.cells.total"] * 100.0, 2),
        np.nan,
    )
    summary["pct.cells.gene.expr"] = np.where(
        summary["n.cells.total"] > 0,
        np.round(summary["n.cells.gene.expr"] / summary["n.cells.total"] * 100.0, 2),
        np.nan,
    )
    rows = summary[
        [
            "cell.group",
            "sample.id",
            "n.cells.total",
            "sj.counts.total",
            "n.cells.sj.expr",
            "pct.cells.sj.expr",
            "gene.counts.total",
            "n.cells.gene.expr",
            "pct.cells.gene.expr",
            "psi",
        ]
    ].to_dict("records")
    marvel_object.value_plots["psi_pseudobulk"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def plot_values_psi_pseudobulk_heatmap_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list: Mapping[str, object],
    coord_introns: list[str],
) -> Marvel10x:
    from .de import _summarize_donor_sparse_counts

    sj_meta = marvel_object.sj_metadata.set_index("coord.intron")
    gene_names = [sj_meta.loc[coord_intron, "gene_short_name.start"] for coord_intron in coord_introns]
    summary = _summarize_donor_sparse_counts(
        sj_count_matrix=marvel_object.sj_count_matrix,
        gene_count_matrix=marvel_object.gene_count_matrix,
        coord_introns=coord_introns,
        gene_short_names=gene_names,
        donor_maps=_normalize_pseudobulk_cell_group_map(cell_group_list),
    )
    rows = []
    for _, row in summary.iterrows():
        psi = np.nan if pd.isna(row["psi"]) else round(float(row["psi"]), 2)
        rows.append(
            {
                "cell_group": row["cell.group"],
                "donor_id": row["sample.id"],
                "coord_intron": row["coord.intron"],
                "psi": psi,
            }
        )
    marvel_object.value_plots["psi_pseudobulk_heatmap"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object
