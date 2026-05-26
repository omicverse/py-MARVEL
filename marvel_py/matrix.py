from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .utils import extract_gtf_attr, ordered_intersection, read_table

_read_table = read_table
_ordered_intersection = ordered_intersection
_extract_gtf_attr = extract_gtf_attr


def _matrix_cache_paths(matrix_path: str | Path) -> tuple[Path, Path]:
    matrix_path = Path(matrix_path)
    return Path(f"{matrix_path}.csr.npz"), Path(f"{matrix_path}.csr.meta.json")


def _load_cached_sparse_matrix(matrix_path: str | Path) -> sparse.csr_matrix | None:
    matrix_path = Path(matrix_path)
    cache_path, meta_path = _matrix_cache_paths(matrix_path)
    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    stat = matrix_path.stat()
    expected = {
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }
    if meta != expected:
        return None

    matrix = sparse.load_npz(cache_path)
    return matrix.tocsr()


def _write_cached_sparse_matrix(matrix_path: str | Path, matrix: sparse.csr_matrix) -> None:
    matrix_path = Path(matrix_path)
    cache_path, meta_path = _matrix_cache_paths(matrix_path)
    stat = matrix_path.stat()
    meta = {
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }

    tmp_cache = Path(f"{cache_path}.tmp.npz")
    tmp_meta = Path(f"{meta_path}.tmp")
    sparse.save_npz(tmp_cache, matrix, compressed=False)
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp_cache.replace(cache_path)
    tmp_meta.replace(meta_path)


def _normalize_donor_cell_group_list(
    cell_group_list: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    if not isinstance(cell_group_list, dict) or not cell_group_list:
        raise ValueError("cell_group_list must contain at least one cell group")

    normalized: dict[str, dict[str, list[str]]] = {}
    donor_ids_seen: set[str] = set()
    duplicate_donor_ids: set[str] = set()
    for cell_group, donor_map in cell_group_list.items():
        if not isinstance(donor_map, dict) or not donor_map:
            raise ValueError(f"cell_group_list[{cell_group!r}] must map donor ids to non-empty cell lists")
        normalized[cell_group] = {}
        for donor_id, cell_ids in donor_map.items():
            normalized_ids = [str(cell_id) for cell_id in cell_ids]
            if not normalized_ids:
                raise ValueError(f"cell_group_list[{cell_group!r}][{donor_id!r}] must contain at least one cell")
            if donor_id in donor_ids_seen:
                duplicate_donor_ids.add(str(donor_id))
            donor_ids_seen.add(str(donor_id))
            normalized[cell_group][str(donor_id)] = normalized_ids
    if duplicate_donor_ids:
        duplicate_sorted = sorted(duplicate_donor_ids)
        raise ValueError(f"Duplicate donor ids across cell groups: {duplicate_sorted}")
    return normalized


def _load_sparse_matrix(
    matrix_path: str | Path,
    feature_path: str | Path,
    feature_column: str,
    pheno_path: str | Path,
    pheno_column: str,
) -> "NamedMatrix":
    matrix = _load_cached_sparse_matrix(matrix_path)
    if matrix is None:
        matrix = mmread(str(matrix_path))
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
        else:
            matrix = sparse.csr_matrix(np.asarray(matrix))
        _write_cached_sparse_matrix(matrix_path, matrix)

    feature = read_table(feature_path)
    pheno = read_table(pheno_path)
    row_ids = feature[feature_column].astype(str).to_numpy()
    col_ids = pheno[pheno_column].astype(str).to_numpy()

    if matrix.shape != (len(row_ids), len(col_ids)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match feature/pheno lengths "
            f"({len(row_ids)}, {len(col_ids)}) for {matrix_path}"
        )

    return NamedMatrix(matrix=matrix, row_ids=row_ids, col_ids=col_ids)


def _coerce_table(table) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return read_table(table)


def _coerce_gtf(gtf) -> pd.DataFrame:
    if isinstance(gtf, pd.DataFrame):
        gtf_df = gtf.copy()
    else:
        gtf_df = pd.read_csv(
            gtf,
            sep="\t",
            header=None,
            comment="#",
            dtype=str,
            names=[f"V{i}" for i in range(1, 10)],
        )
    expected_cols = [f"V{i}" for i in range(1, 10)]
    if list(gtf_df.columns) != expected_cols and len(gtf_df.columns) == len(expected_cols):
        gtf_df.columns = expected_cols
    return gtf_df


def _coerce_sparse_matrix(matrix) -> sparse.csr_matrix:
    if isinstance(matrix, (str, Path)):
        loaded = mmread(str(matrix))
        if sparse.issparse(loaded):
            return loaded.tocsr()
        return sparse.csr_matrix(np.asarray(loaded))
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def _named_matrix_from_data(
    matrix,
    feature,
    feature_column: str,
    pheno,
    pheno_column: str,
) -> "NamedMatrix":
    sparse_matrix = _coerce_sparse_matrix(matrix)
    feature_df = _coerce_table(feature)
    pheno_df = _coerce_table(pheno)
    row_ids = feature_df[feature_column].astype(str).to_numpy()
    col_ids = pheno_df[pheno_column].astype(str).to_numpy()

    if sparse_matrix.shape != (len(row_ids), len(col_ids)):
        raise ValueError(
            f"Matrix shape {sparse_matrix.shape} does not match feature/pheno lengths "
            f"({len(row_ids)}, {len(col_ids)})"
        )

    return NamedMatrix(matrix=sparse_matrix, row_ids=row_ids, col_ids=col_ids)


@dataclass
class NamedMatrix:
    matrix: sparse.csr_matrix
    row_ids: np.ndarray
    col_ids: np.ndarray
    _row_index_cache: pd.Index | None = field(default=None, init=False, repr=False, compare=False)
    _col_index_cache: pd.Index | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def _row_index(self) -> pd.Index:
        if self._row_index_cache is None:
            self._row_index_cache = pd.Index(self.row_ids)
        return self._row_index_cache

    @property
    def _col_index(self) -> pd.Index:
        if self._col_index_cache is None:
            self._col_index_cache = pd.Index(self.col_ids)
        return self._col_index_cache

    def row_indexer(self, ids: Iterable[str]) -> np.ndarray:
        ids = list(ids)
        positions = self._row_index.get_indexer(ids)
        if (positions < 0).any():
            missing = np.array(ids, dtype=object)[positions < 0]
            raise KeyError(f"Missing row ids: {missing[:5].tolist()}")
        return positions

    def col_indexer(self, ids: Iterable[str]) -> np.ndarray:
        ids = list(ids)
        positions = self._col_index.get_indexer(ids)
        if (positions < 0).any():
            missing = np.array(ids, dtype=object)[positions < 0]
            raise KeyError(f"Missing col ids: {missing[:5].tolist()}")
        return positions

    def subset_rows(self, ids: Iterable[str]) -> "NamedMatrix":
        ids = list(ids)
        idx = self.row_indexer(ids)
        return NamedMatrix(self.matrix[idx, :].tocsr(), np.asarray(ids), self.col_ids.copy())

    def subset_cols(self, ids: Iterable[str]) -> "NamedMatrix":
        ids = list(ids)
        idx = self.col_indexer(ids)
        return NamedMatrix(self.matrix[:, idx].tocsr(), self.row_ids.copy(), np.asarray(ids))


@dataclass
class Marvel10x:
    sample_metadata: pd.DataFrame
    gene_metadata: pd.DataFrame
    gene_norm_matrix: NamedMatrix
    gene_count_matrix: NamedMatrix
    sj_count_matrix: NamedMatrix
    pca: pd.DataFrame
    gtf: pd.DataFrame
    sj_metadata: pd.DataFrame | None = None
    pct_expr_gene: pd.DataFrame | None = None
    pct_expr_sj: pd.DataFrame | None = None
    de_sj: pd.DataFrame | None = None
    de_sj_cell_group_g1: list[str] | None = None
    de_sj_cell_group_g2: list[str] | None = None
    de_sj_donor: pd.DataFrame | None = None
    de_sj_donor_cell_group_list: dict[str, dict[str, list[str]]] | None = None
    de_gene: pd.DataFrame | None = None
    de_plots: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    iso_switch: dict[str, pd.DataFrame] = field(default_factory=dict)
    adhoc_gene: dict[str, object] = field(default_factory=dict)
    value_plots: dict[str, dict[str, pd.DataFrame | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.de_plots = {str(key): value for key, value in self.de_plots.items()}
        self.iso_switch = {str(key): value for key, value in self.iso_switch.items()}
        self.adhoc_gene = {} if self.adhoc_gene is None else dict(self.adhoc_gene)
        self.value_plots = {} if self.value_plots is None else dict(self.value_plots)

    @classmethod
    def from_paths(
        cls,
        *,
        gene_norm_matrix: str | Path,
        gene_norm_pheno: str | Path,
        gene_norm_feature: str | Path,
        gene_count_matrix: str | Path,
        gene_count_pheno: str | Path,
        gene_count_feature: str | Path,
        sj_count_matrix: str | Path,
        sj_count_pheno: str | Path,
        sj_count_feature: str | Path,
        pca: str | Path,
        gtf: str | Path,
    ) -> "Marvel10x":
        sample_metadata = read_table(gene_norm_pheno)
        gene_metadata = read_table(gene_norm_feature)
        pca_df = read_table(pca)
        gtf_df = pd.read_csv(
            gtf,
            sep="\t",
            header=None,
            comment="#",
            dtype=str,
            names=[f"V{i}" for i in range(1, 10)],
        )

        return cls(
            sample_metadata=sample_metadata,
            gene_norm_matrix=_load_sparse_matrix(
                gene_norm_matrix, gene_norm_feature, "gene_short_name", gene_norm_pheno, "cell.id"
            ),
            gene_count_matrix=_load_sparse_matrix(
                gene_count_matrix, gene_count_feature, "gene_short_name", gene_count_pheno, "cell.id"
            ),
            sj_count_matrix=_load_sparse_matrix(
                sj_count_matrix, sj_count_feature, "coord.intron", sj_count_pheno, "cell.id"
            ),
            pca=pca_df,
            gtf=gtf_df,
            gene_metadata=gene_metadata,
        )

    @classmethod
    def from_data(
        cls,
        *,
        gene_norm_matrix,
        gene_norm_pheno,
        gene_norm_feature,
        gene_count_matrix,
        gene_count_pheno,
        gene_count_feature,
        sj_count_matrix,
        sj_count_pheno,
        sj_count_feature,
        pca,
        gtf,
    ) -> "Marvel10x":
        sample_metadata = _coerce_table(gene_norm_pheno)
        gene_metadata = _coerce_table(gene_norm_feature)
        pca_df = _coerce_table(pca)
        gtf_df = _coerce_gtf(gtf)

        return cls(
            sample_metadata=sample_metadata,
            gene_norm_matrix=_named_matrix_from_data(
                gene_norm_matrix, gene_norm_feature, "gene_short_name", gene_norm_pheno, "cell.id"
            ),
            gene_count_matrix=_named_matrix_from_data(
                gene_count_matrix, gene_count_feature, "gene_short_name", gene_count_pheno, "cell.id"
            ),
            sj_count_matrix=_named_matrix_from_data(
                sj_count_matrix, sj_count_feature, "coord.intron", sj_count_pheno, "cell.id"
            ),
            pca=pca_df,
            gtf=gtf_df,
            gene_metadata=gene_metadata,
        )

    def annotate_genes(self) -> "Marvel10x":
        from .annotation import _annotate_genes_10x_inplace

        return _annotate_genes_10x_inplace(self)

    def annotate_sj(self) -> "Marvel10x":
        from .annotation import _annotate_sj_10x_inplace

        return _annotate_sj_10x_inplace(self)

    @staticmethod
    def _classify_sj(start_value: str | float | None, end_value: str | float | None) -> str:
        from .annotation import _classify_sj

        return _classify_sj(start_value, end_value)

    def validate_sj(self, keep_novel_sj: bool = False) -> "Marvel10x":
        from .qc import _validate_sj_10x_inplace

        return _validate_sj_10x_inplace(self, keep_novel_sj=keep_novel_sj)

    def filter_genes(self, gene_type: str = "protein_coding") -> "Marvel10x":
        from .qc import _filter_genes_10x_inplace

        return _filter_genes_10x_inplace(self, gene_type=gene_type)

    def check_alignment(self) -> "Marvel10x":
        from .qc import _check_alignment_10x_inplace

        return _check_alignment_10x_inplace(self)

    def get_cell_groups(self, group_column: str, group1_value: str, group2_value: str) -> tuple[list[str], list[str]]:
        meta = self.sample_metadata.copy()
        meta["cell.id"] = meta["cell.id"].astype(str)
        group1 = meta.loc[meta[group_column] == group1_value, "cell.id"].tolist()
        group2 = meta.loc[meta[group_column] == group2_value, "cell.id"].tolist()
        if not group1 or not group2:
            raise ValueError(f"Empty group detected for {group1_value=} or {group2_value=}")
        return group1, group2

    def plot_pct_expr_cells_genes(
        self, cell_group_g1: list[str], cell_group_g2: list[str], min_pct_cells: float = 1
    ) -> pd.DataFrame:
        df = pd.concat(
            [
                self._expression_rate_table(self.gene_norm_matrix, cell_group_g1, "gene_short_name", "cell.group.g1"),
                self._expression_rate_table(self.gene_norm_matrix, cell_group_g2, "gene_short_name", "cell.group.g2"),
            ],
            ignore_index=True,
        )
        df = df[df["pct.cells.expr"] > min_pct_cells].reset_index(drop=True)
        self.pct_expr_gene = df
        return df

    def plot_pct_expr_cells_sj(
        self,
        cell_group_g1: list[str],
        cell_group_g2: list[str],
        min_pct_cells_genes: float = 10,
        min_pct_cells_sj: float = 10,
        downsample: bool = False,
        downsample_pct_sj: float = 10,
        seed: int = 1,
        downsample_coord_introns: list[str] | None = None,
    ) -> pd.DataFrame:
        gene_results = pd.concat(
            [
                self._expression_rate_table(self.gene_norm_matrix, cell_group_g1, "gene_short_name", "cell.group.g1"),
                self._expression_rate_table(self.gene_norm_matrix, cell_group_g2, "gene_short_name", "cell.group.g2"),
            ],
            ignore_index=True,
        )
        gene_results = gene_results[gene_results["pct.cells.expr"] > min_pct_cells_genes]

        sj_metadata = self.sj_metadata.copy()
        sj_count_matrix = self.sj_count_matrix
        if downsample_coord_introns is not None:
            coord = list(dict.fromkeys(str(coord) for coord in downsample_coord_introns))
            sj_count_matrix = sj_count_matrix.subset_rows(coord)
            sj_metadata = sj_metadata[sj_metadata["coord.intron"].isin(coord)].copy()
        elif downsample:
            rng = np.random.default_rng(seed)
            size = int(round(sj_count_matrix.matrix.shape[0] * (downsample_pct_sj / 100.0)))
            size = max(1, min(size, sj_count_matrix.matrix.shape[0]))
            coord = rng.choice(sj_count_matrix.row_ids, size=size, replace=False).tolist()
            sj_count_matrix = sj_count_matrix.subset_rows(coord)
            sj_metadata = sj_metadata[sj_metadata["coord.intron"].isin(coord)].copy()

        results = []
        for group_name, cell_group in (("cell.group.g1", cell_group_g1), ("cell.group.g2", cell_group_g2)):
            genes = gene_results.loc[gene_results["cell.group"] == group_name, "gene_short_name"].tolist()
            coord = sj_metadata.loc[sj_metadata["gene_short_name.start"].isin(genes), "coord.intron"].tolist()
            table = self._expression_rate_table(
                sj_count_matrix.subset_rows(coord),
                cell_group,
                "coord.intron",
                group_name,
            )
            results.append(table)

        df = pd.concat(results, ignore_index=True)
        df = df[df["pct.cells.expr"] > min_pct_cells_sj].reset_index(drop=True)
        self.pct_expr_sj = df
        return df

    def compare_values_sj(
        self,
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
        from .de import _droplet_compare_values_sj_inplace

        return _droplet_compare_values_sj_inplace(
            self,
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            coord_introns=coord_introns,
            min_pct_cells_genes=min_pct_cells_genes,
            min_pct_cells_sj=min_pct_cells_sj,
            min_gene_norm=min_gene_norm,
            seed=seed,
            n_iterations=n_iterations,
            downsample=downsample,
            permutation_cell_ids=permutation_cell_ids,
            bounded_pval=bounded_pval,
        )

    def compare_values_sj_donor_level(
        self,
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
        from .de import _droplet_compare_values_sj_donor_level_inplace

        return _droplet_compare_values_sj_donor_level_inplace(
            self,
            cell_group_list=cell_group_list,
            coord_introns=coord_introns,
            min_pct_cells_gene_expr=min_pct_cells_gene_expr,
            min_n_cells_gene_expr=min_n_cells_gene_expr,
            min_gene_counts_total=min_gene_counts_total,
            min_sj_count=min_sj_count,
            min_donor_gene_expr=min_donor_gene_expr,
            min_donor_sj_expr=min_donor_sj_expr,
        )

    def compare_values_genes(
        self,
        log2_transform: bool = True,
        method: str = "wilcox",
    ) -> pd.DataFrame:
        from .de import _droplet_compare_values_genes_inplace

        return _droplet_compare_values_genes_inplace(self, log2_transform=log2_transform, method=method)

    def save_outputs(self, output_dir: str | Path, summary: dict) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.gene_metadata.to_csv(output_dir / "preprocessed_gene_metadata.tsv", sep="\t", index=False)
        if self.sj_metadata is not None:
            self.sj_metadata.to_csv(output_dir / "preprocessed_sj_metadata.tsv", sep="\t", index=False)
        if self.pct_expr_gene is not None:
            self.pct_expr_gene.to_csv(output_dir / "pct_expr_gene.tsv", sep="\t", index=False)
        if self.pct_expr_sj is not None:
            self.pct_expr_sj.to_csv(output_dir / "pct_expr_sj.tsv", sep="\t", index=False)
        if self.de_sj is not None:
            self.de_sj.to_csv(output_dir / "de_sj.tsv", sep="\t", index=False)
        if self.de_gene is not None:
            self.de_gene.to_csv(output_dir / "de_gene.tsv", sep="\t", index=False)

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _expression_rate_table(
        self, named_matrix: NamedMatrix, cell_ids: list[str], feature_column: str, group_name: str
    ) -> pd.DataFrame:
        col_idx = named_matrix.col_indexer(cell_ids)
        sub = named_matrix.matrix[:, col_idx]
        n_cells_expr = np.asarray(sub.getnnz(axis=1)).ravel()
        n_cells_total = len(cell_ids)
        return pd.DataFrame(
            {
                "cell.group": group_name,
                feature_column: named_matrix.row_ids,
                "n.cells.total": n_cells_total,
                "n.cells.expr": n_cells_expr,
                "pct.cells.expr": np.round(n_cells_expr / n_cells_total * 100.0, 2),
            }
        )

    def _sj_above_threshold(self, cell_ids: list[str], genes: list[str], min_pct_cells_sj: float) -> list[str]:
        coord = self.sj_metadata.loc[self.sj_metadata["gene_short_name.start"].isin(genes), "coord.intron"].tolist()
        table = self._expression_rate_table(self.sj_count_matrix.subset_rows(coord), cell_ids, "coord.intron", "group")
        return table.loc[table["pct.cells.expr"] > min_pct_cells_sj, "coord.intron"].tolist()

    def _build_sj_results(self, coord_introns: list[str], group1: list[str], group2: list[str]) -> pd.DataFrame:
        sj_meta = self.sj_metadata.set_index("coord.intron").loc[coord_introns].reset_index()
        gene_ids = list(dict.fromkeys(sj_meta["gene_short_name.start"].tolist()))
        gene_index = {gene: i for i, gene in enumerate(gene_ids)}
        sj_gene_idx = np.array([gene_index[gene] for gene in sj_meta["gene_short_name.start"]])

        sj_sub = self.sj_count_matrix.subset_rows(coord_introns).subset_cols(group1 + group2).matrix
        gene_sub = self.gene_count_matrix.subset_rows(gene_ids).subset_cols(group1 + group2).matrix
        g1_idx = np.arange(len(group1))
        g2_idx = np.arange(len(group1), len(group1) + len(group2))

        group1_df = self._compute_sj_group_metrics(
            coord_introns, sj_meta["gene_short_name.start"].tolist(), sj_sub, gene_sub, sj_gene_idx, g1_idx, "g1"
        )
        group2_df = self._compute_sj_group_metrics(
            coord_introns, sj_meta["gene_short_name.start"].tolist(), sj_sub, gene_sub, sj_gene_idx, g2_idx, "g2"
        )
        results = group1_df.merge(group2_df, on=["coord.intron", "gene_short_name"], how="left")
        results.attrs["cell.group.g1"] = group1
        results.attrs["cell.group.g2"] = group2
        return results

    @staticmethod
    def _compute_sj_group_metrics(
        coord_introns: list[str],
        sj_gene_names: list[str],
        sj_matrix: sparse.csr_matrix,
        gene_matrix: sparse.csr_matrix,
        sj_gene_idx: np.ndarray,
        group_idx: np.ndarray,
        suffix: str,
    ) -> pd.DataFrame:
        sj_group = sj_matrix[:, group_idx]
        gene_group = gene_matrix[:, group_idx]
        n_cells_total = len(group_idx)
        n_cells_expr_sj = np.asarray(sj_group.getnnz(axis=1)).ravel()
        pct_cells_expr_sj = np.round(n_cells_expr_sj / n_cells_total * 100.0, 2)
        sj_count_total = np.asarray(sj_group.sum(axis=1)).ravel()

        n_cells_expr_gene = np.asarray(gene_group.getnnz(axis=1)).ravel()
        pct_cells_expr_gene = np.round(n_cells_expr_gene / n_cells_total * 100.0, 2)
        gene_count_total = np.asarray(gene_group.sum(axis=1)).ravel()
        gene_total_for_sj = gene_count_total[sj_gene_idx]
        gene_nexpr_for_sj = n_cells_expr_gene[sj_gene_idx]
        gene_pct_for_sj = pct_cells_expr_gene[sj_gene_idx]
        psi = np.round(
            np.divide(
                sj_count_total,
                gene_total_for_sj,
                out=np.zeros_like(sj_count_total, dtype=float),
                where=gene_total_for_sj != 0,
            )
            * 100.0,
            2,
        )

        return pd.DataFrame(
            {
                "coord.intron": coord_introns,
                "gene_short_name": sj_gene_names,
                f"n.cells.total.{suffix}": n_cells_total,
                f"n.cells.expr.sj.{suffix}": n_cells_expr_sj,
                f"pct.cells.expr.sj.{suffix}": pct_cells_expr_sj,
                f"n.cells.expr.gene.{suffix}": gene_nexpr_for_sj,
                f"pct.cells.expr.gene.{suffix}": gene_pct_for_sj,
                f"sj.count.total.{suffix}": sj_count_total,
                f"gene.count.total.{suffix}": gene_total_for_sj,
                f"psi.{suffix}": psi,
            }
        )

    @staticmethod
    def _compute_delta(
        sj_matrix: sparse.csr_matrix,
        gene_matrix: sparse.csr_matrix,
        sj_gene_idx: np.ndarray,
        group1_idx: np.ndarray,
        group2_idx: np.ndarray,
    ) -> np.ndarray:
        sj_g1 = np.asarray(sj_matrix[:, group1_idx].sum(axis=1)).ravel()
        sj_g2 = np.asarray(sj_matrix[:, group2_idx].sum(axis=1)).ravel()
        gene_g1 = np.asarray(gene_matrix[:, group1_idx].sum(axis=1)).ravel()[sj_gene_idx]
        gene_g2 = np.asarray(gene_matrix[:, group2_idx].sum(axis=1)).ravel()[sj_gene_idx]
        psi_g1 = np.round(
            np.divide(sj_g1, gene_g1, out=np.zeros_like(sj_g1, dtype=float), where=gene_g1 != 0) * 100.0,
            2,
        )
        psi_g2 = np.round(
            np.divide(sj_g2, gene_g2, out=np.zeros_like(sj_g2, dtype=float), where=gene_g2 != 0) * 100.0,
            2,
        )
        return psi_g2 - psi_g1
