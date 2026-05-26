from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from .modality import annotate_splicing_outliers, assign_modality_from_tables
from .psi import (
    PLATE_EVENT_TYPES,
    compute_psi_event,
    compute_psi_posterior_event,
    empty_feature,
    empty_psi,
)
from .stats import safe_anderson_pvalue
from .utils import ordered_intersection, read_table


@dataclass
class MarvelPlate:
    splice_pheno: pd.DataFrame
    splice_junction: pd.DataFrame
    intron_counts: pd.DataFrame | None
    splice_feature: dict[str, pd.DataFrame]
    gene_feature: pd.DataFrame
    exp: pd.DataFrame
    gtf: pd.DataFrame | None = None
    splice_feature_validated: dict[str, pd.DataFrame] = field(default_factory=dict)
    psi: dict[str, pd.DataFrame] = field(default_factory=dict)
    psi_posterior: dict[str, pd.DataFrame] = field(default_factory=dict)
    counts: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    modality_results: pd.DataFrame | None = None
    modality_prop: pd.DataFrame | None = None
    modality_change: pd.DataFrame | None = None
    de_splicing: dict[str, pd.DataFrame] = field(default_factory=dict)
    de_gene: pd.DataFrame | None = None
    de_spliced_gene: pd.DataFrame | None = None
    de_plots: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    n_events: dict[str, pd.DataFrame] = field(default_factory=dict)
    distance_to_canonical: dict[str, pd.DataFrame] = field(default_factory=dict)
    parsed_gtf: pd.DataFrame | None = None
    variable_splicing: dict[str, object] | None = None
    de_pctase: dict[str, pd.DataFrame] | None = None
    de_absase: dict[str, pd.DataFrame] | None = None
    de_a3ss_dist_to_ss: dict[str, pd.DataFrame] | None = None
    pca_results: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    value_plots: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    _splice_junction_numeric_cache: tuple[tuple[int, tuple[int, int], tuple[str, ...]], pd.DataFrame] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _intron_norm_cache: dict[tuple[float, tuple[int, tuple[int, int], tuple[str, ...]]], pd.DataFrame] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.splice_feature = {event: self.splice_feature.get(event, empty_feature()).copy() for event in PLATE_EVENT_TYPES}
        validated = {}
        for event in PLATE_EVENT_TYPES:
            df = self.splice_feature_validated.get(event)
            validated[event] = empty_feature() if df is None else df.copy()
        self.splice_feature_validated = validated

        psi_tables = {}
        psi_posterior_tables = {}
        sample_cols = self.splice_pheno["sample.id"].astype(str).tolist() if "sample.id" in self.splice_pheno.columns else []
        for event in PLATE_EVENT_TYPES:
            df = self.psi.get(event)
            if df is None:
                df = pd.DataFrame(columns=["tran_id", *sample_cols])
            psi_tables[event] = df.copy()
            posterior_df = self.psi_posterior.get(event)
            if posterior_df is None:
                posterior_df = pd.DataFrame(columns=["tran_id", *sample_cols])
            psi_posterior_tables[event] = posterior_df.copy()
        self.psi = psi_tables
        self.psi_posterior = psi_posterior_tables
        self.distance_to_canonical = {
            event: self.distance_to_canonical.get(event, pd.DataFrame()).copy() for event in PLATE_EVENT_TYPES
        }
        self.pca_results = {str(key): value for key, value in self.pca_results.items()}
        self.value_plots = {str(key): value for key, value in self.value_plots.items()}

    @classmethod
    def from_paths(
        cls,
        *,
        splice_pheno: str | Path,
        splice_junction: str | Path,
        gene_feature: str | Path,
        exp: str | Path,
        intron_counts: str | Path | None = None,
        gtf: str | Path | None = None,
        feature_se: str | Path | None = None,
        feature_mxe: str | Path | None = None,
        feature_ri: str | Path | None = None,
        feature_a5ss: str | Path | None = None,
        feature_a3ss: str | Path | None = None,
        feature_afe: str | Path | None = None,
        feature_ale: str | Path | None = None,
        psi_se: str | Path | None = None,
        psi_mxe: str | Path | None = None,
        psi_ri: str | Path | None = None,
        psi_a5ss: str | Path | None = None,
        psi_a3ss: str | Path | None = None,
        psi_afe: str | Path | None = None,
        psi_ale: str | Path | None = None,
    ) -> "MarvelPlate":
        feature_paths = {
            "SE": feature_se,
            "MXE": feature_mxe,
            "RI": feature_ri,
            "A5SS": feature_a5ss,
            "A3SS": feature_a3ss,
            "AFE": feature_afe,
            "ALE": feature_ale,
        }
        psi_paths = {
            "SE": psi_se,
            "MXE": psi_mxe,
            "RI": psi_ri,
            "A5SS": psi_a5ss,
            "A3SS": psi_a3ss,
            "AFE": psi_afe,
            "ALE": psi_ale,
        }

        feature_tables = {
            event: read_table(path, dtype=str) if path is not None else empty_feature()
            for event, path in feature_paths.items()
        }
        psi_tables = {
            event: read_table(path, dtype=str) if path is not None else empty_psi()
            for event, path in psi_paths.items()
        }
        if gtf is not None:
            gtf_df = pd.read_csv(
                gtf,
                sep="\t",
                header=None,
                comment="#",
                dtype=str,
                names=[f"V{i}" for i in range(1, 10)],
            )
        else:
            gtf_df = None

        return cls(
            splice_pheno=read_table(splice_pheno, dtype=str),
            splice_junction=read_table(splice_junction),
            intron_counts=read_table(intron_counts) if intron_counts is not None else None,
            splice_feature=feature_tables,
            gene_feature=read_table(gene_feature, dtype=str),
            exp=read_table(exp),
            gtf=gtf_df,
            psi=psi_tables,
        )

    def subset_samples(self, sample_ids: list[str]) -> "MarvelPlate":
        sample_ids = [str(sample_id) for sample_id in sample_ids]
        overlap = ordered_intersection(self.splice_pheno["sample.id"].astype(str).tolist(), sample_ids)
        self.splice_pheno = self.splice_pheno[self.splice_pheno["sample.id"].astype(str).isin(overlap)].copy()
        return self

    def transform_exp_values(
        self,
        offset: float = 1.0,
        transformation: str = "log2",
        threshold_lower: float = 1.0,
    ) -> "MarvelPlate":
        from .qc import _transform_exp_values_inplace

        return _transform_exp_values_inplace(
            self,
            offset=offset,
            transformation=transformation,
            threshold_lower=threshold_lower,
        )

    def check_alignment(self, level: str) -> "MarvelPlate":
        from .qc import _check_alignment_inplace

        return _check_alignment_inplace(self, level=level)

    def compute_psi(
        self,
        event_type: str,
        coverage_threshold: float,
        uneven_coverage_multiplier: float = 10.0,
        read_length: float = 1.0,
    ) -> "MarvelPlate":
        from .splicing import _compute_psi_inplace

        return _compute_psi_inplace(
            self,
            coverage_threshold=coverage_threshold,
            event_type=event_type,
            uneven_coverage_multiplier=uneven_coverage_multiplier,
            read_length=read_length,
        )

    def compute_psi_posterior(
        self,
        event_type: str | None = None,
    ) -> "MarvelPlate":
        from .splicing import _compute_psi_posterior_inplace

        return _compute_psi_posterior_inplace(self, event_type=event_type)

    def count_events(self, sample_ids: list[str], min_cells: int, label: str | None = None) -> pd.DataFrame:
        sample_ids = [str(sample_id) for sample_id in sample_ids]
        rows = []
        for event_type in PLATE_EVENT_TYPES:
            psi_df = self.psi.get(event_type)
            feature_df = self.splice_feature_validated.get(event_type)
            if psi_df is None or feature_df is None or psi_df.empty or feature_df.empty:
                continue
            columns = [column for column in psi_df.columns if column != "tran_id" and column in sample_ids]
            if not columns:
                continue
            values = psi_df.set_index("tran_id")[columns].apply(pd.to_numeric, errors="coerce")
            counts = values.notna().sum(axis=1)
            rows.append(
                {
                    "event_type": event_type,
                    "freq": int((counts >= min_cells).sum()),
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result["pct"] = result["freq"] / result["freq"].sum() * 100.0
        key = str(min_cells) if label is None else label
        self.n_events[key] = result
        return result

    def compare_values_genes(
        self,
        cell_group_g1: list[str],
        cell_group_g2: list[str],
        min_cells: int = 25,
        pct_cells: float | None = None,
        method: str = "wilcox",
        method_adjust: str = "fdr_bh",
        custom_gene_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        from .de import _plate_compare_values_genes_inplace

        return _plate_compare_values_genes_inplace(
            self,
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            min_cells=min_cells,
            pct_cells=pct_cells,
            method=method,
            method_adjust=method_adjust,
            custom_gene_ids=custom_gene_ids,
        )

    def compare_values_splicing(
        self,
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
        from .de import _plate_compare_values_splicing_inplace

        return _plate_compare_values_splicing_inplace(
            self,
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            method=method,
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

    def _compare_values_splicing_dts(
        self,
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
        _ = method_adjust, sigma_sq, bimodal_adjust, bimodal_adjust_fc, bimodal_adjust_diff
        rng = np.random.default_rng(seed if seed_dts is None else seed_dts)
        event_types = PLATE_EVENT_TYPES if event_types is None else [event.upper() for event in event_types]
        group1 = [str(sample_id) for sample_id in cell_group_g1]
        group2 = [str(sample_id) for sample_id in cell_group_g2]

        psi_tables = []
        feature_tables = []
        for event_type in event_types:
            psi_df = self.psi.get(event_type)
            feature_df = self.splice_feature_validated.get(event_type)
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
            from .de import _bootstrap_abs_mean_diff_pvalue_blocked

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
            self.de_splicing["dts"] = result
            return result

        result = result.dropna(subset=["p.val"]).sort_values("p.val").reset_index(drop=True)
        result["mean.g1"] = result["mean.g1"] * 100.0
        result["mean.g2"] = result["mean.g2"] * 100.0
        result["mean.diff"] = result["mean.diff"] * 100.0
        result["statistic"] = result["statistic"] * 100.0
        result["p.val.adj"] = multipletests(result["p.val"], method=method_adjust)[1]
        result = feature_all.merge(result, on="tran_id", how="inner")

        if assign_modality:
            modality_g1 = self.assign_modality(
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
            modality_g2 = self.assign_modality(
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
        self.de_splicing["dts"] = result
        return result

    def assign_modality(
        self,
        sample_ids: list[str],
        min_cells: int = 25,
        sigma_sq: float = 0.001,
        bimodal_adjust: bool = True,
        bimodal_adjust_fc: float = 3.0,
        bimodal_adjust_diff: float = 50.0,
        seed: int = 1,
        tran_ids: list[str] | None = None,
        update_store: bool = True,
    ) -> pd.DataFrame:
        from .modality import _assign_modality_inplace

        return _assign_modality_inplace(
            self,
            sample_ids=sample_ids,
            min_cells=min_cells,
            sigma_sq=sigma_sq,
            bimodal_adjust=bimodal_adjust,
            bimodal_adjust_fc=bimodal_adjust_fc,
            bimodal_adjust_diff=bimodal_adjust_diff,
            seed=seed,
            tran_ids=tran_ids,
            update_store=update_store,
        )

    def compare_values_spliced_genes(
        self,
        cell_group_g1: list[str],
        cell_group_g2: list[str],
        psi_method: list[str],
        psi_pval: list[float],
        psi_delta: float,
        method_de_gene: str = "wilcox",
        method_adjust_de_gene: str = "fdr_bh",
    ) -> pd.DataFrame:
        if len(psi_method) != len(psi_pval):
            raise ValueError("psi_method and psi_pval must have the same length")

        gene_frames = []
        for method_name, pval_cutoff in zip(psi_method, psi_pval):
            results = self.de_splicing.get(method_name.lower())
            if results is None:
                raise ValueError(f"Missing splicing DE results for method={method_name}")
            subset = results[
                (results["p.val.adj"] < pval_cutoff)
                & (results["mean.diff"].abs() > psi_delta)
                & (~results["outlier"])
            ][["gene_id"]]
            gene_frames.append(subset)

        if not gene_frames:
            raise ValueError("No splicing DE results available")

        gene_ids = pd.concat(gene_frames, ignore_index=True)["gene_id"].drop_duplicates().astype(str).tolist()
        de_gene_before = self.de_gene
        result = self.compare_values_genes(
            cell_group_g1=cell_group_g1,
            cell_group_g2=cell_group_g2,
            min_cells=3,
            method=method_de_gene,
            method_adjust=method_adjust_de_gene,
            custom_gene_ids=gene_ids,
        )
        self.de_spliced_gene = result.copy()
        self.de_gene = de_gene_before
        return result

    def get_sample_ids(self, column: str, values: list[str]) -> list[str]:
        mask = self.splice_pheno[column].isin(values)
        return self.splice_pheno.loc[mask, "sample.id"].astype(str).tolist()

    def save_outputs(self, output_dir: str | Path, summary: dict[str, object]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.splice_pheno.to_csv(output_dir / "splice_pheno.tsv", sep="\t", index=False)
        self.splice_junction.to_csv(output_dir / "splice_junction.tsv", sep="\t", index=False)
        if self.intron_counts is not None:
            self.intron_counts.to_csv(output_dir / "intron_counts.tsv", sep="\t", index=False)
        self.gene_feature.to_csv(output_dir / "gene_feature.tsv", sep="\t", index=False)
        self.exp.to_csv(output_dir / "exp.tsv", sep="\t", index=False)

        for event_type in PLATE_EVENT_TYPES:
            feature_df = self.splice_feature_validated.get(event_type)
            if feature_df is not None and not feature_df.empty:
                feature_df.to_csv(output_dir / f"psi_feature_{event_type.lower()}.tsv", sep="\t", index=False)
            psi_df = self.psi.get(event_type)
            if psi_df is not None and not psi_df.empty:
                psi_df.to_csv(output_dir / f"psi_{event_type.lower()}.tsv", sep="\t", index=False)

        for method_name, table in self.de_splicing.items():
            table.to_csv(output_dir / f"de_splicing_{method_name}.tsv", sep="\t", index=False)
        if self.modality_results is not None and not self.modality_results.empty:
            self.modality_results.to_csv(output_dir / "modality_results.tsv", sep="\t", index=False)
        if self.de_gene is not None:
            self.de_gene.to_csv(output_dir / "de_gene.tsv", sep="\t", index=False)
        if self.de_spliced_gene is not None:
            self.de_spliced_gene.to_csv(output_dir / "de_spliced_gene.tsv", sep="\t", index=False)
        for label, table in self.n_events.items():
            table.to_csv(output_dir / f"n_events_min_cells_{label}.tsv", sep="\t", index=False)

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    @staticmethod
    def _run_de_test(values_x: np.ndarray, values_y: np.ndarray, method: str) -> tuple[float, float]:
        if len(values_x) == 0 or len(values_y) == 0:
            return math.nan, math.nan
        method = method.lower()
        if method == "wilcox":
            try:
                return math.nan, float(mannwhitneyu(values_x, values_y, alternative="two-sided").pvalue)
            except ValueError:
                return math.nan, 1.0
        if method == "t.test":
            result = ttest_ind(values_x, values_y, equal_var=False, nan_policy="omit")
            return float(result.statistic), float(result.pvalue)
        if method == "ks":
            result = ks_2samp(values_x, values_y)
            return float(result.statistic), float(result.pvalue)
        if method == "ad":
            return safe_anderson_pvalue(values_x, values_y)
        raise ValueError(f"Unsupported DE method: {method}")
