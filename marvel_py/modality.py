from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from .psi import PLATE_EVENT_TYPES

if TYPE_CHECKING:
    from .models import MarvelPlate

__all__ = [
    "assign_modality",
    "count_events",
    "modality_change",
    "prop_modality",
    "prop_modality_bar",
    "prop_modality_doughnut",
]


_PHASE1_MODALITY_TYPES = {"basic", "extended", "complete"}


def _collapse_modality_label(value):
    if isinstance(value, str):
        if value.startswith("Included."):
            return "Included"
        if value.startswith("Excluded."):
            return "Excluded"
    return value


def fit_beta_params(values: np.ndarray, seed: int) -> dict[str, float]:
    def _fit(prepared: np.ndarray) -> dict[str, float]:
        if len(prepared) == 0:
            return {"alpha": math.nan, "beta": math.nan, "log.likelihood": math.nan, "variance": math.nan}
        try:
            alpha, beta, _, _ = beta_dist.fit(prepared, floc=0.0, fscale=1.0)
            log_likelihood = float(np.sum(beta_dist.logpdf(prepared, alpha, beta, loc=0.0, scale=1.0)))
            return {
                "alpha": float(alpha),
                "beta": float(beta),
                "log.likelihood": log_likelihood,
                "variance": float(np.var(prepared, ddof=1)) if len(prepared) > 1 else 0.0,
            }
        except Exception:
            return {"alpha": math.nan, "beta": math.nan, "log.likelihood": math.nan, "variance": math.nan}

    cleaned = np.asarray(values, dtype=float)
    cleaned = cleaned[np.isfinite(cleaned)]
    if len(cleaned) == 0:
        return {"alpha": math.nan, "beta": math.nan, "log.likelihood": math.nan, "variance": math.nan}

    rng = np.random.default_rng(seed)
    prepared = cleaned.copy()
    one_mask = prepared == 1.0
    zero_mask = prepared == 0.0
    if one_mask.any():
        prepared[one_mask] = rng.uniform(0.98, 0.9999, size=int(one_mask.sum()))
    if zero_mask.any():
        prepared[zero_mask] = rng.uniform(0.0001, 0.02, size=int(zero_mask.sum()))
    result = _fit(prepared)
    if not math.isnan(result["alpha"]):
        return result

    rng = np.random.default_rng(seed)
    prepared = cleaned + rng.uniform(0.0001, 0.01, size=len(cleaned))
    high_mask = prepared >= 1.0
    if high_mask.any():
        prepared[high_mask] = prepared[high_mask] - rng.uniform(0.0001, 0.01, size=int(high_mask.sum()))
    prepared = np.clip(prepared, 0.0001, 0.9999)
    return _fit(prepared)


def assign_modality_from_tables(
    *,
    psi_tables: list[pd.DataFrame],
    feature_tables: list[pd.DataFrame],
    sample_ids: list[str],
    min_cells: int = 25,
    sigma_sq: float = 0.001,
    bimodal_adjust: bool = True,
    bimodal_adjust_fc: float = 3.0,
    bimodal_adjust_diff: float = 50.0,
    seed: int = 1,
    tran_ids: list[str] | None = None,
) -> pd.DataFrame:
    if not psi_tables:
        return pd.DataFrame()

    psi_all = pd.concat(psi_tables, ignore_index=True).set_index("tran_id")
    feature_all = pd.concat(feature_tables, ignore_index=True)
    sample_columns = [str(sample_id) for sample_id in sample_ids]
    available_columns = [str(column) for column in psi_all.columns if column != "tran_id"]
    missing_columns = [column for column in sample_columns if column not in available_columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing sample columns in PSI tables: {missing}")

    values = psi_all.loc[:, sample_columns].apply(pd.to_numeric, errors="coerce")

    n_cells = values.notna().sum(axis=1)
    values = values.loc[n_cells >= min_cells]
    feature_all = feature_all[feature_all["tran_id"].astype(str).isin(values.index.astype(str))].copy()
    if tran_ids is not None:
        tran_ids_keep = [str(tran_id) for tran_id in tran_ids if str(tran_id) in values.index]
        values = values.loc[tran_ids_keep]
        feature_all = feature_all.set_index("tran_id").loc[tran_ids_keep].reset_index()
    else:
        feature_all = feature_all.set_index("tran_id").loc[values.index.astype(str)].reset_index()

    params = []
    for tran_id in values.index.astype(str).tolist():
        stats = fit_beta_params(values.loc[tran_id].to_numpy(dtype=float), seed=seed)
        stats["tran_id"] = tran_id
        params.append(stats)
    result = feature_all.merge(pd.DataFrame(params), on="tran_id", how="left")
    result["n.cells"] = result["tran_id"].map(n_cells.to_dict()).astype(int)

    result["modality"] = pd.NA
    missing = result["alpha"].isna()
    result.loc[missing, "modality"] = "Missing"
    mask = result["modality"].isna() & ((result["alpha"] <= 0.4) | (result["beta"] <= 0.4))
    result.loc[mask, "modality"] = "Bimodal"
    mask = result["modality"].isna() & (result["alpha"] >= 2.0) & (result["beta"] <= 1.0)
    result.loc[mask, "modality"] = "Included"
    mask = result["modality"].isna() & ((result["alpha"] / result["beta"]) > 2.0)
    result.loc[mask, "modality"] = "Included"
    mask = result["modality"].isna() & (result["beta"] >= 2.0) & (result["alpha"] <= 1.0)
    result.loc[mask, "modality"] = "Excluded"
    mask = result["modality"].isna() & ((result["beta"] / result["alpha"]) > 2.0)
    result.loc[mask, "modality"] = "Excluded"
    mask = result["modality"].isna() & (result["alpha"] >= 1.6) & (result["beta"] >= 1.6)
    result.loc[mask, "modality"] = "Middle"
    result.loc[result["modality"].isna(), "modality"] = "Multimodal"

    result["modality.var"] = pd.NA
    result.loc[missing, "modality.var"] = "Missing"
    mask = result["modality.var"].isna() & (result["variance"] <= sigma_sq) & (result["modality"] == "Included")
    result.loc[mask, "modality.var"] = "Included.Primary"
    mask = result["modality.var"].isna() & (result["modality"] == "Included")
    result.loc[mask, "modality.var"] = "Included.Dispersed"
    mask = result["modality.var"].isna() & (result["variance"] <= sigma_sq) & (result["modality"] == "Excluded")
    result.loc[mask, "modality.var"] = "Excluded.Primary"
    mask = result["modality.var"].isna() & (result["modality"] == "Excluded")
    result.loc[mask, "modality.var"] = "Excluded.Dispersed"
    mask = result["modality.var"].isna()
    result.loc[mask, "modality.var"] = result.loc[mask, "modality"]

    result["modality.bimodal.adj"] = result["modality.var"]
    if bimodal_adjust and (result["modality.var"] == "Bimodal").any():
        lower = values.lt(0.25).sum(axis=1).astype(float)
        higher = values.gt(0.75).sum(axis=1).astype(float)
        tail_total = lower + higher
        pct_lower = np.where(tail_total > 0, lower / tail_total * 100.0, np.nan)
        pct_higher = np.where(tail_total > 0, higher / tail_total * 100.0, np.nan)
        ratio_hi_lo = np.divide(pct_higher, pct_lower, out=np.full_like(pct_higher, np.inf), where=pct_lower > 0)
        ratio_lo_hi = np.divide(pct_lower, pct_higher, out=np.full_like(pct_lower, np.inf), where=pct_higher > 0)
        pct_fc = np.where((pct_lower > 0) & (pct_higher > 0), np.where(pct_higher > pct_lower, ratio_hi_lo, ratio_lo_hi), np.inf)
        pct_diff = np.where(np.isfinite(pct_lower) & np.isfinite(pct_higher), np.abs(pct_higher - pct_lower), np.inf)
        psi_average = values.mean(axis=1, skipna=True).to_numpy(dtype=float)

        result["pct.fc"] = result["tran_id"].map(dict(zip(values.index.astype(str), pct_fc, strict=False)))
        result["pct.diff"] = result["tran_id"].map(dict(zip(values.index.astype(str), pct_diff, strict=False)))
        result["psi.average"] = result["tran_id"].map(dict(zip(values.index.astype(str), psi_average, strict=False)))
        result["bimodal.class"] = pd.NA
        bimodal_mask = result["modality.var"] == "Bimodal"
        pass_mask = (
            bimodal_mask
            & (result["alpha"] <= 0.4)
            & (result["beta"] <= 0.4)
            & (result["pct.fc"] <= bimodal_adjust_fc)
            & (result["pct.diff"] <= bimodal_adjust_diff)
        )
        result.loc[bimodal_mask & pass_mask, "bimodal.class"] = "pass"
        fail_mask = bimodal_mask & (~pass_mask)
        result.loc[fail_mask, "bimodal.class"] = "fail"
        included_primary = fail_mask & (result["psi.average"] >= 0.5) & (result["variance"] <= sigma_sq)
        included_disp = fail_mask & (result["psi.average"] >= 0.5) & (~included_primary)
        excluded_primary = fail_mask & (result["psi.average"] < 0.5) & (result["variance"] <= sigma_sq)
        excluded_disp = fail_mask & (result["psi.average"] < 0.5) & (~excluded_primary)
        result.loc[included_primary, "modality.bimodal.adj"] = "Included.Primary"
        result.loc[included_disp, "modality.bimodal.adj"] = "Included.Dispersed"
        result.loc[excluded_primary, "modality.bimodal.adj"] = "Excluded.Primary"
        result.loc[excluded_disp, "modality.bimodal.adj"] = "Excluded.Dispersed"

    result.loc[result["modality"] == "Missing", "modality"] = pd.NA
    result.loc[result["modality.var"] == "Missing", "modality.var"] = pd.NA
    result.loc[result["modality.bimodal.adj"] == "Missing", "modality.bimodal.adj"] = pd.NA
    return result


def count_events_from_tables(
    *,
    marvel_object,
    sample_ids: list[str],
    min_cells: int,
    label: str | None = None,
) -> pd.DataFrame:
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    rows = []
    for event_type in PLATE_EVENT_TYPES:
        psi_df = marvel_object.psi.get(event_type)
        feature_df = marvel_object.splice_feature_validated.get(event_type)
        if psi_df is None or feature_df is None or psi_df.empty or feature_df.empty:
            continue
        available_columns = [str(column) for column in psi_df.columns if column != "tran_id"]
        missing_columns = [column for column in sample_ids if column not in available_columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing sample columns in PSI tables: {missing}")
        columns = [column for column in psi_df.columns if column != "tran_id" and column in sample_ids]
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
        total = float(result["freq"].sum())
        result["pct"] = np.where(total > 0.0, result["freq"] / total * 100.0, 0.0)
    key = str(min_cells) if label is None else label
    marvel_object.n_events[key] = result
    return result


def summarize_modality(
    results: pd.DataFrame,
    *,
    modality_column: str,
    event_type: list[str],
    across_event_type: bool,
) -> pd.DataFrame:
    columns = ["event_type", "modality", "freq", "pct"] if across_event_type else ["modality", "freq", "pct"]
    if results.empty:
        return pd.DataFrame(columns=columns)

    df = results[results["event_type"].astype(str).isin([str(item) for item in event_type])].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    if across_event_type:
        table = (
            df.groupby(["event_type", modality_column], dropna=False)
            .size()
            .rename("freq")
            .reset_index()
            .rename(columns={modality_column: "modality"})
        )
        table["pct"] = table.groupby("event_type")["freq"].transform(lambda values: values / values.sum() * 100.0)
        return table

    table = (
        df.groupby(modality_column, dropna=False)
        .size()
        .rename("freq")
        .reset_index()
        .rename(columns={modality_column: "modality"})
    )
    table["pct"] = table["freq"] / table["freq"].sum() * 100.0
    return table


def prop_modality_from_results(
    *,
    marvel_object,
    modality_column: str,
    modality_type: str,
    event_type: list[str],
    across_event_type: bool,
) -> pd.DataFrame:
    if marvel_object.modality_results is None:
        raise ValueError("assign_modality must run before prop_modality")
    modality_type = str(modality_type).lower()
    if modality_type not in _PHASE1_MODALITY_TYPES:
        raise ValueError(
            f"Unsupported modality_type for Phase 1: {modality_type}. "
            "Use basic, extended, or complete."
        )
    if modality_type == "complete":
        modality_type = "extended"

    results = marvel_object.modality_results.copy()
    if modality_type == "basic" and modality_column in results.columns:
        results.loc[:, modality_column] = results[modality_column].map(_collapse_modality_label)

    event_type = [str(item).upper() for item in event_type]
    result = summarize_modality(
        results,
        modality_column=modality_column,
        event_type=event_type,
        across_event_type=across_event_type,
    )
    marvel_object.modality_prop = result
    return result


def annotate_splicing_outliers(
    *,
    result: pd.DataFrame,
    values: pd.DataFrame,
    group1: list[str],
    group2: list[str],
    n_cells_outliers: int,
) -> pd.DataFrame:
    result = result.copy()
    result["n.cells.outliers.g1"] = 0
    result["n.cells.outliers.g2"] = 0
    result["outliers"] = False

    included_mask = (
        result["modality.bimodal.adj.g1"].fillna("").str.contains("Included", regex=False)
        & result["modality.bimodal.adj.g2"].fillna("").str.contains("Included", regex=False)
    )
    if included_mask.any():
        tran_ids = result.loc[included_mask, "tran_id"].astype(str).tolist()
        counts_g1 = values.loc[tran_ids, group1].apply(lambda row: int(np.sum(np.abs(row.to_numpy(dtype=float) - 1.0) > 1e-12)), axis=1)
        counts_g2 = values.loc[tran_ids, group2].apply(lambda row: int(np.sum(np.abs(row.to_numpy(dtype=float) - 1.0) > 1e-12)), axis=1)
        result.loc[included_mask, "n.cells.outliers.g1"] = result.loc[included_mask, "tran_id"].map(counts_g1.to_dict()).astype(int)
        result.loc[included_mask, "n.cells.outliers.g2"] = result.loc[included_mask, "tran_id"].map(counts_g2.to_dict()).astype(int)
        outlier_mask = included_mask & (result["n.cells.outliers.g1"] < n_cells_outliers) & (result["n.cells.outliers.g2"] < n_cells_outliers)
        result.loc[outlier_mask, "outliers"] = True

    excluded_mask = (
        result["modality.bimodal.adj.g1"].fillna("").str.contains("Excluded", regex=False)
        & result["modality.bimodal.adj.g2"].fillna("").str.contains("Excluded", regex=False)
    )
    if excluded_mask.any():
        tran_ids = result.loc[excluded_mask, "tran_id"].astype(str).tolist()
        counts_g1 = values.loc[tran_ids, group1].apply(lambda row: int(np.sum(np.abs(row.to_numpy(dtype=float)) > 1e-12)), axis=1)
        counts_g2 = values.loc[tran_ids, group2].apply(lambda row: int(np.sum(np.abs(row.to_numpy(dtype=float)) > 1e-12)), axis=1)
        result.loc[excluded_mask, "n.cells.outliers.g1"] = result.loc[excluded_mask, "tran_id"].map(counts_g1.to_dict()).astype(int)
        result.loc[excluded_mask, "n.cells.outliers.g2"] = result.loc[excluded_mask, "tran_id"].map(counts_g2.to_dict()).astype(int)
        outlier_mask = excluded_mask & (result["n.cells.outliers.g1"] < n_cells_outliers) & (result["n.cells.outliers.g2"] < n_cells_outliers)
        result.loc[outlier_mask, "outliers"] = True

    return result


def _assign_modality_inplace(
    marvel_object: MarvelPlate,
    *,
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
    psi_tables = []
    feature_tables = []
    for event_type, psi_df in marvel_object.psi.items():
        feature_df = marvel_object.splice_feature_validated.get(event_type)
        if psi_df is None or feature_df is None or psi_df.empty or feature_df.empty:
            continue
        psi_tables.append(psi_df)
        feature_tables.append(feature_df)

    result = assign_modality_from_tables(
        psi_tables=psi_tables,
        feature_tables=feature_tables,
        sample_ids=[str(sample_id) for sample_id in sample_ids],
        min_cells=min_cells,
        sigma_sq=sigma_sq,
        bimodal_adjust=bimodal_adjust,
        bimodal_adjust_fc=bimodal_adjust_fc,
        bimodal_adjust_diff=bimodal_adjust_diff,
        seed=seed,
        tran_ids=tran_ids,
    )
    if update_store:
        marvel_object.modality_results = result.copy()
    return result


def _prop_modality_bar_inplace(
    marvel_object: MarvelPlate,
    *,
    modality_column: str,
    modality_type: str,
    event_type: list[str],
    across_event_type: bool,
) -> MarvelPlate:
    prop_modality_from_results(
        marvel_object=marvel_object,
        modality_column=modality_column,
        modality_type=modality_type,
        event_type=event_type,
        across_event_type=across_event_type,
    )
    return marvel_object


def _modality_change_inplace(
    marvel_object: MarvelPlate,
    *,
    event_type: str,
) -> MarvelPlate:
    if marvel_object.modality_results is None:
        raise ValueError("assign_modality must run before modality_change")

    results = marvel_object.modality_results.copy()
    results["event_type"] = results["event_type"].astype(str)
    results = results[results["event_type"] == str(event_type).upper()].copy()
    source_col = "modality.pre"
    target_col = "modality.post"
    if source_col not in results.columns or target_col not in results.columns:
        fallback_col = "modality.bimodal.adj"
        if fallback_col not in results.columns:
            raise ValueError("modality_change requires modality.pre/modality.post or modality.bimodal.adj")
        results[source_col] = results[fallback_col]
        results[target_col] = results[fallback_col]

    summary = (
        results.groupby([source_col, target_col], dropna=False)
        .size()
        .rename("freq")
        .reset_index()
        .rename(columns={source_col: "modality_pre", target_col: "modality_post"})
    )
    total = float(summary["freq"].sum())
    summary["pct"] = summary["freq"] / total * 100.0 if total > 0.0 else 0.0
    marvel_object.modality_change = summary
    return marvel_object


def count_events(marvel_object: MarvelPlate, *, sample_ids: list[str], min_cells: int) -> MarvelPlate:
    count_events_from_tables(marvel_object=marvel_object, sample_ids=sample_ids, min_cells=min_cells)
    return marvel_object


def assign_modality(
    marvel_object: MarvelPlate,
    *,
    sample_ids: list[str],
    min_cells: int,
    seed: int = 1,
) -> MarvelPlate:
    _assign_modality_inplace(
        marvel_object,
        sample_ids=sample_ids,
        min_cells=min_cells,
        seed=seed,
    )
    return marvel_object


def prop_modality(
    marvel_object: MarvelPlate,
    *,
    modality_column: str,
    modality_type: str,
    event_type: list[str],
    across_event_type: bool,
    prop_test: str | None = None,
    prop_adj: str | None = None,
    xlabels_size: float | None = None,
) -> MarvelPlate:
    if prop_test is not None:
        raise NotImplementedError("prop_test is not implemented in Phase 1")
    if prop_adj is not None:
        raise NotImplementedError("prop_adj is not implemented in Phase 1")
    if xlabels_size is not None:
        raise NotImplementedError("xlabels_size is not implemented in Phase 1")
    prop_modality_from_results(
        marvel_object=marvel_object,
        modality_column=modality_column,
        modality_type=modality_type,
        event_type=event_type,
        across_event_type=across_event_type,
    )
    return marvel_object


def prop_modality_bar(
    marvel_object: MarvelPlate,
    *,
    modality_column: str,
    modality_type: str,
    event_type: list[str],
    across_event_type: bool,
) -> MarvelPlate:
    return _prop_modality_bar_inplace(
        marvel_object,
        modality_column=modality_column,
        modality_type=modality_type,
        event_type=event_type,
        across_event_type=across_event_type,
    )


def prop_modality_doughnut(
    marvel_object: MarvelPlate,
    *,
    modality_column: str,
    modality_type: str,
    event_type: list[str],
    across_event_type: bool,
) -> MarvelPlate:
    return _prop_modality_bar_inplace(
        marvel_object,
        modality_column=modality_column,
        modality_type=modality_type,
        event_type=event_type,
        across_event_type=across_event_type,
    )


def modality_change(
    marvel_object: MarvelPlate,
    *,
    event_type: str,
) -> MarvelPlate:
    return _modality_change_inplace(marvel_object, event_type=event_type)
