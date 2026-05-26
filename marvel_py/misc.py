from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .de import _normalize_methods
from .models import MarvelPlate
from .alignment import _invalidate_plate_state
from .utils import extract_gtf_attr

__all__ = [
    "identify_variable_events",
    "parse_gtf",
    "pct_ase",
    "prepare_bed_file_ri",
    "preprocess_rmats",
    "preprocess_rmats_a3ss",
    "preprocess_rmats_a5ss",
    "preprocess_rmats_mxe",
    "preprocess_rmats_ri",
    "preprocess_rmats_se",
    "remove_cryptic_ss",
    "remove_cryptic_ss_afe",
    "remove_cryptic_ss_ale",
    "subset_cryptic_a3ss",
    "subset_cryptic_ss",
    "subset_cryptic_ss_a3ss",
    "subset_cryptic_ss_a5ss",
]

EVENT_TYPE_ORDER = ["SE", "MXE", "RI", "A5SS", "A3SS", "AFE", "ALE"]
FEATURE_COLUMNS = ["tran_id", "gene_id", "gene_short_name", "gene_type"]
BED_CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def parse_gtf(marvel_object: MarvelPlate) -> MarvelPlate:
    if marvel_object.gtf is None:
        raise ValueError("gtf is required for parse_gtf")
    gtf = marvel_object.gtf.copy()
    if "V9" not in gtf.columns:
        raise ValueError("gtf must contain V9 attribute column for parse_gtf")
    attrs = gtf["V9"].astype(str)
    gtf["gene_id"] = attrs.map(lambda value: extract_gtf_attr(value, "gene_id") or "")
    gtf["transcript_id"] = attrs.map(lambda value: extract_gtf_attr(value, "transcript_id") or "")
    gtf["transcript_type"] = attrs.map(
        lambda value: extract_gtf_attr(value, "transcript_type")
        or extract_gtf_attr(value, "transcript_biotype")
        or ""
    )
    gtf = gtf.drop(columns=["V9"])
    marvel_object.parsed_gtf = gtf
    return marvel_object


def _mgcv_like_pspline_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing: float,
    n_basis: int = 10,
    degree: int = 3,
    penalty_order: int = 1,
) -> np.ndarray:
    n_basis = max(int(n_basis), int(degree) + 2)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_min == x_max:
        return np.full_like(x, float(np.nanmean(y)), dtype=float)

    n_internal = n_basis - degree - 1
    internal = np.linspace(x_min, x_max, n_internal + 2)[1:-1] if n_internal > 0 else np.array([], dtype=float)
    knots = np.concatenate(
        [
            np.repeat(x_min, degree + 1),
            np.asarray(internal, dtype=float),
            np.repeat(x_max, degree + 1),
        ]
    )
    design = BSpline.design_matrix(x, knots, degree, extrapolate=True).toarray()
    difference = np.diff(np.eye(design.shape[1]), n=penalty_order, axis=0)
    penalty = difference.T @ difference
    lhs = design.T @ design + float(smoothing) * penalty
    rhs = design.T @ y
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs
    return design @ coefficients


def _predict_sd_from_mean(mean: np.ndarray, sd: np.ndarray, smoothing: float) -> np.ndarray:
    finite = np.isfinite(mean) & np.isfinite(sd)
    if finite.sum() == 0:
        return np.zeros_like(sd, dtype=float)
    if finite.sum() < 3 or np.nanstd(sd[finite]) == 0:
        baseline = float(np.nanmean(sd[finite]))
        return np.full_like(sd, baseline, dtype=float)

    x = mean[finite].astype(float)
    y = sd[finite].astype(float)
    if len(np.unique(x)) < 3:
        unique_x, inverse = np.unique(x, return_inverse=True)
        unique_y = np.array([float(np.nanmean(y[inverse == idx])) for idx in range(len(unique_x))])
        predicted = np.interp(mean.astype(float), unique_x, unique_y)
    else:
        predicted = _mgcv_like_pspline_predict(
            mean.astype(float),
            sd.astype(float),
            smoothing=float(smoothing),
        )
    return np.clip(np.asarray(predicted, dtype=float), 0.0, None)


def identify_variable_events(
    marvel_object: MarvelPlate,
    *,
    sample_ids: list[str] | None = None,
    cell_group_column: str,
    cell_group_order: list[str] | None,
    min_cells: int = 25,
    smoothing: float = 0.6,
) -> MarvelPlate:
    psi_tables = [table for table in marvel_object.psi.values() if table is not None and not table.empty]
    feature_tables = [
        table for table in marvel_object.splice_feature_validated.values() if table is not None and not table.empty
    ]
    if not psi_tables or not feature_tables:
        raise ValueError("compute_psi must run before identify_variable_events")
    if cell_group_column not in marvel_object.splice_pheno.columns:
        raise ValueError(f"Unknown cell_group_column: {cell_group_column}")

    psi_df = pd.concat([table.copy() for table in psi_tables], ignore_index=True)
    if "tran_id" not in psi_df.columns:
        raise ValueError("PSI tables must contain tran_id")
    psi_df["tran_id"] = psi_df["tran_id"].astype(str)
    psi_df = psi_df.drop_duplicates("tran_id", keep="first").set_index("tran_id")

    pheno = marvel_object.splice_pheno.copy()
    pheno["sample.id"] = pheno["sample.id"].astype(str)
    pheno["pca.cell.group.label"] = pheno[cell_group_column].astype(str)
    if sample_ids is not None:
        requested = [str(sample_id) for sample_id in sample_ids]
        pheno = pheno[pheno["sample.id"].isin(requested)].copy()
    if cell_group_order is None:
        group_order = pheno["pca.cell.group.label"].drop_duplicates().astype(str).tolist()
    else:
        group_order = [str(group) for group in cell_group_order]
    pheno = pheno[pheno["pca.cell.group.label"].isin(group_order)].copy()
    sample_order = [sample_id for sample_id in pheno["sample.id"].tolist() if sample_id in psi_df.columns]
    if not sample_order:
        raise ValueError("No matching sample ids were found in PSI tables")

    values = psi_df.loc[:, sample_order].apply(pd.to_numeric, errors="coerce")
    keep = values.notna().sum(axis=1) >= int(min_cells)
    values = values.loc[keep]
    if values.empty:
        marvel_object.variable_splicing = {
            "tran_ids": [],
            "table": pd.DataFrame(columns=["tran_id", "mean", "sd", "sd_pred", "variable"]),
            "plot": None,
        }
        return marvel_object

    result = pd.DataFrame(
        {
            "tran_id": values.index.astype(str),
            "mean": values.mean(axis=1, skipna=True).to_numpy(dtype=float),
            "sd": values.std(axis=1, skipna=True, ddof=1).fillna(0.0).to_numpy(dtype=float),
        }
    )
    result["sd_pred"] = _predict_sd_from_mean(
        result["mean"].to_numpy(dtype=float),
        result["sd"].to_numpy(dtype=float),
        smoothing,
    )
    result["variable"] = np.where(result["sd"] > result["sd_pred"], "Yes", "No")

    features = pd.concat([table.copy() for table in feature_tables], ignore_index=True)
    if "tran_id" in features.columns:
        features["tran_id"] = features["tran_id"].astype(str)
        feature_cols = [column for column in FEATURE_COLUMNS if column in features.columns]
        features = features.loc[:, feature_cols].drop_duplicates("tran_id", keep="first")
        result = result.merge(features, on="tran_id", how="left")

    tran_ids = result.loc[result["variable"] == "Yes", "tran_id"].astype(str).tolist()
    marvel_object.variable_splicing = {"tran_ids": tran_ids, "table": result, "plot": None}
    return marvel_object


def _get_outlier_mask(frame: pd.DataFrame) -> pd.Series:
    if "outliers" in frame.columns:
        values = frame["outliers"]
    elif "outlier" in frame.columns:
        values = frame["outlier"]
    else:
        return pd.Series(False, index=frame.index)
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "t", "1"})


def pct_ase(
    marvel_object: MarvelPlate,
    *,
    method,
    psi_pval,
    psi_mean_diff: float,
    ylabels_size: float = 8.0,
    barlabels_size: float = 3.0,
    x_offset: float = 0.0,
    direction_color: list[str] | None = None,
    mode: str = "percentage",
) -> MarvelPlate:
    _ = ylabels_size, barlabels_size, x_offset, direction_color
    methods = _normalize_methods(method, level="splicing")
    if isinstance(psi_pval, (float, int)):
        pvals = [float(psi_pval)] * len(methods)
    else:
        pvals = [float(value) for value in psi_pval]
    if len(methods) != len(pvals):
        raise ValueError("method and psi_pval must have the same length")
    if mode not in {"percentage", "absolute"}:
        raise ValueError("mode must be one of {'percentage', 'absolute'}")

    sig_frames = []
    for method_name, pval_cutoff in zip(methods, pvals):
        if method_name not in marvel_object.de_splicing:
            raise ValueError(f"Missing differential splicing table for method={method_name}")
        results = marvel_object.de_splicing[method_name].copy()
        outlier_mask = _get_outlier_mask(results)
        up = results[(results["p.val.adj"] < pval_cutoff) & (results["mean.diff"] > psi_mean_diff) & (~outlier_mask)]
        down = results[(results["p.val.adj"] < pval_cutoff) & (results["mean.diff"] < (-1 * psi_mean_diff)) & (~outlier_mask)]
        small = pd.concat([up, down], ignore_index=True)
        if small.empty:
            continue
        small = small.loc[:, ["tran_id", "event_type", "mean.diff"]].copy()
        sig_frames.append(small)

    results_sig = (
        pd.concat(sig_frames, ignore_index=True).drop_duplicates()
        if sig_frames
        else pd.DataFrame(columns=["tran_id", "event_type", "mean.diff"])
    )
    if not results_sig.empty:
        results_sig["direction"] = np.where(results_sig["mean.diff"] > psi_mean_diff, "up", "down")

    base = marvel_object.de_splicing[methods[0]].copy()
    annotated = (
        base.merge(results_sig.loc[:, ["tran_id", "direction"]], on="tran_id", how="left")
        if not results_sig.empty
        else base.assign(direction=np.nan)
    )
    event_types = [event for event in EVENT_TYPE_ORDER if event in annotated["event_type"].astype(str).unique().tolist()]
    summary_rows = []
    for event_type in event_types:
        small = annotated[annotated["event_type"].astype(str) == event_type]
        n_total = len(small)
        n_sig_up = int((small["direction"] == "up").sum())
        n_sig_down = int((small["direction"] == "down").sum())
        pct_up = round((n_sig_up / n_total) * 100, 1) if n_total else 0.0
        pct_down = round((n_sig_down / n_total) * 100, 1) if n_total else 0.0
        summary_rows.append(
            {
                "event_type": event_type,
                "n.total": n_total,
                "n.sig.up": n_sig_up,
                "pct.sig.up": pct_up,
                "n.sig.down": n_sig_down,
                "pct.sig.down": pct_down,
            }
        )
    summary = pd.DataFrame(summary_rows)

    if mode == "percentage":
        table = pd.concat(
            [
                summary.loc[:, ["event_type", "pct.sig.up"]].rename(columns={"pct.sig.up": "pct"}).assign(direction="up"),
                summary.loc[:, ["event_type", "pct.sig.down"]].rename(columns={"pct.sig.down": "pct"}).assign(direction="down"),
            ],
            ignore_index=True,
        )
        table = table.loc[:, ["event_type", "direction", "pct"]]
        marvel_object.de_pctase = {"summary": summary, "table": table, "plot": None}
    else:
        table = pd.concat(
            [
                summary.loc[:, ["event_type", "n.sig.up"]].rename(columns={"n.sig.up": "n"}).assign(direction="up"),
                summary.loc[:, ["event_type", "n.sig.down"]].rename(columns={"n.sig.down": "n"}).assign(direction="down"),
            ],
            ignore_index=True,
        )
        table = table.loc[:, ["event_type", "direction", "n"]]
        marvel_object.de_absase = {"summary": summary, "table": table, "plot": None}
    return marvel_object


def _parse_a3ss_dist_to_ss(tran_id: str) -> int:
    right = str(tran_id).split("@", 1)[1]
    parts = right.split(":")
    pair = parts[1]
    first, second = pair.split("|", 1)
    return abs(int(first) - int(second))


def subset_cryptic_a3ss(
    marvel_object: MarvelPlate,
    *,
    method,
    distance_to_ss: tuple[int, int] | list[int] = (1, 100),
) -> MarvelPlate:
    if marvel_object.de_a3ss_dist_to_ss is None:
        marvel_object.de_a3ss_dist_to_ss = {}
    methods = _normalize_methods(method, level="splicing")
    lower, upper = int(distance_to_ss[0]), int(distance_to_ss[1])
    for method_name in methods:
        if method_name not in marvel_object.de_splicing:
            raise ValueError(f"Missing differential splicing table for method={method_name}")
        df = marvel_object.de_splicing[method_name].copy().reset_index(drop=True)
        df["row.id"] = range(1, len(df) + 1)
        small = df[df["event_type"].astype(str) == "A3SS"].copy()
        if small.empty:
            marvel_object.de_a3ss_dist_to_ss[method_name] = pd.DataFrame(
                columns=["tran_id", "event_type", "gene_id", "gene_short_name", "gene_type", "dist.to.ss"]
            )
            marvel_object.de_splicing[method_name] = df.drop(columns=["row.id"])
            continue
        results = small.loc[:, ["tran_id", "event_type", "gene_id", "gene_short_name", "gene_type"]].copy()
        results["dist.to.ss"] = results["tran_id"].astype(str).map(_parse_a3ss_dist_to_ss)
        annotated = df.merge(results.loc[:, ["tran_id", "dist.to.ss"]], on="tran_id", how="left")
        keep = (
            ((annotated["event_type"].astype(str) == "A3SS") & (annotated["dist.to.ss"] >= lower) & (annotated["dist.to.ss"] <= upper))
            | (annotated["event_type"].astype(str) != "A3SS")
        )
        annotated = annotated.loc[keep].sort_values("row.id").drop(columns=["row.id", "dist.to.ss"]).reset_index(drop=True)
        marvel_object.de_splicing[method_name] = annotated
        marvel_object.de_a3ss_dist_to_ss[method_name] = results.reset_index(drop=True)
    return marvel_object


def _with_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["row_id"] = range(1, len(result) + 1)
    return result


def _parse_a5ss_distance(tran_id: str, strand: str) -> int:
    left, _ = str(tran_id).split(f":{strand}@", 1)
    splice_pair = left.split(":")[2]
    canonical_token, alt_token = splice_pair.split("|", 1)
    if strand == "+":
        canonical = int(canonical_token)
        alt = int(alt_token)
    else:
        canonical = int(alt_token)
        alt = int(canonical_token)
    return abs(canonical - alt) + 1


def _parse_a3ss_distance(tran_id: str, strand: str) -> int:
    _, right = str(tran_id).split(f":{strand}@", 1)
    splice_pair = right.split(":")[1]
    canonical_token, alt_token = splice_pair.split("|", 1)
    if strand == "+":
        canonical = int(alt_token)
        alt = int(canonical_token)
    else:
        canonical = int(canonical_token)
        alt = int(alt_token)
    return abs(canonical - alt) + 1


def _parse_afe_distance(tran_id: str, strand: str) -> int:
    left, _ = str(tran_id).split(f":{strand}@", 1)
    canonical_token = left.split("|", 1)[0]
    alt_token = left.split("|", 1)[1]
    canonical_parts = canonical_token.split(":")
    alt_parts = alt_token.split(":")
    if strand == "+":
        canonical = int(canonical_parts[2])
        alt = int(alt_parts[1])
    else:
        canonical = int(alt_parts[0])
        alt = int(canonical_parts[1])
    return abs(canonical - alt) + 1


def _parse_ale_distance(tran_id: str, strand: str) -> int:
    _, right = str(tran_id).split(f":{strand}@", 1)
    left_token, right_token = right.split("|", 1)
    left_parts = left_token.split(":")
    right_parts = right_token.split(":")
    if strand == "+":
        canonical = int(right_parts[0])
        alt = int(left_parts[1])
    else:
        canonical = int(left_parts[2])
        alt = int(right_parts[1])
    return abs(canonical - alt) + 1


def _annotate_distance(frame: pd.DataFrame, *, event_type: str) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        if "dist_to_canonical" not in result.columns:
            result["dist_to_canonical"] = pd.Series(dtype=float)
        return result

    parse_fn = {
        "A5SS": _parse_a5ss_distance,
        "A3SS": _parse_a3ss_distance,
        "AFE": _parse_afe_distance,
        "ALE": _parse_ale_distance,
    }[event_type]
    result = frame.copy()
    strands = result["tran_id"].astype(str).map(lambda value: "+" if ":+@" in value else "-")
    result["dist_to_canonical"] = [
        parse_fn(tran_id, strand) for tran_id, strand in zip(result["tran_id"].astype(str), strands)
    ]
    return result


def _subset_by_distance(
    frame: pd.DataFrame,
    *,
    event_type: str,
    distance_to_canonical: float,
    keep_cryptic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        annotated = _annotate_distance(frame, event_type=event_type)
        return frame.copy(), annotated

    working = _with_row_ids(frame)
    pos = working[working["tran_id"].astype(str).str.contains(":+@", regex=False)].copy()
    neg = working[working["tran_id"].astype(str).str.contains(":-@", regex=False)].copy()
    pos = _annotate_distance(pos, event_type=event_type)
    neg = _annotate_distance(neg, event_type=event_type)
    annotated = pd.concat([pos, neg], ignore_index=True).sort_values("row_id").reset_index(drop=True)

    if keep_cryptic:
        mask = annotated["dist_to_canonical"].astype(float) <= float(distance_to_canonical)
    else:
        mask = annotated["dist_to_canonical"].astype(float) > float(distance_to_canonical)

    filtered = annotated.loc[mask].copy()
    filtered = filtered.drop(columns=["row_id", "dist_to_canonical"], errors="ignore").reset_index(drop=True)
    annotated = annotated.drop(columns=["row_id"], errors="ignore").reset_index(drop=True)
    return filtered, annotated


def _store_distance(
    marvel_object: MarvelPlate,
    *,
    slot: str,
    filtered: pd.DataFrame,
    annotated: pd.DataFrame,
) -> MarvelPlate:
    marvel_object.splice_feature[slot] = filtered
    marvel_object.distance_to_canonical[slot] = annotated
    _invalidate_plate_state(marvel_object, event_types={slot}, clear_psi=True)
    return marvel_object


def subset_cryptic_ss_a5ss(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
) -> MarvelPlate:
    filtered, annotated = _subset_by_distance(
        marvel_object.splice_feature["A5SS"],
        event_type="A5SS",
        distance_to_canonical=distance_to_canonical,
        keep_cryptic=True,
    )
    return _store_distance(marvel_object, slot="A5SS", filtered=filtered, annotated=annotated)


def subset_cryptic_ss_a3ss(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
) -> MarvelPlate:
    filtered, annotated = _subset_by_distance(
        marvel_object.splice_feature["A3SS"],
        event_type="A3SS",
        distance_to_canonical=distance_to_canonical,
        keep_cryptic=True,
    )
    return _store_distance(marvel_object, slot="A3SS", filtered=filtered, annotated=annotated)


def subset_cryptic_ss(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
    event_type: str,
) -> MarvelPlate:
    event_type = str(event_type).upper()
    if event_type == "A5SS":
        return subset_cryptic_ss_a5ss(marvel_object, distance_to_canonical=distance_to_canonical)
    if event_type == "A3SS":
        return subset_cryptic_ss_a3ss(marvel_object, distance_to_canonical=distance_to_canonical)
    raise ValueError("event_type must be one of {'A5SS', 'A3SS'}")


def remove_cryptic_ss_afe(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
) -> MarvelPlate:
    filtered, annotated = _subset_by_distance(
        marvel_object.splice_feature["AFE"],
        event_type="AFE",
        distance_to_canonical=distance_to_canonical,
        keep_cryptic=False,
    )
    return _store_distance(marvel_object, slot="AFE", filtered=filtered, annotated=annotated)


def remove_cryptic_ss_ale(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
) -> MarvelPlate:
    filtered, annotated = _subset_by_distance(
        marvel_object.splice_feature["ALE"],
        event_type="ALE",
        distance_to_canonical=distance_to_canonical,
        keep_cryptic=False,
    )
    return _store_distance(marvel_object, slot="ALE", filtered=filtered, annotated=annotated)


def remove_cryptic_ss(
    marvel_object: MarvelPlate,
    *,
    distance_to_canonical: float = 100,
    event_type: str,
) -> MarvelPlate:
    event_type = str(event_type).upper()
    if event_type == "AFE":
        return remove_cryptic_ss_afe(marvel_object, distance_to_canonical=distance_to_canonical)
    if event_type == "ALE":
        return remove_cryptic_ss_ale(marvel_object, distance_to_canonical=distance_to_canonical)
    raise ValueError("event_type must be one of {'AFE', 'ALE'}")


def _strip_x_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column)[1:] if str(column).startswith("X") else str(column) for column in result.columns]
    return result


def _build_gene_type_ref(gtf: pd.DataFrame) -> pd.DataFrame:
    result = gtf.copy()
    if "V3" not in result.columns or "V9" not in result.columns:
        raise ValueError("GTF must contain V3 and V9 columns")
    genes = result[result["V3"].astype(str) == "gene"].copy()
    genes["gene_id"] = genes["V9"].astype(str).map(lambda value: extract_gtf_attr(value, "gene_id"))
    genes["gene_type"] = genes["V9"].astype(str).map(lambda value: extract_gtf_attr(value, "gene_type"))
    return genes.loc[:, ["gene_id", "gene_type"]].drop_duplicates()


def _collapse_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    unique = frame.drop_duplicates()
    counts = unique["tran_id"].value_counts()
    singletons = unique[unique["tran_id"].isin(counts[counts == 1].index)].copy()
    duplicates = unique[unique["tran_id"].isin(counts[counts > 1].index)].copy()
    if duplicates.empty:
        return unique.loc[:, FEATURE_COLUMNS].reset_index(drop=True)

    collapsed_rows = []
    for tran_id, group in duplicates.groupby("tran_id", sort=False):
        collapsed_rows.append(
            {
                "tran_id": tran_id,
                "gene_id": "|".join(group["gene_id"].astype(str).tolist()),
                "gene_short_name": "|".join(group["gene_short_name"].astype(str).tolist()),
                "gene_type": "|".join(group["gene_type"].astype(str).tolist()),
            }
        )
    collapsed = pd.DataFrame(collapsed_rows)
    merged = pd.concat([singletons.loc[:, FEATURE_COLUMNS], collapsed], ignore_index=True)
    return merged.loc[:, FEATURE_COLUMNS].reset_index(drop=True)


def _annotate_gene_type(frame: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    ref = _build_gene_type_ref(gtf)
    merged = frame.merge(ref, on="gene_id", how="left")
    return _collapse_duplicates(merged)


def prepare_bed_file_ri(file: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    result = df.loc[:, ["chr", "upstreamEE", "downstreamES"]].drop_duplicates().copy()
    result["chr"] = pd.Categorical(result["chr"], categories=BED_CHROM_ORDER, ordered=True)
    result = result.sort_values(["chr", "upstreamEE"], kind="mergesort").reset_index(drop=True)
    result["chr"] = result["chr"].astype(str)
    return result


def preprocess_rmats_se(file: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    pos = df[df["strand"].astype(str) == "+"].copy()
    neg = df[df["strand"].astype(str) == "-"].copy()
    pos["exonStart_0base"] = pos["exonStart_0base"].astype(int) + 1
    pos["upstreamES"] = pos["upstreamES"].astype(int) + 1
    pos["downstreamES"] = pos["downstreamES"].astype(int) + 1
    pos["tran_id"] = (
        pos["chr"].astype(str) + ":" + pos["upstreamES"].astype(str) + ":" + pos["upstreamEE"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["exonStart_0base"].astype(str) + ":" + pos["exonEnd"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["downstreamES"].astype(str) + ":" + pos["downstreamEE"].astype(str)
    )
    neg["exonStart_0base"] = neg["exonStart_0base"].astype(int) + 1
    neg["upstreamES"] = neg["upstreamES"].astype(int) + 1
    neg["downstreamES"] = neg["downstreamES"].astype(int) + 1
    neg["tran_id"] = (
        neg["chr"].astype(str) + ":" + neg["downstreamES"].astype(str) + ":" + neg["downstreamEE"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["exonStart_0base"].astype(str) + ":" + neg["exonEnd"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["upstreamES"].astype(str) + ":" + neg["upstreamEE"].astype(str)
    )
    merged = pd.concat([pos, neg], ignore_index=True)
    merged = merged.loc[:, ["tran_id", "GeneID", "geneSymbol"]].rename(columns={"GeneID": "gene_id", "geneSymbol": "gene_short_name"})
    return _annotate_gene_type(merged, gtf)


def preprocess_rmats_mxe(file: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    pos = df[df["strand"].astype(str) == "+"].copy()
    neg = df[df["strand"].astype(str) == "-"].copy()
    pos["1stExonStart_0base"] = pos["1stExonStart_0base"].astype(int) + 1
    pos["2ndExonStart_0base"] = pos["2ndExonStart_0base"].astype(int) + 1
    pos["upstreamES"] = pos["upstreamES"].astype(int) + 1
    pos["downstreamES"] = pos["downstreamES"].astype(int) + 1
    pos["tran_id"] = (
        pos["chr"].astype(str) + ":" + pos["upstreamES"].astype(str) + ":" + pos["upstreamEE"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["1stExonStart_0base"].astype(str) + ":" + pos["1stExonEnd"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["2ndExonStart_0base"].astype(str) + ":" + pos["2ndExonEnd"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["downstreamES"].astype(str) + ":" + pos["downstreamEE"].astype(str)
    )
    neg["1stExonStart_0base"] = neg["1stExonStart_0base"].astype(int) + 1
    neg["2ndExonStart_0base"] = neg["2ndExonStart_0base"].astype(int) + 1
    neg["upstreamES"] = neg["upstreamES"].astype(int) + 1
    neg["downstreamES"] = neg["downstreamES"].astype(int) + 1
    neg["tran_id"] = (
        neg["chr"].astype(str) + ":" + neg["downstreamES"].astype(str) + ":" + neg["downstreamEE"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["2ndExonStart_0base"].astype(str) + ":" + neg["2ndExonEnd"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["1stExonStart_0base"].astype(str) + ":" + neg["1stExonEnd"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["upstreamES"].astype(str) + ":" + neg["upstreamEE"].astype(str)
    )
    merged = pd.concat([pos, neg], ignore_index=True)
    merged = merged.loc[:, ["tran_id", "GeneID", "geneSymbol"]].rename(columns={"GeneID": "gene_id", "geneSymbol": "gene_short_name"})
    return _annotate_gene_type(merged, gtf)


def preprocess_rmats_ri(file: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    pos = df[df["strand"].astype(str) == "+"].copy()
    neg = df[df["strand"].astype(str) == "-"].copy()
    pos["riExonStart_0base"] = pos["riExonStart_0base"].astype(int) + 1
    pos["downstreamES"] = pos["downstreamES"].astype(int) + 1
    pos["tran_id"] = (
        pos["chr"].astype(str) + ":" + pos["riExonStart_0base"].astype(str) + ":" + pos["upstreamEE"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["downstreamES"].astype(str) + ":" + pos["riExonEnd"].astype(str)
    )
    neg["riExonStart_0base"] = neg["riExonStart_0base"].astype(int) + 1
    neg["downstreamES"] = neg["downstreamES"].astype(int) + 1
    neg["tran_id"] = (
        neg["chr"].astype(str) + ":" + neg["riExonEnd"].astype(str) + ":" + neg["downstreamES"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["upstreamEE"].astype(str) + ":" + neg["riExonStart_0base"].astype(str)
    )
    merged = pd.concat([pos, neg], ignore_index=True)
    merged = merged.loc[:, ["tran_id", "GeneID", "geneSymbol"]].rename(columns={"GeneID": "gene_id", "geneSymbol": "gene_short_name"})
    return _annotate_gene_type(merged, gtf)


def preprocess_rmats_a5ss(file: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    pos = df[df["strand"].astype(str) == "+"].copy()
    neg = df[df["strand"].astype(str) == "-"].copy()
    pos["longExonStart_0base"] = pos["longExonStart_0base"].astype(int) + 1
    pos["flankingES"] = pos["flankingES"].astype(int) + 1
    pos["tran_id"] = (
        pos["chr"].astype(str) + ":" + pos["longExonStart_0base"].astype(str) + ":" + pos["shortEE"].astype(str)
        + "|" + pos["longExonEnd"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["flankingES"].astype(str) + ":" + pos["flankingEE"].astype(str)
    )
    neg["longExonStart_0base"] = neg["longExonStart_0base"].astype(int) + 1
    neg["shortES"] = neg["shortES"].astype(int) + 1
    neg["flankingES"] = neg["flankingES"].astype(int) + 1
    neg["tran_id"] = (
        neg["chr"].astype(str) + ":" + neg["longExonEnd"].astype(str) + ":" + neg["longExonStart_0base"].astype(str)
        + "|" + neg["shortES"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["flankingES"].astype(str) + ":" + neg["flankingEE"].astype(str)
    )
    merged = pd.concat([pos, neg], ignore_index=True)
    merged = merged.loc[:, ["tran_id", "GeneID", "geneSymbol"]].rename(columns={"GeneID": "gene_id", "geneSymbol": "gene_short_name"})
    return _annotate_gene_type(merged, gtf)


def preprocess_rmats_a3ss(file: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    df = _strip_x_prefix(file)
    pos = df[df["strand"].astype(str) == "+"].copy()
    neg = df[df["strand"].astype(str) == "-"].copy()
    pos["longExonStart_0base"] = pos["longExonStart_0base"].astype(int) + 1
    pos["shortES"] = pos["shortES"].astype(int) + 1
    pos["flankingES"] = pos["flankingES"].astype(int) + 1
    pos["tran_id"] = (
        pos["chr"].astype(str) + ":" + pos["flankingES"].astype(str) + ":" + pos["flankingEE"].astype(str)
        + ":+@" + pos["chr"].astype(str) + ":" + pos["longExonStart_0base"].astype(str) + "|" + pos["shortES"].astype(str)
        + ":" + pos["longExonEnd"].astype(str)
    )
    neg["longExonStart_0base"] = neg["longExonStart_0base"].astype(int) + 1
    neg["flankingES"] = neg["flankingES"].astype(int) + 1
    neg["tran_id"] = (
        neg["chr"].astype(str) + ":" + neg["flankingES"].astype(str) + ":" + neg["flankingEE"].astype(str)
        + ":-@" + neg["chr"].astype(str) + ":" + neg["shortEE"].astype(str) + "|" + neg["longExonEnd"].astype(str)
        + ":" + neg["longExonStart_0base"].astype(str)
    )
    merged = pd.concat([pos, neg], ignore_index=True)
    merged = merged.loc[:, ["tran_id", "GeneID", "geneSymbol"]].rename(columns={"GeneID": "gene_id", "geneSymbol": "gene_short_name"})
    return _annotate_gene_type(merged, gtf)


def preprocess_rmats(file: pd.DataFrame, gtf: pd.DataFrame, *, event_type: str) -> pd.DataFrame:
    event_type = str(event_type).upper()
    if event_type == "SE":
        return preprocess_rmats_se(file, gtf)
    if event_type == "MXE":
        return preprocess_rmats_mxe(file, gtf)
    if event_type == "RI":
        return preprocess_rmats_ri(file, gtf)
    if event_type == "A5SS":
        return preprocess_rmats_a5ss(file, gtf)
    if event_type == "A3SS":
        return preprocess_rmats_a3ss(file, gtf)
    raise ValueError("event_type must be one of {'SE', 'MXE', 'RI', 'A5SS', 'A3SS'}")
