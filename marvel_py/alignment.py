from __future__ import annotations

import pandas as pd

from .models import MarvelPlate
from .utils import ordered_intersection
from .psi import PLATE_EVENT_TYPES, empty_feature


def _sample_columns(frame: pd.DataFrame, sample_ids: set[str], *, exclude: set[str]) -> list[str]:
    return [column for column in frame.columns if column not in exclude and str(column) in sample_ids]


def _align_count_tables(
    count_tables: dict[str, pd.DataFrame] | None,
    *,
    sample_ids: set[str],
    tran_ids: list[str] | None,
) -> dict[str, pd.DataFrame] | None:
    if count_tables is None:
        return None

    aligned: dict[str, pd.DataFrame] = {}
    tran_ids = None if tran_ids is None else [str(tran_id) for tran_id in tran_ids]
    for key, frame in count_tables.items():
        if frame is None or frame.empty or "tran_id" not in frame.columns:
            aligned[key] = frame.copy() if isinstance(frame, pd.DataFrame) else frame
            continue

        sample_cols = _sample_columns(frame, sample_ids, exclude={"tran_id"})
        trimmed = frame.loc[:, ["tran_id", *sample_cols]].copy()
        if tran_ids is not None:
            row_ids = ordered_intersection(tran_ids, trimmed["tran_id"].astype(str).tolist())
            trimmed = trimmed.set_index("tran_id").loc[row_ids].reset_index()
        aligned[key] = trimmed
    return aligned


def _invalidate_plate_state(
    marvel_object: MarvelPlate,
    *,
    event_types: set[str] | None = None,
    clear_psi: bool = False,
    clear_gene: bool = False,
    preserved_validated: dict[str, pd.DataFrame] | None = None,
    preserve_counts: bool = False,
) -> None:
    event_types = set(PLATE_EVENT_TYPES) if event_types is None else set(event_types)
    for event_type in event_types:
        if preserved_validated is not None and event_type in preserved_validated:
            marvel_object.splice_feature_validated[event_type] = preserved_validated[event_type].copy()
        else:
            marvel_object.splice_feature_validated[event_type] = empty_feature()
        if not preserve_counts:
            marvel_object.counts.pop(event_type, None)
        if clear_psi:
            marvel_object.psi[event_type] = marvel_object.psi[event_type].iloc[0:0].copy()
            marvel_object.psi_posterior[event_type] = marvel_object.psi_posterior[event_type].iloc[0:0].copy()

    marvel_object.modality_results = None
    marvel_object.modality_prop = None
    marvel_object.de_splicing = {}
    marvel_object.de_plots = {}
    marvel_object.n_events = {}
    marvel_object.de_spliced_gene = None
    if clear_gene:
        marvel_object.de_gene = None


def check_alignment(marvel_object: MarvelPlate, *, level: str) -> MarvelPlate:
    return marvel_object.check_alignment(level)


def subset_samples(marvel_object: MarvelPlate, *, sample_ids: list[str]) -> MarvelPlate:
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    sample_id_set = set(sample_ids)
    pheno_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
    overlap = [sample_id for sample_id in pheno_ids if sample_id in sample_id_set]
    overlap_set = set(overlap)

    marvel_object.splice_pheno = marvel_object.splice_pheno[
        marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
    ].copy()

    if "coord.intron" in marvel_object.splice_junction.columns:
        sample_cols = _sample_columns(marvel_object.splice_junction, overlap_set, exclude={"coord.intron"})
        marvel_object.splice_junction = marvel_object.splice_junction.loc[:, ["coord.intron", *sample_cols]].copy()

    if marvel_object.intron_counts is not None and "coord.intron" in marvel_object.intron_counts.columns:
        sample_cols = _sample_columns(marvel_object.intron_counts, overlap_set, exclude={"coord.intron"})
        marvel_object.intron_counts = marvel_object.intron_counts.loc[:, ["coord.intron", *sample_cols]].copy()

    if "gene_id" in marvel_object.exp.columns:
        sample_cols = _sample_columns(marvel_object.exp, overlap_set, exclude={"gene_id"})
        marvel_object.exp = marvel_object.exp.loc[:, ["gene_id", *sample_cols]].copy()

    preserved_validated: dict[str, pd.DataFrame] = {}
    preserved_counts: dict[str, dict[str, pd.DataFrame]] = {}
    for event_type in PLATE_EVENT_TYPES:
        psi_df = marvel_object.psi.get(event_type)
        if psi_df is None or "tran_id" not in psi_df.columns:
            continue
        sample_cols = _sample_columns(psi_df, overlap_set, exclude={"tran_id"})
        trimmed_psi = psi_df.loc[:, ["tran_id", *sample_cols]].copy()
        marvel_object.psi[event_type] = trimmed_psi

        posterior_df = marvel_object.psi_posterior.get(event_type)
        if posterior_df is not None and not posterior_df.empty and "tran_id" in posterior_df.columns:
            posterior_sample_cols = _sample_columns(posterior_df, overlap_set, exclude={"tran_id"})
            posterior_tran_ids = ordered_intersection(
                trimmed_psi["tran_id"].astype(str).tolist(),
                posterior_df["tran_id"].astype(str).tolist(),
            )
            marvel_object.psi_posterior[event_type] = (
                posterior_df.set_index("tran_id").loc[posterior_tran_ids, posterior_sample_cols].reset_index()
            )

        feature_df = marvel_object.splice_feature_validated.get(event_type)
        source_feature_df = marvel_object.splice_feature.get(event_type)
        if (
            feature_df is not None
            and not feature_df.empty
            and "tran_id" in feature_df.columns
            and source_feature_df is not None
            and not source_feature_df.empty
            and "tran_id" in source_feature_df.columns
        ):
            # Keep validated rows only when they still map to the original feature set.
            tran_overlap = trimmed_psi["tran_id"].astype(str).tolist()
            source_tran_ids = set(source_feature_df["tran_id"].astype(str).tolist())
            valid_tran_ids = feature_df["tran_id"].astype(str).isin(tran_overlap) & feature_df["tran_id"].astype(str).isin(source_tran_ids)
            preserved_validated[event_type] = feature_df[valid_tran_ids].copy()

        count_tables = marvel_object.counts.get(event_type)
        if count_tables:
            preserved_counts[event_type] = _align_count_tables(
                count_tables,
                sample_ids=overlap_set,
                tran_ids=trimmed_psi["tran_id"].astype(str).tolist(),
            )

    _invalidate_plate_state(
        marvel_object,
        clear_gene=True,
        clear_psi=False,
        preserved_validated=preserved_validated,
        preserve_counts=True,
    )
    for event_type, count_tables in preserved_counts.items():
        marvel_object.counts[event_type] = count_tables
    return marvel_object


def transform_exp_values(
    marvel_object: MarvelPlate,
    *,
    offset: float = 1.0,
    transformation: str = "log2",
    threshold_lower: float = 1.0,
) -> MarvelPlate:
    return marvel_object.transform_exp_values(
        offset=offset,
        transformation=transformation,
        threshold_lower=threshold_lower,
    )
