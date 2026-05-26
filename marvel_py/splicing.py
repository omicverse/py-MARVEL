from __future__ import annotations

from .detect import detect_terminal_events
from .models import MarvelPlate
from .alignment import _invalidate_plate_state
from .psi import PLATE_EVENT_TYPES, compute_psi_event, compute_psi_posterior_event, ri_intron_matrix, sj_matrix

__all__ = [
    "compute_psi",
    "compute_psi_posterior",
    "detect_events",
]


def _plate_table_cache_key(frame) -> tuple[int, tuple[int, int], tuple[str, ...]]:
    return (id(frame), tuple(frame.shape), tuple(str(column) for column in frame.columns))


def _cached_plate_sj_matrix(marvel_object: MarvelPlate):
    key = _plate_table_cache_key(marvel_object.splice_junction)
    cached = marvel_object._splice_junction_numeric_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    numeric = sj_matrix(marvel_object.splice_junction)
    marvel_object._splice_junction_numeric_cache = (key, numeric)
    return numeric


def _cached_plate_intron_norm(marvel_object: MarvelPlate, read_length: float):
    if marvel_object.intron_counts is None:
        return None
    source_key = _plate_table_cache_key(marvel_object.intron_counts)
    cache_key = (float(read_length), source_key)
    cached = marvel_object._intron_norm_cache.get(cache_key)
    if cached is not None:
        return cached
    numeric = ri_intron_matrix(marvel_object.intron_counts, read_length)
    marvel_object._intron_norm_cache = {cache_key: numeric}
    return numeric


def _compute_psi_inplace(
    marvel_object: MarvelPlate,
    *,
    coverage_threshold: float,
    event_type: str,
    uneven_coverage_multiplier: float = 10.0,
    read_length: float = 1.0,
) -> MarvelPlate:
    normalized_event_type = str(event_type).upper()
    sj_numeric = _cached_plate_sj_matrix(marvel_object)
    intron_norm = _cached_plate_intron_norm(marvel_object, read_length) if normalized_event_type == "RI" else None
    feature, counts, psi = compute_psi_event(
        event_type=normalized_event_type,
        splice_feature=marvel_object.splice_feature,
        splice_junction=marvel_object.splice_junction,
        intron_counts=marvel_object.intron_counts,
        splice_junction_numeric=sj_numeric,
        intron_norm=intron_norm,
        coverage_threshold=coverage_threshold,
        uneven_coverage_multiplier=uneven_coverage_multiplier,
        read_length=read_length,
    )
    marvel_object.splice_feature_validated[normalized_event_type] = feature
    marvel_object.counts[normalized_event_type] = counts
    marvel_object.psi[normalized_event_type] = psi
    return marvel_object


def _compute_psi_posterior_inplace(
    marvel_object: MarvelPlate,
    *,
    event_type: str | None = None,
) -> MarvelPlate:
    event_types = PLATE_EVENT_TYPES if event_type is None else [str(event_type).upper()]
    for current_event_type in event_types:
        if current_event_type not in marvel_object.counts or not marvel_object.counts.get(current_event_type):
            raise ValueError(f"compute_psi must run before compute_psi_posterior for {current_event_type}")
        marvel_object.psi_posterior[current_event_type] = compute_psi_posterior_event(
            event_type=current_event_type,
            counts=marvel_object.counts,
        )
    return marvel_object


def compute_psi(
    marvel_object: MarvelPlate,
    *,
    coverage_threshold: float,
    event_type: str,
    uneven_coverage_multiplier: float = 10.0,
    read_length: float = 1.0,
    thread: int | None = None,
) -> MarvelPlate:
    _ = thread
    return _compute_psi_inplace(
        marvel_object,
        event_type=event_type,
        coverage_threshold=coverage_threshold,
        uneven_coverage_multiplier=uneven_coverage_multiplier,
        read_length=read_length,
    )


def compute_psi_posterior(
    marvel_object: MarvelPlate,
    *,
    event_type: str | None = None,
) -> MarvelPlate:
    return _compute_psi_posterior_inplace(marvel_object, event_type=event_type)


def detect_events(
    marvel_object: MarvelPlate,
    *,
    event_type: str,
    min_cells: int,
    min_expr: float,
    track_progress: bool = False,
) -> MarvelPlate:
    _ = track_progress
    event_type = event_type.upper()
    if event_type not in {"AFE", "ALE"}:
        raise ValueError("Stage 1 detect_events supports only AFE and ALE")
    if marvel_object.gtf is None:
        raise ValueError("GTF is required for detect_events")
    sample_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
    marvel_object.splice_feature[event_type] = detect_terminal_events(
        gtf=marvel_object.gtf,
        exp=marvel_object.exp,
        gene_feature=marvel_object.gene_feature,
        splice_junction=marvel_object.splice_junction,
        sample_ids=sample_ids,
        min_cells=min_cells,
        min_expr=min_expr,
        event_type=event_type,
    )
    _invalidate_plate_state(marvel_object, event_types={event_type}, clear_psi=True)
    return marvel_object
