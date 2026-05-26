from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .matrix import Marvel10x
from .models import MarvelPlate
from .psi import PLATE_EVENT_TYPES, empty_feature
from .utils import maybe_read_table

__all__ = [
    "create_marvel_object",
    "create_marvel_object_10x",
    "maybe_read_table",
]


def _normalize_gtf(gtf) -> pd.DataFrame | None:
    if gtf is None:
        return None

    if isinstance(gtf, pd.DataFrame):
        gtf_df = maybe_read_table(gtf, dtype=str)
    else:
        gtf_df = maybe_read_table(gtf, header=None, comment="#", dtype=str)
    expected_cols = [f"V{i}" for i in range(1, 10)]
    if list(gtf_df.columns) != expected_cols and len(gtf_df.columns) == len(expected_cols):
        gtf_df = gtf_df.copy()
        gtf_df.columns = expected_cols
    return gtf_df


def _normalize_splice_feature(splice_feature: Any) -> dict[str, pd.DataFrame]:
    if isinstance(splice_feature, dict):
        canonical_event_types = set(PLATE_EVENT_TYPES)
        invalid_keys = [key for key in splice_feature if key not in canonical_event_types]
        if invalid_keys:
            raise KeyError(
                "splice_feature contains invalid event type keys: " + ", ".join(sorted(map(str, invalid_keys)))
            )
        tables = {}
        for event_type in PLATE_EVENT_TYPES:
            value = splice_feature.get(event_type)
            tables[event_type] = empty_feature() if value is None else maybe_read_table(value, dtype=str)
        return tables
    raise TypeError("splice_feature must be a dict keyed by MARVEL event type")


def create_marvel_object(
    *,
    splice_junction,
    splice_pheno,
    splice_feature,
    intron_counts=None,
    gene_feature,
    exp,
    gtf=None,
) -> MarvelPlate:
    return MarvelPlate(
        splice_pheno=maybe_read_table(splice_pheno, dtype=str),
        splice_junction=maybe_read_table(splice_junction),
        intron_counts=None if intron_counts is None else maybe_read_table(intron_counts),
        splice_feature=_normalize_splice_feature(splice_feature),
        gene_feature=maybe_read_table(gene_feature, dtype=str),
        exp=maybe_read_table(exp),
        gtf=_normalize_gtf(gtf),
    )


def create_marvel_object_10x(
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
    gtf=None,
    pca=None,
) -> Marvel10x:
    path_inputs = [
        gene_norm_matrix,
        gene_norm_pheno,
        gene_norm_feature,
        gene_count_matrix,
        gene_count_pheno,
        gene_count_feature,
        sj_count_matrix,
        sj_count_pheno,
        sj_count_feature,
    ]
    if (
        all(isinstance(value, (str, Path)) for value in path_inputs)
        and (gtf is None or isinstance(gtf, (str, Path)))
        and (pca is None or isinstance(pca, (str, Path)))
    ):
        return Marvel10x.from_paths(
            gene_norm_matrix=gene_norm_matrix,
            gene_norm_pheno=gene_norm_pheno,
            gene_norm_feature=gene_norm_feature,
            gene_count_matrix=gene_count_matrix,
            gene_count_pheno=gene_count_pheno,
            gene_count_feature=gene_count_feature,
            sj_count_matrix=sj_count_matrix,
            sj_count_pheno=sj_count_pheno,
            sj_count_feature=sj_count_feature,
            pca=pca,
            gtf=gtf,
        )
    return Marvel10x.from_data(
        gene_norm_matrix=gene_norm_matrix,
        gene_norm_pheno=gene_norm_pheno,
        gene_norm_feature=gene_norm_feature,
        gene_count_matrix=gene_count_matrix,
        gene_count_pheno=gene_count_pheno,
        gene_count_feature=gene_count_feature,
        sj_count_matrix=sj_count_matrix,
        sj_count_pheno=sj_count_pheno,
        sj_count_feature=sj_count_feature,
        pca=pca,
        gtf=gtf,
    )
