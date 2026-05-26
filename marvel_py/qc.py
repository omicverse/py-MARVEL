from __future__ import annotations

import numpy as np
import pandas as pd

from .matrix import Marvel10x
from .models import MarvelPlate
from .utils import ordered_intersection
from .alignment import subset_samples

__all__ = [
    "check_alignment",
    "subset_samples",
    "transform_exp_values",
    "validate_sj_10x",
    "filter_genes_10x",
    "check_alignment_10x",
]


def _transform_exp_values_inplace(
    marvel_object: MarvelPlate,
    *,
    offset: float = 1.0,
    transformation: str = "log2",
    threshold_lower: float = 1.0,
) -> MarvelPlate:
    sample_cols = [column for column in marvel_object.exp.columns if column != "gene_id"]
    max_value = float(np.nanmax(marvel_object.exp[sample_cols].to_numpy(dtype=float))) if sample_cols else 0.0
    if max_value < 20:
        return marvel_object

    values = marvel_object.exp[sample_cols].astype(float) + offset
    if transformation == "log2":
        values = np.log2(values)
    elif transformation == "log10":
        values = np.log10(values)
    else:
        raise ValueError(f"Unsupported transformation: {transformation}")
    values = values.mask(values < threshold_lower, 0.0)
    marvel_object.exp.loc[:, sample_cols] = values
    return marvel_object


def _check_alignment_inplace(marvel_object: MarvelPlate, *, level: str) -> MarvelPlate:
    if level == "SJ":
        pheno_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
        sj_ids = [str(column) for column in marvel_object.splice_junction.columns if column != "coord.intron"]
        overlap = ordered_intersection(pheno_ids, sj_ids)
        marvel_object.splice_pheno = marvel_object.splice_pheno[
            marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
        ].copy()
        marvel_object.splice_junction = marvel_object.splice_junction.loc[:, ["coord.intron", *overlap]].copy()

        if marvel_object.intron_counts is not None:
            intron_ids = [str(column) for column in marvel_object.intron_counts.columns if column != "coord.intron"]
            overlap = ordered_intersection(overlap, intron_ids)
            marvel_object.splice_pheno = marvel_object.splice_pheno[
                marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
            ].copy()
            marvel_object.splice_junction = marvel_object.splice_junction.loc[:, ["coord.intron", *overlap]].copy()
            marvel_object.intron_counts = marvel_object.intron_counts.loc[:, ["coord.intron", *overlap]].copy()
        return marvel_object

    if level == "splicing":
        event_types = [event for event, df in marvel_object.splice_feature_validated.items() if not df.empty]
        if not event_types:
            return marvel_object
        pheno_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
        psi_ids = []
        for event in event_types:
            psi_ids.extend([str(column) for column in marvel_object.psi[event].columns if column != "tran_id"])
        overlap = ordered_intersection(pheno_ids, pd.unique(pd.Series(psi_ids)).tolist())
        marvel_object.splice_pheno = marvel_object.splice_pheno[
            marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
        ].copy()
        for event in event_types:
            marvel_object.psi[event] = marvel_object.psi[event].loc[:, ["tran_id", *overlap]].copy()
            feature_df = marvel_object.splice_feature_validated[event]
            tran_overlap = ordered_intersection(
                feature_df["tran_id"].astype(str).tolist(),
                marvel_object.psi[event]["tran_id"].astype(str).tolist(),
            )
            marvel_object.splice_feature_validated[event] = (
                feature_df.set_index("tran_id").loc[tran_overlap].reset_index()
            )
            marvel_object.psi[event] = marvel_object.psi[event].set_index("tran_id").loc[tran_overlap].reset_index()
        return marvel_object

    if level == "gene":
        pheno_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
        exp_ids = [str(column) for column in marvel_object.exp.columns if column != "gene_id"]
        overlap = ordered_intersection(pheno_ids, exp_ids)
        marvel_object.splice_pheno = marvel_object.splice_pheno[
            marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
        ].copy()
        marvel_object.exp = marvel_object.exp.loc[:, ["gene_id", *overlap]].copy()

        gene_overlap = ordered_intersection(
            marvel_object.gene_feature["gene_id"].astype(str).tolist(),
            marvel_object.exp["gene_id"].astype(str).tolist(),
        )
        marvel_object.gene_feature = marvel_object.gene_feature.set_index("gene_id").loc[gene_overlap].reset_index()
        marvel_object.exp = marvel_object.exp.set_index("gene_id").loc[gene_overlap].reset_index()
        return marvel_object

    if level == "splicing and gene":
        psi_ids = []
        for event, df in marvel_object.psi.items():
            if not df.empty:
                psi_ids.extend([str(column) for column in df.columns if column != "tran_id"])
        if not psi_ids:
            return marvel_object
        exp_ids = [str(column) for column in marvel_object.exp.columns if column != "gene_id"]
        overlap = ordered_intersection(exp_ids, pd.unique(pd.Series(psi_ids)).tolist())
        marvel_object.exp = marvel_object.exp.loc[:, ["gene_id", *overlap]].copy()
        for event, df in marvel_object.psi.items():
            if not df.empty:
                marvel_object.psi[event] = df.loc[:, ["tran_id", *overlap]].copy()
        marvel_object.splice_pheno = marvel_object.splice_pheno[
            marvel_object.splice_pheno["sample.id"].astype(str).isin(overlap)
        ].copy()
        return marvel_object

    raise ValueError(f"Unsupported alignment level: {level}")


def _validate_sj_10x_inplace(marvel_object: Marvel10x, *, keep_novel_sj: bool = False) -> Marvel10x:
    if marvel_object.sj_metadata is None:
        raise ValueError("annotate_sj must run before validate_sj")

    sj_types_1 = {"start_known.single.gene|end_known.single.gene|same"}
    sj_types_2 = {
        "start_known.single.gene|end_unknown.gene",
        "start_unknown.gene|end_known.single.gene",
    }

    sj_metadata = marvel_object.sj_metadata.copy()
    if keep_novel_sj:
        sj_metadata = sj_metadata[sj_metadata["sj.type"].isin(sj_types_1 | sj_types_2)].copy()
        start_unknown = sj_metadata["sj.type"] == "start_unknown.gene|end_known.single.gene"
        end_unknown = sj_metadata["sj.type"] == "start_known.single.gene|end_unknown.gene"
        sj_metadata.loc[start_unknown, "gene_short_name.start"] = sj_metadata.loc[
            start_unknown, "gene_short_name.end"
        ]
        sj_metadata.loc[end_unknown, "gene_short_name.end"] = sj_metadata.loc[
            end_unknown, "gene_short_name.start"
        ]
    else:
        sj_metadata = sj_metadata[sj_metadata["sj.type"].isin(sj_types_1)].copy()

    coord_intron = sj_metadata["coord.intron"].astype(str).tolist()
    marvel_object.sj_metadata = sj_metadata.reset_index(drop=True)
    marvel_object.sj_count_matrix = marvel_object.sj_count_matrix.subset_rows(coord_intron)
    return marvel_object


def _filter_genes_10x_inplace(marvel_object: Marvel10x, *, gene_type: str = "protein_coding") -> Marvel10x:
    if marvel_object.sj_metadata is None:
        raise ValueError("validate_sj must run before filter_genes")

    gene_metadata = marvel_object.gene_metadata[marvel_object.gene_metadata["gene_type"] == gene_type].copy()
    keep_genes = gene_metadata["gene_short_name"].astype(str).tolist()
    marvel_object.gene_metadata = gene_metadata.reset_index(drop=True)
    marvel_object.gene_norm_matrix = marvel_object.gene_norm_matrix.subset_rows(keep_genes)

    sj_metadata = marvel_object.sj_metadata[marvel_object.sj_metadata["gene_short_name.start"].isin(keep_genes)].copy()
    keep_sj = sj_metadata["coord.intron"].astype(str).tolist()
    marvel_object.sj_metadata = sj_metadata.reset_index(drop=True)
    marvel_object.sj_count_matrix = marvel_object.sj_count_matrix.subset_rows(keep_sj)
    return marvel_object


def _check_alignment_10x_inplace(marvel_object: Marvel10x) -> Marvel10x:
    if marvel_object.sj_metadata is None:
        raise ValueError("filter_genes must run before check_alignment")

    gene_overlap = ordered_intersection(marvel_object.gene_norm_matrix.row_ids, marvel_object.gene_count_matrix.row_ids)
    marvel_object.gene_norm_matrix = marvel_object.gene_norm_matrix.subset_rows(gene_overlap)
    marvel_object.gene_count_matrix = marvel_object.gene_count_matrix.subset_rows(gene_overlap)
    marvel_object.gene_metadata = (
        marvel_object.gene_metadata.set_index("gene_short_name").loc[gene_overlap].reset_index()
    )

    genes_with_sj = ordered_intersection(
        marvel_object.gene_metadata["gene_short_name"].astype(str).tolist(),
        marvel_object.sj_metadata["gene_short_name.start"].astype(str).tolist(),
    )
    marvel_object.gene_norm_matrix = marvel_object.gene_norm_matrix.subset_rows(genes_with_sj)
    marvel_object.gene_count_matrix = marvel_object.gene_count_matrix.subset_rows(genes_with_sj)
    marvel_object.gene_metadata = (
        marvel_object.gene_metadata.set_index("gene_short_name").loc[genes_with_sj].reset_index()
    )

    marvel_object.sj_metadata = marvel_object.sj_metadata[
        marvel_object.sj_metadata["gene_short_name.start"].isin(genes_with_sj)
    ].copy()
    marvel_object.sj_metadata = (
        marvel_object.sj_metadata.set_index("coord.intron").loc[marvel_object.sj_metadata["coord.intron"]].reset_index()
    )
    marvel_object.sj_count_matrix = marvel_object.sj_count_matrix.subset_rows(marvel_object.sj_metadata["coord.intron"].tolist())

    cell_overlap = ordered_intersection(
        marvel_object.gene_norm_matrix.col_ids,
        ordered_intersection(marvel_object.gene_count_matrix.col_ids, marvel_object.sj_count_matrix.col_ids),
    )
    marvel_object.gene_norm_matrix = marvel_object.gene_norm_matrix.subset_cols(cell_overlap)
    marvel_object.gene_count_matrix = marvel_object.gene_count_matrix.subset_cols(cell_overlap)
    marvel_object.sj_count_matrix = marvel_object.sj_count_matrix.subset_cols(cell_overlap)

    sample_metadata = marvel_object.sample_metadata.copy()
    sample_metadata["cell.id"] = sample_metadata["cell.id"].astype(str)
    marvel_object.sample_metadata = sample_metadata.set_index("cell.id").loc[cell_overlap].reset_index()

    if marvel_object.pca is not None and "cell.id" in marvel_object.pca.columns:
        pca = marvel_object.pca.copy()
        pca["cell.id"] = pca["cell.id"].astype(str)
        overlap = ordered_intersection(pca["cell.id"].tolist(), cell_overlap)
        marvel_object.pca = pca.set_index("cell.id").loc[overlap].reset_index()
    return marvel_object


def check_alignment(marvel_object: MarvelPlate, *, level: str) -> MarvelPlate:
    return marvel_object.check_alignment(level)


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


def validate_sj_10x(marvel_object: Marvel10x, *, keep_novel_sj: bool = False) -> Marvel10x:
    marvel_object.validate_sj(keep_novel_sj=keep_novel_sj)
    return marvel_object


def filter_genes_10x(marvel_object: Marvel10x, *, gene_type: str = "protein_coding") -> Marvel10x:
    marvel_object.filter_genes(gene_type=gene_type)
    return marvel_object


def check_alignment_10x(marvel_object: Marvel10x) -> Marvel10x:
    marvel_object.check_alignment()
    return marvel_object
