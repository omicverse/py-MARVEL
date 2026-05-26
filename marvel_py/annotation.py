from __future__ import annotations

import pandas as pd

from .utils import extract_gtf_attr, ordered_intersection
from .matrix import Marvel10x
from .misc import parse_gtf

__all__ = [
    "annotate_genes_10x",
    "annotate_sj_10x",
    "parse_gtf",
]


def _classify_sj(start_value: str | float | None, end_value: str | float | None) -> str:
    start_is_na = pd.isna(start_value)
    end_is_na = pd.isna(end_value)
    start_multi = (not start_is_na) and ("|" in str(start_value))
    end_multi = (not end_is_na) and ("|" in str(end_value))

    if not start_is_na and not end_is_na and not start_multi and not end_multi and start_value == end_value:
        return "start_known.single.gene|end_known.single.gene|same"
    if not start_is_na and not end_is_na and not start_multi and not end_multi and start_value != end_value:
        return "start_known.single.gene|end_known.single.gene|different"
    if start_is_na and end_is_na:
        return "start_unknown.gene|end_unknown.gene"
    if start_is_na and not end_is_na and not end_multi:
        return "start_unknown.gene|end_known.single.gene"
    if start_is_na and not end_is_na and end_multi:
        return "start_unknown.gene|end_known.multi.gene"
    if end_is_na and not start_is_na and not start_multi:
        return "start_known.single.gene|end_unknown.gene"
    if end_is_na and not start_is_na and start_multi:
        return "start_known.multi.gene|end_unknown.gene"
    if not start_is_na and not end_is_na and start_multi and end_multi:
        return "start_known.multi.gene|end_known.multi.gene"
    if not start_is_na and not end_is_na and start_multi and not end_multi:
        return "start_known.multi.gene|start_known.single.gene"
    return "start_known.single.gene|end_known.multi.gene"


def _annotate_genes_10x_inplace(marvel_object: Marvel10x) -> Marvel10x:
    gtf_genes = marvel_object.gtf[marvel_object.gtf["V3"] == "gene"].copy()
    gtf_genes["gene_short_name"] = gtf_genes["V9"].map(
        lambda value: extract_gtf_attr(value, "gene_name")
    )
    gtf_genes["gene_type"] = gtf_genes["V9"].map(
        lambda value: extract_gtf_attr(value, "gene_biotype") or extract_gtf_attr(value, "gene_type")
    )
    gtf_genes = gtf_genes[["gene_short_name", "gene_type"]].drop_duplicates()

    overlap = ordered_intersection(
        marvel_object.gene_metadata["gene_short_name"].astype(str).tolist(),
        gtf_genes["gene_short_name"].astype(str).tolist(),
    )
    gene_metadata = marvel_object.gene_metadata.copy()
    gene_metadata = gene_metadata[gene_metadata["gene_short_name"].isin(overlap)].copy()
    gene_metadata["gene_short_name"] = gene_metadata["gene_short_name"].astype(str)
    gene_metadata = gene_metadata.set_index("gene_short_name").loc[overlap].reset_index()
    gene_metadata = gene_metadata.merge(gtf_genes, on="gene_short_name", how="left")

    marvel_object.gene_metadata = gene_metadata
    marvel_object.gene_norm_matrix = marvel_object.gene_norm_matrix.subset_rows(overlap)
    return marvel_object


def _annotate_sj_10x_inplace(marvel_object: Marvel10x) -> Marvel10x:
    sj_ids = marvel_object.sj_count_matrix.row_ids.astype(str)
    parts = [value.split(":") for value in sj_ids]
    df = pd.DataFrame(
        {
            "coord.intron": sj_ids,
            "chr": [value[0] for value in parts],
            "start": [value[1] for value in parts],
            "end": [value[2] for value in parts],
        }
    )

    exon_gtf = marvel_object.gtf[marvel_object.gtf["V3"] == "exon"].copy()
    exon_gtf["gene_short_name"] = exon_gtf["V9"].map(lambda value: extract_gtf_attr(value, "gene_name"))
    exon_gtf["start_intron"] = (exon_gtf["V5"].astype(int) + 1).astype(str)
    exon_gtf["end_intron"] = (exon_gtf["V4"].astype(int) - 1).astype(str)

    has_chr_prefix = exon_gtf.iloc[0]["V1"].startswith("chr")
    chr_prefix = "" if has_chr_prefix else "chr"

    start_map = (
        exon_gtf.assign(chr_pos=chr_prefix + exon_gtf["V1"].astype(str) + ":" + exon_gtf["start_intron"])
        .groupby("chr_pos")["gene_short_name"]
        .apply(lambda values: "|".join(sorted(set(values))))
        .to_dict()
    )
    end_map = (
        exon_gtf.assign(chr_pos=chr_prefix + exon_gtf["V1"].astype(str) + ":" + exon_gtf["end_intron"])
        .groupby("chr_pos")["gene_short_name"]
        .apply(lambda values: "|".join(sorted(set(values))))
        .to_dict()
    )

    df["chr.start"] = df["chr"] + ":" + df["start"]
    df["chr.end"] = df["chr"] + ":" + df["end"]
    df["gene_short_name.start"] = df["chr.start"].map(start_map)
    df["gene_short_name.end"] = df["chr.end"].map(end_map)
    df["sj.type"] = [
        _classify_sj(start_value, end_value)
        for start_value, end_value in zip(df["gene_short_name.start"], df["gene_short_name.end"])
    ]
    marvel_object.sj_metadata = df[
        ["coord.intron", "gene_short_name.start", "gene_short_name.end", "sj.type"]
    ].copy()
    return marvel_object


def annotate_genes_10x(marvel_object: Marvel10x) -> Marvel10x:
    marvel_object.annotate_genes()
    return marvel_object


def annotate_sj_10x(marvel_object: Marvel10x) -> Marvel10x:
    marvel_object.annotate_sj()
    return marvel_object
