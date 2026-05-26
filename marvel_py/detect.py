from __future__ import annotations

import pandas as pd

from .utils import extract_gtf_attr


FEATURE_COLUMNS = ["tran_id", "event_type", "gene_id", "gene_short_name", "gene_type"]


def _empty_feature() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _parse_exons(gtf: pd.DataFrame) -> pd.DataFrame:
    exons = gtf[gtf["V3"] == "exon"].copy()
    exons["gene_id"] = exons["V9"].map(lambda value: extract_gtf_attr(value, "gene_id"))
    exons["transcript_id"] = exons["V9"].map(lambda value: extract_gtf_attr(value, "transcript_id"))
    exons["chrom"] = exons["V1"].astype(str)
    exons["strand"] = exons["V7"].astype(str)
    exons["start"] = exons["V4"].astype(int)
    exons["end"] = exons["V5"].astype(int)
    return exons[["chrom", "gene_id", "transcript_id", "strand", "start", "end"]]


def _filtered_exons(
    *,
    gtf: pd.DataFrame,
    exp: pd.DataFrame,
    sample_ids: list[str],
    min_cells: int,
    min_expr: float,
) -> pd.DataFrame:
    exons = _parse_exons(gtf)
    expr = exp.set_index("gene_id")[sample_ids].apply(pd.to_numeric, errors="coerce")
    keep_genes = expr.index[(expr >= min_expr).sum(axis=1) >= min_cells]
    return exons[exons["gene_id"].isin(keep_genes)].copy()


def _supported_junctions(splice_junction: pd.DataFrame) -> set[str]:
    if "coord.intron" not in splice_junction.columns:
        return set()

    sample_cols = [column for column in splice_junction.columns if column != "coord.intron"]
    if not sample_cols:
        return set()

    values = splice_junction[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    supported = splice_junction.loc[(values > 0).any(axis=1), "coord.intron"]
    return set(supported.astype(str).tolist())


def _terminal_coord(*, alt: pd.Series, common: pd.Series, event_type: str, strand: str) -> str:
    chrom = str(alt["chrom"])
    if event_type == "AFE" and strand == "+":
        return _coord(chrom, int(alt["end"]) + 1, int(common["start"]) - 1)
    if event_type == "AFE" and strand == "-":
        return _coord(chrom, int(common["end"]) + 1, int(alt["start"]) - 1)
    if event_type == "ALE" and strand == "+":
        return _coord(chrom, int(common["end"]) + 1, int(alt["start"]) - 1)
    return _coord(chrom, int(alt["end"]) + 1, int(common["start"]) - 1)


def _terminal_junctions(exons: pd.DataFrame, *, event_type: str, supported_junctions: set[str]) -> pd.DataFrame:
    records = []

    for (_, _), transcript_exons in exons.groupby(["gene_id", "transcript_id"], sort=False):
        strand = transcript_exons["strand"].iloc[0]
        ordered = transcript_exons.sort_values("start", ascending=strand == "+").reset_index(drop=True)
        if len(ordered) < 2:
            continue

        if event_type == "AFE":
            alt = ordered.iloc[0]
            common = ordered.iloc[1]
            group_coord = common["start"] if strand == "+" else common["end"]
        else:
            alt = ordered.iloc[-1]
            common = ordered.iloc[-2]
            group_coord = common["end"] if strand == "+" else common["start"]

        coord_intron = _terminal_coord(alt=alt, common=common, event_type=event_type, strand=strand)
        if coord_intron not in supported_junctions:
            continue

        records.append(
            {
                "gene_id": alt["gene_id"],
                "chrom": alt["chrom"],
                "strand": strand,
                "common_start": int(common["start"]),
                "common_end": int(common["end"]),
                "alt_start": int(alt["start"]),
                "alt_end": int(alt["end"]),
                "group_coord": int(group_coord),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["gene_id", "chrom", "strand", "common_start", "common_end", "alt_start", "alt_end", "group_coord"]
        )

    return pd.DataFrame(records).drop_duplicates()


def _sort_group(group: pd.DataFrame, *, event_type: str, strand: str) -> pd.DataFrame:
    if event_type == "AFE" and strand == "+":
        return group.sort_values(["alt_end", "alt_start"], ascending=[True, True]).reset_index(drop=True)
    if event_type == "AFE" and strand == "-":
        return group.sort_values(["alt_start", "alt_end"], ascending=[False, False]).reset_index(drop=True)
    if event_type == "ALE" and strand == "+":
        return group.sort_values(["alt_start", "alt_end"], ascending=[False, False]).reset_index(drop=True)
    return group.sort_values(["alt_end", "alt_start"], ascending=[True, True]).reset_index(drop=True)


def _coord(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start}:{end}"


def _build_tran_id(base: pd.Series, other: pd.Series, *, event_type: str) -> str:
    common = _coord(base["chrom"], int(base["common_start"]), int(base["common_end"]))
    base_alt = f"{int(base['alt_start'])}:{int(base['alt_end'])}"
    other_alt = f"{int(other['alt_start'])}:{int(other['alt_end'])}"
    strand = str(base["strand"])

    if event_type == "AFE":
        return f"{base['chrom']}:{base_alt}|{other_alt}:{strand}@{common}"
    return f"{common}:{strand}@{base['chrom']}:{other_alt}|{base_alt}"


def detect_terminal_events(
    *,
    gtf: pd.DataFrame,
    exp: pd.DataFrame,
    gene_feature: pd.DataFrame,
    splice_junction: pd.DataFrame,
    sample_ids: list[str],
    min_cells: int,
    min_expr: float,
    event_type: str,
) -> pd.DataFrame:
    event_type = event_type.upper()
    exons = _filtered_exons(gtf=gtf, exp=exp, sample_ids=sample_ids, min_cells=min_cells, min_expr=min_expr)
    if exons.empty:
        return _empty_feature()

    junctions = _terminal_junctions(
        exons,
        event_type=event_type,
        supported_junctions=_supported_junctions(splice_junction),
    )
    if junctions.empty:
        return _empty_feature()

    records = []
    group_cols = ["gene_id", "chrom", "strand", "common_start", "common_end", "group_coord"]
    for _, group in junctions.groupby(group_cols, sort=False):
        if len(group) < 2:
            continue

        strand = str(group["strand"].iloc[0])
        group = _sort_group(group, event_type=event_type, strand=strand)
        base = group.iloc[0]

        for idx in range(1, len(group)):
            records.append({"tran_id": _build_tran_id(base, group.iloc[idx], event_type=event_type), "gene_id": base["gene_id"]})

    if not records:
        return _empty_feature()

    detected = pd.DataFrame(records).drop_duplicates()
    gene_meta = gene_feature[["gene_id", "gene_short_name", "gene_type"]].drop_duplicates()
    detected = detected.merge(gene_meta, on="gene_id", how="left")
    detected.insert(1, "event_type", event_type)
    return detected[FEATURE_COLUMNS].reset_index(drop=True)
