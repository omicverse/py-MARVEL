from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


PLATE_EVENT_TYPES = ("SE", "MXE", "RI", "A5SS", "A3SS", "AFE", "ALE")

POSTERIOR_COUNT_KEYS = {
    "SE": ("sj_included_1", "sj_included_2", "sj_excluded"),
    "MXE": ("sj_included_1", "sj_included_2", "sj_excluded_1", "sj_excluded_2"),
    "RI": ("counts_included", "counts_excluded"),
    "A5SS": ("sj_included", "sj_excluded"),
    "A3SS": ("sj_included", "sj_excluded"),
    "AFE": ("sj_included", "sj_excluded"),
    "ALE": ("sj_included", "sj_excluded"),
}


def split_event(tran_id: str) -> tuple[list[str], str]:
    exons, strand = split_event_cached(tran_id)
    return list(exons), strand


@lru_cache(maxsize=65536)
def split_event_cached(tran_id: str) -> tuple[tuple[str, ...], str]:
    if ":+@" in tran_id:
        return tuple(tran_id.split(":+@")), "+"
    if ":-@" in tran_id:
        return tuple(tran_id.split(":-@")), "-"
    raise ValueError(f"Unrecognized tran_id: {tran_id}")


@lru_cache(maxsize=262144)
def parse_coord(coord: str) -> tuple[str, int, int]:
    chrom, start, end = coord.split(":")
    return chrom, int(start), int(end)


def coord(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start}:{end}"


def coord_a3ss_r_compat(chrom: str, start: int, end: int) -> str:
    # MARVEL's installed R implementation can stringify rare round-number A3SS
    # coordinates like 500000 as 5e+05 during lookup, which drops those events.
    start_str = f"{float(start):.0e}" if start >= 100000 and start % 100000 == 0 else str(start)
    end_str = f"{float(end):.0e}" if end >= 100000 and end % 100000 == 0 else str(end)
    return f"{chrom}:{start_str}:{end_str}"


def as_r_char_number(value: int | str) -> str:
    # The installed MARVEL plate RI code stores exons.ref start/end as character.
    # R then compares those character columns against numeric intron bounds, which
    # coerces the numeric scalar to character and performs a lexicographic compare.
    return str(value)


def empty_feature() -> pd.DataFrame:
    return pd.DataFrame(columns=["tran_id", "event_type", "gene_id", "gene_short_name", "gene_type"])


def empty_psi() -> pd.DataFrame:
    return pd.DataFrame(columns=["tran_id"])


def normalize_feature_event_type(feature: pd.DataFrame, event_type: str) -> pd.DataFrame:
    feature_df = feature.reset_index(drop=True).copy()
    if "event_type" in feature_df.columns:
        feature_df["event_type"] = event_type
        columns = ["tran_id", "event_type", *[column for column in feature_df.columns if column not in {"tran_id", "event_type"}]]
        return feature_df.loc[:, columns]
    feature_df.insert(1, "event_type", event_type)
    return feature_df


def compute_psi_event(
    *,
    event_type: str,
    splice_feature: dict[str, pd.DataFrame],
    splice_junction: pd.DataFrame,
    intron_counts: pd.DataFrame | None,
    coverage_threshold: float,
    splice_junction_numeric: pd.DataFrame | None = None,
    intron_norm: pd.DataFrame | None = None,
    uneven_coverage_multiplier: float = 10.0,
    read_length: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    event_type = event_type.upper()
    sj = splice_junction_numeric if splice_junction_numeric is not None else sj_matrix(splice_junction)

    if event_type == "SE":
        return _compute_psi_se(splice_feature["SE"], sj, coverage_threshold, uneven_coverage_multiplier)
    if event_type == "MXE":
        return _compute_psi_mxe(splice_feature["MXE"], sj, coverage_threshold, uneven_coverage_multiplier)
    if event_type == "RI":
        if intron_counts is None:
            return empty_feature(), {}, empty_psi()
        if intron_norm is None:
            intron_norm = ri_intron_matrix(intron_counts, read_length)
        return _compute_psi_ri(splice_feature, sj, intron_norm, coverage_threshold)
    if event_type == "A5SS":
        return _compute_psi_a5ss(splice_feature["A5SS"], sj, coverage_threshold)
    if event_type == "A3SS":
        return _compute_psi_a3ss(splice_feature["A3SS"], sj, coverage_threshold)
    if event_type == "AFE":
        return _compute_psi_afe(splice_feature["AFE"], sj, coverage_threshold)
    if event_type == "ALE":
        return _compute_psi_ale(splice_feature["ALE"], sj, coverage_threshold)
    raise ValueError(f"Unsupported event type: {event_type}")


def compute_psi_posterior_event(*, event_type: str, counts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    event_type = event_type.upper()
    if event_type not in PLATE_EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type}")

    count_tables = counts.get(event_type)
    if not count_tables:
        return empty_psi()

    keys = POSTERIOR_COUNT_KEYS[event_type]
    if any(key not in count_tables for key in keys):
        return empty_psi()

    observed_numerator, observed_denominator = _posterior_observed_counts(event_type, count_tables)
    if observed_numerator.empty:
        return empty_psi()

    num_total = observed_numerator.sum(axis=1, skipna=True)
    den_total = observed_denominator.sum(axis=1, skipna=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        num_prior = num_total / den_total
        den_prior = den_total / num_total
    num_final = observed_numerator.add(num_prior, axis=0)
    den_final = observed_denominator.add(den_prior, axis=0)
    psi = num_final / den_final
    return psi.reset_index(names="tran_id")


def _posterior_observed_counts(
    event_type: str,
    count_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_type = event_type.upper()
    if event_type == "SE":
        sj_included_1 = _count_table_to_numeric(count_tables["sj_included_1"])
        sj_included_2 = _count_table_to_numeric(count_tables["sj_included_2"])
        sj_excluded = _count_table_to_numeric(count_tables["sj_excluded"])
        num_obs = sj_included_1 + sj_included_2
        den_obs = num_obs + (2.0 * sj_excluded)
        return num_obs, den_obs

    if event_type == "MXE":
        sj_included_1 = _count_table_to_numeric(count_tables["sj_included_1"])
        sj_included_2 = _count_table_to_numeric(count_tables["sj_included_2"])
        sj_excluded_1 = _count_table_to_numeric(count_tables["sj_excluded_1"])
        sj_excluded_2 = _count_table_to_numeric(count_tables["sj_excluded_2"])
        num_obs = sj_included_1 + sj_included_2
        den_obs = num_obs + sj_excluded_1 + sj_excluded_2
        return num_obs, den_obs

    if event_type == "RI":
        counts_included = _count_table_to_numeric(count_tables["counts_included"])
        counts_excluded = _count_table_to_numeric(count_tables["counts_excluded"])
        num_obs = counts_included
        den_obs = counts_included + counts_excluded
        return num_obs, den_obs

    sj_included = _count_table_to_numeric(count_tables["sj_included"])
    sj_excluded = _count_table_to_numeric(count_tables["sj_excluded"])
    num_obs = sj_included
    den_obs = sj_included + sj_excluded
    return num_obs, den_obs


def _count_table_to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.set_index("tran_id").iloc[:, 0:0].copy()
    df = frame.copy()
    if "tran_id" not in df.columns:
        raise ValueError("count table is missing tran_id")
    df = df.set_index("tran_id")
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def sj_matrix(splice_junction: pd.DataFrame) -> pd.DataFrame:
    df = splice_junction.copy()
    df = df.set_index("coord.intron")
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def ri_intron_matrix(intron_counts: pd.DataFrame | None, read_length: float) -> pd.DataFrame:
    if intron_counts is None:
        raise ValueError("intron_counts is required for RI PSI computation")
    df = intron_counts.copy()
    df = df.set_index("coord.intron")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    lengths = pd.Series(
        {coord_value: parse_coord(coord_value)[2] - parse_coord(coord_value)[1] + 1 for coord_value in df.index},
        dtype=float,
    )
    return df * read_length / lengths.to_numpy()[:, None]


def _compute_psi_se(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
    uneven_coverage_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included_1 = []
    included_2 = []
    excluded = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = parse_coord(exons[0])
        exon_2 = parse_coord(exons[1])
        exon_3 = parse_coord(exons[2])
        chrom = exon_1[0]
        if strand == "+":
            coord_i1 = coord(chrom, exon_1[2] + 1, exon_2[1] - 1)
            coord_i2 = coord(chrom, exon_2[2] + 1, exon_3[1] - 1)
            coord_e = coord(chrom, exon_1[2] + 1, exon_3[1] - 1)
        else:
            coord_i1 = coord(chrom, exon_3[2] + 1, exon_2[1] - 1)
            coord_i2 = coord(chrom, exon_2[2] + 1, exon_1[1] - 1)
            coord_e = coord(chrom, exon_3[2] + 1, exon_1[1] - 1)
        if coord_i1 in sj.index and coord_i2 in sj.index and coord_e in sj.index:
            valid_rows.append(row)
            included_1.append(coord_i1)
            included_2.append(coord_i2)
            excluded.append(coord_e)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "SE")
    sj_i1 = sj.loc[included_1].copy()
    sj_i2 = sj.loc[included_2].copy()
    sj_e = sj.loc[excluded].copy()
    sj_i1.index = feature_df["tran_id"]
    sj_i2.index = feature_df["tran_id"]
    sj_e.index = feature_df["tran_id"]

    psi = (sj_i1 + sj_i2) / (sj_i1 + sj_i2 + 2.0 * sj_e)
    cov = ((sj_i1 >= coverage_threshold) & (sj_i2 >= coverage_threshold)) | (sj_e >= coverage_threshold)
    psi = psi.mask(~cov)
    uneven = (sj_i1 / sj_i2 >= uneven_coverage_multiplier) | (sj_i2 / sj_i1 >= uneven_coverage_multiplier)
    psi = psi.mask(uneven)
    return (
        feature_df,
        {
            "sj_included_1": sj_i1.reset_index(names="tran_id"),
            "sj_included_2": sj_i2.reset_index(names="tran_id"),
            "sj_excluded": sj_e.reset_index(names="tran_id"),
        },
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_mxe(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
    uneven_coverage_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included_1 = []
    included_2 = []
    excluded_1 = []
    excluded_2 = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = parse_coord(exons[0])
        exon_2 = parse_coord(exons[1])
        exon_3 = parse_coord(exons[2])
        exon_4 = parse_coord(exons[3])
        chrom = exon_1[0]
        if strand == "+":
            coord_i1 = coord(chrom, exon_1[2] + 1, exon_2[1] - 1)
            coord_i2 = coord(chrom, exon_2[2] + 1, exon_4[1] - 1)
            coord_e1 = coord(chrom, exon_1[2] + 1, exon_3[1] - 1)
            coord_e2 = coord(chrom, exon_3[2] + 1, exon_4[1] - 1)
        else:
            coord_i1 = coord(chrom, exon_2[2] + 1, exon_1[1] - 1)
            coord_i2 = coord(chrom, exon_4[2] + 1, exon_2[1] - 1)
            coord_e1 = coord(chrom, exon_3[2] + 1, exon_1[1] - 1)
            coord_e2 = coord(chrom, exon_4[2] + 1, exon_3[1] - 1)
        if all(coord_value in sj.index for coord_value in (coord_i1, coord_i2, coord_e1, coord_e2)):
            valid_rows.append(row)
            included_1.append(coord_i1)
            included_2.append(coord_i2)
            excluded_1.append(coord_e1)
            excluded_2.append(coord_e2)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "MXE")
    sj_i1 = sj.loc[included_1].copy()
    sj_i2 = sj.loc[included_2].copy()
    sj_e1 = sj.loc[excluded_1].copy()
    sj_e2 = sj.loc[excluded_2].copy()
    for table in (sj_i1, sj_i2, sj_e1, sj_e2):
        table.index = feature_df["tran_id"]

    psi = (sj_i1 + sj_i2) / (sj_i1 + sj_i2 + sj_e1 + sj_e2)
    cov = ((sj_i1 >= coverage_threshold) & (sj_i2 >= coverage_threshold)) | (
        (sj_e1 >= coverage_threshold) & (sj_e2 >= coverage_threshold)
    )
    psi = psi.mask(~cov)
    uneven_included = (sj_i1 / sj_i2 >= uneven_coverage_multiplier) | (sj_i2 / sj_i1 >= uneven_coverage_multiplier)
    uneven_excluded = (sj_e1 / sj_e2 >= uneven_coverage_multiplier) | (sj_e2 / sj_e1 >= uneven_coverage_multiplier)
    psi = psi.mask(uneven_included | uneven_excluded)
    return (
        feature_df,
        {
            "sj_included_1": sj_i1.reset_index(names="tran_id"),
            "sj_included_2": sj_i2.reset_index(names="tran_id"),
            "sj_excluded_1": sj_e1.reset_index(names="tran_id"),
            "sj_excluded_2": sj_e2.reset_index(names="tran_id"),
        },
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_a5ss(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included = []
    excluded = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = exons[0]
        exon_2 = parse_coord(exons[1])
        if strand == "+":
            chrom = exon_2[0]
            left, right = exon_1.split("|")
            left_parts = left.split(":")
            right_parts = right.split(":")
            coord_i = coord(chrom, int(right_parts[-1]) + 1, exon_2[1] - 1)
            coord_e = coord(chrom, int(left_parts[2]) + 1, exon_2[1] - 1)
        else:
            left, right = exon_1.split("|")
            left_parts = left.split(":")
            chrom = left_parts[0]
            coord_i = coord(chrom, exon_2[2] + 1, int(left_parts[2]) - 1)
            coord_e = coord(chrom, exon_2[2] + 1, int(right) - 1)
        if coord_i in sj.index and coord_e in sj.index:
            valid_rows.append(row)
            included.append(coord_i)
            excluded.append(coord_e)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "A5SS")
    sj_i = sj.loc[included].copy()
    sj_e = sj.loc[excluded].copy()
    sj_i.index = feature_df["tran_id"]
    sj_e.index = feature_df["tran_id"]
    psi = sj_i / (sj_i + sj_e)
    cov = (sj_i >= coverage_threshold) | (sj_e >= coverage_threshold)
    psi = psi.mask(~cov)
    return (
        feature_df,
        {"sj_included": sj_i.reset_index(names="tran_id"), "sj_excluded": sj_e.reset_index(names="tran_id")},
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_a3ss(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included = []
    excluded = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = parse_coord(exons[0])
        exon_2 = exons[1]
        chrom = exon_1[0]
        left, right = exon_2.split("|")
        left_parts = left.split(":")
        right_parts = right.split(":")
        if strand == "+":
            coord_i = coord(chrom, exon_1[2] + 1, int(left_parts[1]) - 1)
            coord_e = coord(chrom, exon_1[2] + 1, int(right_parts[0]) - 1)
        else:
            coord_i = coord_a3ss_r_compat(chrom, int(right_parts[0]) + 1, exon_1[1] - 1)
            coord_e = coord_a3ss_r_compat(chrom, int(left_parts[1]) + 1, exon_1[1] - 1)
        if coord_i in sj.index and coord_e in sj.index:
            valid_rows.append(row)
            included.append(coord_i)
            excluded.append(coord_e)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "A3SS")
    sj_i = sj.loc[included].copy()
    sj_e = sj.loc[excluded].copy()
    sj_i.index = feature_df["tran_id"]
    sj_e.index = feature_df["tran_id"]
    psi = sj_i / (sj_i + sj_e)
    cov = (sj_i >= coverage_threshold) | (sj_e >= coverage_threshold)
    psi = psi.mask(~cov)
    return (
        feature_df,
        {"sj_included": sj_i.reset_index(names="tran_id"), "sj_excluded": sj_e.reset_index(names="tran_id")},
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_ale(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included = []
    excluded = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = parse_coord(exons[0])
        exon_2 = exons[1]
        chrom = exon_1[0]
        left, right = exon_2.split("|")
        left_parts = left.split(":")
        right_parts = right.split(":")
        if strand == "+":
            start = exon_1[2] + 1
            coord_i = coord(chrom, start, int(left_parts[1]) - 1)
            coord_e = coord(chrom, start, int(right_parts[0]) - 1)
        else:
            end = exon_1[1] - 1
            coord_i = coord(chrom, int(left_parts[2]) + 1, end)
            coord_e = coord(chrom, int(right_parts[1]) + 1, end)
        if coord_i in sj.index and coord_e in sj.index:
            valid_rows.append(row)
            included.append(coord_i)
            excluded.append(coord_e)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "ALE")
    sj_i = sj.loc[included].copy()
    sj_e = sj.loc[excluded].copy()
    sj_i.index = feature_df["tran_id"]
    sj_e.index = feature_df["tran_id"]
    total = sj_i + sj_e
    psi = sj_i / total
    psi = psi.mask(total < coverage_threshold)
    return (
        feature_df,
        {"sj_included": sj_i.reset_index(names="tran_id"), "sj_excluded": sj_e.reset_index(names="tran_id")},
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_afe(
    feature: pd.DataFrame,
    sj: pd.DataFrame,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    valid_rows = []
    included = []
    excluded = []

    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = exons[0]
        exon_2 = parse_coord(exons[1])
        left, right = exon_1.split("|")
        left_parts = left.split(":")
        right_parts = right.split(":")
        chrom = exon_2[0]
        if strand == "+":
            end = exon_2[1] - 1
            coord_i = coord(chrom, int(right_parts[1]) + 1, end)
            coord_e = coord(chrom, int(left_parts[2]) + 1, end)
        else:
            start = exon_2[2] + 1
            coord_i = coord(chrom, start, int(right_parts[0]) - 1)
            coord_e = coord(chrom, start, int(left_parts[1]) - 1)
        if coord_i in sj.index and coord_e in sj.index:
            valid_rows.append(row)
            included.append(coord_i)
            excluded.append(coord_e)

    if not valid_rows:
        return empty_feature(), {}, empty_psi()

    feature_df = normalize_feature_event_type(pd.DataFrame(valid_rows), "AFE")
    sj_i = sj.loc[included].copy()
    sj_e = sj.loc[excluded].copy()
    sj_i.index = feature_df["tran_id"]
    sj_e.index = feature_df["tran_id"]
    total = sj_i + sj_e
    psi = sj_i / total
    psi = psi.mask(total < coverage_threshold)
    return (
        feature_df,
        {"sj_included": sj_i.reset_index(names="tran_id"), "sj_excluded": sj_e.reset_index(names="tran_id")},
        psi.reset_index(names="tran_id"),
    )


def _compute_psi_ri(
    splice_feature: dict[str, pd.DataFrame],
    sj: pd.DataFrame,
    intron_norm: pd.DataFrame,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    feature = splice_feature["RI"].copy()
    exons_ref = ri_reference_exons(splice_feature)
    parsed_rows = []
    for _, row in feature.iterrows():
        exons, strand = split_event(row["tran_id"])
        exon_1 = parse_coord(exons[0])
        exon_2 = parse_coord(exons[1])
        chrom = exon_1[0]
        if strand == "+":
            start = exon_1[2] + 1
            end = exon_2[1] - 1
        else:
            start = exon_2[1] + 1
            end = exon_1[2] - 1
        parsed = row.to_dict()
        parsed["chr"] = chrom
        parsed["start"] = start
        parsed["end"] = end
        parsed_rows.append(parsed)

    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        return empty_feature(), {}, empty_psi()

    parsed_df = parsed_df.loc[_ri_keep_without_reference_overlap(parsed_df, exons_ref)].copy()
    if parsed_df.empty:
        return empty_feature(), {}, empty_psi()

    parsed_df["coord.intron"] = parsed_df.apply(
        lambda row: coord(row["chr"], int(row["start"]), int(row["end"])),
        axis=1,
    )

    parsed_df["chr.start"] = parsed_df["chr"] + ":" + parsed_df["start"].astype(str)
    collapsed = []
    for _, group in parsed_df.groupby("chr.start"):
        collapsed.append(group.loc[group["end"].idxmax()])
    parsed_df = pd.DataFrame(collapsed).copy()

    parsed_df["chr.end"] = parsed_df["chr"] + ":" + parsed_df["end"].astype(str)
    collapsed = []
    for _, group in parsed_df.groupby("chr.end"):
        collapsed.append(group.loc[group["start"].idxmin()])
    parsed_df = pd.DataFrame(collapsed).copy()

    parsed_df = parsed_df[parsed_df["coord.intron"].isin(intron_norm.index)].copy()
    if parsed_df.empty:
        return empty_feature(), {}, empty_psi()

    counts_excluded = ri_excluded_counts(sj=sj, parsed_df=parsed_df)

    feature_df = parsed_df.copy()
    feature_df["event_type"] = "RI"
    feature_df = feature_df[["tran_id", "event_type", "gene_id", "gene_short_name", "gene_type"]]

    counts_included = intron_norm.loc[parsed_df["coord.intron"]].copy()
    counts_included.index = feature_df["tran_id"]
    psi = counts_included / (counts_included + counts_excluded)
    cov = counts_included + counts_excluded
    psi = psi.mask(cov < coverage_threshold)
    return (
        feature_df.reset_index(drop=True),
        {
            "counts_included": counts_included.reset_index(names="tran_id"),
            "counts_excluded": counts_excluded.reset_index(names="tran_id"),
        },
        psi.reset_index(names="tran_id"),
    )


def _ri_keep_without_reference_overlap(parsed_df: pd.DataFrame, exons_ref: pd.DataFrame) -> np.ndarray:
    keep = np.ones(len(parsed_df), dtype=bool)
    if parsed_df.empty or exons_ref.empty:
        return keep

    ref_by_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, ref in exons_ref.groupby("chr", sort=False):
        ref_by_chrom[str(chrom)] = (
            ref["start"].astype(str).to_numpy(),
            ref["end"].astype(str).to_numpy(),
        )

    for chrom, row_positions in parsed_df.groupby("chr", sort=False).groups.items():
        ref_arrays = ref_by_chrom.get(str(chrom))
        if ref_arrays is None:
            continue
        ref_starts, ref_ends = ref_arrays
        starts = parsed_df.loc[row_positions, "start"].astype(int).map(as_r_char_number).to_numpy()
        ends = parsed_df.loc[row_positions, "end"].astype(int).map(as_r_char_number).to_numpy()
        for row_position, intron_start, intron_end in zip(row_positions, starts, ends):
            keep[row_position] = not ((ref_starts >= intron_start) & (ref_ends <= intron_end)).any()

    return keep


def ri_excluded_counts(sj: pd.DataFrame, parsed_df: pd.DataFrame, chunk_size: int = 64) -> pd.DataFrame:
    sj_coord = pd.DataFrame(
        [parse_coord(coord_value) for coord_value in sj.index],
        columns=["chr", "start", "end"],
        index=sj.index,
    )
    sample_cols = sj.columns.tolist()
    excluded_parts = []

    for chrom, group in parsed_df.groupby("chr", sort=False):
        sj_chr = sj_coord[sj_coord["chr"] == chrom]
        if sj_chr.empty:
            values = np.zeros((len(group), len(sample_cols)), dtype=float)
            excluded_parts.append(pd.DataFrame(values, index=group["tran_id"], columns=sample_cols))
            continue

        chrom_values = _ri_excluded_counts_for_chromosome(
            sj_values=sj.loc[sj_chr.index].to_numpy(dtype=float, copy=False),
            sj_starts=sj_chr["start"].to_numpy(dtype=np.int64, copy=False),
            sj_ends=sj_chr["end"].to_numpy(dtype=np.int64, copy=False),
            intron_starts=group["start"].to_numpy(dtype=np.int64, copy=False),
            intron_ends=group["end"].to_numpy(dtype=np.int64, copy=False),
        )
        excluded_parts.append(pd.DataFrame(chrom_values, index=group["tran_id"], columns=sample_cols))

    if not excluded_parts:
        return pd.DataFrame(columns=sample_cols)

    return pd.concat(excluded_parts, axis=0).loc[parsed_df["tran_id"]].copy()


def _ri_excluded_counts_for_chromosome(
    *,
    sj_values: np.ndarray,
    sj_starts: np.ndarray,
    sj_ends: np.ndarray,
    intron_starts: np.ndarray,
    intron_ends: np.ndarray,
) -> np.ndarray:
    if len(intron_starts) == 0:
        return np.zeros((0, sj_values.shape[1]), dtype=float)
    if len(sj_starts) == 0:
        return np.zeros((len(intron_starts), sj_values.shape[1]), dtype=float)

    result = np.zeros((len(intron_starts), sj_values.shape[1]), dtype=float)
    for query_idx, (intron_start, intron_end) in enumerate(zip(intron_starts, intron_ends)):
        mask = (sj_starts <= intron_start) & (sj_ends >= intron_end)
        if mask.any():
            result[query_idx] = sj_values[mask].sum(axis=0)

    return result


def ri_reference_exons(splice_feature: dict[str, pd.DataFrame]) -> pd.DataFrame:
    exon_coords: list[str] = []

    for event_type in ("SE", "MXE"):
        for tran_id in splice_feature[event_type]["tran_id"].astype(str).tolist():
            exons, _ = split_event(tran_id)
            exon_coords.extend(exons)

    for tran_id in splice_feature["RI"]["tran_id"].astype(str).tolist():
        exons, strand = split_event(tran_id)
        if strand == "+":
            exon_coords.extend(exons)
        else:
            exon_1 = exons[0].split(":")
            chrom = exon_1[0]
            exon_coords.extend([f"{chrom}:{exon_1[2]}:{exon_1[1]}"])

    for tran_id in splice_feature["A5SS"]["tran_id"].astype(str).tolist():
        exons, strand = split_event(tran_id)
        exon_1 = exons[0].split(":")
        chrom = exon_1[0]
        exon_2 = exons[1]
        alt_a, alt_b = exon_1[2].split("|")
        if strand == "+":
            exon_coords.extend(
                [
                    f"{chrom}:{exon_1[1]}:{alt_a}",
                    f"{chrom}:{exon_1[1]}:{alt_b}",
                    exon_2,
                ]
            )
        else:
            exon_coords.extend(
                [
                    f"{chrom}:{alt_a}:{exon_1[1]}",
                    f"{chrom}:{alt_b}:{exon_1[1]}",
                    exon_2,
                ]
            )

    for tran_id in splice_feature["A3SS"]["tran_id"].astype(str).tolist():
        exons, strand = split_event(tran_id)
        exon_1 = exons[0]
        exon_2 = exons[1].split(":")
        chrom = exon_2[0]
        alt_a, alt_b = exon_2[1].split("|")
        trailing = exon_2[2]
        if strand == "+":
            exon_coords.extend(
                [
                    exon_1,
                    f"{chrom}:{alt_a}:{trailing}",
                    f"{chrom}:{alt_b}:{trailing}",
                ]
            )
        else:
            exon_coords.extend(
                [
                    exon_1,
                    f"{chrom}:{trailing}:{alt_a}",
                    f"{chrom}:{trailing}:{alt_b}",
                ]
            )

    unique_coords = pd.unique(pd.Series(exon_coords)).tolist()
    ref = pd.DataFrame([coord_value.split(":") for coord_value in unique_coords], columns=["chr", "start", "end"])
    return ref.reset_index(drop=True)
