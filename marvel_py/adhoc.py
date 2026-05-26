from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .matrix import Marvel10x
from .utils import extract_gtf_attr

__all__ = [
    "adhoc_gene_tabulate_expression_gene_10x",
    "adhoc_gene_tabulate_expression_psi_10x",
    "adhoc_gene_de_gene_10x",
    "adhoc_gene_de_psi_10x",
    "adhoc_gene_plot_de_values_10x",
    "adhoc_gene_plot_sj_position_10x",
]


def _ensure_adhoc_gene_state(marvel_object: Marvel10x) -> dict[str, object]:
    state = marvel_object.adhoc_gene
    if not isinstance(state, dict):
        state = {}
        marvel_object.adhoc_gene = state
    state.setdefault("expression", {})
    state.setdefault("de", {})
    return state


def _ordered_cell_group_items(cell_group_list: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    if not isinstance(cell_group_list, dict):
        raise TypeError("cell_group_list must be a mapping of cell groups to cell ids")
    if not cell_group_list:
        raise ValueError("cell_group_list must contain at least one cell group")
    ordered_items: list[tuple[str, list[str]]] = []
    for group_name, cell_ids in cell_group_list.items():
        cell_ids = [str(cell_id) for cell_id in list(cell_ids)]
        if not cell_ids:
            raise ValueError(f"cell_group_list[{group_name!r}] must contain at least one cell")
        ordered_items.append((str(group_name), cell_ids))
    return ordered_items


def _ordered_group_pairs(groups: list[str]) -> list[tuple[str, str]]:
    return list(combinations([str(group) for group in groups], 2))


def _build_sj_position_payload(
    marvel_object: Marvel10x,
    *,
    coord_intron: str,
    coord_intron_ext: int = 25,
    show_protein_coding_only: bool = True,
) -> dict[str, object]:
    if marvel_object.sj_metadata is None:
        raise ValueError("sj_metadata is required for adhoc_gene_plot_sj_position_10x")
    if marvel_object.gtf is None or marvel_object.gtf.empty:
        raise ValueError("gtf is required for adhoc_gene_plot_sj_position_10x")

    sj_rows = marvel_object.sj_metadata.loc[
        marvel_object.sj_metadata["coord.intron"].astype(str) == str(coord_intron)
    ].copy()
    if sj_rows.empty:
        raise ValueError(f"coord.intron not found in sj_metadata: {coord_intron!r}")

    gene_short_name = str(sj_rows["gene_short_name.start"].iloc[0])
    gtf = marvel_object.gtf.copy()
    gene_mask = gtf["V9"].map(lambda value: extract_gtf_attr(str(value), "gene_name") == gene_short_name)
    gene_gtf = gtf.loc[gene_mask].copy().reset_index(drop=True)

    def _as_seqname(value: object) -> object:
        text = str(value)
        if text.startswith("chr"):
            text = text[3:]
        try:
            return int(text)
        except ValueError:
            return text

    def _extract_transcript_biotype(value: object) -> str:
        biotype = extract_gtf_attr(str(value), "transcript_biotype")
        if biotype is None:
            biotype = extract_gtf_attr(str(value), "transcript_type")
        return str(biotype) if biotype is not None else "character(0)"

    transcript_order: list[tuple[str, str]] = []
    seen_transcripts: set[str] = set()
    for value in gene_gtf["V9"].astype(str).tolist():
        transcript_id = extract_gtf_attr(value, "transcript_id")
        if not transcript_id or transcript_id in seen_transcripts:
            continue
        seen_transcripts.add(transcript_id)
        transcript_order.append((transcript_id, _extract_transcript_biotype(value)))

    if show_protein_coding_only:
        protein_coding_order = [item for item in transcript_order if "protein_coding" in item[1]]
        if protein_coding_order:
            transcript_order = protein_coding_order

    def _make_group_frame(
        rows: pd.DataFrame,
        *,
        group: int,
        group_name: str,
        cds: bool,
    ) -> pd.DataFrame:
        columns = ["group", "group_name", "seqnames", "start", "end", "width", "strand", "exon_id", "exon_name", "exon_rank"]
        if rows.empty:
            return pd.DataFrame(columns=columns)
        result_rows = []
        for exon_rank, (_, row) in enumerate(rows.iterrows(), start=1):
            exon_id = extract_gtf_attr(str(row["V9"]), "exon_id")
            if not exon_id:
                exon_id = "character(0)" if cds else ""
            result_rows.append(
                {
                    "group": group,
                    "group_name": group_name,
                    "seqnames": _as_seqname(row["V1"]),
                    "start": int(row["V4"]),
                    "end": int(row["V5"]),
                    "width": int(row["V5"]) - int(row["V4"]) + 1,
                    "strand": str(row["V7"]),
                    "exon_id": exon_id,
                    "exon_name": exon_id,
                    "exon_rank": exon_rank,
                }
            )
        return pd.DataFrame(result_rows, columns=columns)

    def _build_transcript_rows(transcript_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        transcript_rows = gene_gtf.loc[
            gene_gtf["V9"].astype(str).map(lambda value: extract_gtf_attr(value, "transcript_id") == transcript_id)
        ].copy()
        exons = transcript_rows.loc[transcript_rows["V3"].astype(str) == "exon"].copy().reset_index(drop=True)
        cds = transcript_rows.loc[transcript_rows["V3"].astype(str) == "CDS"].copy().reset_index(drop=True)
        sj_exons: pd.DataFrame | None = None
        if not exons.empty:
            try:
                start_sj = int(str(coord_intron).split(":")[1])
                end_sj = int(str(coord_intron).split(":")[2])
            except Exception as exc:  # pragma: no cover - guarded by existing tests
                raise ValueError(f"Invalid coord.intron: {coord_intron!r}") from exc
            exon_sj_start = exons.loc[exons["V5"].astype(int) == start_sj - 1].copy()
            exon_sj_end = exons.loc[exons["V4"].astype(int) == end_sj + 1].copy()
            if not exon_sj_start.empty and not exon_sj_end.empty:
                selected = exons.loc[
                    exons["V5"].astype(int).isin(exon_sj_start["V5"].astype(int).tolist())
                    | exons["V4"].astype(int).isin(exon_sj_end["V4"].astype(int).tolist())
                ].copy()
            else:
                selected = pd.DataFrame()
            if not selected.empty:
                selected = selected.drop_duplicates(subset=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]).reset_index(drop=True)
                selected["V4"] = selected["V4"].astype(int)
                selected["V5"] = selected["V5"].astype(int)
                for idx, row in selected.iterrows():
                    if int(row["V5"]) == start_sj - 1:
                        selected.at[idx, "V4"] = int(row["V5"]) - int(coord_intron_ext)
                    if int(row["V4"]) == end_sj + 1:
                        selected.at[idx, "V5"] = int(row["V4"]) + int(coord_intron_ext)
                sj_exons = selected.reset_index(drop=True)
        return exons, cds, sj_exons

    exon_groups: list[pd.DataFrame] = []
    cds_groups: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    group_index = 1

    for transcript_id, biotype in transcript_order:
        transcript_name = f"{transcript_id} ({biotype})"
        exons, cds, _ = _build_transcript_rows(transcript_id)
        exon_frame = _make_group_frame(exons, group=group_index, group_name=transcript_name, cds=False)
        if not exon_frame.empty:
            exon_groups.append(exon_frame)
            metadata_rows.append({"transcript_id": transcript_name, "gene_short_name": gene_short_name, "strand": 1 if str(gene_gtf["V7"].iloc[0]) == "+" else -1})
            group_index += 1
        if not cds.empty:
            cds_groups.append(_make_group_frame(cds, group=group_index - 1 if not exon_frame.empty else group_index, group_name=transcript_name, cds=True))

    sj_group_index = len(exon_groups) + 1
    for transcript_id, biotype in transcript_order:
        transcript_name = f"{transcript_id} ({biotype})"
        _, _, sj_exons = _build_transcript_rows(transcript_id)
        if sj_exons is None or sj_exons.empty:
            continue
        sj_name = f"{transcript_name}_SJ"
        sj_frame = _make_group_frame(sj_exons, group=sj_group_index, group_name=sj_name, cds=False)
        if not sj_frame.empty:
            exon_groups.append(sj_frame)
            metadata_rows.append({"transcript_id": sj_name, "gene_short_name": gene_short_name, "strand": 1 if str(gene_gtf["V7"].iloc[0]) == "+" else -1})
            sj_group_index += 1

    columns = ["group", "group_name", "seqnames", "start", "end", "width", "strand", "exon_id", "exon_name", "exon_rank"]
    exonfile = pd.concat(exon_groups, ignore_index=True) if exon_groups else pd.DataFrame(columns=columns)
    cdsfile = pd.concat(cds_groups, ignore_index=True) if cds_groups else pd.DataFrame(columns=columns)
    metadata = pd.DataFrame(metadata_rows, columns=["transcript_id", "gene_short_name", "strand"])
    return {
        "metadata": metadata.reset_index(drop=True),
        "exonfile": exonfile.reset_index(drop=True),
        "cdsfile": cdsfile.reset_index(drop=True),
        "plot": None,
    }


def adhoc_gene_tabulate_expression_gene_10x(
    marvel_object: Marvel10x,
    *,
    cell_group_list,
    gene_short_name,
    log2_transform: bool = True,
    min_pct_cells: float = 10.0,
    downsample: bool = False,
    seed: int = 1,
):
    _ = downsample, seed
    state = _ensure_adhoc_gene_state(marvel_object)
    ordered_items = _ordered_cell_group_items(cell_group_list)
    normalized_cell_group_list = {group_name: cell_ids for group_name, cell_ids in ordered_items}
    state["cell_group_list"] = normalized_cell_group_list
    state["gene_short_name"] = str(gene_short_name)

    gene_matrix = marvel_object.gene_norm_matrix.subset_rows([str(gene_short_name)])
    values = pd.Series(
        np.asarray(gene_matrix.matrix.toarray()).ravel(),
        index=gene_matrix.col_ids.astype(str).tolist(),
        dtype=float,
    )
    if log2_transform:
        values = np.log2(values + 1.0)

    rows = []
    for group_name, cell_ids in ordered_items:
        group_values = values.loc[cell_ids]
        mean_expr = float(group_values.mean()) if not group_values.empty else np.nan
        pct_cells_expr = float((group_values != 0).mean() * 100.0) if not group_values.empty else np.nan
        if pct_cells_expr < min_pct_cells:
            pct_cells_expr = np.nan
        rows.append(
            {
                "group": group_name,
                "mean.expr": mean_expr,
                "pct.cells.expr": pct_cells_expr,
            }
        )

    expression = state.setdefault("expression", {})
    expression["gene"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def adhoc_gene_tabulate_expression_psi_10x(
    marvel_object: Marvel10x,
    *,
    min_pct_cells: float = 10.0,
):
    state = _ensure_adhoc_gene_state(marvel_object)
    if "cell_group_list" not in state or "gene_short_name" not in state:
        raise ValueError("adhoc_gene_tabulate_expression_gene_10x must run before adhoc_gene_tabulate_expression_psi_10x")

    cell_group_list = state["cell_group_list"]
    gene_short_name = str(state["gene_short_name"])
    ordered_items = _ordered_cell_group_items(cell_group_list)

    if marvel_object.sj_metadata is None:
        raise ValueError("sj_metadata is required for adhoc_gene_tabulate_expression_psi_10x")

    coord_introns = (
        marvel_object.sj_metadata.loc[
            marvel_object.sj_metadata["gene_short_name.start"].astype(str) == gene_short_name,
            "coord.intron",
        ]
        .astype(str)
        .tolist()
    )
    if not coord_introns:
        raise ValueError(f"No splice junctions found for gene_short_name={gene_short_name!r}")

    sj_matrix = marvel_object.sj_count_matrix.subset_rows(coord_introns)
    gene_matrix = marvel_object.gene_count_matrix.subset_rows([gene_short_name])
    gene_counts = pd.Series(
        np.asarray(gene_matrix.matrix.toarray()).ravel(),
        index=gene_matrix.col_ids.astype(str).tolist(),
        dtype=float,
    )

    sj_df = pd.DataFrame(
        np.asarray(sj_matrix.matrix.toarray()),
        index=sj_matrix.row_ids.astype(str).tolist(),
        columns=sj_matrix.col_ids.astype(str).tolist(),
    )

    rows = []
    for group_name, cell_ids in ordered_items:
        df_small = sj_df.loc[:, cell_ids]
        sj_count_total = df_small.sum(axis=1)
        n_cells_total = df_small.shape[1]
        n_cells_expr_sj = (df_small != 0).sum(axis=1)
        pct_cells_expr_sj = (n_cells_expr_sj / max(n_cells_total, 1) * 100.0).round(2)
        results = pd.DataFrame(
            {
                "group": group_name,
                "coord.intron": df_small.index.astype(str),
                "n.cells.total": n_cells_total,
                "n.cells.expr.sj": n_cells_expr_sj.astype(int),
                "pct.cells.expr.sj": pct_cells_expr_sj.astype(float),
                "sj.count.total": sj_count_total.astype(float),
            }
        ).reset_index(drop=True)
        rows.append(results)

    results_sj = pd.concat(rows, ignore_index=True)

    gene_rows = []
    for group_name, cell_ids in ordered_items:
        gene_small = gene_counts.loc[cell_ids]
        gene_count_total = float(gene_small.sum())
        n_cells_total = len(cell_ids)
        n_cells_expr_gene = int((gene_small != 0).sum())
        pct_cells_expr_gene = round(n_cells_expr_gene / max(n_cells_total, 1) * 100.0, 2)
        gene_rows.append(
            pd.DataFrame(
                {
                    "group": [group_name],
                    "gene_short_name": [gene_short_name],
                    "n.cells.total": [n_cells_total],
                    "n.cells.expr.gene": [n_cells_expr_gene],
                    "pct.cells.expr.gene": [pct_cells_expr_gene],
                    "gene.count.total": [gene_count_total],
                }
            )
        )

    results_gene = pd.concat(gene_rows, ignore_index=True)
    results = results_sj.merge(results_gene[["group", "gene.count.total"]], on="group", how="left")
    results["psi"] = (results["sj.count.total"] / results["gene.count.total"] * 100.0).round(2)
    results.loc[results["pct.cells.expr.sj"] < min_pct_cells, "psi"] = np.nan
    results = results.loc[results["psi"].notna()].copy()
    if results.empty:
        raise ValueError("No complete PSI rows could be computed from the available inputs")

    coord_support = results.groupby("coord.intron", sort=False)["n.cells.expr.sj"].sum().sort_values(ascending=False)
    coord_order = coord_support.index.astype(str).tolist()
    results["coord.intron"] = pd.Categorical(results["coord.intron"].astype(str), categories=coord_order, ordered=True)
    results["group"] = pd.Categorical(
        results["group"].astype(str),
        categories=[group for group, _ in ordered_items],
        ordered=True,
    )
    results = results.sort_values(["group", "coord.intron"]).reset_index(drop=True)
    coord_levels = list(results["coord.intron"].cat.categories)
    coord_labels = {coord_intron: f"SJ-{index}" for index, coord_intron in enumerate(coord_levels, start=1)}
    results["figure.column"] = results["coord.intron"].astype(str).map(coord_labels)
    results["group"] = results["group"].astype(str)
    results["coord.intron"] = results["coord.intron"].astype(str)

    expression = state.setdefault("expression", {})
    expression["psi"] = {
        "table": results[
            [
                "group",
                "figure.column",
                "coord.intron",
                "n.cells.total",
                "n.cells.expr.sj",
                "pct.cells.expr.sj",
                "sj.count.total",
                "gene.count.total",
                "psi",
            ]
        ].reset_index(drop=True),
        "plot": None,
    }
    return marvel_object


def adhoc_gene_de_gene_10x(marvel_object: Marvel10x) -> Marvel10x:
    state = _ensure_adhoc_gene_state(marvel_object)
    gene_state = state.get("expression", {}).get("gene")
    if not gene_state or "table" not in gene_state:
        raise ValueError("adhoc_gene_tabulate_expression_gene_10x must run before adhoc_gene_de_gene_10x")

    table = gene_state["table"]
    if table is None or table.empty:
        raise ValueError("adhoc_gene_tabulate_expression_gene_10x must produce a non-empty gene table")

    groups = table["group"].astype(str).tolist()
    group_pairs = _ordered_group_pairs(groups)
    rows = []
    for group1, group2 in group_pairs:
        mean_g1 = float(table.loc[table["group"].astype(str) == group1, "mean.expr"].iloc[0])
        mean_g2 = float(table.loc[table["group"].astype(str) == group2, "mean.expr"].iloc[0])
        rows.append(
            {
                "group.pair": f"{group2} vs {group1}",
                "log2fc": mean_g2 - mean_g1,
            }
        )

    state.setdefault("de", {})
    state["de"]["gene"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def adhoc_gene_de_psi_10x(marvel_object: Marvel10x) -> Marvel10x:
    state = _ensure_adhoc_gene_state(marvel_object)
    psi_state = state.get("expression", {}).get("psi")
    if not psi_state or "table" not in psi_state:
        raise ValueError("adhoc_gene_tabulate_expression_psi_10x must run before adhoc_gene_de_psi_10x")

    table = psi_state["table"]
    if table is None or table.empty:
        raise ValueError("adhoc_gene_tabulate_expression_psi_10x must produce a non-empty psi table")

    ordered_groups = list(state.get("cell_group_list", {}).keys())
    group_pairs = _ordered_group_pairs(ordered_groups)
    coord_order = (
        [str(value) for value in getattr(table["coord.intron"], "cat", None).categories]
        if pd.api.types.is_categorical_dtype(table["coord.intron"])
        else table["coord.intron"].astype(str).drop_duplicates().tolist()
    )
    rows = []
    for coord_intron in coord_order:
        coord_table = table.loc[table["coord.intron"].astype(str) == coord_intron]
        if len(coord_table) < len(ordered_groups):
            continue
        figure_column = coord_table["figure.column"].iloc[0]
        psi_by_group = coord_table.set_index("group")["psi"].astype(float).to_dict()
        coord_rows = []
        for group1, group2 in group_pairs:
            missing_groups = [group for group in (group1, group2) if group not in psi_by_group]
            if missing_groups:
                coord_rows = []
                break
            coord_rows.append(
                {
                    "group.pair": f"{group2} vs {group1}",
                    "figure.column": figure_column,
                    "coord.intron": coord_intron,
                    "delta": psi_by_group[group2] - psi_by_group[group1],
                }
            )
        rows.extend(coord_rows)

    if not rows:
        raise ValueError("No complete PSI DE rows could be computed from the available PSI table")

    state.setdefault("de", {})
    state["de"]["psi"] = {"table": pd.DataFrame(rows), "plot": None}
    return marvel_object


def adhoc_gene_plot_de_values_10x(
    marvel_object: Marvel10x,
    *,
    coord_intron: str,
    log2fc_gene: float = 0.5,
    delta_sj: float = 5.0,
    label_size: float = 2.0,
    point_size: float = 2.0,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
) -> Marvel10x:
    _ = label_size, point_size, xmin, xmax, ymin, ymax

    state = _ensure_adhoc_gene_state(marvel_object)
    gene_state = state.get("de", {}).get("gene")
    psi_state = state.get("de", {}).get("psi")
    if not gene_state or "table" not in gene_state:
        raise ValueError("adhoc_gene_de_gene_10x must run before adhoc_gene_plot_de_values_10x")
    if not psi_state or "table" not in psi_state:
        raise ValueError("adhoc_gene_de_psi_10x must run before adhoc_gene_plot_de_values_10x")

    gene_table = gene_state["table"]
    psi_table = psi_state["table"]
    if gene_table is None or gene_table.empty:
        raise ValueError("adhoc_gene_de_gene_10x must produce a non-empty gene table")
    if psi_table is None or psi_table.empty:
        raise ValueError("adhoc_gene_de_psi_10x must produce a non-empty psi table")

    psi_subset = psi_table.loc[psi_table["coord.intron"].astype(str) == str(coord_intron)].copy()
    if psi_subset.empty:
        raise ValueError(f"No psi DE rows found for coord.intron={coord_intron!r}")

    table = gene_table.merge(
        psi_subset[["group.pair", "delta"]],
        on="group.pair",
        how="inner",
    )
    if table.empty:
        raise ValueError(f"No overlapping gene and psi DE rows found for coord.intron={coord_intron!r}")

    table["change"] = "Gene/SJ n.s."
    pos_gene = table["log2fc"] > log2fc_gene
    neg_gene = table["log2fc"] < -log2fc_gene
    pos_sj = table["delta"] > delta_sj
    neg_sj = table["delta"] < -delta_sj

    table.loc[pos_gene & pos_sj | neg_gene & neg_sj, "change"] = "Coordinated"
    table.loc[pos_gene & neg_sj | neg_gene & pos_sj, "change"] = "Opposing"
    table.loc[(table["change"] == "Gene/SJ n.s.") & (table["delta"].abs() > delta_sj), "change"] = "Iso-Swicth"

    state.setdefault("de", {})
    state["de"]["volcano"] = {
        "table": table[["group.pair", "log2fc", "delta", "change"]].reset_index(drop=True),
        "plot": None,
        "coord.intron": str(coord_intron),
    }
    return marvel_object


def adhoc_gene_plot_sj_position_10x(
    marvel_object: Marvel10x,
    *,
    coord_intron: str,
    coord_intron_ext: int = 25,
    rescale_introns: bool = False,
    show_protein_coding_only: bool = True,
    anno_label_size: float = 3.0,
    anno_colors: tuple[str, str, str] | None = None,
) -> Marvel10x:
    _ = coord_intron_ext, rescale_introns, anno_label_size, anno_colors

    state = _ensure_adhoc_gene_state(marvel_object)
    state["sj_position"] = _build_sj_position_payload(
        marvel_object,
        coord_intron=coord_intron,
        coord_intron_ext=coord_intron_ext,
        show_protein_coding_only=show_protein_coding_only,
    )
    return marvel_object
