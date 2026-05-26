from __future__ import annotations

import pandas as pd

from .matrix import Marvel10x
from .models import MarvelPlate

__all__ = [
    "classify_iso_switch",
    "iso_switch",
    "iso_switch_10x",
    "iso_switch_plot_expr",
    "label_droplet_sj",
]


def label_droplet_sj(
    df: pd.DataFrame,
    *,
    pval: float,
    delta: float,
    min_gene_norm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.copy()
    labeled["sig"] = "n.s."

    expressed = labeled["mean.expr.gene.norm.g1.g2"] > min_gene_norm
    up = (labeled["pval"] < pval) & (labeled["delta"] > delta) & expressed
    down = (labeled["pval"] < pval) & (labeled["delta"] < -delta) & expressed

    labeled.loc[up, "sig"] = "up"
    labeled.loc[down, "sig"] = "down"
    labeled["label"] = pd.NA

    summary = labeled["sig"].value_counts(dropna=False).rename_axis("sig").reset_index(name="freq")
    summary = summary.sort_values(
        by="sig",
        key=lambda values: pd.Categorical(values, categories=["up", "down", "n.s."], ordered=True),
    ).reset_index(drop=True)
    return labeled, summary


def classify_iso_switch(
    df: pd.DataFrame,
    *,
    pval_sj: float,
    delta_sj: float,
    min_gene_norm: float,
    pval_adj_gene: float,
    log2fc_gene: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled, _ = label_droplet_sj(df, pval=pval_sj, delta=delta_sj, min_gene_norm=min_gene_norm)
    labeled = labeled[labeled["sig"].isin(["up", "down"])].copy()

    labeled["sig.sj"] = labeled["sig"]
    labeled["sig.gene"] = "n.s."
    labeled.loc[
        (labeled["pval.adj.gene.norm"] < pval_adj_gene) & (labeled["log2fc.gene.norm"] > log2fc_gene),
        "sig.gene",
    ] = "up"
    labeled.loc[
        (labeled["pval.adj.gene.norm"] < pval_adj_gene) & (labeled["log2fc.gene.norm"] < -log2fc_gene),
        "sig.gene",
    ] = "down"

    labeled["cor"] = pd.NA
    labeled.loc[(labeled["sig.sj"] == "up") & (labeled["sig.gene"] == "up"), "cor"] = "Coordinated"
    labeled.loc[(labeled["sig.sj"] == "down") & (labeled["sig.gene"] == "down"), "cor"] = "Coordinated"
    labeled.loc[(labeled["sig.sj"] == "down") & (labeled["sig.gene"] == "up"), "cor"] = "Opposing"
    labeled.loc[(labeled["sig.sj"] == "up") & (labeled["sig.gene"] == "down"), "cor"] = "Opposing"
    labeled.loc[(labeled["sig.sj"] == "up") & (labeled["sig.gene"] == "n.s."), "cor"] = "Iso-Switch"
    labeled.loc[(labeled["sig.sj"] == "down") & (labeled["sig.gene"] == "n.s."), "cor"] = "Iso-Switch"

    gene_summary = []
    for gene_short_name, group in labeled.groupby("gene_short_name", sort=False):
        relations = [value for value in group["cor"].dropna().astype(str).tolist() if value]
        if not relations:
            relation = "Complex"
        elif len(set(relations)) == 1:
            relation = relations[0]
        else:
            relation = "Complex"
        gene_summary.append({"sj.gene.cor": relation, "gene_short_name": gene_short_name})

    gene_summary_df = pd.DataFrame(gene_summary)
    if gene_summary_df.empty:
        summary = pd.DataFrame(columns=["sj.gene.cor", "freq", "pct"])
        labeled["cor.complete"] = pd.Series(dtype=object)
        return labeled, summary

    cor_lookup = gene_summary_df.set_index("gene_short_name")["sj.gene.cor"]
    labeled["cor.complete"] = labeled["gene_short_name"].map(cor_lookup)

    unique_gene_summary = gene_summary_df.drop_duplicates("gene_short_name")
    summary = (
        unique_gene_summary["sj.gene.cor"]
        .value_counts()
        .rename_axis("sj.gene.cor")
        .reset_index(name="freq")
    )
    summary["pct"] = summary["freq"] / summary["freq"].sum() * 100.0
    summary["sj.gene.cor"] = pd.Categorical(
        summary["sj.gene.cor"],
        categories=["Coordinated", "Opposing", "Iso-Switch", "Complex"],
        ordered=True,
    )
    summary = summary.sort_values("sj.gene.cor").reset_index(drop=True)
    summary["sj.gene.cor"] = summary["sj.gene.cor"].astype(str)
    return labeled, summary


def iso_switch(
    marvel_object: MarvelPlate,
    *,
    method: str,
    psi_pval: float = 0.1,
    psi_delta: float = 0.0,
    gene_pval: float = 0.1,
    gene_log2fc: float = 0.5,
    event_type: str | None = None,
    custom_tran_ids: list[str] | None = None,
) -> MarvelPlate:
    method_key = str(method).lower()
    if method_key not in marvel_object.de_splicing:
        raise ValueError(f"Missing splicing DE results for method={method}")
    if marvel_object.de_gene is None:
        raise ValueError("compare_values(level='gene') must run before iso_switch")

    splicing = marvel_object.de_splicing[method_key].copy()
    if event_type is not None and "event_type" in splicing.columns:
        splicing = splicing[splicing["event_type"].astype(str) == str(event_type).upper()].copy()
    if custom_tran_ids is not None and "tran_id" in splicing.columns:
        allowed = {str(tran_id) for tran_id in custom_tran_ids}
        splicing = splicing[splicing["tran_id"].astype(str).isin(allowed)].copy()

    if "p.val.adj" not in splicing.columns or "mean.diff" not in splicing.columns:
        raise ValueError("iso_switch requires p.val.adj and mean.diff in splicing DE table")

    splicing["sig.sj"] = "n.s."
    splicing.loc[(splicing["p.val.adj"] < psi_pval) & (splicing["mean.diff"] > psi_delta), "sig.sj"] = "up"
    splicing.loc[(splicing["p.val.adj"] < psi_pval) & (splicing["mean.diff"] < -psi_delta), "sig.sj"] = "down"
    splicing = splicing[splicing["sig.sj"].isin(["up", "down"])].copy()

    genes = marvel_object.de_gene.copy()
    if "gene_short_name" not in genes.columns or "p.val.adj" not in genes.columns or "log2fc" not in genes.columns:
        raise ValueError("iso_switch requires gene_short_name, p.val.adj, and log2fc in gene DE table")
    genes = genes[["gene_short_name", "p.val.adj", "log2fc"]].rename(
        columns={"p.val.adj": "gene_p.val.adj", "log2fc": "gene_log2fc"}
    )

    table = splicing.merge(genes, on="gene_short_name", how="left")
    table["sig.gene"] = "n.s."
    table.loc[
        (table["gene_p.val.adj"] < gene_pval) & (table["gene_log2fc"] > gene_log2fc),
        "sig.gene",
    ] = "up"
    table.loc[
        (table["gene_p.val.adj"] < gene_pval) & (table["gene_log2fc"] < -gene_log2fc),
        "sig.gene",
    ] = "down"

    table["cor.complete"] = "Complex"
    table.loc[(table["sig.sj"] == "up") & (table["sig.gene"] == "up"), "cor.complete"] = "Coordinated"
    table.loc[(table["sig.sj"] == "down") & (table["sig.gene"] == "down"), "cor.complete"] = "Coordinated"
    table.loc[(table["sig.sj"] == "up") & (table["sig.gene"] == "down"), "cor.complete"] = "Opposing"
    table.loc[(table["sig.sj"] == "down") & (table["sig.gene"] == "up"), "cor.complete"] = "Opposing"
    table.loc[(table["sig.sj"] != "n.s.") & (table["sig.gene"] == "n.s."), "cor.complete"] = "Iso-Switch"

    summary = (
        table[["gene_short_name", "cor.complete"]]
        .drop_duplicates()
        .groupby("cor.complete", dropna=False)
        .size()
        .rename("freq")
        .reset_index()
    )
    total = float(summary["freq"].sum())
    summary["pct"] = summary["freq"] / total * 100.0 if total > 0.0 else 0.0
    marvel_object.iso_switch = {"table": table, "summary": summary, "plot": None}
    return marvel_object


def iso_switch_plot_expr(
    marvel_object: MarvelPlate,
    *,
    gene_short_name: str | None = None,
) -> MarvelPlate:
    if not marvel_object.iso_switch:
        raise ValueError("iso_switch must run before iso_switch_plot_expr")
    table = marvel_object.iso_switch["table"].copy()
    if gene_short_name is not None and "gene_short_name" in table.columns:
        table = table[table["gene_short_name"].astype(str) == str(gene_short_name)].copy()
    marvel_object.iso_switch["plot_expr"] = {"table": table, "plot": None}
    return marvel_object


def iso_switch_10x(
    marvel_object: Marvel10x,
    *,
    pval_sj: float,
    delta_sj: float,
    min_gene_norm: float,
    pval_adj_gene: float,
    log2fc_gene: float,
) -> Marvel10x:
    if marvel_object.de_sj is None:
        raise ValueError("compare_values_sj_10x or compare_values_sj must run before iso_switch_10x")

    table, summary = classify_iso_switch(
        marvel_object.de_sj,
        pval_sj=pval_sj,
        delta_sj=delta_sj,
        min_gene_norm=min_gene_norm,
        pval_adj_gene=pval_adj_gene,
        log2fc_gene=log2fc_gene,
    )
    marvel_object.iso_switch = {"table": table, "summary": summary, "plot": None}
    return marvel_object
