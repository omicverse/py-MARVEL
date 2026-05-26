from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "external"
REPORT_ROOT = BENCHMARK_ROOT / "report"
FIGURES_ROOT = REPO_ROOT / "benchmark" / "results" / "figures"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _load_summary() -> dict:
    return json.loads((BENCHMARK_ROOT / "summary.json").read_text(encoding="utf-8"))


def _metrics_rows(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section in summary["sections"]:
        section_name = section["section"]
        for artifact in section["artifacts"]:
            artifact_name = artifact["artifact"]
            metrics = artifact["numeric_metrics"]
            for metric_name, values in metrics.items():
                rows.append(
                    {
                        "section": section_name,
                        "artifact": artifact_name,
                        "metric": metric_name,
                        "n": values["n"],
                        "max_abs_diff": values["max_abs_diff"],
                        "mean_abs_diff": values["mean_abs_diff"],
                        "rmse": values["rmse"],
                        "pearson_r": np.nan if values["pearson_r"] is None else values["pearson_r"],
                        "rows_r": artifact["rows_r"],
                        "rows_python": artifact["rows_python"],
                    }
                )
    return pd.DataFrame(rows)


def _artifact_rows(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section in summary["sections"]:
        section_name = section["section"]
        for artifact in section["artifacts"]:
            metric_values = artifact["numeric_metrics"]
            max_metric_diff = max((v["max_abs_diff"] for v in metric_values.values()), default=np.nan)
            mean_metric_diff = np.nanmean([v["mean_abs_diff"] for v in metric_values.values()]) if metric_values else np.nan
            min_corr = np.nanmin(
                [
                    np.nan if v["pearson_r"] is None else v["pearson_r"]
                    for v in metric_values.values()
                ]
            ) if metric_values else np.nan
            rows.append(
                {
                    "section": section_name,
                    "artifact": artifact["artifact"],
                    "rows_r": artifact["rows_r"],
                    "rows_python": artifact["rows_python"],
                    "row_delta": artifact["rows_python"] - artifact["rows_r"],
                    "max_metric_abs_diff": max_metric_diff,
                    "mean_metric_abs_diff": mean_metric_diff,
                    "min_metric_pearson_r": min_corr,
                    "metric_count": len(metric_values),
                }
            )
    return pd.DataFrame(rows)


def _save_tables(metrics_df: pd.DataFrame, artifact_df: pd.DataFrame) -> None:
    metrics_df.to_csv(REPORT_ROOT / "metric_summary.tsv", sep="\t", index=False)
    artifact_df.to_csv(REPORT_ROOT / "artifact_summary.tsv", sep="\t", index=False)


def _plot_top_metric_diffs(metrics_df: pd.DataFrame) -> None:
    top = metrics_df.sort_values("max_abs_diff", ascending=False).head(20).copy()
    top["label"] = top["section"] + "/" + top["artifact"] + "/" + top["metric"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["label"], top["max_abs_diff"], color="#c44e52")
    ax.invert_yaxis()
    ax.set_xlabel("Max absolute difference")
    ax.set_title("Top 20 Metric Differences")
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "top20_metric_max_abs_diff.png", dpi=180)
    plt.close(fig)


def _plot_top_metric_corr(metrics_df: pd.DataFrame) -> None:
    corr_df = metrics_df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=True).head(20).copy()
    corr_df["label"] = corr_df["section"] + "/" + corr_df["artifact"] + "/" + corr_df["metric"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(corr_df["label"], corr_df["pearson_r"], color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.01)
    ax.set_xlabel("Pearson r")
    ax.set_title("Lowest 20 Metric Correlations")
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "lowest20_metric_pearson_r.png", dpi=180)
    plt.close(fig)


def _plot_artifact_row_deltas(artifact_df: pd.DataFrame) -> None:
    plot_df = artifact_df.copy()
    plot_df["label"] = plot_df["section"] + "/" + plot_df["artifact"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#dd8452" if x != 0 else "#55a868" for x in plot_df["row_delta"]]
    ax.bar(plot_df["label"], plot_df["row_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Python rows - R rows")
    ax.set_title("Artifact Row Count Differences")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "artifact_row_delta.png", dpi=180)
    plt.close(fig)


def _plot_section_heatmap(metrics_df: pd.DataFrame, section: str) -> None:
    section_df = metrics_df[metrics_df["section"] == section].copy()
    if section_df.empty:
        return

    heatmap_df = section_df.pivot_table(
        index="artifact",
        columns="metric",
        values="max_abs_diff",
        aggfunc="max",
        fill_value=np.nan,
    )
    if heatmap_df.empty:
        return

    display_df = np.log10(heatmap_df.replace(0, np.nan))
    finite_values = display_df.to_numpy()
    finite_values = finite_values[np.isfinite(finite_values)]
    vmin = np.nanmin(finite_values) if finite_values.size else -6
    vmax = np.nanmax(finite_values) if finite_values.size else 0

    fig_w = max(8, 0.45 * len(heatmap_df.columns))
    fig_h = max(4, 0.55 * len(heatmap_df.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(display_df.to_numpy(), aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=9)
    ax.set_title(f"{section.capitalize()} Metric Difference Heatmap (log10 max_abs_diff)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(max_abs_diff)")
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / f"{section}_metric_heatmap.png", dpi=180)
    plt.close(fig)


def _runtime_rows(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section in summary.get("sections", []):
        runtime = section.get("runtime_seconds") or {}
        section_name = str(section.get("section", "unknown"))
        for implementation, seconds in runtime.items():
            if seconds is None:
                continue
            rows.append(
                {
                    "section": section_name,
                    "implementation": "R" if implementation == "r" else "Python",
                    "seconds": float(seconds),
                }
            )
    return pd.DataFrame(rows)


def _plot_runtime_comparison(summary: dict) -> None:
    runtime_df = _runtime_rows(summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if runtime_df.empty:
        ax.text(0.5, 0.5, "No runtime data in benchmark summary", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(REPORT_ROOT / "runtime_r_vs_python.png", dpi=180)
        plt.close(fig)
        return

    sections = list(dict.fromkeys(runtime_df["section"].astype(str).tolist()))
    implementations = ["R", "Python"]
    x = np.arange(len(sections), dtype=float)
    width = 0.36
    colors = {"R": "#4C78A8", "Python": "#F58518"}
    for idx, implementation in enumerate(implementations):
        values = []
        for section in sections:
            match = runtime_df[
                (runtime_df["section"] == section) & (runtime_df["implementation"] == implementation)
            ]
            values.append(float(match["seconds"].iloc[0]) if not match.empty else np.nan)
        offset = (idx - 0.5) * width
        ax.bar(x + offset, values, width=width, label=implementation, color=colors[implementation], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([section.capitalize() for section in sections])
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("R vs Python Runtime")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "runtime_r_vs_python.png", dpi=180)
    plt.close(fig)


def _plot_variable_splicing_comparison() -> None:
    comparison_root = _reset_dir(BENCHMARK_ROOT / "comparison" / "plate_variable_splicing")
    r_root = BENCHMARK_ROOT / "r_runs" / "plate"
    py_root = BENCHMARK_ROOT / "python_runs" / "plate"
    r_table_path = r_root / "variable_splicing_table.tsv"
    py_table_path = py_root / "variable_splicing_table.tsv"
    r_ids_path = r_root / "variable_splicing_tran_ids.tsv"
    py_ids_path = py_root / "variable_splicing_tran_ids.tsv"
    if not all(path.exists() for path in (r_table_path, py_table_path, r_ids_path, py_ids_path)):
        return

    r_table = pd.read_csv(r_table_path, sep="\t")
    py_table = pd.read_csv(py_table_path, sep="\t")
    r_ids = set(pd.read_csv(r_ids_path, sep="\t")["tran_id"].astype(str))
    py_ids = set(pd.read_csv(py_ids_path, sep="\t")["tran_id"].astype(str))
    shared_ids = sorted(r_ids & py_ids)
    r_only_ids = sorted(r_ids - py_ids)
    py_only_ids = sorted(py_ids - r_ids)

    pd.DataFrame({"tran_id": shared_ids}).to_csv(comparison_root / "shared_variable_tran_ids.tsv", sep="\t", index=False)
    pd.DataFrame({"tran_id": r_only_ids}).to_csv(comparison_root / "r_only_variable_tran_ids.tsv", sep="\t", index=False)
    pd.DataFrame({"tran_id": py_only_ids}).to_csv(comparison_root / "python_only_variable_tran_ids.tsv", sep="\t", index=False)

    keep = ["tran_id", "mean", "sd", "sd_pred", "variable"]
    merged = r_table[[column for column in keep if column in r_table.columns]].merge(
        py_table[[column for column in keep if column in py_table.columns]],
        on="tran_id",
        suffixes=("_r", "_python"),
        how="inner",
    )
    merged["selection"] = np.select(
        [
            merged["tran_id"].astype(str).isin(shared_ids),
            merged["tran_id"].astype(str).isin(r_only_ids),
            merged["tran_id"].astype(str).isin(py_only_ids),
        ],
        ["shared", "R only", "Python only"],
        default="not variable",
    )
    merged.to_csv(comparison_root / "merged_variable_splicing_table.tsv", sep="\t", index=False)

    summary = {
        "r_variable_events": len(r_ids),
        "python_variable_events": len(py_ids),
        "shared_variable_events": len(shared_ids),
        "r_only_variable_events": len(r_only_ids),
        "python_only_variable_events": len(py_only_ids),
        "overlap_fraction_of_r": len(shared_ids) / len(r_ids) if r_ids else None,
        "overlap_fraction_of_python": len(shared_ids) / len(py_ids) if py_ids else None,
    }
    (comparison_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = ["Shared", "R only", "Python only"]
    values = [len(shared_ids), len(r_only_ids), len(py_only_ids)]
    ax.bar(labels, values, color=["#55a868", "#4C78A8", "#F58518"], alpha=0.9)
    ax.set_ylabel("Variable events")
    ax.set_title("Plate IdentifyVariableEvents Overlap")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(comparison_root / "overlap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = {"shared": "#55a868", "R only": "#4C78A8", "Python only": "#F58518", "not variable": "0.72"}
    for label in ["not variable", "shared", "R only", "Python only"]:
        subset = merged[merged["selection"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["sd_pred_r"],
            subset["sd_pred_python"],
            s=8 if label == "not variable" else 14,
            alpha=0.25 if label == "not variable" else 0.75,
            color=colors[label],
            label=label,
        )
    finite = merged[["sd_pred_r", "sd_pred_python"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lo = float(finite.min().min())
        hi = float(finite.max().max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("R sd_pred")
    ax.set_ylabel("Python sd_pred")
    ax.set_title("Plate Variable Splicing Smooth Fit")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(comparison_root / "scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_df = merged.sort_values("mean_r").copy()
    for label in ["not variable", "shared", "R only", "Python only"]:
        subset = plot_df[plot_df["selection"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["mean_r"],
            subset["sd_r"],
            s=7 if label == "not variable" else 13,
            alpha=0.18 if label == "not variable" else 0.75,
            color=colors[label],
            label=label,
        )
    ax.plot(plot_df["mean_r"], plot_df["sd_pred_r"], color="#4C78A8", linewidth=1.2, label="R threshold")
    ax.plot(
        plot_df["mean_r"],
        plot_df["sd_pred_python"],
        color="#F58518",
        linewidth=1.2,
        linestyle="--",
        label="Python threshold",
    )
    ax.set_xlabel("Mean PSI")
    ax.set_ylabel("SD PSI")
    ax.set_title("Plate Variable Splicing Threshold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(comparison_root / "threshold_scatter.png", dpi=180)
    plt.close(fig)


def _copy_png_outputs(summary: dict) -> None:
    _reset_dir(FIGURES_ROOT)
    for png_path in sorted(REPORT_ROOT.glob("*.png")):
        shutil.copy2(png_path, FIGURES_ROOT / png_path.name)
    comparison_root = BENCHMARK_ROOT / "comparison" / "plate_variable_splicing"
    comparison_outputs = {
        "overlap.png": "plate__variable_splicing_overlap.png",
        "scatter.png": "plate__variable_splicing_selection_scatter.png",
        "threshold_scatter.png": "plate__variable_splicing_threshold_scatter.png",
    }
    for input_name, output_name in comparison_outputs.items():
        png_path = comparison_root / input_name
        if png_path.exists():
            shutil.copy2(png_path, FIGURES_ROOT / output_name)
    for section in summary.get("sections", []):
        section_name = str(section.get("section", "unknown"))
        for artifact in section.get("artifacts", []):
            artifact_name = str(artifact.get("artifact", "unknown"))
            scatter_path = BENCHMARK_ROOT / section_name / artifact_name / "scatter.png"
            if scatter_path.exists():
                output_name = f"{section_name}__{artifact_name}__scatter.png"
                shutil.copy2(scatter_path, FIGURES_ROOT / output_name)


def _write_report_md(metrics_df: pd.DataFrame, artifact_df: pd.DataFrame) -> None:
    top_diff = metrics_df.sort_values("max_abs_diff", ascending=False).head(10)
    low_corr = metrics_df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=True).head(10)
    row_delta = artifact_df.loc[artifact_df["row_delta"] != 0, ["section", "artifact", "row_delta"]]

    lines = [
        "# External Benchmark Difference Report",
        "",
        "Generated from `benchmark/external/summary.json` and per-artifact `metrics.json` files.",
        "",
        "## Outputs",
        "- `top20_metric_max_abs_diff.png`",
        "- `lowest20_metric_pearson_r.png`",
        "- `artifact_row_delta.png`",
        "- `plate_metric_heatmap.png`",
        "- `droplet_metric_heatmap.png`",
        "- `metric_summary.tsv`",
        "- `artifact_summary.tsv`",
        "",
        "## Top Max Differences",
    ]
    for _, row in top_diff.iterrows():
        lines.append(
            f"- `{row['section']}/{row['artifact']}/{row['metric']}`: max_abs_diff={row['max_abs_diff']:.6g}, rmse={row['rmse']:.6g}"
        )

    lines.append("")
    lines.append("## Lowest Correlations")
    for _, row in low_corr.iterrows():
        lines.append(
            f"- `{row['section']}/{row['artifact']}/{row['metric']}`: pearson_r={row['pearson_r']:.6g}, max_abs_diff={row['max_abs_diff']:.6g}"
        )

    lines.append("")
    lines.append("## Row Count Deltas")
    if row_delta.empty:
        lines.append("- None")
    else:
        for _, row in row_delta.iterrows():
            lines.append(f"- `{row['section']}/{row['artifact']}`: python_rows - r_rows = {int(row['row_delta'])}")

    (REPORT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _reset_dir(REPORT_ROOT)
    summary = _load_summary()
    metrics_df = _metrics_rows(summary)
    artifact_df = _artifact_rows(summary)
    _save_tables(metrics_df, artifact_df)
    _plot_top_metric_diffs(metrics_df)
    _plot_top_metric_corr(metrics_df)
    _plot_artifact_row_deltas(artifact_df)
    _plot_section_heatmap(metrics_df, "plate")
    _plot_section_heatmap(metrics_df, "droplet")
    _plot_runtime_comparison(summary)
    _plot_variable_splicing_comparison()
    _write_report_md(metrics_df, artifact_df)
    _copy_png_outputs(summary)


if __name__ == "__main__":
    main()
