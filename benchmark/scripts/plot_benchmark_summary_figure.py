from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "benchmark" / "runs"


def _latest_run_dir() -> Path:
    runs = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not runs:
        raise FileNotFoundError(f"no benchmark run directories found under {RUNS_ROOT}")
    return runs[-1]


def _artifact_label(row: pd.Series) -> str:
    return f"{row['section']}/{row['artifact']}"


def _section_colors(values: pd.Series) -> list[str]:
    palette = {"plate": "#4C78A8", "droplet": "#F58518"}
    return [palette.get(str(value), "0.45") for value in values]


def _format_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_summary_figure(run_dir: Path) -> dict[str, Path]:
    report_dir = run_dir / "results" / "report"
    artifact_path = report_dir / "artifact_summary.tsv"
    metric_path = report_dir / "metric_summary.tsv"
    if not artifact_path.exists():
        raise FileNotFoundError(f"missing artifact summary: {artifact_path}")
    if not metric_path.exists():
        raise FileNotFoundError(f"missing metric summary: {metric_path}")

    artifact_df = pd.read_csv(artifact_path, sep="\t")
    metric_df = pd.read_csv(metric_path, sep="\t")
    artifact_df["label"] = artifact_df.apply(_artifact_label, axis=1)
    artifact_df = artifact_df.sort_values(["section", "artifact"]).reset_index(drop=True)

    top_metric_df = (
        metric_df.sort_values("max_abs_diff", ascending=False)
        .head(10)
        .copy()
        .sort_values("max_abs_diff")
        .reset_index(drop=True)
    )
    top_metric_df["label"] = (
        top_metric_df["section"] + "/" + top_metric_df["artifact"] + "/" + top_metric_df["metric"]
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0], wspace=0.48, hspace=0.55)
    ax_delta = fig.add_subplot(gs[0, 0])
    ax_diff = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])

    labels = artifact_df["label"].tolist()
    y = np.arange(len(artifact_df))
    colors = _section_colors(artifact_df["section"])

    ax_delta.barh(y, artifact_df["row_delta"], color=colors, alpha=0.9)
    ax_delta.axvline(0, color="0.25", linewidth=0.9)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels, fontsize=7)
    ax_delta.set_xlabel("Python rows - R rows")
    ax_delta.set_title("A  Row Count Delta", loc="left", fontweight="bold")
    _format_axis(ax_delta)

    max_diff = artifact_df["max_metric_abs_diff"].astype(float).replace(0, np.nan)
    floor = max(float(max_diff.min(skipna=True)) * 0.1, 1e-16)
    log_diff = np.log10(artifact_df["max_metric_abs_diff"].astype(float).clip(lower=floor))
    ax_diff.barh(y, log_diff, color=colors, alpha=0.9)
    ax_diff.set_yticks(y)
    ax_diff.set_yticklabels(labels, fontsize=7)
    ax_diff.set_xlabel("log10(max absolute numeric difference)")
    ax_diff.set_title("B  Artifact-Level Difference", loc="left", fontweight="bold")
    _format_axis(ax_diff)

    corr = artifact_df["min_metric_pearson_r"].astype(float)
    ax_corr.scatter(corr, y, color=colors, s=42, edgecolor="white", linewidth=0.7)
    ax_corr.axvline(0.99, color="0.65", linestyle="--", linewidth=0.9)
    ax_corr.set_xlim(max(0.65, np.nanmin(corr) - 0.04), 1.01)
    ax_corr.set_yticks(y)
    ax_corr.set_yticklabels(labels, fontsize=7)
    ax_corr.set_xlabel("Minimum Pearson r across numeric metrics")
    ax_corr.set_title("C  Numeric Concordance", loc="left", fontweight="bold")
    _format_axis(ax_corr)

    top_colors = _section_colors(top_metric_df["section"])
    top_values = top_metric_df["max_abs_diff"].astype(float).clip(lower=floor)
    ax_top.barh(np.arange(len(top_metric_df)), np.log10(top_values), color=top_colors, alpha=0.9)
    ax_top.set_yticks(np.arange(len(top_metric_df)))
    ax_top.set_yticklabels(top_metric_df["label"], fontsize=6.6)
    ax_top.set_xlabel("log10(max absolute difference)")
    ax_top.set_title("D  Top Metric-Level Differences", loc="left", fontweight="bold")
    _format_axis(ax_top)

    handles = [
        mpl.patches.Patch(facecolor="#4C78A8", label="plate"),
        mpl.patches.Patch(facecolor="#F58518", label="droplet"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 1.0))
    fig.suptitle("R vs Python MARVEL Benchmark Summary", y=1.035, fontsize=12, fontweight="bold")

    outputs = {"png": report_dir / "benchmark_summary_figure.png"}
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact R-vs-Python benchmark summary figure.")
    parser.add_argument("--run-dir", type=Path, default=None, help="benchmark/runs/<run_id> directory")
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir()
    outputs = plot_summary_figure(run_dir)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
