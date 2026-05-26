# Benchmark

This directory keeps only the lightweight benchmark materials needed for review.

## Contents

- `scripts/`: benchmark code used to rebuild the R-vs-Python comparison.
- `results/figures/`: retained benchmark result images from the validated external replay.

Raw R/Python replay tables, copied input/output matrices, per-artifact TSVs, logs, and archived duplicate run directories were removed to reduce disk use. Regenerate them with `scripts/run_external_benchmark_archive.py` when a full replay is needed.

## Retained Result Figures

- `benchmark_summary_figure.png`
- `benchmark_summary_figure.pdf`
- `benchmark_summary_figure.svg`
- `artifact_row_delta.png`
- `top20_metric_max_abs_diff.png`
- `lowest20_metric_pearson_r.png`
- `plate_metric_heatmap.png`
- `droplet_metric_heatmap.png`
