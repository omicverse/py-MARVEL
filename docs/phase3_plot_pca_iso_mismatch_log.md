# Phase 3 Plot/PCA/Iso Mismatch Log

## Audit Rules

- Every in-scope Phase 3 family must be classified in the committed `comparison_status` manifests as either `exact` or `partial`.
- Every `partial` family must have exactly one status row below.
- Allowed dispositions are `accepted` and `fix-required`.
- No uncategorized drift is allowed to remain outside this log.

## 2026-04-21 Status

| Date | Function Family | Reference Artifact | Python Artifact | Observed Difference | Disposition |
| --- | --- | --- | --- | --- | --- |
| 2026-04-21 | PropModality | `tests/r_reference/phase3_plate/prop_modality.tsv` | `tests/test_phase3_plate_r_compare.py` | Upstream R keeps `modality` as Missing/NA on the tiny shared fixture while Python resolves concrete dispersed labels | accepted |
| 2026-04-21 | ModalityChange | `tests/r_reference/phase3_plate/modality_change.tsv` | `tests/test_phase3_modality_iso_contract.py` | Reference is exported from `marvel.demo.rds`; current Python contract covers slot population, not full demo-state parity | accepted |
| 2026-04-21 | IsoSwitch | `tests/r_reference/phase3_plate/iso_switch.tsv` | `tests/test_phase3_modality_iso_contract.py` | Reference is exported from `marvel.demo.rds`; current Python contract covers classification-slot behavior on simplified inputs, not full demo-state parity | accepted |
| 2026-04-21 | IsoSwitch.PlotExpr | `tests/r_reference/phase3_plate/iso_switch_plot_expr.tsv` | `tests/test_phase3_modality_iso_contract.py` | Upstream artifact is a stable raw plotting table; Python currently stores filtered iso-switch plot payloads instead of reproducing the ggplot object contract | accepted |
| 2026-04-21 | PlotDEValues.SJ.10x | `tests/r_reference/phase3_droplet/plot_de_values_sj.tsv` | `tests/test_phase3_droplet_r_compare.py` | `pval` and derived mean-expression slices inherit the accepted `CompareValues.SJ.10x` drift from Phase 2; classification columns remain stable parity targets | accepted |
| 2026-04-21 | PlotValues.PSI.Pseudobulk.Heatmap.10x | `tests/r_reference/phase3_droplet/plot_values_psi_pseudobulk_heatmap.tsv` | `tests/test_phase3_droplet_r_compare.py` | Python exposes a donor-map-driven long-table helper instead of the exact `x.column` / `y.column` heatmap API from R | accepted |
| 2026-04-21 | IsoSwitch.10x | `tests/r_reference/phase3_droplet/iso_switch_10x.tsv` | `tests/test_phase3_droplet_r_compare.py` | The iso-switch table inherits the accepted Phase 2 splice-junction p-value drift; stable classification columns and summary counts remain exact | accepted |
