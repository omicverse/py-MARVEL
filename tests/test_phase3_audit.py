from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_ROOT = REPO_ROOT / "tests" / "r_reference" / "phase3_plate"
DROPLET_ROOT = REPO_ROOT / "tests" / "r_reference" / "phase3_droplet"
MISMATCH_LOG = REPO_ROOT / "docs" / "phase3_plot_pca_iso_mismatch_log.md"
PLATE_README = PLATE_ROOT / "README.md"
DROPLET_README = DROPLET_ROOT / "README.md"


def _parse_status_rows(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("| 2026-04-21 |"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 6:
            continue
        _, function_family, reference_artifact, python_artifact, observed_difference, disposition = parts
        rows[function_family] = {
            "reference_artifact": reference_artifact,
            "python_artifact": python_artifact,
            "observed_difference": observed_difference,
            "disposition": disposition,
        }
    return rows


def test_phase3_mismatch_log_exists_and_matches_manifest_partial_families() -> None:
    assert MISMATCH_LOG.exists(), f"missing mismatch log: {MISMATCH_LOG}"

    text = MISMATCH_LOG.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "# Phase 3 Plot/PCA/Iso Mismatch Log" in text
    assert "## Audit Rules" in text
    assert "## 2026-04-21 Status" in text
    for forbidden in ("todo", "tbd", "placeholder"):
        assert forbidden not in lowered

    plate_manifest = json.loads((PLATE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    droplet_manifest = json.loads((DROPLET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    partial_from_manifests = {
        name
        for manifest in (plate_manifest, droplet_manifest)
        for name, status in manifest["comparison_status"].items()
        if status == "partial"
    }

    status_rows = _parse_status_rows(text)
    assert set(status_rows) == partial_from_manifests

    expected_partial = {
        "PropModality",
        "ModalityChange",
        "IsoSwitch",
        "IsoSwitch.PlotExpr",
        "PlotDEValues.SJ.10x",
        "PlotValues.PSI.Pseudobulk.Heatmap.10x",
        "IsoSwitch.10x",
    }
    assert partial_from_manifests == expected_partial

    assert status_rows["PropModality"]["reference_artifact"] == "`tests/r_reference/phase3_plate/prop_modality.tsv`"
    assert "Missing/NA" in status_rows["PropModality"]["observed_difference"]
    assert status_rows["PropModality"]["disposition"] == "accepted"

    assert status_rows["ModalityChange"]["python_artifact"] == "`tests/test_phase3_modality_iso_contract.py`"
    assert "marvel.demo.rds" in status_rows["ModalityChange"]["observed_difference"]

    assert status_rows["IsoSwitch"]["python_artifact"] == "`tests/test_phase3_modality_iso_contract.py`"
    assert "simplified inputs" in status_rows["IsoSwitch"]["observed_difference"]

    assert status_rows["IsoSwitch.PlotExpr"]["reference_artifact"] == "`tests/r_reference/phase3_plate/iso_switch_plot_expr.tsv`"
    assert "ggplot object" in status_rows["IsoSwitch.PlotExpr"]["observed_difference"]

    assert status_rows["PlotDEValues.SJ.10x"]["reference_artifact"] == "`tests/r_reference/phase3_droplet/plot_de_values_sj.tsv`"
    assert "CompareValues.SJ.10x" in status_rows["PlotDEValues.SJ.10x"]["observed_difference"]

    assert status_rows["PlotValues.PSI.Pseudobulk.Heatmap.10x"]["reference_artifact"] == "`tests/r_reference/phase3_droplet/plot_values_psi_pseudobulk_heatmap.tsv`"
    assert "x.column" in status_rows["PlotValues.PSI.Pseudobulk.Heatmap.10x"]["observed_difference"]

    assert status_rows["IsoSwitch.10x"]["reference_artifact"] == "`tests/r_reference/phase3_droplet/iso_switch_10x.tsv`"
    assert "summary counts remain exact" in status_rows["IsoSwitch.10x"]["observed_difference"]

    for row in status_rows.values():
        assert row["disposition"] == "accepted"


def test_phase3_reference_readmes_and_manifests_are_explicit_about_coverage() -> None:
    plate_readme = PLATE_README.read_text(encoding="utf-8")
    droplet_readme = DROPLET_README.read_text(encoding="utf-8")
    log_text = MISMATCH_LOG.read_text(encoding="utf-8")

    assert "prop_modality.tsv" in plate_readme
    assert "modality_change.tsv" in plate_readme
    assert "iso_switch_plot_expr.tsv" in plate_readme

    assert "plot_values_pca_*" in droplet_readme
    assert "plot_values_psi_pseudobulk_heatmap.tsv" in droplet_readme
    assert "CompareValues.SJ.10x" in droplet_readme

    required_fragments = [
        "tests/r_reference/phase3_plate/prop_modality.tsv",
        "tests/r_reference/phase3_droplet/plot_de_values_sj.tsv",
        "tests/r_reference/phase3_droplet/plot_values_psi_pseudobulk_heatmap.tsv",
        "accepted",
        "fix-required",
    ]
    for fragment in required_fragments:
        assert fragment in log_text
