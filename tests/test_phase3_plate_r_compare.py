from __future__ import annotations

import numpy as np
from pandas.testing import assert_frame_equal

from marvel_py import compare_values, plot_de_values, plot_values, prop_modality, run_pca
from tests._phase3_plate_reference import REFERENCE_ROOT, build_shared_plate_object, read_manifest, read_reference


def _align_component_signs(actual, expected, columns: list[str]) -> None:
    for column in columns:
        actual_values = actual[column].to_numpy(dtype=float)
        expected_values = expected[column].to_numpy(dtype=float)
        if np.allclose(actual_values, expected_values, rtol=1e-9, atol=1e-9):
            continue
        if np.allclose(actual_values, -expected_values, rtol=1e-9, atol=1e-9):
            actual[column] = -actual[column]
            continue
        raise AssertionError(f"{column} is neither equal to nor a global sign flip of the R reference")


def test_phase3_plate_reference_outputs_exist() -> None:
    assert (REFERENCE_ROOT / "run_pca_gene_coords.tsv").exists()
    assert (REFERENCE_ROOT / "run_pca_gene_explained.tsv").exists()
    assert (REFERENCE_ROOT / "run_pca_splicing_coords.tsv").exists()
    assert (REFERENCE_ROOT / "run_pca_splicing_explained.tsv").exists()
    assert (REFERENCE_ROOT / "plot_values_gene.tsv").exists()
    assert (REFERENCE_ROOT / "plot_de_gene_global.tsv").exists()
    assert (REFERENCE_ROOT / "prop_modality.tsv").exists()
    assert (REFERENCE_ROOT / "modality_change.tsv").exists()
    assert (REFERENCE_ROOT / "iso_switch.tsv").exists()
    assert (REFERENCE_ROOT / "iso_switch_plot_expr.tsv").exists()


def test_phase3_manifest_marks_exact_and_partial_coverage_explicitly() -> None:
    manifest = read_manifest()
    assert manifest["comparison_status"]["RunPCA"] == "exact"
    assert manifest["comparison_status"]["PlotValues"] == "exact"
    assert manifest["comparison_status"]["PlotDEValues"] == "exact"
    assert manifest["comparison_status"]["PropModality"] == "partial"
    assert manifest["comparison_status"]["ModalityChange"] == "partial"
    assert manifest["comparison_status"]["IsoSwitch"] == "partial"
    assert manifest["comparison_status"]["IsoSwitch.PlotExpr"] == "partial"


def test_phase3_run_pca_gene_matches_r_reference_up_to_component_sign_flip() -> None:
    marvel = build_shared_plate_object()
    marvel = run_pca(
        marvel,
        cell_group_column="cell.type",
        features=["GENE1", "GENE2"],
        level="gene",
        min_cells=1,
    )

    actual_coords = (
        marvel.pca_results["gene"]["coords"]
        .rename(columns={"sample.id": "sample_id"})
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    expected_coords = read_reference("run_pca_gene_coords.tsv", dtype={"sample_id": str}).sort_values("sample_id").reset_index(drop=True)
    _align_component_signs(actual_coords, expected_coords, ["PC1", "PC2"])
    assert_frame_equal(actual_coords, expected_coords, check_dtype=False, rtol=1e-9, atol=1e-9)

    actual_explained = marvel.pca_results["gene"]["explained_variance"].reset_index(drop=True)
    expected_explained = read_reference("run_pca_gene_explained.tsv").reset_index(drop=True)
    assert_frame_equal(actual_explained, expected_explained, check_dtype=False, rtol=1e-9, atol=1e-9)


def test_phase3_run_pca_splicing_matches_r_reference_up_to_component_sign_flip() -> None:
    marvel = build_shared_plate_object()
    marvel = run_pca(
        marvel,
        cell_group_column="cell.type",
        features=[
            "chr1:100:119:+@chr1:200:219:+@chr1:300:319",
            "chr2:500:519:-@chr2:400:419:-@chr2:300:319",
        ],
        level="splicing",
        min_cells=1,
        method_impute="random",
        seed=1,
    )

    actual_coords = (
        marvel.pca_results["splicing"]["coords"]
        .rename(columns={"sample.id": "sample_id"})
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    expected_coords = read_reference("run_pca_splicing_coords.tsv", dtype={"sample_id": str}).sort_values("sample_id").reset_index(drop=True)
    _align_component_signs(actual_coords, expected_coords, ["PC1", "PC2"])
    assert_frame_equal(actual_coords, expected_coords, check_dtype=False, rtol=1e-9, atol=1e-9)

    actual_explained = marvel.pca_results["splicing"]["explained_variance"].reset_index(drop=True)
    expected_explained = read_reference("run_pca_splicing_explained.tsv").reset_index(drop=True)
    assert_frame_equal(actual_explained, expected_explained, check_dtype=False, rtol=1e-9, atol=1e-9)


def test_phase3_plot_values_gene_matches_r_reference() -> None:
    marvel = build_shared_plate_object()
    marvel = plot_values(
        marvel,
        cell_group_list={"iPSC": ["s1", "s2"], "Endoderm": ["s3", "s4"]},
        feature="GENE1",
        level="gene",
    )

    assert_frame_equal(
        marvel.value_plots["gene"]["table"].reset_index(drop=True),
        read_reference("plot_values_gene.tsv").reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_phase3_plot_de_gene_global_matches_r_reference() -> None:
    marvel = build_shared_plate_object()
    marvel = compare_values(
        marvel,
        cell_group_g1=["s1", "s2"],
        cell_group_g2=["s3", "s4"],
        min_cells=1,
        method="wilcox",
        method_adjust="fdr",
        level="gene",
    )
    marvel = plot_de_values(marvel, level="gene.global", pval=0.1, log2fc=0.5)

    assert_frame_equal(
        marvel.de_plots["gene.global"]["table"].reset_index(drop=True),
        read_reference("plot_de_gene_global.tsv").reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_phase3_prop_modality_reference_captures_known_r_drift() -> None:
    marvel = build_shared_plate_object()
    marvel = prop_modality(
        marvel,
        modality_column="modality.bimodal.adj",
        modality_type="extended",
        event_type=["SE"],
        across_event_type=False,
    )

    reference = read_reference("prop_modality.tsv")
    assert reference["freq"].sum() == 2
    assert reference["pct"].sum() == 100.0
    assert reference["modality"].isna().all()

    actual_modalities = set(marvel.modality_prop["modality"])
    assert actual_modalities == {"Excluded.Dispersed", "Included.Dispersed"}


def test_phase3_demo_references_for_modality_change_and_iso_switch_are_non_empty() -> None:
    modality_change = read_reference("modality_change.tsv")
    iso_switch = read_reference("iso_switch.tsv")
    iso_switch_plot_expr = read_reference("iso_switch_plot_expr.tsv")

    assert not modality_change.empty
    assert not iso_switch.empty
    assert not iso_switch_plot_expr.empty

    assert {"tran_id", "event_type", "modality.change"}.issubset(modality_change.columns)
    assert {"gene_id", "gene_short_name", "cor"}.issubset(iso_switch.columns)
    assert {"gene_short_name", "mean.diff", "log2fc.gene", "cor"}.issubset(iso_switch_plot_expr.columns)
