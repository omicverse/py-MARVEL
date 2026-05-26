"""
marvel_py: standalone Python reimplementation of MARVEL public workflows.

The package exposes a single-layer public namespace that mirrors the style used
by `monocle2_py`: users import workflow classes and public functions directly
from `marvel_py`, while implementation details stay in module files underneath.
"""

import weakref

from .adhoc import (
    adhoc_gene_de_gene_10x,
    adhoc_gene_de_psi_10x,
    adhoc_gene_plot_de_values_10x,
    adhoc_gene_plot_sj_position_10x,
    adhoc_gene_tabulate_expression_gene_10x,
    adhoc_gene_tabulate_expression_psi_10x,
)
from .annotation import annotate_genes_10x, annotate_sj_10x, parse_gtf
from .de import (
    _normalize_methods,
    compare_values,
    compare_values_genes_10x,
    compare_values_sj_10x,
    compare_values_sj_donor_level_10x,
)
from .io import create_marvel_object, create_marvel_object_10x, maybe_read_table
from .adata import DEFAULT_INPUT_KEY, DEFAULT_RESULT_KEY, MARVEL, Splicing, setup_10x_anndata, setup_plate_anndata
from .iso import iso_switch, iso_switch_10x, iso_switch_plot_expr
from .modality import (
    assign_modality,
    count_events,
    modality_change,
    prop_modality,
    prop_modality_bar,
    prop_modality_doughnut,
)
from .plot import (
    plot_de_values,
    plot_de_values_genes_10x,
    plot_de_values_sj_10x,
    plot_pct_expr_cells_genes_10x,
    plot_pct_expr_cells_sj_10x,
    plot_values,
    plot_values_gene_pseudobulk_10x,
    plot_values_gene_single_cell_10x,
    plot_values_pca_cell_group_10x,
    plot_values_pca_gene_10x,
    plot_values_pca_psi_10x,
    plot_values_psi_pseudobulk_10x,
    plot_values_psi_pseudobulk_heatmap_10x,
    run_pca,
)
from .qc import check_alignment, check_alignment_10x, filter_genes_10x, subset_samples, transform_exp_values, validate_sj_10x
from .splicing import compute_psi, compute_psi_posterior, detect_events
from .models import MarvelPlate
from .matrix import Marvel10x, NamedMatrix
from .misc import (
    identify_variable_events,
    pct_ase,
    prepare_bed_file_ri,
    preprocess_rmats,
    preprocess_rmats_a3ss,
    preprocess_rmats_a5ss,
    preprocess_rmats_mxe,
    preprocess_rmats_ri,
    preprocess_rmats_se,
    remove_cryptic_ss,
    remove_cryptic_ss_afe,
    remove_cryptic_ss_ale,
    subset_cryptic_a3ss,
    subset_cryptic_ss,
    subset_cryptic_ss_a3ss,
    subset_cryptic_ss_a5ss,
)

__version__ = "0.1.0"

__all__ = [
    "MarvelPlate",
    "Marvel10x",
    "NamedMatrix",
    "MARVEL",
    "Splicing",
    "setup_plate_anndata",
    "setup_10x_anndata",
    "maybe_read_table",
    # plate object creation and workflow
    "create_marvel_object",
    "check_alignment",
    "subset_samples",
    "transform_exp_values",
    "detect_events",
    "compute_psi",
    "compute_psi_posterior",
    "assign_modality",
    "count_events",
    "prop_modality",
    "compare_values",
    "run_pca",
    "plot_values",
    "plot_de_values",
    "modality_change",
    "prop_modality_bar",
    "prop_modality_doughnut",
    "iso_switch",
    "iso_switch_plot_expr",
    "remove_cryptic_ss",
    "remove_cryptic_ss_afe",
    "remove_cryptic_ss_ale",
    "subset_cryptic_ss",
    "subset_cryptic_ss_a5ss",
    "subset_cryptic_ss_a3ss",
    "subset_cryptic_a3ss",
    "prepare_bed_file_ri",
    "preprocess_rmats",
    "preprocess_rmats_se",
    "preprocess_rmats_mxe",
    "preprocess_rmats_ri",
    "preprocess_rmats_a5ss",
    "preprocess_rmats_a3ss",
    "parse_gtf",
    "identify_variable_events",
    "pct_ase",
    # droplet / 10x workflow
    "create_marvel_object_10x",
    "annotate_genes_10x",
    "annotate_sj_10x",
    "validate_sj_10x",
    "filter_genes_10x",
    "check_alignment_10x",
    "compare_values_sj_10x",
    "compare_values_genes_10x",
    "compare_values_sj_donor_level_10x",
    "plot_pct_expr_cells_genes_10x",
    "plot_pct_expr_cells_sj_10x",
    "plot_de_values_genes_10x",
    "plot_de_values_sj_10x",
    "plot_values_gene_pseudobulk_10x",
    "plot_values_gene_single_cell_10x",
    "plot_values_pca_cell_group_10x",
    "plot_values_pca_gene_10x",
    "plot_values_pca_psi_10x",
    "plot_values_psi_pseudobulk_10x",
    "plot_values_psi_pseudobulk_heatmap_10x",
    "iso_switch_10x",
    "adhoc_gene_tabulate_expression_gene_10x",
    "adhoc_gene_tabulate_expression_psi_10x",
    "adhoc_gene_de_gene_10x",
    "adhoc_gene_de_psi_10x",
    "adhoc_gene_plot_de_values_10x",
    "adhoc_gene_plot_sj_position_10x",
]


_BACKEND_FUNCTIONS = {
    "check_alignment": check_alignment,
    "subset_samples": subset_samples,
    "transform_exp_values": transform_exp_values,
    "detect_events": detect_events,
    "compute_psi": compute_psi,
    "compute_psi_posterior": compute_psi_posterior,
    "assign_modality": assign_modality,
    "count_events": count_events,
    "prop_modality": prop_modality,
    "prop_modality_bar": prop_modality_bar,
    "prop_modality_doughnut": prop_modality_doughnut,
    "compare_values": compare_values,
    "run_pca": run_pca,
    "plot_values": plot_values,
    "plot_de_values": plot_de_values,
    "modality_change": modality_change,
    "iso_switch": iso_switch,
    "iso_switch_plot_expr": iso_switch_plot_expr,
    "remove_cryptic_ss": remove_cryptic_ss,
    "remove_cryptic_ss_afe": remove_cryptic_ss_afe,
    "remove_cryptic_ss_ale": remove_cryptic_ss_ale,
    "subset_cryptic_ss": subset_cryptic_ss,
    "subset_cryptic_ss_a5ss": subset_cryptic_ss_a5ss,
    "subset_cryptic_ss_a3ss": subset_cryptic_ss_a3ss,
    "subset_cryptic_a3ss": subset_cryptic_a3ss,
    "identify_variable_events": identify_variable_events,
    "pct_ase": pct_ase,
    "annotate_genes_10x": annotate_genes_10x,
    "annotate_sj_10x": annotate_sj_10x,
    "validate_sj_10x": validate_sj_10x,
    "filter_genes_10x": filter_genes_10x,
    "check_alignment_10x": check_alignment_10x,
    "compare_values_sj_10x": compare_values_sj_10x,
    "compare_values_genes_10x": compare_values_genes_10x,
    "compare_values_sj_donor_level_10x": compare_values_sj_donor_level_10x,
    "plot_pct_expr_cells_genes_10x": plot_pct_expr_cells_genes_10x,
    "plot_pct_expr_cells_sj_10x": plot_pct_expr_cells_sj_10x,
    "plot_de_values_genes_10x": plot_de_values_genes_10x,
    "plot_de_values_sj_10x": plot_de_values_sj_10x,
    "plot_values_gene_pseudobulk_10x": plot_values_gene_pseudobulk_10x,
    "plot_values_gene_single_cell_10x": plot_values_gene_single_cell_10x,
    "plot_values_pca_cell_group_10x": plot_values_pca_cell_group_10x,
    "plot_values_pca_gene_10x": plot_values_pca_gene_10x,
    "plot_values_pca_psi_10x": plot_values_pca_psi_10x,
    "plot_values_psi_pseudobulk_10x": plot_values_psi_pseudobulk_10x,
    "plot_values_psi_pseudobulk_heatmap_10x": plot_values_psi_pseudobulk_heatmap_10x,
    "iso_switch_10x": iso_switch_10x,
    "adhoc_gene_tabulate_expression_gene_10x": adhoc_gene_tabulate_expression_gene_10x,
    "adhoc_gene_tabulate_expression_psi_10x": adhoc_gene_tabulate_expression_psi_10x,
    "adhoc_gene_de_gene_10x": adhoc_gene_de_gene_10x,
    "adhoc_gene_de_psi_10x": adhoc_gene_de_psi_10x,
    "adhoc_gene_plot_de_values_10x": adhoc_gene_plot_de_values_10x,
    "adhoc_gene_plot_sj_position_10x": adhoc_gene_plot_sj_position_10x,
}
_RUNTIME_OBJECTS = {}
_RUNTIME_FINALIZERS = {}
_ANNDATA_SYNC_FUNCTIONS = {
    "check_alignment",
    "subset_samples",
    "transform_exp_values",
    "filter_genes_10x",
    "check_alignment_10x",
}


def _drop_runtime_object(key: int) -> None:
    _RUNTIME_OBJECTS.pop(key, None)
    _RUNTIME_FINALIZERS.pop(key, None)


def _cache_runtime_object(adata, marvel_object) -> None:
    key = id(adata)
    _RUNTIME_OBJECTS[key] = marvel_object
    if key not in _RUNTIME_FINALIZERS:
        _RUNTIME_FINALIZERS[key] = weakref.finalize(adata, _drop_runtime_object, key)


def _is_anndata(obj) -> bool:
    try:
        from anndata import AnnData
    except ImportError:
        return False
    return isinstance(obj, AnnData)


def _infer_anndata_mode(adata) -> str:
    for key in (DEFAULT_INPUT_KEY, DEFAULT_RESULT_KEY):
        value = adata.uns.get(key)
        if isinstance(value, dict) and value.get("mode") is not None:
            return str(value["mode"])
    return "plate"


def _controller_from_anndata(adata) -> MARVEL:
    controller = MARVEL(adata, mode=_infer_anndata_mode(adata))
    cached = _RUNTIME_OBJECTS.get(id(adata))
    if cached is not None:
        controller.object = cached
    else:
        controller.build()
    return controller


def _dispatch(name: str, obj, *args, **kwargs):
    backend = _BACKEND_FUNCTIONS[name]
    if not _is_anndata(obj):
        return backend(obj, *args, **kwargs)

    controller = _controller_from_anndata(obj)
    result = backend(controller.object, *args, **kwargs)
    if result is not None:
        controller.object = result
    controller.write(sync=name in _ANNDATA_SYNC_FUNCTIONS)
    _cache_runtime_object(obj, controller.object)
    return obj


def _make_dispatch(name: str):
    def wrapper(marvel_object, *args, **kwargs):
        return _dispatch(name, marvel_object, *args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = _BACKEND_FUNCTIONS[name].__doc__
    return wrapper


for _name in _BACKEND_FUNCTIONS:
    globals()[_name] = _make_dispatch(_name)
del _name
