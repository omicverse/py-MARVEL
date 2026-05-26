from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread

from .io import create_marvel_object, create_marvel_object_10x
from .matrix import _load_sparse_matrix

DEFAULT_INPUT_KEY = "marvel_input"
DEFAULT_RESULT_KEY = "marvel"


def _read_table(table: Any, *, dtype: Any = None) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.read_csv(table, sep="\t", dtype=dtype)


def _ensure_sample_id(obs: pd.DataFrame, sample_id_key: str = "sample.id") -> pd.DataFrame:
    obs = obs.copy()
    if sample_id_key in obs.columns:
        obs["sample.id"] = obs[sample_id_key].astype(str).to_numpy()
    elif "sample.id" in obs.columns:
        obs["sample.id"] = obs["sample.id"].astype(str).to_numpy()
    else:
        obs["sample.id"] = obs.index.astype(str)
    return obs


def _ensure_gene_id(var: pd.DataFrame, gene_id_key: str = "gene_id") -> pd.DataFrame:
    var = var.copy()
    if gene_id_key in var.columns:
        var["gene_id"] = var[gene_id_key].astype(str).to_numpy()
    elif "gene_id" in var.columns:
        var["gene_id"] = var["gene_id"].astype(str).to_numpy()
    else:
        var["gene_id"] = var.index.astype(str)
    return var


def _matrix_from_adata(adata: AnnData, *, layer: Optional[str] = None, raw: bool = False):
    if raw:
        if adata.raw is None:
            raise ValueError("raw=True was requested, but adata.raw is None")
        return adata.raw.X
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not found in adata.layers")
        return adata.layers[layer]
    return adata.X


def _exp_table_from_adata(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    raw: bool = False,
    gene_id_key: str = "gene_id",
) -> pd.DataFrame:
    matrix = _matrix_from_adata(adata, layer=layer, raw=raw)
    gene_ids = _ensure_gene_id(adata.raw.var if raw else adata.var, gene_id_key)["gene_id"].astype(str).tolist()
    sample_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str).tolist()
    arr = matrix.T.toarray() if sparse.issparse(matrix) else np.asarray(matrix).T
    return pd.DataFrame(arr, columns=sample_ids).assign(gene_id=gene_ids)[["gene_id", *sample_ids]]


def _sparse_matrix(matrix: Any):
    if isinstance(matrix, (str, Path)):
        loaded = mmread(str(matrix))
        return loaded.tocsr() if sparse.issparse(loaded) else sparse.csr_matrix(np.asarray(loaded))
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def _gene_named_matrix_from_10x_inputs(matrix: Any, pheno: Any, feature: Any):
    if isinstance(matrix, (str, Path)) and isinstance(pheno, (str, Path)) and isinstance(feature, (str, Path)):
        return _load_sparse_matrix(matrix, feature, "gene_short_name", pheno, "cell.id")
    return None


def _normalize_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in {"plate", "bulk", "smartseq", "smart-seq", "smartseq2", "smart-seq2"}:
        return "plate"
    if key in {"10x", "droplet", "single-cell", "single_cell"}:
        return "droplet"
    raise ValueError("mode must be 'plate', 'droplet', or '10x'")


def _extract_result_tables(marvel_object: Any) -> dict[str, Any]:
    keys = (
        "psi",
        "psi_posterior",
        "counts",
        "splice_feature",
        "splice_feature_validated",
        "modality_results",
        "modality_prop",
        "modality_change",
        "de_splicing",
        "de_gene",
        "de_spliced_gene",
        "de_plots",
        "n_events",
        "distance_to_canonical",
        "parsed_gtf",
        "variable_splicing",
        "de_pctase",
        "de_absase",
        "de_a3ss_dist_to_ss",
        "pca_results",
        "value_plots",
        "pct_expr_gene",
        "pct_expr_sj",
        "de_sj",
        "de_sj_donor",
        "iso_switch",
        "adhoc_gene",
    )
    tables = {}
    for key in keys:
        if hasattr(marvel_object, key):
            value = getattr(marvel_object, key)
            if value is not None:
                tables[key] = value
    return tables


def _write_plate_psi_obsm(adata: AnnData, marvel_object: Any, *, result_key: str) -> dict[str, str]:
    obsm_keys: dict[str, str] = {}
    sample_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str).tolist()
    for event_type, psi_df in getattr(marvel_object, "psi", {}).items():
        if psi_df is None or psi_df.empty or "tran_id" not in psi_df.columns:
            continue
        event_ids = psi_df["tran_id"].astype(str).tolist()
        values = psi_df.set_index("tran_id")
        mat = np.full((adata.n_obs, len(event_ids)), np.nan, dtype=np.float32)
        for row_idx, sample_id in enumerate(sample_ids):
            if sample_id in values.columns:
                mat[row_idx, :] = pd.to_numeric(values[sample_id], errors="coerce").to_numpy(dtype=np.float32)
        key = f"X_{result_key}_psi_{str(event_type).lower()}"
        adata.obsm[key] = mat
        obsm_keys[str(event_type).upper()] = key
    return obsm_keys


def _reindex_obs_by_sample_id(adata: AnnData, sample_ids: Sequence[str]) -> None:
    obs_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str)
    obs_name_by_sample = dict(zip(obs_ids, adata.obs_names.astype(str)))
    ordered_obs_names = [obs_name_by_sample[sample_id] for sample_id in map(str, sample_ids) if sample_id in obs_name_by_sample]
    if ordered_obs_names and ordered_obs_names != adata.obs_names.astype(str).tolist():
        adata._inplace_subset_obs(ordered_obs_names)


def _reindex_var_by_gene_id(adata: AnnData, gene_ids: Sequence[str]) -> None:
    var_ids = _ensure_gene_id(adata.var)["gene_id"].astype(str)
    var_name_by_gene = dict(zip(var_ids, adata.var_names.astype(str)))
    ordered_var_names = [var_name_by_gene[gene_id] for gene_id in map(str, gene_ids) if gene_id in var_name_by_gene]
    if ordered_var_names and ordered_var_names != adata.var_names.astype(str).tolist():
        adata._inplace_subset_var(ordered_var_names)


def _sync_plate_anndata(adata: AnnData, marvel_object: Any) -> None:
    if hasattr(marvel_object, "splice_pheno") and "sample.id" in marvel_object.splice_pheno.columns:
        sample_ids = marvel_object.splice_pheno["sample.id"].astype(str).tolist()
        _reindex_obs_by_sample_id(adata, sample_ids)
        if len(adata.obs) == len(sample_ids) and adata.obs_names.astype(str).tolist() == sample_ids:
            adata.obs = marvel_object.splice_pheno.set_index("sample.id", drop=False).loc[sample_ids].copy()

    if hasattr(marvel_object, "gene_feature") and "gene_id" in marvel_object.gene_feature.columns:
        gene_ids = marvel_object.gene_feature["gene_id"].astype(str).tolist()
        _reindex_var_by_gene_id(adata, gene_ids)
        if len(adata.var) == len(gene_ids) and adata.var_names.astype(str).tolist() == gene_ids:
            adata.var = marvel_object.gene_feature.set_index("gene_id", drop=False).loc[gene_ids].copy()

    if hasattr(marvel_object, "exp") and "gene_id" in marvel_object.exp.columns and adata.n_obs and adata.n_vars:
        sample_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str).tolist()
        gene_ids = _ensure_gene_id(adata.var)["gene_id"].astype(str).tolist()
        exp = marvel_object.exp.set_index("gene_id")
        if set(sample_ids).issubset(exp.columns) and set(gene_ids).issubset(exp.index.astype(str)):
            exp.index = exp.index.astype(str)
            adata.X = exp.loc[gene_ids, sample_ids].apply(pd.to_numeric, errors="coerce").T.to_numpy()


def _sync_droplet_anndata(adata: AnnData, marvel_object: Any, *, count_layer: str = "counts") -> None:
    if hasattr(marvel_object, "sample_metadata") and "cell.id" in marvel_object.sample_metadata.columns:
        sample_metadata = marvel_object.sample_metadata.copy()
        sample_metadata["sample.id"] = sample_metadata["cell.id"].astype(str)
        sample_ids = sample_metadata["sample.id"].tolist()
        _reindex_obs_by_sample_id(adata, sample_ids)
        if len(adata.obs) == len(sample_ids) and adata.obs_names.astype(str).tolist() == sample_ids:
            adata.obs = sample_metadata.set_index("sample.id", drop=False).loc[sample_ids].copy()

    if hasattr(marvel_object, "gene_metadata") and "gene_short_name" in marvel_object.gene_metadata.columns:
        gene_metadata = marvel_object.gene_metadata.copy()
        gene_metadata["gene_id"] = gene_metadata["gene_short_name"].astype(str)
        gene_ids = gene_metadata["gene_id"].tolist()
        _reindex_var_by_gene_id(adata, gene_ids)
        if len(adata.var) == len(gene_ids) and adata.var_names.astype(str).tolist() == gene_ids:
            adata.var = gene_metadata.set_index("gene_id", drop=False).loc[gene_ids].copy()

    spec = adata.uns.get(DEFAULT_INPUT_KEY, {})
    sync_matrices = bool(spec.get("load_matrices")) if isinstance(spec, dict) else False
    if not sync_matrices:
        return

    if hasattr(marvel_object, "gene_norm_matrix"):
        norm = marvel_object.gene_norm_matrix
        cell_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str).tolist()
        gene_ids = _ensure_gene_id(adata.var)["gene_id"].astype(str).tolist()
        if cell_ids and gene_ids:
            adata.X = norm.subset_rows(gene_ids).subset_cols(cell_ids).matrix.T.tocsr()
    if hasattr(marvel_object, "gene_count_matrix") and count_layer in adata.layers:
        counts = marvel_object.gene_count_matrix
        cell_ids = _ensure_sample_id(adata.obs)["sample.id"].astype(str).tolist()
        gene_ids = _ensure_gene_id(adata.var)["gene_id"].astype(str).tolist()
        common_gene_ids = [gene_id for gene_id in gene_ids if gene_id in set(counts.row_ids.astype(str))]
        if cell_ids and common_gene_ids and len(common_gene_ids) == len(gene_ids):
            adata.layers[count_layer] = counts.subset_rows(gene_ids).subset_cols(cell_ids).matrix.T.tocsr()


def sync_anndata(adata: AnnData, marvel_object: Any, *, mode: str, result_key: str = DEFAULT_RESULT_KEY) -> None:
    """Synchronize mutable MARVEL backend state into AnnData axes and matrices."""
    if _normalize_mode(mode) == "plate":
        _sync_plate_anndata(adata, marvel_object)
        _write_plate_psi_obsm(adata, marvel_object, result_key=result_key)
    else:
        _sync_droplet_anndata(adata, marvel_object)


def setup_plate_anndata(
    *,
    exp: Any,
    splice_pheno: Any,
    splice_junction: Any,
    splice_feature: dict[str, Any],
    gene_feature: Any,
    intron_counts: Any = None,
    gtf: Any = None,
    input_key: str = DEFAULT_INPUT_KEY,
) -> AnnData:
    """Create an AnnData container for a plate/Smart-seq MARVEL workflow."""
    exp_df = _read_table(exp)
    pheno_df = _ensure_sample_id(_read_table(splice_pheno, dtype=str))
    gene_df = _ensure_gene_id(_read_table(gene_feature, dtype=str))

    if "gene_id" not in exp_df.columns:
        raise KeyError("exp must contain a 'gene_id' column")
    sample_ids = pheno_df["sample.id"].astype(str).tolist()
    missing_samples = [sample_id for sample_id in sample_ids if sample_id not in exp_df.columns]
    if missing_samples:
        raise KeyError(f"exp is missing sample columns: {missing_samples[:5]}")

    exp_indexed = exp_df.set_index("gene_id")
    gene_ids = gene_df["gene_id"].astype(str).tolist()
    common_gene_ids = [gene_id for gene_id in gene_ids if gene_id in exp_indexed.index]
    if not common_gene_ids:
        raise ValueError("No overlapping gene ids between exp and gene_feature")

    X = exp_indexed.loc[common_gene_ids, sample_ids].apply(pd.to_numeric, errors="coerce").T.to_numpy()
    obs = pheno_df.set_index("sample.id", drop=False).loc[sample_ids].copy()
    var = gene_df.set_index("gene_id", drop=False).loc[common_gene_ids].copy()
    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns[input_key] = {
        "mode": "plate",
        "splice_junction": splice_junction,
        "splice_feature": dict(splice_feature),
        "intron_counts": intron_counts,
        "gtf": gtf,
    }
    return adata


def setup_10x_anndata(
    *,
    gene_norm_matrix: Any,
    gene_norm_pheno: Any,
    gene_norm_feature: Any,
    gene_count_matrix: Any,
    gene_count_pheno: Any,
    gene_count_feature: Any,
    sj_count_matrix: Any,
    sj_count_pheno: Any,
    sj_count_feature: Any,
    pca: Any,
    gtf: Any,
    count_layer: str = "counts",
    input_key: str = DEFAULT_INPUT_KEY,
    load_matrices: bool = False,
) -> AnnData:
    """Create an AnnData container for a droplet/10x MARVEL workflow."""
    gene_pheno = _ensure_sample_id(_read_table(gene_norm_pheno, dtype=str), sample_id_key="cell.id")
    gene_feature = _ensure_gene_id(_read_table(gene_norm_feature, dtype=str), gene_id_key="gene_short_name")
    if load_matrices:
        gene_norm_named = _gene_named_matrix_from_10x_inputs(gene_norm_matrix, gene_norm_pheno, gene_norm_feature)
        gene_count_named = _gene_named_matrix_from_10x_inputs(gene_count_matrix, gene_count_pheno, gene_count_feature)
        gene_norm = gene_norm_named.matrix if gene_norm_named is not None else _sparse_matrix(gene_norm_matrix)
        gene_count = gene_count_named.matrix if gene_count_named is not None else _sparse_matrix(gene_count_matrix)
    else:
        shape = (len(gene_feature), len(gene_pheno))
        gene_norm = sparse.csr_matrix(shape, dtype=np.float32)
        gene_count = sparse.csr_matrix(shape, dtype=np.float32)

    obs = gene_pheno.set_index("sample.id", drop=False).copy()
    var = gene_feature.set_index("gene_id", drop=False).copy()
    adata = AnnData(X=gene_norm.T.tocsr(), obs=obs, var=var)
    adata.layers[count_layer] = gene_count.T.tocsr()
    adata.uns[input_key] = {
        "mode": "droplet",
        "gene_norm_matrix": gene_norm_matrix,
        "gene_norm_pheno": gene_norm_pheno,
        "gene_norm_feature": gene_norm_feature,
        "gene_count_matrix": gene_count_matrix,
        "gene_count_pheno": gene_count_pheno,
        "gene_count_feature": gene_count_feature,
        "sj_count_matrix": sj_count_matrix,
        "sj_count_pheno": sj_count_pheno,
        "sj_count_feature": sj_count_feature,
        "pca": pca,
        "gtf": gtf,
        "load_matrices": load_matrices,
    }
    return adata


class MARVEL:
    """AnnData-native MARVEL workflow controller.

    Parameters
    ----------
    adata
        Annotated data matrix. Gene expression is stored in ``adata.X`` and
        MARVEL-specific SJ/event inputs are stored in ``adata.uns[input_key]``.
    mode
        ``'plate'`` for plate/Smart-seq data or ``'10x'``/``'droplet'`` for
        droplet data.
    input_key
        ``adata.uns`` key with backend inputs.
    result_key
        ``adata.uns`` namespace used for results. Defaults to ``'marvel'``.
    layer
        Optional expression layer used for plate expression export.
    raw
        Use ``adata.raw`` for plate expression export.
    """

    def __init__(
        self,
        adata: AnnData,
        mode: str = "plate",
        *,
        input_key: str = DEFAULT_INPUT_KEY,
        result_key: str = DEFAULT_RESULT_KEY,
        layer: Optional[str] = None,
        raw: bool = False,
    ) -> None:
        if not isinstance(adata, AnnData):
            raise TypeError("MARVEL expects an AnnData object as the first argument")
        self.adata = adata
        self.mode = _normalize_mode(mode)
        self.input_key = input_key
        self.result_key = result_key
        self.layer = layer
        self.raw = raw
        self.object: Any = None

    @classmethod
    def from_plate(cls, **kwargs: Any) -> "MARVEL":
        input_key = kwargs.pop("input_key", DEFAULT_INPUT_KEY)
        result_key = kwargs.pop("result_key", DEFAULT_RESULT_KEY)
        adata = setup_plate_anndata(input_key=input_key, **kwargs)
        return cls(adata, mode="plate", input_key=input_key, result_key=result_key).build()

    @classmethod
    def from_10x(cls, **kwargs: Any) -> "MARVEL":
        input_key = kwargs.pop("input_key", DEFAULT_INPUT_KEY)
        result_key = kwargs.pop("result_key", DEFAULT_RESULT_KEY)
        adata = setup_10x_anndata(input_key=input_key, **kwargs)
        return cls(adata, mode="droplet", input_key=input_key, result_key=result_key).build()

    from_droplet = from_10x

    def __repr__(self) -> str:
        status = [f"MARVEL({self.adata.n_obs} cells x {self.adata.n_vars} genes, mode={self.mode!r})"]
        if self.object is not None:
            status.append(f"  backend: {self.object.__class__.__name__}")
        if self.result_key in self.adata.uns:
            status.append(f"  results: adata.uns[{self.result_key!r}]")
        return "\n".join(status)

    @property
    def result(self) -> Any:
        return self.object

    @property
    def marvel_object(self) -> Any:
        return self.object

    def build(self) -> "MARVEL":
        if self.mode == "plate":
            self.object = self._build_plate_object()
        else:
            self.object = self._build_droplet_object()
        self.write(sync=False)
        return self

    def run(
        self,
        *,
        events: Sequence[str] = ("SE", "MXE", "RI", "A5SS", "A3SS"),
        coverage_threshold: float = 10.0,
        uneven_coverage_multiplier: float = 10.0,
        read_length: float = 1.0,
    ) -> "MARVEL":
        """Run a compact default workflow and write results into AnnData."""
        self.build()
        if self.mode == "plate":
            self.check_alignment(level="SJ")
            for event_type in events:
                event_type = str(event_type).upper()
                feature = self.object.splice_feature.get(event_type)
                if feature is not None and len(feature) > 0:
                    self.compute_psi(
                        event_type=event_type,
                        coverage_threshold=coverage_threshold,
                        uneven_coverage_multiplier=uneven_coverage_multiplier,
                        read_length=read_length,
                    )
        else:
            self.annotate_genes_10x()
            self.annotate_sj_10x()
            self.validate_sj_10x()
            self.filter_genes_10x()
            self.check_alignment_10x()
        return self

    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        if self.object is None:
            self.build()
        if key is None:
            return self.object
        if hasattr(self.object, key):
            return getattr(self.object, key)
        return default

    def write(self, *, include_object: bool = False, sync: bool = True) -> "MARVEL":
        if self.object is None:
            self.build()
            return self
        if sync:
            sync_anndata(self.adata, self.object, mode=self.mode, result_key=self.result_key)
        obsm = _write_plate_psi_obsm(self.adata, self.object, result_key=self.result_key) if self.mode == "plate" else {}
        payload = {
            "mode": self.mode,
            "backend": "marvel_py",
            "input_key": self.input_key,
            "tables": _extract_result_tables(self.object),
        }
        if obsm:
            payload["obsm"] = obsm
        if include_object:
            payload["object"] = self.object
        self.adata.uns[self.result_key] = payload
        return self

    def call(self, function_name: str, *args: Any, update: bool = True, **kwargs: Any) -> Any:
        import marvel_py as mp

        if self.object is None:
            self.build()
        if not hasattr(mp, function_name):
            raise AttributeError(f"marvel_py has no public function {function_name!r}")
        result = getattr(mp, function_name)(self.object, *args, **kwargs)
        if update and result is not None:
            self.object = result
            self.write()
            return self
        return result

    def _build_plate_object(self):
        spec = self.adata.uns.get(self.input_key)
        if not isinstance(spec, dict):
            raise KeyError(f"adata.uns[{self.input_key!r}] must contain plate MARVEL inputs")
        missing = [key for key in ("splice_junction", "splice_feature") if key not in spec]
        if missing:
            raise KeyError(f"adata.uns[{self.input_key!r}] missing keys: {missing}")
        return create_marvel_object(
            splice_junction=spec["splice_junction"],
            splice_pheno=spec.get("splice_pheno", _ensure_sample_id(self.adata.obs)),
            splice_feature=spec["splice_feature"],
            intron_counts=spec.get("intron_counts"),
            gene_feature=spec.get("gene_feature", _ensure_gene_id(self.adata.var)),
            exp=spec.get("exp", _exp_table_from_adata(self.adata, layer=self.layer, raw=self.raw)),
            gtf=spec.get("gtf"),
        )

    def _build_droplet_object(self):
        spec = self.adata.uns.get(self.input_key)
        if not isinstance(spec, dict):
            raise KeyError(f"adata.uns[{self.input_key!r}] must contain droplet MARVEL inputs")
        required = (
            "gene_norm_matrix",
            "gene_norm_pheno",
            "gene_norm_feature",
            "gene_count_matrix",
            "gene_count_pheno",
            "gene_count_feature",
            "sj_count_matrix",
            "sj_count_pheno",
            "sj_count_feature",
            "pca",
            "gtf",
        )
        missing = [key for key in required if key not in spec]
        if missing:
            raise KeyError(f"adata.uns[{self.input_key!r}] missing keys: {missing}")
        return create_marvel_object_10x(**{key: spec[key] for key in required})

    def _step(self, name: str, *args: Any, **kwargs: Any) -> "MARVEL":
        import marvel_py as mp

        if self.object is None:
            self.build()
        result = getattr(mp, name)(self.object, *args, **kwargs)
        if result is not None:
            self.object = result
        self.write()
        return self

    # Plate workflow
    def check_alignment(self, **kwargs: Any) -> "MARVEL":
        return self._step("check_alignment", **kwargs)

    def subset_samples(self, **kwargs: Any) -> "MARVEL":
        return self._step("subset_samples", **kwargs)

    def transform_exp_values(self, **kwargs: Any) -> "MARVEL":
        return self._step("transform_exp_values", **kwargs)

    def detect_events(self, **kwargs: Any) -> "MARVEL":
        return self._step("detect_events", **kwargs)

    def compute_psi(self, **kwargs: Any) -> "MARVEL":
        return self._step("compute_psi", **kwargs)

    def compute_psi_posterior(self, **kwargs: Any) -> "MARVEL":
        return self._step("compute_psi_posterior", **kwargs)

    def assign_modality(self, **kwargs: Any) -> "MARVEL":
        return self._step("assign_modality", **kwargs)

    def count_events(self, **kwargs: Any) -> "MARVEL":
        return self._step("count_events", **kwargs)

    def prop_modality(self, **kwargs: Any) -> "MARVEL":
        return self._step("prop_modality", **kwargs)

    def compare_values(self, **kwargs: Any) -> "MARVEL":
        return self._step("compare_values", **kwargs)

    def run_pca(self, **kwargs: Any) -> "MARVEL":
        return self._step("run_pca", **kwargs)

    def plot_values(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_values", **kwargs)

    def plot_de_values(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_de_values", **kwargs)

    def modality_change(self, **kwargs: Any) -> "MARVEL":
        return self._step("modality_change", **kwargs)

    def iso_switch(self, **kwargs: Any) -> "MARVEL":
        return self._step("iso_switch", **kwargs)

    def identify_variable_events(self, **kwargs: Any) -> "MARVEL":
        return self._step("identify_variable_events", **kwargs)

    def pct_ase(self, **kwargs: Any) -> "MARVEL":
        return self._step("pct_ase", **kwargs)

    # Droplet / 10x workflow
    def annotate_genes_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("annotate_genes_10x", **kwargs)

    def annotate_sj_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("annotate_sj_10x", **kwargs)

    def validate_sj_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("validate_sj_10x", **kwargs)

    def filter_genes_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("filter_genes_10x", **kwargs)

    def check_alignment_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("check_alignment_10x", **kwargs)

    def compare_values_sj_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("compare_values_sj_10x", **kwargs)

    def compare_values_genes_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("compare_values_genes_10x", **kwargs)

    def compare_values_sj_donor_level_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("compare_values_sj_donor_level_10x", **kwargs)

    def plot_pct_expr_cells_genes_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_pct_expr_cells_genes_10x", **kwargs)

    def plot_pct_expr_cells_sj_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_pct_expr_cells_sj_10x", **kwargs)

    def plot_de_values_genes_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_de_values_genes_10x", **kwargs)

    def plot_de_values_sj_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("plot_de_values_sj_10x", **kwargs)

    def iso_switch_10x(self, **kwargs: Any) -> "MARVEL":
        return self._step("iso_switch_10x", **kwargs)


Splicing = MARVEL
