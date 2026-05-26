from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

import marvel_py as mp
from marvel_py import MARVEL, setup_10x_anndata, setup_plate_anndata


def test_setup_plate_anndata_creates_scanpy_shaped_container():
    adata = setup_plate_anndata(
        exp=pd.DataFrame({"gene_id": ["gene1", "gene2"], "cell1": [1, 2], "cell2": [3, 4]}),
        splice_pheno=pd.DataFrame({"sample.id": ["cell1", "cell2"], "cell.type": ["A", "B"]}),
        splice_junction=pd.DataFrame({"coord.intron": ["chr1:1-2"]}),
        splice_feature={"SE": pd.DataFrame({"tran_id": ["event1"]})},
        gene_feature=pd.DataFrame({"gene_id": ["gene1", "gene2"], "gene_short_name": ["G1", "G2"]}),
    )

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["cell1", "cell2"]
    assert adata.var_names.tolist() == ["gene1", "gene2"]
    assert adata.uns["marvel_input"]["mode"] == "plate"


def test_marvel_plate_builds_from_anndata_and_writes_uns():
    adata = setup_plate_anndata(
        exp=pd.DataFrame({"gene_id": ["gene1"], "cell1": [1.0], "cell2": [2.0]}),
        splice_pheno=pd.DataFrame({"sample.id": ["cell1", "cell2"]}),
        splice_junction=pd.DataFrame({"coord.intron": ["chr1:1-2"]}),
        splice_feature={"SE": pd.DataFrame({"tran_id": ["event1"]})},
        gene_feature=pd.DataFrame({"gene_id": ["gene1"]}),
    )

    marvel = MARVEL(adata, mode="plate").build()

    assert marvel.object.__class__.__name__ == "MarvelPlate"
    assert adata.uns["marvel"]["mode"] == "plate"
    assert adata.uns["marvel"]["backend"] == "marvel_py"
    assert "tables" in adata.uns["marvel"]


def test_public_function_api_accepts_anndata_and_keeps_state():
    adata = setup_plate_anndata(
        exp=pd.DataFrame({"gene_id": ["gene1"], "cell1": [64.0], "cell2": [32.0]}),
        splice_pheno=pd.DataFrame({"sample.id": ["cell1", "cell2"]}),
        splice_junction=pd.DataFrame({"coord.intron": ["chr1:1-2"], "cell1": [1], "cell2": [2]}),
        splice_feature={"SE": pd.DataFrame({"tran_id": ["event1"]})},
        gene_feature=pd.DataFrame({"gene_id": ["gene1"]}),
    )

    returned = mp.subset_samples(adata, sample_ids=["cell1"])
    mp.transform_exp_values(adata, offset=1.0, transformation="log2", threshold_lower=1.0)

    assert returned is adata
    assert "object" not in adata.uns["marvel"]
    backend = mp._controller_from_anndata(adata).object
    assert backend.splice_pheno["sample.id"].tolist() == ["cell1"]
    assert backend.exp.columns.tolist() == ["gene_id", "cell1"]
    assert backend.exp["cell1"].iloc[0] == np.log2(65.0)
    assert adata.obs_names.tolist() == ["cell1"]
    assert adata.shape == (1, 1)
    assert adata.X[0, 0] == np.log2(65.0)


def test_marvel_plate_writes_psi_to_obsm():
    adata = AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame(index=["gene1"]),
    )
    adata.uns["marvel_input"] = {
        "splice_junction": pd.DataFrame({"coord.intron": ["chr1:1-2"]}),
        "splice_feature": {"SE": pd.DataFrame({"tran_id": ["event1"]})},
    }
    marvel = MARVEL(adata, mode="plate").build()
    marvel.object.psi["SE"] = pd.DataFrame({"tran_id": ["event1"], "cell1": [0.25], "cell2": [0.75]})

    marvel.write()

    assert "X_marvel_psi_se" in adata.obsm
    np.testing.assert_allclose(adata.obsm["X_marvel_psi_se"], np.array([[0.25], [0.75]], dtype=np.float32))
    assert adata.uns["marvel"]["obsm"]["SE"] == "X_marvel_psi_se"


def test_setup_10x_anndata_creates_counts_layer_and_input_namespace():
    matrix = sparse.csr_matrix([[1, 0], [2, 3]])  # genes x cells
    adata = setup_10x_anndata(
        gene_norm_matrix=matrix,
        gene_norm_pheno=pd.DataFrame({"cell.id": ["cell1", "cell2"]}),
        gene_norm_feature=pd.DataFrame({"gene_short_name": ["gene1", "gene2"]}),
        gene_count_matrix=matrix,
        gene_count_pheno=pd.DataFrame({"cell.id": ["cell1", "cell2"]}),
        gene_count_feature=pd.DataFrame({"gene_short_name": ["gene1", "gene2"]}),
        sj_count_matrix=sparse.csr_matrix([[1, 0]]),
        sj_count_pheno=pd.DataFrame({"cell.id": ["cell1", "cell2"]}),
        sj_count_feature=pd.DataFrame({"coord.intron": ["chr1:1-2"]}),
        pca=pd.DataFrame({"cell.id": ["cell1", "cell2"], "PC1": [0.0, 1.0]}),
        gtf=pd.DataFrame({f"V{i}": [] for i in range(1, 10)}),
        load_matrices=True,
    )

    assert adata.shape == (2, 2)
    assert "counts" in adata.layers
    assert adata.layers["counts"].shape == (2, 2)
    assert adata.uns["marvel_input"]["mode"] == "droplet"
