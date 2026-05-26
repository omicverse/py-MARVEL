from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marvel_py import assign_modality, compute_psi, transform_exp_values
from marvel_py.io import create_marvel_object
from tests._phase1_r_reference import load_shared_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "tests" / "r_reference" / "phase3_plate"


def read_reference(name: str, *, dtype=None) -> pd.DataFrame:
    return pd.read_csv(REFERENCE_ROOT / name, sep="\t", dtype=dtype)


def read_manifest() -> dict[str, object]:
    return json.loads((REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def build_shared_plate_object():
    marvel = create_marvel_object(**load_shared_inputs())
    marvel = transform_exp_values(
        marvel,
        offset=1.0,
        transformation="log2",
        threshold_lower=1.0,
    )
    marvel = compute_psi(marvel, coverage_threshold=1, event_type="SE")
    marvel = assign_modality(marvel, sample_ids=["s1", "s2", "s3", "s4"], min_cells=1, seed=1)
    return marvel
