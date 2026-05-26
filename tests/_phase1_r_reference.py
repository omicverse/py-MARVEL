from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "tests" / "r_reference" / "plate_phase1"
INPUT_ROOT = REFERENCE_ROOT / "inputs"
EVENT_TYPES = ("SE", "MXE", "RI", "A5SS", "A3SS", "AFE", "ALE")


def read_input(name: str, *, dtype=None) -> pd.DataFrame:
    return pd.read_csv(INPUT_ROOT / name, sep="\t", dtype=dtype)


def read_reference(name: str, *, dtype=None) -> pd.DataFrame:
    return pd.read_csv(REFERENCE_ROOT / name, sep="\t", dtype=dtype)


def load_shared_inputs() -> dict[str, object]:
    return {
        "splice_pheno": read_input("splice_pheno.tsv", dtype=str),
        "splice_junction": read_input("splice_junction.tsv"),
        "splice_feature": {
            "SE": read_input("splice_feature_se.tsv", dtype=str),
            **{event_type: None for event_type in EVENT_TYPES if event_type != "SE"},
        },
        "intron_counts": None,
        "gene_feature": read_input("gene_feature.tsv", dtype=str),
        "exp": read_input("exp.tsv"),
        "gtf": None,
    }
