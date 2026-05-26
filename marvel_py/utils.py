from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*) "([^"]*)"')


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def maybe_read_table(value: Any, **kwargs) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if kwargs:
            unsupported = sorted(set(kwargs) - {"dtype"})
            if unsupported:
                keys = ", ".join(unsupported)
                raise ValueError(f"maybe_read_table() does not accept DataFrame kwargs other than dtype: {keys}")
            dtype = kwargs.pop("dtype")
            result = value.copy()
            return result.astype(dtype) if dtype is not None else result
        return value.copy()
    if isinstance(value, (str, Path)):
        return read_table(value, **kwargs)
    raise TypeError(f"Expected a DataFrame or tabular path, got {type(value)!r}")


def ordered_intersection(left: Iterable[str], right: Iterable[str]) -> list[str]:
    right_set = set(right)
    return [value for value in left if value in right_set]


def extract_gtf_attr(value: str, key: str) -> str | None:
    attrs = dict(ATTR_RE.findall(value))
    return attrs.get(key)
