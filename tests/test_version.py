from __future__ import annotations

import re
from pathlib import Path

import marvel_py


def test_package_version_is_0_1_1() -> None:
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject_text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == "0.1.1"
    assert marvel_py.__version__ == "0.1.1"
