from __future__ import annotations

import importlib.metadata

import revertex as m


def test_package():
    assert importlib.metadata.version("revertex") == m.__version__
