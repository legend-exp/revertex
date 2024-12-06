from __future__ import annotations

import importlib.metadata

import revert as m


def test_package():
    assert importlib.metadata.version("revert") == m.__version__
