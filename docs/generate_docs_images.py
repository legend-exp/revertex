"""Render the musun-gs sampling surfaces overlaid on the LEGEND-1000 geometry.

For each preset in :data:`revertex.generators.musun_gs.DEFAULT_DIMENSIONS` and
each view (``front`` and ``top``), this script:

1. builds the LEGEND-1000 geometry with :mod:`pygeoml1000` and renders it in a
   **parallel (orthographic) projection** to a temporary PNG via
   :mod:`pygeomtools.viewer`;
2. overlays the muon sampling cuboid as a clean rectangle with dimension labels,
   drawn on top of the rendered image with :mod:`matplotlib`.

Because the projection is orthographic and the camera is focused on the cuboid
centre, the world -> pixel mapping is a simple affine transform fully determined
by the camera and the parallel scale (which we set explicitly, see
``_projector``). The cuboid is centred on the focus, so its projected rectangle
is symmetric about the image centre and the overlay stays aligned regardless of
the viewer's internal sign conventions.

The dimensions are imported from the source of truth so the figures cannot drift
from the code.

Run it manually from the ``docs`` directory whenever the sampling dimensions
change, then **commit** the generated PNGs (they are built into the HTML on every
push; this script is not run in CI):

.. code:: console

    > cd docs
    > python generate_docs_images.py            # needs a display, or:
    > xvfb-run -a python generate_docs_images.py

Requirements (not installed in the CI docs environment):

- ``revertex`` and ``legend-pygeom-l1000`` (``pip install '.[imagegen]'``)
- a VTK-capable display (or ``xvfb-run`` / offscreen VTK) — a system dependency
- ``legend-metadata`` available for ``pygeoml1000.core.construct()``
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pyg4ometry import config as meshconfig
from pygeoml1000 import core
from pygeomtools import viewer, write_pygeom

from revertex.generators.musun_gs import DEFAULT_DIMENSIONS

logging.basicConfig()
meshconfig.setGlobalMeshSliceAndStack(100)

# cm -> mm (Geant4/pyg4ometry native length unit)
_CM_TO_MM = 10.0

# square render window (px). a square window keeps the parallel projection
# isotropic (world half-width == world half-height == parallel scale).
_WINDOW_PX = 1000

# camera stand-off distance [mm]. irrelevant for parallel projection (only the
# parallel scale sets the zoom), just needs to sit outside the geometry.
_CAMERA_DISTANCE_MM = 1.0e6

# fraction of the frame the cuboid should fill (the rest is margin).
_FILL_FRACTION = 0.82

_ORANGE = "#ff6600"

# view definitions: viewing direction, screen "up", and which cuboid dimension
# maps to the horizontal / vertical screen axis (for the dimension labels).
_VIEWS = {
    "front": {
        "dir": (1.0, 0.0, 0.0),  # look along +x
        "up": (0.0, 0.0, 1.0),  # z up
        "h": ("dy_cm", "y"),
        "v": ("dz_cm", "z"),
    },
    "top": {
        "dir": (0.0, 0.0, -1.0),  # look straight down
        "up": (0.0, 1.0, 0.0),  # y up
        "h": ("dx_cm", "x"),
        "v": ("dy_cm", "y"),
    },
}


def _corners_mm(dims: dict) -> np.ndarray:
    """Return the 8 cuboid corner coordinates in mm."""
    c = (
        np.array([dims["center_x_cm"], dims["center_y_cm"], dims["center_z_cm"]])
        * _CM_TO_MM
    )
    h = np.array([dims["dx_cm"], dims["dy_cm"], dims["dz_cm"]]) * _CM_TO_MM / 2
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    return c + signs * h


def _projector(pos, focus, up, scale: float, w: int, h: int):
    """Build a world(mm) -> pixel mapping for an orthographic VTK camera.

    ``scale`` is the VTK parallel scale (world half-height of the viewport). For
    a square window the horizontal half-width equals ``scale`` too.
    """
    focus = np.asarray(focus, float)
    cam_z = np.asarray(pos, float) - focus  # out of screen (== -direction)
    cam_z /= np.linalg.norm(cam_z)
    up = np.asarray(up, float)
    up = up - np.dot(up, cam_z) * cam_z
    up /= np.linalg.norm(up)
    right = np.cross(up, cam_z)  # VTK convention: right = up x viewPlaneNormal
    aspect = w / h

    def world_to_px(p) -> tuple[float, float]:
        d = np.asarray(p, float) - focus
        sx = float(np.dot(d, right))
        sy = float(np.dot(d, up))
        px = (sx / (scale * aspect) * 0.5 + 0.5) * w
        py = (0.5 - sy / scale * 0.5) * h
        return px, py

    return world_to_px


def _render_geometry(registry, pos, focus, up, scale: float, out_png: str) -> None:
    """Render the (already coloured) geometry in parallel projection to a PNG."""
    vis_scene = {
        "window_size": [_WINDOW_PX, _WINDOW_PX],
        "default": {
            "focus": list(map(float, focus)),
            "up": list(map(float, up)),
            "camera": list(map(float, pos)),
            # a numeric "parallel" turns on orthographic projection AND sets the
            # parallel scale (see pygeomtools.viewer._set_camera).
            "parallel": float(scale),
        },
        "color_overrides": {
            # soften the outer water tank so the cryostat/strings stay visible
            "water.*": [0.0, 0.5, 1.0, 0.05],
        },
        "export_scale": 1,
        "export_and_exit": out_png,
    }
    viewer.visualize(registry, vis_scene)


def _draw_overlay(bg_png: str, out_png: str, w2px, dims: dict, view: dict) -> None:
    """Overlay the projected cuboid rectangle and dimension labels."""
    img = plt.imread(bg_png)
    h_px, w_px = img.shape[0], img.shape[1]

    pts = np.array([w2px(c) for c in _corners_mm(dims)])
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(img, extent=[0, w_px, h_px, 0])
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=_ORANGE, lw=1.0)
    )

    pad = 0.03 * w_px

    def _label(key_axis) -> str:
        key, axis = key_axis
        return f"${axis}$ = {dims[key] / 100:g} m"

    # horizontal dimension arrow + label, below the rectangle
    ax.annotate(
        "",
        xy=(x0, y1 + pad),
        xytext=(x1, y1 + pad),
        arrowprops={"arrowstyle": "<->", "color": _ORANGE, "lw": 1.5},
    )
    ax.text(
        (x0 + x1) / 2,
        y1 + 1.6 * pad,
        _label(view["h"]),
        ha="center",
        va="top",
        color=_ORANGE,
        fontsize=11,
    )
    # vertical dimension arrow + label, left of the rectangle
    ax.annotate(
        "",
        xy=(x0 - pad, y0),
        xytext=(x0 - pad, y1),
        arrowprops={"arrowstyle": "<->", "color": _ORANGE, "lw": 1.5},
    )
    ax.text(
        x0 - 1.6 * pad,
        (y0 + y1) / 2,
        _label(view["v"]),
        ha="right",
        va="center",
        rotation=90,
        color=_ORANGE,
        fontsize=11,
    )

    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def export_image(preset: str, dims: dict) -> None:
    """Render one preset of ``DEFAULT_DIMENSIONS`` over the l1000 geometry."""
    registry = core.construct(
        detail_level="cosmogenic", assemblies=["cavern", "watertank", "cryostat"]
    )
    write_pygeom(registry, None)

    corners = _corners_mm(dims)
    center = corners.mean(axis=0)

    for view_name, view in _VIEWS.items():
        direction = np.asarray(view["dir"], float)
        direction /= np.linalg.norm(direction)
        pos = center - direction * _CAMERA_DISTANCE_MM
        focus = center

        # size the parallel scale so the cuboid fills _FILL_FRACTION of the frame.
        w2px_unit = _projector(pos, focus, view["up"], 1.0, _WINDOW_PX, _WINDOW_PX)
        # projected extent (in "scale=1" world units) of the corners about centre
        ext = np.array([w2px_unit(c) for c in corners]) - _WINDOW_PX / 2
        half_extent = np.abs(ext).max() / (_WINDOW_PX / 2)  # in units of "scale"
        scale = half_extent / _FILL_FRACTION
        w2px = _projector(pos, focus, view["up"], scale, _WINDOW_PX, _WINDOW_PX)

        out_png = f"source/generators/images/musun_gs_sampling_{preset}_{view_name}.png"
        with tempfile.TemporaryDirectory() as tmp:
            bg_png = str(Path(tmp) / "bg.png")
            _render_geometry(registry, pos, focus, view["up"], scale, bg_png)
            _draw_overlay(bg_png, out_png, w2px, dims, view)


for _preset, _dims in DEFAULT_DIMENSIONS.items():
    export_image(_preset, _dims)
