# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""raster_rectify: crooked raster log calibration, 2-D flattening, and ML dataset capture.

Models:
- DepthTieLine: a depth reference line across the sheet with tilt/angle or multi-point curvature
  (for warped or folded scans), optional gap flag (skipping intervals with missing/discontinuous
  sections), and synthetic flag (temporary header/footer nodes used during straightening).
- TrackEdgeGuide: left and right boundary polylines for each track (gamma, resistivity, porosity, or a
  center depth column), with curve-bridging vertical line tracking between user-picked nodes.
- RasterCalibration: combines depth tie-lines and track boundaries into a full 2-D
  grid_rectify.Lattice that maps angled/curved depth lines to 90 degrees (horizontal) against
  straightened vertical track boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from pylogdigitizer.grid_rectify import Lattice, rectify_image


@dataclass
class DepthTieLine:
    """A depth reference line across the log sheet.

    ``nodes`` is a list of ``(x, y)`` sheet points sorted by ``x``. Two points define a tilted
    straight line; >= 3 points define a piecewise curve across a warped or folded scan.
    """
    depth: float
    nodes: list[tuple[float, float]]
    is_gap: bool = False          # True = interval below this line up to next line is a skipped gap
    synthetic: bool = False       # True = temporary header/footer node used only for straightening

    def __post_init__(self):
        pts = sorted((float(x), float(y)) for x, y in self.nodes)
        if len(pts) < 2:
            raise ValueError("DepthTieLine needs at least 2 (x, y) points")
        self.depth = float(self.depth)
        self.nodes = pts

    @classmethod
    def from_angle(cls, depth: float, x_ref: float, y_ref: float, angle_deg: float,
                   sheet_width: int, is_gap: bool = False, synthetic: bool = False) -> "DepthTieLine":
        """Create a tilted straight tie-line through ``(x_ref, y_ref)`` at ``angle_deg`` from horizontal."""
        slope = float(np.tan(np.deg2rad(angle_deg)))
        y0 = y_ref + slope * (0.0 - x_ref)
        y1 = y_ref + slope * (float(sheet_width - 1) - x_ref)
        return cls(depth=depth, nodes=[(0.0, y0), (float(sheet_width - 1), y1)],
                   is_gap=is_gap, synthetic=synthetic)

    @property
    def angle_deg(self) -> float:
        (x0, y0), (x1, y1) = self.nodes[0], self.nodes[-1]
        return float(np.rad2deg(np.arctan2(y1 - y0, max(1e-6, x1 - x0))))

    def y_at(self, x) -> np.ndarray:
        """Evaluate sheet row ``y`` at column(s) ``x`` with linear end extrapolation."""
        xx = np.asarray(x, float)
        nx = np.array([p[0] for p in self.nodes], float)
        ny = np.array([p[1] for p in self.nodes], float)
        yy = np.interp(xx, nx, ny)
        lo, hi = xx < nx[0], xx > nx[-1]
        if lo.any():
            s0 = (ny[1] - ny[0]) / max(1e-6, nx[1] - nx[0])
            yy = np.where(lo, ny[0] + (xx - nx[0]) * s0, yy)
        if hi.any():
            s1 = (ny[-1] - ny[-2]) / max(1e-6, nx[-1] - nx[-2])
            yy = np.where(hi, ny[-1] + (xx - nx[-1]) * s1, yy)
        return yy

    def shift(self, dy: float = 0.0, d_angle_deg: float = 0.0, x_pivot: float | None = None) -> "DepthTieLine":
        """Shift vertically by ``dy`` px and/or rotate around ``x_pivot`` by ``d_angle_deg``."""
        nx = np.array([p[0] for p in self.nodes], float)
        ny = np.array([p[1] for p in self.nodes], float) + float(dy)
        if abs(d_angle_deg) > 1e-9:
            xp = float(np.mean(nx)) if x_pivot is None else float(x_pivot)
            ny = ny + np.tan(np.deg2rad(d_angle_deg)) * (nx - xp)
        return DepthTieLine(self.depth, list(zip(nx, ny)), is_gap=self.is_gap, synthetic=self.synthetic)

    def to_dict(self) -> dict:
        return dict(depth=self.depth, nodes=[list(p) for p in self.nodes],
                    is_gap=self.is_gap, synthetic=self.synthetic)

    @classmethod
    def from_dict(cls, d: dict) -> "DepthTieLine":
        return cls(depth=d["depth"], nodes=[tuple(p) for p in d["nodes"]],
                   is_gap=bool(d.get("is_gap", False)), synthetic=bool(d.get("synthetic", False)))


@dataclass
class TrackEdgeGuide:
    """Left and right boundary polylines for one log track (or center depth column)."""
    name: str
    kind: str = "track"   # "track" or "depth"
    left_nodes: list[tuple[float, float]] = field(default_factory=list)   # (x, y) sorted by y
    right_nodes: list[tuple[float, float]] = field(default_factory=list)  # (x, y) sorted by y

    def __post_init__(self):
        self.left
